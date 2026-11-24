"""Universal slot reference relations with one common lane envelope."""

from fractions import Fraction

from ..ir.gadget_ir import AUX_BITS, Expr, Freedom, MicroprogramSlot
from .blocks import _dot, _exp

OP_DENSE_GQA = "dense_gqa"
OP_SWA_SINK = "swa_sink"
OP_MOE_ROUTER_EXPERT = "moe_router_expert"
OP_MLA_ABSORBED = "mla_absorbed"
OP_GDN_LREC = "gdn_lrec"


def swa_indices(position, window, sink):
    if position < 0 or window < 1 or sink < 0:
        raise ValueError("invalid causal mask")
    end = position + 1
    start = max(0, end-window)
    return tuple(range(min(sink, start))) + tuple(range(start, end))


def _nearest(value):
    return round(value)


def _exp_cell(delta, scale):
    if delta > 0:
        return None
    low, high = _exp(Fraction(delta, scale))
    a, b = _nearest(low*scale), _nearest(high*scale)
    return a if a == b else None


def _softmax(scores, values, scale):
    maximum = max(scores)
    weights = tuple(_exp_cell(score-maximum, scale) for score in scores)
    if None in weights:
        return None
    return _nearest(Fraction(_dot(weights, values), sum(weights)))


def install_microprogram(program, name, *, tokens, experts, top_k, window,
                         sink, conv_width, tau=0, scale=256, extent=None):
    """Install five branch relations over fixed-point scalar reference lanes.

    Dense and MLA each represent a head component and GDN one state
    component. Shapes are profile caps shared across every selected opcode.

    Every algebraic branch condition is an explicit `program.equal` gate
    multiplied by the branch selector (so an unselected branch constrains
    nothing); see `_lift`. The five `<slot>.<opcode>.relation` predicates
    keep only the exponential enclosures and nearest-rounding divisions and
    do NOT re-check the lifted conditions.
    """
    extent = tokens if extent is None else extent
    if (min(tokens, experts, top_k, window, conv_width, extent) < 1
            or top_k > experts or extent > tokens or sink < 0 or scale < 2 or tau < 0):
        raise ValueError("invalid common slot profile")

    def dense(normalize, query, *row):
        # `normalize` in {0, 1} is the gate `<slot>.dense_gqa.normalize_boolean`.
        keys, values = row[:tokens], row[tokens:2*tokens]
        if normalize:
            # Scalar Q/K normalization, including the zero lane.
            def normalize_scalar(x):
                return scale if x > 0 else -scale if x < 0 else 0
            query = normalize_scalar(query)
            keys = tuple(map(normalize_scalar, keys))
        scores = tuple(_nearest(Fraction(query*k, scale)) for k in keys[:extent])
        return row[-1] == _softmax(scores, values[:extent], scale)

    def swa(position, query, *row):
        # The guard keeps the mask lookup total; 0 <= position < extent is also the gate pair
        # `<slot>.swa_sink.position_low` / `.position_high`.
        if not 0 <= position < extent:
            return False
        active = swa_indices(position, window, sink)
        scores = tuple(_nearest(Fraction(query*row[i], scale)) for i in active)
        return row[-1] == _softmax(scores, tuple(row[tokens+i] for i in active), scale)

    def moe(*row):
        # Bitmap / argmax booleanity, one attaining maximum, the ascending index list, top-k slack,
        # zero / positive weights and the denominator sum are gates (`_lift`); what remains is the
        # exponential cell of every enabled expert and the nearest-rounded mixture.
        scores, active, _, weights, outputs = (
            row[i*experts:(i+1)*experts] for i in range(5)
        )
        maximum, denominator, result = row[-3:]
        if denominator <= 0:        # totality of the division below; implied by the gates
            return False
        return (all(not enabled or weight == _exp_cell(score-maximum, scale)
                    for score, enabled, weight in zip(scores, active, weights, strict=True))
                and result == _nearest(Fraction(_dot(weights, outputs), denominator)))

    def mla(query, rope_query, absorbed_value, *row):
        latent, rope_keys = row[:tokens], row[tokens:2*tokens]
        scores = tuple(_nearest(Fraction(query*k+rope_query*r, scale))
                       for k, r in zip(latent[:extent], rope_keys[:extent], strict=True))
        result = _softmax(scores, latent[:extent], scale)
        return result is not None and row[-1] == _nearest(Fraction(result*absorbed_value, scale))

    def gdn(position, in_range, previous, decay, beta, key, value, *row):
        # Position range, in_range = [position < extent], the out-of-range identities, the decay / beta
        # bounds and the zero history tail are gates (`_lift`); the two nearest-rounded updates remain.
        if not in_range:
            return True
        kernel, history = row[:conv_width], row[conv_width:2*conv_width]
        convolution, state = row[-2:]
        if convolution != _nearest(Fraction(_dot(kernel, history), scale)):
            return False
        decayed = Fraction(decay*previous, scale)
        correction = Fraction(beta*key, scale**2) * (value+convolution - key*decayed/scale)
        return state == _nearest(decayed+correction)

    branches = (
        (OP_DENSE_GQA, 2*tokens+3, dense),
        (OP_SWA_SINK, 2*tokens+3, swa),
        (OP_MOE_ROUTER_EXPERT, 5*experts+top_k+3, moe),
        (OP_MLA_ABSORBED, 2*tokens+4, mla),
        (OP_GDN_LREC, 2*conv_width+9, gdn),
    )
    slot = MicroprogramSlot.install(program, name, branches, bits=32)
    _lift(program, name, slot, tokens=tokens, experts=experts, top_k=top_k, conv_width=conv_width,
          tau=tau, scale=scale, extent=extent)
    return slot


def _lift(program, name, slot, *, tokens, experts, top_k, conv_width, tau, scale, extent):
    """Selector-gated algebraic gates of the dense / SWA / MoE / GDN branches (degree <= 2).

    `g = s * lane` product cells (`Program.product`, 7-bit) carry the selector into degree-2 gates;
    inequalities are `Program.nonneg` slack cells whose honest value is 0 when the branch is unselected.
    Gate names are listed in `circuit/lower.py`.
    """
    select = dict(zip((OP_DENSE_GQA, OP_SWA_SINK, OP_MOE_ROUTER_EXPERT, OP_MLA_ABSORBED, OP_GDN_LREC),
                      (Expr("wire", (selector,)) for selector in slot.selectors), strict=True))
    lane = tuple(Expr("wire", (lane,)) for lane in slot.lanes)
    span = (1 << 32) - 1            # |difference| of two 32-bit lanes

    # ---- dense_gqa: normalize in {0, 1}
    prefix = f"{name}.{OP_DENSE_GQA}"
    gated = program.product(prefix + ".normalize_gate", select[OP_DENSE_GQA], lane[0], name)
    program.equal(prefix + ".normalize_boolean", gated * (lane[0] - 1))

    # ---- swa_sink: 0 <= position < extent
    prefix, s = f"{name}.{OP_SWA_SINK}", select[OP_SWA_SINK]
    program.nonneg(prefix + ".position_low", s * lane[0], extent - 1, name)
    program.nonneg(prefix + ".position_high", s * (lane[0] * -1 + (extent - 1)), extent - 1, name)

    # ---- moe_router_expert
    prefix, m = f"{name}.{OP_MOE_ROUTER_EXPERT}", select[OP_MOE_ROUTER_EXPERT]
    score, active, argmax, weight = (lane[i*experts:(i+1)*experts] for i in range(4))
    index = lane[5*experts:5*experts+top_k]
    maximum, denominator = lane[5*experts+top_k], lane[5*experts+top_k+1]
    on = tuple(program.product(f"{prefix}.active_gate.{i}", m, active[i], name) for i in range(experts))
    best = tuple(program.product(f"{prefix}.argmax_gate.{i}", m, argmax[i], name) for i in range(experts))
    for i in range(experts):
        program.equal(f"{prefix}.argmax_boolean.{i}", best[i] * (argmax[i] - 1))
        program.equal(f"{prefix}.argmax_attains.{i}", best[i] * (score[i] - maximum))
        program.nonneg(f"{prefix}.maximum_bound.{i}", on[i] * (maximum - score[i]), span, name)
        program.nonneg(f"{prefix}.weight_positive.{i}", on[i] * (weight[i] - 1), span, name)
        if top_k < experts:
            # With top_k == experts the index gates below force index[r] = r, hence every gated bitmap
            # cell to equal the selector: the bitmap is all ones and these two families are vacuous.
            program.equal(f"{prefix}.argmax_selected.{i}", best[i] * (active[i] - 1))
            program.equal(f"{prefix}.weight_zero.{i}", (m - on[i]) * weight[i])
    program.equal(prefix + ".argmax_one", _sum(best), m)
    program.equal(prefix + ".denominator", m * (denominator - _sum(weight)))
    # Index list = ascending enumeration of the selected bitmap: a top_k x experts one-hot matrix whose
    # rows sum to the selector and whose columns sum to the gated bitmap, with row r selecting index[r]
    # and strictly increasing indices.  Consequences that therefore need NO gate of their own (a separate
    # gate would be implied, i.e. dead under row deletion): every gated bitmap cell is a sum of distinct
    # one-hot rows, so the bitmap is boolean, and it has exactly top_k ones (sum of the row sums).
    onehot = tuple(tuple(
        program.aux(f"{prefix}.index_onehot.{r}.{i}", AUX_BITS, "one-hot index cell", name,
                    lambda w, s=m, a=index[r], i=i: int(s.evaluate(w) == 1 and a.evaluate(w) == i))
        for i in range(experts)) for r in range(top_k))
    for r in range(top_k):
        for i in range(experts):
            program.equal(f"{prefix}.index_onehot_boolean.{r}.{i}", onehot[r][i] * (onehot[r][i] - 1))
        program.equal(f"{prefix}.index_row.{r}", _sum(onehot[r]), m)
        program.equal(f"{prefix}.index_value.{r}",
                      _sum([m * index[r]] + [onehot[r][i] * -i for i in range(1, experts)]))
        if r:
            program.nonneg(f"{prefix}.index_order.{r}", m * (index[r] - index[r-1] - 1), experts, name)
    for i in range(experts):
        program.equal(f"{prefix}.index_column.{i}", _sum(onehot[r][i] for r in range(top_k)), on[i])
    if top_k < experts:
        # Top-k slack: an integer threshold with selected scores >= threshold - tau and unselected
        # scores <= threshold + tau, i.e. unselected - selected <= 2 tau for every pair.
        def hint(w):
            chosen = [score[i].evaluate(w) for i in range(experts) if on[i].evaluate(w) == 1]
            return min(chosen) + tau if chosen else 0
        # Any threshold in [max unselected - tau, min selected + tau] is a valid certificate.
        threshold = program.aux(prefix + ".topk_threshold", 35, "top-k threshold", name, hint,
                                Freedom.PERMITTED_TIE)
        for i in range(experts):
            program.nonneg(f"{prefix}.topk_margin.{i}",
                           (on[i] * 2 - m) * (score[i] - threshold) + m * tau, 2*span + 2*tau, name)

    # ---- gdn_lrec
    prefix, g = f"{name}.{OP_GDN_LREC}", select[OP_GDN_LREC]
    position, in_range, previous, decay, beta = lane[:5]
    history = lane[7+conv_width:7+2*conv_width]
    convolution, state = lane[7+2*conv_width], lane[8+2*conv_width]
    inside = program.product(prefix + ".in_range_gate", g, in_range, name)
    outside = program.aux(prefix + ".out_range_gate", AUX_BITS, "gated complement", name,
                          lambda w: (g - inside).evaluate(w))
    program.equal(prefix + ".out_range_gate.def", outside, g - inside)
    program.equal(prefix + ".in_range_boolean", inside * (in_range - 1))
    program.nonneg(prefix + ".position_low", g * position, tokens - 1, name)
    program.nonneg(prefix + ".position_high", g * (position * -1 + (tokens - 1)), tokens - 1, name)
    # in_range = [position < extent]: extent - 1 - position >= 0 inside, position - extent >= 0 outside.
    program.nonneg(prefix + ".in_range_compare",
                   inside * (position * -1 + (extent - 1)) + outside * (position - extent), tokens, name)
    program.equal(prefix + ".out_range_convolution", outside * convolution)
    program.equal(prefix + ".out_range_state", outside * (state - previous))
    for label, cell in (("decay", decay), ("beta", beta)):
        program.nonneg(f"{prefix}.{label}_low", inside * cell, scale, name)
        program.nonneg(f"{prefix}.{label}_high", inside * (cell * -1 + scale), scale, name)
    for j in range(1, conv_width):
        # reached = [position >= j] when selected; an in-range history cell beyond the position is 0.
        reached = program.aux(f"{prefix}.tail_reached.{j}", AUX_BITS, "gated comparison", name,
                              lambda w, j=j: int(g.evaluate(w) == 1 and position.evaluate(w) >= j))
        program.equal(f"{prefix}.tail_reached_boolean.{j}", reached * (reached - 1))
        program.equal(f"{prefix}.tail_reached_selected.{j}", reached * (g - 1))
        program.nonneg(f"{prefix}.tail_compare.{j}",
                       reached * (position * 2 - (2*j - 1)) + g * (position * -1 + (j - 1)),
                       tokens + conv_width, name)
        beyond = program.aux(f"{prefix}.tail_gate.{j}", AUX_BITS, "gated complement", name,
                             lambda w, reached=reached: (inside - inside * reached).evaluate(w))
        program.equal(f"{prefix}.tail_gate.{j}.def", beyond, inside - inside * reached)
        program.equal(f"{prefix}.history_tail.{j}", beyond * history[j])


def _sum(terms):
    total = None
    for term in terms:
        total = term if total is None else total + term
    return total
