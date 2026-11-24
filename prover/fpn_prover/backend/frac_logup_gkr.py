"""Fractional LogUp with a Fiat–Shamir GKR reduction.

`prove`/`verify` are the public-input reference (SHAKE transcript, unchanged API).
`masked_prove`/`masked_verify` are the ZK-masked variant of relation 3
(specs/11-wire-format.md §5.4): Libra masks on every layer sumcheck, committed before
any challenge, root_num == 0 and root_den != 0 enforced, raw wire bytes absorbed into
`Fpn2Transcript`.
"""

from dataclasses import dataclass
from hashlib import shake_256

from flashprove_next.field import (
    ONE, P, ZERO, eq1, hadd, hf, hinv, hmul, hsub, lagrange_eval,
)


def encode(values):
    data = bytearray()
    for value in values:
        if type(value) is not tuple or len(value) != 3:
            raise ValueError("expected Fp3 element")
        for limb in value:
            if type(limb) is not int or not 0 <= limb < P:
                raise ValueError("noncanonical field limb")
            data.extend(limb.to_bytes(8, "little"))
    return bytes(data)


class Transcript:
    def __init__(self, tag):
        self.state = shake_256(tag)

    def absorb(self, data):
        self.state.update(len(data).to_bytes(8, "little") + data)

    def challenge(self):
        while True:
            word = self.state.digest(24)
            self.absorb(word)
            limbs = tuple(int.from_bytes(word[i:i+8], "little") for i in (0, 8, 16))
            if max(limbs) < P:
                return limbs


def combine(left, right):
    n, d = left
    m, e = right
    return hadd(hmul(n, e), hmul(m, d)), hmul(d, e)


def _mix(a, b, r):
    return hadd(a, hmul(r, hsub(b, a)))


def _fold(values, r):
    return [_mix(a, b, r) for a, b in zip(values[::2], values[1::2])]


def _evaluate(values, point):
    for r in point:
        values = _fold(values, r)
    return values[0]


def _begin(queries, table, multiplicities, salt):
    queries, table, multiplicities = tuple(queries), tuple(table), tuple(multiplicities)
    if len(table) != len(multiplicities) or not queries or not table:
        raise ValueError("invalid lookup dimensions")
    for group in (queries, table, multiplicities):
        if any(type(x) is not int or not 0 <= x < P for x in group):
            raise ValueError("invalid lookup scalar")
    if sum(multiplicities) != len(queries):
        raise ValueError("invalid multiplicity total")
    transcript = Transcript(b"fpn-frac-logup-gkr-v1")
    transcript.absorb(salt)
    for group in (queries, table, multiplicities):
        transcript.absorb(encode(map(hf, group)))
    beta = transcript.challenge()
    leaves = []
    for value, weight in zip(queries + table, (1,)*len(queries) + tuple(-x for x in multiplicities)):
        denominator = hsub(beta, hf(value))
        scale = transcript.challenge() if salt else ONE
        if denominator == ZERO or scale == ZERO:
            raise ValueError("zero leaf denominator")
        leaves.append((hmul(hf(weight), scale), hmul(denominator, scale)))
    size = 1 << (len(leaves)-1).bit_length()
    leaves.extend([(ZERO, ONE)] * (size - len(leaves)))
    return transcript, leaves


@dataclass(frozen=True)
class LayerProof:
    rounds: tuple
    children: tuple


@dataclass(frozen=True)
class Proof:
    root: tuple
    layers: tuple


def _weights(point):
    weights = [ONE]
    for r in reversed(point):
        weights = [hmul(w, factor) for w in weights for factor in (hsub(ONE, r), r)]
    return weights


def _round(columns, alpha):
    result = []
    for i in range(4):
        weights, nl, dl, nr, dr = (_fold(column, hf(i)) for column in columns)
        total = ZERO
        for w, a, b, c, d in zip(weights, nl, dl, nr, dr):
            n, den = combine((a, b), (c, d))
            total = hadd(total, hmul(w, hadd(n, hmul(alpha, den))))
        result.append(total)
    return tuple(result)


def _descend(transcript, children, point):
    transcript.absorb(encode(children))
    branch = transcript.challenge()
    nl, dl, nr, dr = children
    return (branch, *point), (_mix(nl, nr, branch), _mix(dl, dr, branch))


def prove(queries, table, multiplicities, salt=b""):
    transcript, leaves = _begin(queries, table, multiplicities, salt)
    tree = [leaves]
    while len(tree[-1]) > 1:
        nodes = tree[-1]
        tree.append([combine(a, b) for a, b in zip(nodes[::2], nodes[1::2])])
    root = tree[-1][0]
    transcript.absorb(encode(root))
    point = ()
    proofs = []
    for nodes in reversed(tree[:-1]):
        alpha = transcript.challenge()
        columns = [_weights(point), *map(list, zip(*nodes[::2])), *map(list, zip(*nodes[1::2]))]
        rounds = []
        reduced = []
        for _ in point:
            samples = _round(columns, alpha)
            transcript.absorb(encode(samples))
            r = transcript.challenge()
            rounds.append(samples)
            reduced.append(r)
            columns = [_fold(column, r) for column in columns]
        children = tuple(column[0] for column in columns[1:])
        proofs.append(LayerProof(tuple(rounds), children))
        point, _ = _descend(transcript, children, reduced)
    return Proof(root, tuple(proofs))


def verify(queries, table, multiplicities, proof, salt=b""):
    try:
        transcript, leaves = _begin(queries, table, multiplicities, salt)
        if len(proof.layers) != len(leaves).bit_length() - 1:
            return False
        claim = proof.root
        if len(claim) != 2 or claim[0] != ZERO or claim[1] == ZERO:
            return False
        transcript.absorb(encode(claim))
        point = ()
        for layer in proof.layers:
            if len(layer.rounds) != len(point) or len(layer.children) != 4:
                return False
            alpha = transcript.challenge()
            target = hadd(claim[0], hmul(alpha, claim[1]))
            reduced = []
            for samples in layer.rounds:
                if len(samples) != 4 or hadd(samples[0], samples[1]) != target:
                    return False
                transcript.absorb(encode(samples))
                r = transcript.challenge()
                target = lagrange_eval(samples, r)
                reduced.append(r)
            weight = ONE
            for a, b in zip(point, reduced):
                weight = hmul(weight, eq1(a, b))
            n, d = combine(layer.children[:2], layer.children[2:])
            if target != hmul(weight, hadd(n, hmul(alpha, d))):
                return False
            point, claim = _descend(transcript, layer.children, reduced)
        return claim == tuple(_evaluate(list(column), point) for column in zip(*leaves))
    except (ValueError, TypeError, AttributeError, IndexError):
        return False


# ============================================================================= ZK-masked FracLogUp-GKR (relation 3)
# Every layer sumcheck with k >= 1 rounds is masked Libra-style by
#     G_k(x) = sum_i g_{k,i}(x_i),  deg g_{k,i} = 3,
# whose 4k coefficients are cells of the committed vector F (bound under root_0 before any
# challenge).  The prover sends G_sum = sum_b G_k(b) before lambda_k is squeezed, runs the sumcheck
# on  eq(pt, b) (n(b) + alpha d(b)) + lambda_k G_k(b)  and finally sends g_r = G_k(r); the verifier
# defers g_r to the PCS batch, so an adaptively chosen g_r is caught by the commitment.
# The root must satisfy root_num == 0 and root_den != 0 (zero denominators cannot hide inside it).

class GkrReject(Exception):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class MaskedLayer:
    g_sum: object      # None for layer 0
    rounds: tuple      # k rounds of 4 samples (t = 0..3)
    children: tuple    # (n_left, d_left, n_right, d_right)
    g_r: object        # None for layer 0

    def wire(self):
        from .fpn2 import enc3, enc3s
        out = b"" if self.g_sum is None else enc3(self.g_sum)
        out += b"".join(enc3s(s) for s in self.rounds) + enc3s(self.children)
        return out + (b"" if self.g_r is None else enc3(self.g_r))


def upoly(coeffs, x):
    acc = ZERO
    for c in reversed(coeffs):
        acc = hadd(c, hmul(x, acc))
    return acc


def _scal(c, a):
    return tuple(c * x % P for x in a)


def libra_sum(g):
    """sum over b in {0,1}^k of G(b) = 2^(k-1) sum_i (g_i(0) + g_i(1))."""
    k = len(g)
    acc = ZERO
    for co in g:
        acc = hadd(acc, hadd(co[0], upoly(co, ONE)))
    return _scal(pow(2, k - 1, P), acc)


def libra_round(g, i, rs, t):
    """Round-i polynomial of the mask at X = t (rs = earlier challenges)."""
    k = len(g)
    acc = upoly(g[i], t)
    for j in range(i):
        acc = hadd(acc, upoly(g[j], rs[j]))
    acc = _scal(pow(2, k - 1 - i, P), acc)
    if i < k - 1:
        tail = ZERO
        for co in g[i + 1:]:
            tail = hadd(tail, hadd(co[0], upoly(co, ONE)))
        acc = hadd(acc, _scal(pow(2, k - 2 - i, P), tail))
    return acc


def build_tree(leaves):
    tree = [list(leaves)]
    while len(tree[-1]) > 1:
        nodes = tree[-1]
        tree.append([combine(a, b) for a, b in zip(nodes[::2], nodes[1::2])])
    return tree


def masked_prove(transcript, tree, libra, hooks=None):
    """Prove the root of `tree` (from build_tree) with Libra-masked layer sumchecks.

    `libra(k, i)` returns the 4 committed coefficients of g_{k,i}.  The root bytes must already be
    absorbed by the caller.  Returns (layers, point, claim, libra_evals) where libra_evals lists
    (k, rs, g_r) claims that the caller discharges through the PCS.
    `hooks` (tests only) may tamper with children and adaptively re-solve g_r.
    """
    from .fpn2 import enc3, enc3s
    hooks = hooks or {}
    point, claim = (), tree[-1][0]
    layers, evals = [], []
    for k, nodes in enumerate(reversed(tree[:-1])):
        alpha = transcript.squeeze_fp3()
        g = g_sum = lam = None
        target = hadd(claim[0], hmul(alpha, claim[1]))
        if k:
            g = [tuple(libra(k, i)) for i in range(k)]
            g_sum = libra_sum(g)
            transcript.absorb_bytes(enc3(g_sum))
            lam = transcript.squeeze_fp3()
            target = hadd(target, hmul(lam, g_sum))
        columns = [_weights(point), *map(list, zip(*nodes[::2])), *map(list, zip(*nodes[1::2]))]
        rounds, rs = [], []
        for i in range(k):
            samples = _round(columns, alpha)
            samples = tuple(hadd(f, hmul(lam, libra_round(g, i, rs, hf(t)))) for t, f in enumerate(samples))
            transcript.absorb_bytes(enc3s(samples))
            r = transcript.squeeze_fp3()
            rounds.append(samples)
            rs.append(r)
            target = lagrange_eval(samples, r)
            columns = [_fold(column, r) for column in columns]
        children = tuple(column[0] for column in columns[1:])
        g_r = None
        if k:
            g_r = ZERO
            for co, r in zip(g, rs):
                g_r = hadd(g_r, upoly(co, r))
        if "children" in hooks:
            children = hooks["children"](k, children)
            if k and hooks.get("adaptive_g_r"):
                weight = ONE
                for a, b in zip(point, rs):
                    weight = hmul(weight, eq1(a, b))
                n, d = combine(children[:2], children[2:])
                rest = hsub(target, hmul(weight, hadd(n, hmul(alpha, d))))
                g_r = hmul(rest, hinv(lam))
        transcript.absorb_bytes(enc3s(children) + (enc3(g_r) if k else b""))
        if k:
            evals.append((k, tuple(rs), g_r))
        layers.append(MaskedLayer(g_sum, tuple(rounds), children, g_r))
        branch = transcript.squeeze_fp3()
        nl, dl, nr, dr = children
        point, claim = (branch, *rs), (_mix(nl, nr, branch), _mix(dl, dr, branch))
    return tuple(layers), point, claim, tuple(evals)


def masked_verify(transcript, root, layers):
    """Replay the masked GKR. Returns (point, claim, libra_evals) or raises GkrReject."""
    from .fpn2 import enc3, enc3s
    if root[0] != ZERO or root[1] == ZERO:
        raise GkrReject("E_GKR_ROOT", "root_num != 0 or root_den == 0")
    point, claim = (), root
    evals = []
    for k, layer in enumerate(layers):
        alpha = transcript.squeeze_fp3()
        target = hadd(claim[0], hmul(alpha, claim[1]))
        lam = None
        if k:
            transcript.absorb_bytes(enc3(layer.g_sum))
            lam = transcript.squeeze_fp3()
            target = hadd(target, hmul(lam, layer.g_sum))
        if len(layer.rounds) != k:
            raise GkrReject("E_GKR", "round count")
        rs = []
        for samples in layer.rounds:
            transcript.absorb_bytes(enc3s(samples))
            if hadd(samples[0], samples[1]) != target:
                raise GkrReject("E_GKR", f"layer {k} round sum")
            r = transcript.squeeze_fp3()
            rs.append(r)
            target = lagrange_eval(samples, r)
        transcript.absorb_bytes(enc3s(layer.children) + (enc3(layer.g_r) if k else b""))
        weight = ONE
        for a, b in zip(point, rs):
            weight = hmul(weight, eq1(a, b))
        n, d = combine(layer.children[:2], layer.children[2:])
        expect = hmul(weight, hadd(n, hmul(alpha, d)))
        if k:
            expect = hadd(expect, hmul(lam, layer.g_r))
            evals.append((k, tuple(rs), layer.g_r))
        if target != expect:
            raise GkrReject("E_GKR", f"layer {k} final")
        branch = transcript.squeeze_fp3()
        nl, dl, nr, dr = layer.children
        point, claim = (branch, *rs), (_mix(nl, nr, branch), _mix(dl, dr, branch))
    return point, claim, tuple(evals)
