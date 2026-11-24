# SPDX-License-Identifier: Apache-2.0
"""Whole-lowered-circuit auditor for relation 4 (`FOLDED_GKR_AIR`).

Everything here judges the COMMITTED public object -- `AirCircuit.table` (`backend.folded_air.AirTable`:
K rows of (q_M, q_L, q_R, q_O, q_C, sig_a, sig_b, sig_c), copy multiplicities, range flags) -- on a raw
variable vector V, exactly as the relation-4 verifier's three checks do:

  * zerocheck      : q_M V[a] V[b] + q_L V[a] + q_R V[b] + q_O V[c] + q_C = 0 on every row;
  * copy constraint: every gate cell is the variable its public wire id names (one V, so by construction;
                     `table_integrity` checks the public multiplicities against the public wiring);
  * range lookup   : every flagged variable is in [0, 128).

`table_failures` re-implements those checks from the table alone: it shares no code with
`lower.air_row_failures`, `lower.Gate.evaluate` or `Program.failures`.
"""
from itertools import combinations, product

from flashprove_next.field import P
from prover.fpn_prover.backend import folded_air as FA
from prover.fpn_prover.circuit import lower as LW

TABLE = 128


def table_integrity(circuit):
    """Public-table well-formedness the copy-constraint and range arguments rely on."""
    table, K = circuit.table, circuit.layout.K
    assert len(table.rows) == len(table.mult) == len(table.flag) == K
    used = len(circuit.rows)
    mult = [0] * K
    for row in table.rows:
        assert len(row) == 8 and all(0 <= q < P for q in row[:5]) and all(1 <= s <= K for s in row[5:])
        for sig in row[5:]:
            mult[sig - 1] += 1
    assert tuple(mult) == tuple(table.mult)
    # IR rows first, then FA.ZK_ROWS zero-selector rows on fresh `#zk` variables, then pad rows on variable 0.
    index = {name: k for k, name in enumerate(circuit.variables)}
    for r, row in enumerate(circuit.rows):
        assert table.rows[r] == tuple(row.q) + tuple(index[w] + 1 for w in row.wires)
        assert any(row.q), "an IR row with all-zero selectors constrains nothing"
    zk = [k for k, name in enumerate(circuit.variables) if name.startswith(LW.ZK_PREFIX)]
    assert len(zk) == 3 * FA.ZK_ROWS
    for row in table.rows[used:used + FA.ZK_ROWS]:
        assert row[:5] == (0, 0, 0, 0, 0)
    assert sorted(s - 1 for row in table.rows[used:used + FA.ZK_ROWS] for s in row[5:]) == zk
    assert all(row == (0, 0, 0, 0, 0, 1, 1, 1) for row in table.rows[used + FA.ZK_ROWS:])
    flagged = {index[n] for plan in circuit.limb_plans for n in plan.names()}
    assert {k for k, f in enumerate(table.flag) if f} == flagged
    # The pad variable only ever sits in a cell whose selector is 0 (its value is uniform randomness).
    for q, wires in ((row.q, row.wires) for row in circuit.rows):
        qm, ql, qr, qo, _ = q
        a, b, c = (w == LW.PAD for w in wires)
        assert not ((a or b) and qm) and not (a and ql) and not (b and qr) and not (c and qo)
    return used


def table_failures(table, V):
    """(violated row indices, range-flagged variable indices outside [0, 128)) of V on the table."""
    rows = []
    for r, (qm, ql, qr, qo, qc, sa, sb, sc) in enumerate(table.rows):
        a, b, c = V[sa - 1], V[sb - 1], V[sc - 1]
        if (qm * a * b + ql * a + qr * b + qo * c + qc) % P:
            rows.append(r)
    return rows, [k for k, flag in enumerate(table.flag) if flag and not 0 <= V[k] < TABLE]


def delete_row(table, r):
    """Mutated circuit: row r's selectors zeroed, wiring (hence copy multiplicities) untouched."""
    rows = list(table.rows)
    rows[r] = (0, 0, 0, 0, 0) + tuple(rows[r][5:])
    return FA.AirTable(tuple(rows), table.mult, table.flag)


def named_variables(circuit):
    """Indices of every witness-carrying variable (wires, limbs, tightening cells, accumulators)."""
    mask = set(circuit.mask_variables)
    return [k for k in range(len(circuit.variables)) if k not in mask]


def vector(circuit, values, rng=None):
    return LW.air_vector(circuit, values, rng)


# ----------------------------------------------------------------------------- gate-isolating witnesses
class Isolator:
    """Finds, for an IR gate G, in-range variable values violating G and no other IR gate.

    The search works on wire values (limbs / tightening cells re-derived, accumulators re-derived by
    `lower.air_values`); the result is always re-judged by `table_failures` on the committed table.
    """

    def __init__(self, circuit, witnesses):
        self.circuit = circuit
        self.witnesses = witnesses                  # complete wire witnesses (base + auxiliary)
        self.bounds = {name: (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
                       for name, bits, *_ in circuit.columns}
        self.plan = {plan.wire: plan for plan in circuit.limb_plans}
        self.constraint = [g for g in circuit.gates if g.origin == "constraint"]
        self.touching = {}
        for gate in self.constraint:
            for wire in self.wires(gate):
                self.touching.setdefault(wire, []).append(gate)

    @staticmethod
    def wires(gate):
        return sorted({w for _, mono in gate.terms for w in mono})

    def options(self, name, value):
        lo, hi = self.bounds[name]
        cand = {value + 1, value - 1, 0, 1, 2, -1, -value}
        return sorted(c for c in cand if lo <= c <= hi and c != value)

    def in_range(self, witness):
        return all(lo <= witness[name] <= hi for name, (lo, hi) in self.bounds.items())

    def violated(self, witness):
        return {g.name for g in self.constraint if g.evaluate(witness)}

    def slack_names(self):
        return {name for name in self.circuit.hints if name.endswith((".slack", ".topk_threshold"))}

    def build(self, base, changes, mode):
        """`base` with `changes`; mode "stale" keeps every other cell, "slack" regenerates the slack
        cells from their hints, "full" regenerates every auxiliary cell that is not itself changed."""
        row = {**base, **changes}
        if mode == "stale":
            return row
        drop = self.slack_names() if mode == "slack" else set(self.circuit.hints)
        return LW.complete_witness(self.circuit, {n: v for n, v in row.items()
                                                  if n in changes or n not in drop})

    def forms(self, base, changes):
        yield self.build(base, changes, "stale")
        yield self.build(base, changes, "slack")
        if any(name not in self.circuit.hints for name in changes):
            yield self.build(base, changes, "full")

    def isolates(self, gate, witness):
        return self.in_range(witness) and self.violated(witness) == {gate.name}

    def within(self, names, gate, witness):
        """Violates `gate` and nothing outside the gate set `names`."""
        return self.in_range(witness) and gate.name in (bad := self.violated(witness)) and bad <= names

    def candidates(self, gate, wide):
        own = self.wires(gate)
        if wide:
            own = sorted({w for g in self.constraint if g.name in wide for w in self.wires(g)})
        near = sorted({w for name in own for g in self.touching[name] for w in self.wires(g)} - set(own))
        for names in [(n,) for n in own + near] + list(combinations(own, 2)):
            for base in self.witnesses:
                for change in product(*(self.options(n, base[n]) for n in names)):
                    yield from self.forms(base, dict(zip(names, change)))

    def search(self, gate):
        """An in-range wire witness whose only violated constraint gate is `gate`, or None: one changed
        cell from the gate or a neighbouring gate, then two changed cells of the gate itself."""
        return next((c for c in self.candidates(gate, None) if self.isolates(gate, c)), None)

    def search_group(self, gate, names):
        """A witness violating `gate` and only gates of its group `names`, or None."""
        return next((c for c in self.candidates(gate, names) if self.within(names, gate, c)), None)

    def group(self, gate):
        """Constraint group: gates with the same name once every numeric component is dropped."""
        return {g.name for g in self.constraint if group_name(g.name) == group_name(gate.name)}

    def limb_mutant(self, gate, base_values):
        """In-range limb / tightening values violating only this recompose or tighten gate."""
        wire = gate.name.split("#")[0]
        plan = self.plan[wire]
        names = plan.names()
        values = dict(base_values)
        if gate.origin == "tighten":
            cell = names[-1]
            values[cell] += 1 if values[cell] < TABLE - 1 else -1
            return values
        if plan.limbs > 1 or not plan.tightened:
            values[names[0]] += 1 if values[names[0]] < TABLE - 1 else -1
            return values
        # One limb plus its tightening cell: shift both so the tightening identity keeps holding.
        step = 1 if values[names[1]] < TABLE - 1 else -1
        values[names[0]] += step
        values[names[1]] += step
        return values


def group_name(name):
    return ".".join(part for part in name.split(".") if not part.isdigit())


def gate_rows(circuit):
    """IR gate name -> list of its row indices (contiguous, the last one closes the gate)."""
    out = {}
    for r, row in enumerate(circuit.rows):
        out.setdefault(row.origin, []).append(r)
    return out


def _define(circuit, values, name):
    d = circuit.defs[name]
    return (d[1] * values[d[2]] * values[d[3]] if d[0] == "prod" else sum(c * values[x] for c, x in d[1])) % P


def row_deletion_vector(circuit, values, rows, r):
    """Variable values violating row r and no other row of its gate.

    `values` violates exactly one IR gate (rows `rows`).  With the gate's accumulators derived forward
    only its closing row fails.  Deleting an earlier row frees the accumulator it defines: solve the
    (affine) closing row for that accumulator and re-derive the accumulators after it.
    """
    values = dict(values)
    for row in rows[:-1]:
        values[circuit.rows[row].wires[2]] = _define(circuit, values, circuit.rows[row].wires[2])
    if r == rows[-1]:
        return values
    free = circuit.rows[r].wires[2]
    assert free in circuit.defs, "a non-closing row defines an accumulator"

    def residual(t):
        trial = dict(values)
        trial[free] = t % P
        for later in rows[rows.index(r) + 1:-1]:
            trial[circuit.rows[later].wires[2]] = _define(circuit, trial, circuit.rows[later].wires[2])
        a, b, c = (trial.get(w, 0) if w != LW.PAD else 0 for w in circuit.rows[rows[-1]].wires)
        qm, ql, qr, qo, qc = circuit.rows[rows[-1]].q
        return trial, (qm * a * b + ql * a + qr * b + qo * c + qc) % P

    _, r0 = residual(0)
    _, r1 = residual(1)
    slope = (r1 - r0) % P
    assert slope, "the closing row does not depend on the freed accumulator"
    trial, zero = residual(-r0 * pow(slope, P - 2, P))
    assert zero == 0
    return trial
