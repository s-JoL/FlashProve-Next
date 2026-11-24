"""Typed scalar cells and an exact reference interpreter for gadget relations."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable

from flashprove_next.field import P


AUX_BITS = 7      # one relation-4 limb: an auxiliary wire of 7k bits costs 1 + k variables, no tightening


class Freedom(str, Enum):
    STRICT = "STRICT"
    BLINDING = "BLINDING"
    PERMITTED_TIE = "PERMITTED_TIE"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    domain_size: int
    semantic_type: str
    producer: str
    signed_bound_bits: int
    freedom_kind: Freedom = Freedom.STRICT

    def __post_init__(self):
        if (
            type(self.name) is not str
            or type(self.domain_size) is not int
            or type(self.semantic_type) is not str
            or type(self.producer) is not str
            or type(self.signed_bound_bits) is not int
            or type(self.freedom_kind) is not Freedom
        ):
            raise TypeError("ColumnSpec requires exact built-in field types")

    @property
    def bounds(self):
        limit = 1 << (self.signed_bound_bits - 1)
        return -limit, limit - 1


@dataclass(frozen=True)
class ColumnLedger(ColumnSpec):
    consumers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple

    def __post_init__(self):
        if type(self) is not Expr or type(self.op) is not str:
            raise TypeError("Expr requires exact Expr type and exact str op")
        if type(self.args) is not tuple:
            object.__setattr__(self, "args", tuple(self.args))
        if self.op == "wire":
            if len(self.args) != 1 or type(self.args[0]) is not str:
                raise TypeError("wire Expr requires (str,)")
        elif self.op == "const":
            if len(self.args) != 1 or type(self.args[0]) is not int:
                raise TypeError("const Expr requires (int,)")
        elif self.op in ("+", "-", "*"):
            if len(self.args) != 2 or not all(type(a) is Expr for a in self.args):
                raise TypeError(f"binary Expr {self.op!r} requires (Expr, Expr)")
        else:
            raise ValueError(f"unknown Expr op: {self.op!r}")

    @property
    def columns(self):
        if self.op == "wire":
            return {self.args[0]}
        return set().union(*(arg.columns for arg in self.args if isinstance(arg, Expr)))

    def evaluate(self, witness):
        if self.op == "wire":
            return witness[self.args[0]]
        if self.op == "const":
            return self.args[0]
        a, b = (arg.evaluate(witness) for arg in self.args)
        if self.op == "+":
            return a + b
        if self.op == "-":
            return a - b
        return a * b

    def bounds(self, columns):
        if self.op == "wire":
            return columns[self.args[0]].bounds
        if self.op == "const":
            return self.args[0], self.args[0]
        (a, b), (c, d) = (arg.bounds(columns) for arg in self.args)
        if self.op == "+":
            return a + c, b + d
        if self.op == "-":
            return a - d, b - c
        products = a*c, a*d, b*c, b*d
        return min(products), max(products)

    def __add__(self, other):
        return Expr("+", (self, expression(other)))

    def __sub__(self, other):
        return Expr("-", (self, expression(other)))

    def __mul__(self, other):
        return Expr("*", (self, expression(other)))


def expression(value):
    if isinstance(value, Expr):
        return value
    if type(value) is int:
        return Expr("const", (value,))
    raise TypeError("expected expression or Python integer")


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    difference: Expr

    def __post_init__(self):
        if type(self) is not ConstraintSpec or type(self.name) is not str or type(self.difference) is not Expr:
            raise TypeError("ConstraintSpec requires exact str name and exact Expr difference")

    @property
    def columns(self):
        return self.difference.columns


@dataclass(frozen=True)
class LookupSpec:
    """Exact reference relation. A backend must lower the predicate separately."""

    name: str
    columns: tuple[str, ...]
    accepts: Callable[..., bool]

    def __post_init__(self):
        if type(self) is not LookupSpec or type(self.name) is not str:
            raise TypeError("LookupSpec requires exact str name")
        if type(self.columns) is not tuple:
            object.__setattr__(self, "columns", tuple(self.columns))
        if not all(type(c) is str for c in self.columns):
            raise TypeError("LookupSpec.columns must be exact str names")


@dataclass
class Program:
    columns: dict[str, ColumnSpec] = field(default_factory=dict)
    constraints: list[ConstraintSpec] = field(default_factory=list)
    lookups: list[LookupSpec] = field(default_factory=list)
    slots: list["MicroprogramSlot"] = field(default_factory=list)
    # Auxiliary wires (gated products, sign-free slack cells, one-hot index cells) are ordinary STRICT
    # range-checked columns whose honest value is a function of earlier columns.  `hints` is prover-side
    # witness generation only: it is not part of the public circuit and no verifier trusts it.
    hints: dict[str, Callable[[dict], int]] = field(default_factory=dict)

    def wire(self, name, bits, semantic_type, producer, freedom=Freedom.STRICT):
        spec = ColumnSpec(name, 1, semantic_type, producer, bits, freedom)
        if not 1 <= bits <= 62 or self.columns.setdefault(name, spec) != spec:
            raise ValueError(f"invalid or conflicting column {name}")
        return Expr("wire", (name,))

    def equal(self, name, lhs, rhs=0):
        self.constraints.append(ConstraintSpec(name, expression(lhs) - rhs))

    def aux(self, name, bits, semantic_type, producer, hint, freedom=Freedom.STRICT):
        """Auxiliary wire with a witness-generation hint clamped into the wire's signed range, so a
        forged base witness surfaces as the named gate that pins the wire, never as `range:`."""
        wire = self.wire(name, bits, semantic_type, producer, freedom)
        low, high = self.columns[name].bounds
        self.hints[name] = lambda witness: min(high, max(low, hint(witness)))
        return wire

    def product(self, name, left, right, producer, bits=AUX_BITS):
        """Auxiliary wire g with the degree-2 gate `<name>.def`: g = left * right."""
        value = expression(left) * right
        wire = self.aux(name, bits, "gated product", producer, value.evaluate)
        self.equal(name + ".def", wire - value)
        return wire

    def nonneg(self, name, value, bound, producer):
        """Gate `<name>`: value = u + 2^(B-1) for a fresh B-bit slack wire `<name>.slack`.

        B is the smallest multiple of the limb width with 2^B > bound, so the wire's signed range check
        (limb membership, no tightening cell) is exactly 0 <= value < 2^B; every honest value must lie
        in [0, bound]."""
        if bound < 0:
            raise ValueError("negative slack bound")
        bits = AUX_BITS * max(1, -(-bound.bit_length() // AUX_BITS))
        half = 1 << (bits - 1)
        value = expression(value)
        slack = self.aux(name + ".slack", bits, "non-negative slack", producer,
                         lambda witness: value.evaluate(witness) - half)
        self.equal(name, value - slack - half)
        return slack

    def fits(self, difference):
        """Whether a gate over the declared column ranges keeps the signed lift bound of `check_bounds`."""
        lower, upper = expression(difference).bounds(self.columns)
        return max(abs(lower), abs(upper)) < min(P, 1 << 62)

    def complete(self, witness):
        """Fill every absent auxiliary wire from its hint (column order; present values are kept)."""
        out = dict(witness)
        for name in self.columns:
            if name in self.hints and name not in out:
                out[name] = self.hints[name](out)
        return out

    def ledger(self):
        consumers = {name: [] for name in self.columns}
        for relation in (*self.constraints, *self.lookups):
            for name in set(relation.columns):
                if name in consumers:
                    consumers[name].append(relation.name)
        return [
            asdict(ColumnLedger(**asdict(spec), consumers=tuple(consumers[spec.name])))
            for spec in self.columns.values()
        ]

    def check_bounds(self):
        # Memoised on the identity of every constraint and every (name, ColumnSpec) pair
        # (strong references, so an edited / replaced relation or widened column always re-runs).
        checked = self.__dict__.get("_bounds_checked")
        col_items = tuple(self.columns.items())
        if (
            checked is not None
            and len(checked[0]) == len(self.constraints)
            and all(a is b for a, b in zip(checked[0], self.constraints))
            and len(checked[1]) == len(col_items)
            and all(k1 == k2 and s1 is s2 for (k1, s1), (k2, s2) in zip(checked[1], col_items))
        ):
            return
        for equation in self.constraints:
            lower, upper = equation.difference.bounds(self.columns)
            if max(abs(lower), abs(upper)) >= min(P, 1 << 62):
                raise ValueError(f"missing signed lift bound: {equation.name}")
        self.__dict__["_bounds_checked"] = (tuple(self.constraints), col_items)

    def failures(self, witness):
        self.check_bounds()
        missing = self.columns.keys() - witness.keys()
        if witness.keys() - self.columns.keys() or not missing <= self.hints.keys():
            return ["witness shape"]
        for name, spec in self.columns.items():
            if name in missing:
                continue
            value = witness[name]
            low, high = spec.bounds
            if type(value) is not int or not low <= value <= high:
                return [f"range: {name}"]
        if missing:
            witness = self.complete(witness)       # hints are clamped into range
        failures = [eq.name for eq in self.constraints if eq.difference.evaluate(witness)]
        failures.extend(
            lookup.name for lookup in self.lookups
            if not lookup.accepts(*(witness[name] for name in lookup.columns))
        )
        return failures


@dataclass(frozen=True)
class MicroprogramSlot:
    selectors: tuple[str, ...]
    lanes: tuple[str, ...]
    widths: tuple[int, ...]
    branch_names: tuple[str, ...]

    @classmethod
    def install(cls, program, name, branches, bits=24):
        """Each branch supplies (opcode name, lane count, reference predicate)."""
        branches = tuple(branches)
        if not branches or len({branch[0] for branch in branches}) != len(branches):
            raise ValueError("slot requires distinct opcodes")
        widths = tuple(width for _, width, _ in branches)
        if min(widths) < 1:
            raise ValueError("empty branch")
        selectors = tuple(f"{name}.{opcode}" for opcode, _, _ in branches)
        wires = [program.wire(selector, 2, "boolean", name) for selector in selectors]
        lanes = tuple(f"{name}.lane{i}" for i in range(max(widths)))
        for lane in lanes:
            program.wire(lane, bits, "shared operand/result", name)
        total = expression(0)
        for selector, wire in zip(selectors, wires):
            program.equal(selector + ".boolean", wire * (wire - 1))
            total = total + wire
        program.equal(name + ".exhaustive", total, 1)
        # Unused-lane zero guard as algebraic gates : selected branch k => lanes >= width_k are 0,
        # i.e. s_k * lane_i = 0 (one Plonkish q_M row each); previously only inside the `guarded` predicate.
        for selector, wire, width in zip(selectors, wires, widths):
            for i in range(width, len(lanes)):
                program.equal(f"{selector}.unused{i}", wire * Expr("wire", (lanes[i],)))
        branch_names = []
        for selector, (_, width, accepts) in zip(selectors, branches):
            def guarded(active, *values, width=width, accepts=accepts):
                # Unused lanes (>= width) are pinned to 0 by the `.unused<i>` gates above, not here.
                return active != 1 or accepts(*values[:width])

            branch_name = selector + ".relation"
            branch_names.append(branch_name)
            program.lookups.append(LookupSpec(branch_name, (selector,) + lanes, guarded))
        slot = cls(selectors, lanes, widths, tuple(branch_names))
        program.slots.append(slot)
        return slot


def topk_tau(scores, bitmap, indices, tau):
    """Enumerate the selected bitmap in ascending index order with 2*tau slack."""
    if (len(scores) != len(bitmap) or tau < 0 or not indices
            or any(bit not in (0, 1) for bit in bitmap)):
        return False
    selected = {i for i, bit in enumerate(bitmap) if bit}
    if tuple(indices) != tuple(sorted(selected)):
        return False
    return all(
        scores[j] - scores[i] <= 2 * tau
        for i in selected for j in range(len(scores)) if j not in selected
    )


def attaining_rmx(scores, active, argmax, maximum, exponentials, denominator):
    """Check the active maximum and denominator given upstream exponential cells."""
    if not (len(scores) == len(active) == len(argmax) == len(exponentials)):
        return False
    return (
        all(bit in (0, 1) for bit in (*active, *argmax))
        and sum(argmax) == 1
        and all(
            chosen <= enabled and (not enabled or score <= maximum)
            and (not chosen or score == maximum)
            and (weight > 0 if enabled else weight == 0)
            for score, enabled, chosen, weight in zip(scores, active, argmax, exponentials)
        )
        and denominator == sum(exponentials) > 0
    )
