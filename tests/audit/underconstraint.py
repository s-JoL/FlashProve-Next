"""Bounded witness perturbation and equation deletion checks."""

from dataclasses import replace
from itertools import chain, product

from prover.fpn_prover.ir.gadget_ir import Freedom, P


def static_audit(program):
    program.check_bounds()
    relations = program.constraints + program.lookups
    names = {relation.name for relation in relations}
    if len(names) != len(relations):
        raise ValueError("duplicate relation")
    if any(set(relation.columns) - program.columns.keys() for relation in relations):
        raise ValueError("unknown column")
    for entry in program.ledger():
        if not entry["producer"] or (
            entry["freedom_kind"] != Freedom.BLINDING and not entry["consumers"]
        ):
            raise ValueError(f"unconnected column {entry['name']}")
    for slot in program.slots:
        required = set(slot.branch_names) | {name + ".boolean" for name in slot.selectors}
        required.add(slot.selectors[0].rsplit(".", 1)[0] + ".exhaustive")
        if not required <= names:
            raise ValueError("uncovered slot branch")


def _forms(program, candidate, changed):
    """A perturbed row as given (auxiliary cells stale) and, when an in-range base cell changed, also
    with every auxiliary cell regenerated from its hint (what a forger running the honest generator on
    forged inputs obtains).  Both are adversarial rows and both must be rejected."""
    yield candidate
    hints = program.hints
    if hints and any(name not in hints for name in changed) and all(
            type(candidate[name]) is int and lo <= candidate[name] <= hi
            for name in changed for lo, hi in (program.columns[name].bounds,)):
        yield program.complete({name: value for name, value in candidate.items() if name not in hints})


def perturbations(program, witness, fixed=()):
    options = {}
    for name, spec in program.columns.items():
        if spec.freedom_kind != Freedom.STRICT or name in fixed:
            continue
        value = witness[name]
        lo, hi = spec.bounds
        options[name] = sorted({value+1, value-1, 0, -value, lo, hi, value+P, value-P} - {value})
        for change in options[name]:
            yield from _forms(program, {**witness, name: change}, (name,))
    for relation in program.constraints + program.lookups:
        names = sorted(set(relation.columns) & options.keys())
        for i, left in enumerate(names):
            for right in names[i+1:]:
                for a, b in product(options[left], options[right]):
                    yield from _forms(program, {**witness, left: a, right: b}, (left, right))


def audit(program, witness, fixed=(), correlated=()):
    """A fixture holds its public inputs fixed. Include larger targeted changes explicitly."""
    static_audit(program)
    witness = program.complete(witness)
    if program.failures(witness):
        raise ValueError("invalid audit baseline")
    for candidate in chain(correlated, perturbations(program, witness, fixed)):
        if any(candidate[name] != witness[name] for name in fixed):
            raise ValueError("perturbation changed a fixed cell")
        if not program.failures(candidate):
            changed = {name: value for name, value in candidate.items() if witness[name] != value}
            raise AssertionError(f"silent strict perturbation: {changed}")


def mutations(program, witness, fixed=(), targeted=()):
    """Report an admitted witness for each deleted equation, or None if search missed it."""
    witness = program.complete(witness)
    if program.failures(witness):
        raise ValueError("invalid mutation baseline")
    candidates = tuple(perturbations(program, witness, fixed)) + tuple(targeted)
    # One interpreter run per candidate: only a row whose sole failure is the equation can witness it.
    sole = {}
    for row in candidates:
        if all(row[name] == witness[name] for name in fixed):
            failures = program.failures(row)
            if len(failures) == 1:
                sole.setdefault(failures[0], []).append(row)
    result = {}
    for index, equation in enumerate(program.constraints):
        weakened = replace(program, constraints=program.constraints[:index]+program.constraints[index+1:])
        result[equation.name] = next((
            row for row in sole.get(equation.name, ()) if not weakened.failures(row)
        ), None)
    return result


def lookup_mutations(program, witness, candidates):
    """Check targeted predicate deletions using candidates with one failure."""
    result = dict.fromkeys(lookup.name for lookup in program.lookups)
    pending = {lookup.name: i for i, lookup in enumerate(program.lookups)}
    for candidate in candidates:
        failures = program.failures(candidate)
        if len(failures) != 1 or failures[0] not in pending:
            continue
        name = failures[0]
        index = pending.pop(name)
        weakened = replace(program, lookups=program.lookups[:index]+program.lookups[index+1:])
        if not weakened.failures(candidate):
            result[name] = candidate
    return result
