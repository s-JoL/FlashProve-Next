"""Rate-1/8 folded polynomial commitments.

* `commit`/`prove`/`verify`: the original batched univariate opening (binary FRI folds,
  SHA-256 Merkle, SHAKE transcript) -- unchanged reference API.
* `zk_commit`/`zk_open`/`zk_verify`: the fused two-round (coset-4) ZK-Basefold of relation 3
  (specs/11-wire-format.md §5.4) with BLAKE2b coset Merkle trees and raw FPN2 wire bytes absorbed
  into `Fpn2Transcript`; `observations`/`require_joint_mask_ext` give the explicit rank check
  im(A) <= im(B) of the reserved non-doubling blinding cells; `byte_budget` is the exact size
  calculator.
"""

from dataclasses import dataclass
from hashlib import sha256
from secrets import randbelow

from flashprove_next.field import ONE, P, ZERO, hadd, hf, hinv, hmul, hsub, lagrange_eval

from .frac_logup_gkr import Transcript, encode

RATE = 8
HALF = hf((P + 1) // 2)


def rank(matrix):
    rows = [list(row) for row in matrix]
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("ragged observation matrix")
    rank_so_far = 0
    for column in range(width):
        pivot = next((i for i in range(rank_so_far, len(rows)) if rows[i][column] % P), None)
        if pivot is None:
            continue
        rows[rank_so_far], rows[pivot] = rows[pivot], rows[rank_so_far]
        row = rows[rank_so_far]
        inverse = pow(row[column] % P, -1, P)
        for other in rows[rank_so_far+1:]:
            factor = other[column] * inverse % P
            for j in range(column, width):
                other[j] = (other[j] - factor * row[j]) % P
        rank_so_far += 1
    return rank_so_far


def require_joint_mask(secret, mask):
    """Check im(A) is contained in im(B) over Fp for observations A*s+B*r."""
    secret, mask = tuple(map(tuple, secret)), tuple(map(tuple, mask))
    if len(secret) != len(mask):
        raise ValueError("observation count mismatch")
    if rank([a + b for a, b in zip(secret, mask)]) != rank(mask):
        raise ValueError("under-rank mask")


def reserve(coefficients, size, positions):
    """Insert independent Fp3 randomness at reserved coefficient positions."""
    positions = set(positions)
    coefficients = tuple(coefficients)
    if not positions <= set(range(size)) or len(coefficients) + len(positions) != size:
        raise ValueError("incorrect reserved layout")
    encode(coefficients)
    values = iter(coefficients)
    return tuple(tuple(randbelow(P) for _ in range(3)) if i in positions else next(values)
                 for i in range(size))


def evaluate(coefficients, point):
    result = ZERO
    for value in reversed(coefficients):
        result = hadd(value, hmul(point, result))
    return result


def _hash(data):
    return sha256(data).digest()


def _tree(rows):
    levels = [[_hash(b"L" + encode(row)) for row in rows]]
    while len(levels[-1]) > 1:
        layer = levels[-1]
        levels.append([_hash(b"N" + a + b) for a, b in zip(layer[::2], layer[1::2])])
    return levels


@dataclass(frozen=True)
class Opening:
    row: tuple
    path: tuple

    def bytes(self):
        return encode(self.row) + b"".join(self.path)


def _open(rows, tree, i):
    return Opening(rows[i], tuple(layer[(i >> level) ^ 1]
                                  for level, layer in enumerate(tree[:-1])))


def _check(opening, root, i, depth, width):
    if len(opening.path) != depth or len(opening.row) != width:
        return False
    node = _hash(b"L" + encode(opening.row))
    for other in opening.path:
        if not isinstance(other, bytes) or len(other) != 32:
            return False
        node = _hash(b"N" + (other + node if i & 1 else node + other))
        i >>= 1
    return node == root


def _fold(a, b, x, r):
    even = hmul(HALF, hadd(a, b))
    odd = hmul(hmul(HALF, hinv(x)), hsub(a, b))
    return hadd(even, hmul(r, odd))


def _batch(values, alpha):
    return evaluate(values, alpha)


def _domain(size):
    if type(size) is not int or size < 2 or size & (size - 1) or size > 1 << 29:
        raise ValueError("invalid coefficient capacity")
    n = RATE * size
    return n, pow(7, (P - 1) // n, P)


def _begin(root, size, point, claims, queries):
    n, omega = _domain(size)
    if not 1 <= queries <= n or not claims or len(root) != 32:
        raise ValueError("invalid opening parameters")
    if point[1:] == (0, 0) and pow(point[0], n, P) == 1:
        raise ValueError("opening point intersects domain")
    transcript = Transcript(b"fpn-batched-folded-pcs-v1")
    transcript.absorb(root)
    transcript.absorb(size.to_bytes(8, "little") + queries.to_bytes(8, "little"))
    transcript.absorb(encode((point, *claims)))
    return transcript, n, omega, transcript.challenge()


@dataclass(frozen=True)
class Proof:
    roots: tuple
    terminal: tuple
    openings: tuple

    def bytes(self):
        """Wire payload with shape inferred from the statement."""
        return (b"".join(self.roots) + encode((self.terminal,))
                + b"".join(item.bytes() for chain in self.openings for item in chain))

    @property
    def byte_count(self):
        return len(self.bytes())


def _index(transcript, n):
    # Use the low bits of an unconditioned 64-bit transcript word.
    word = transcript.state.digest(8)
    transcript.absorb(word)
    return int.from_bytes(word, "little") % n


def _codeword(polynomials, size):
    n, omega = _domain(size)
    if not polynomials or any(len(poly) != size for poly in polynomials):
        raise ValueError("polynomial capacity mismatch")
    for poly in polynomials:
        encode(poly)
    return [tuple(evaluate(poly, hf(pow(omega, i, P))) for poly in polynomials)
            for i in range(n)]


def _pairs(rows):
    half = len(rows) // 2
    return [a + b for a, b in zip(rows[:half], rows[half:])]


def commit(polynomials):
    """Compute the commitment before an opening point is selected."""
    polynomials = tuple(map(tuple, polynomials))
    if not polynomials:
        raise ValueError("empty batch")
    return _tree(_pairs(_codeword(polynomials, len(polynomials[0]))))[-1][0]


def prove(polynomials, point, queries=8):
    """Return commitment, evaluations, and a batch opening proof."""
    polynomials = tuple(map(tuple, polynomials))
    if not polynomials:
        raise ValueError("empty batch")
    size = len(polynomials[0])
    code = _codeword(polynomials, size)
    pairs = _pairs(code)
    trees = [_tree(pairs)]
    layers = [pairs]
    root = trees[0][-1][0]
    claims = tuple(evaluate(poly, point) for poly in polynomials)
    transcript, n, omega, alpha = _begin(root, size, point, claims, queries)
    values = [
        hmul(_batch(tuple(hsub(y, c) for y, c in zip(row, claims)), alpha),
             hinv(hsub(hf(pow(omega, i, P)), point)))
        for i, row in enumerate(code)
    ]
    roots = []
    for _ in range(size.bit_length() - 1):
        r = transcript.challenge()
        half = len(values) // 2
        values = [_fold(values[i], values[i+half], hf(pow(omega, i, P)), r)
                  for i in range(half)]
        omega = omega * omega % P
        if len(values) > RATE:
            pairs = _pairs([(value,) for value in values])
            layers.append(pairs)
            trees.append(_tree(pairs))
            roots.append(trees[-1][-1][0])
            transcript.absorb(roots[-1])
    terminal = values[0]
    transcript.absorb(encode((terminal,)))
    openings = []
    for _ in range(queries):
        i = _index(transcript, n)
        chain = []
        for rows, tree in zip(layers, trees):
            i %= len(rows)
            chain.append(_open(rows, tree, i))
        openings.append(tuple(chain))
    return root, claims, Proof(tuple(roots), terminal, tuple(openings))


def verify(root, size, point, claims, proof, queries=8):
    try:
        transcript, n, omega, alpha = _begin(root, size, point, claims, queries)
        roots = (root, *proof.roots)
        if len(roots) != size.bit_length()-1 or len(proof.openings) != queries:
            return False
        challenges = []
        for next_root in roots[1:]:
            if len(next_root) != 32:
                return False
            challenges.append(transcript.challenge())
            transcript.absorb(next_root)
        challenges.append(transcript.challenge())
        transcript.absorb(encode((proof.terminal,)))
        for chain in proof.openings:
            if len(chain) != len(roots):
                return False
            index = _index(transcript, n)
            length, generator = n, omega
            previous = None
            for layer, (opening, digest, r) in enumerate(zip(chain, roots, challenges)):
                index, side = index % (length // 2), index // (length // 2)
                width = len(claims) if not layer else 1
                if not _check(opening, digest, index, length.bit_length()-2, 2*width):
                    return False
                x = hf(pow(generator, index, P))
                if layer:
                    a, b = opening.row
                    if (a, b)[side] != previous:
                        return False
                else:
                    a, b = (
                        hmul(_batch(tuple(hsub(y, c) for y, c in zip(row, claims)), alpha),
                             hinv(hsub(z, point)))
                        for row, z in ((opening.row[:width], x),
                                       (opening.row[width:], hsub(ZERO, x)))
                    )
                previous = _fold(a, b, x, r)
                length //= 2
                generator = generator * generator % P
            if previous != proof.terminal:
                return False
        return True
    except (ValueError, TypeError, IndexError, AttributeError, ZeroDivisionError):
        return False


# ============================================================================= fused two-round ZK-Basefold (relation 3)
# The committed object is a multilinear F over 2^n Fp3 cells.  Its monomial coefficients c (Mobius
# transform of the hypercube values) are read as a univariate f(X) = sum_i c_i X^i of degree < 2^n
# and Reed-Solomon encoded at rate 1/8 on L = 2^(n+3) points w^j.  One fold fixes the two lowest
# variables at once (coset-4 folding):
#     f(X) = sum_{e<4} X^e f_e(X^4),   g(Y) = f_0 + r_a f_1 + r_b f_2 + r_a r_b f_3,
# so each Merkle leaf holds the coset {f(x i^k)}_{k<4} (i a primitive 4th root of unity) and the
# fold challenges are exactly the Basefold sumcheck challenges of  claim = sum_b F(b) w(b).
# Zero knowledge comes from reserved (non-doubling) uniformly random cells of F: the honest
# prover's observation map  obs = A s + B r  satisfies im A <= im B (`require_joint_mask_ext`).
from .fpn2 import (  # noqa: E402
    Fpn2Transcript, enc3, enc3s, leaf_hash, merkle_levels, merkle_path, merkle_verify,
)

INV4 = pow(4, P - 2, P)


class PcsReject(Exception):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def mobius(values):
    """Hypercube values -> monomial coefficients of the multilinear extension."""
    c = list(values)
    n = len(c).bit_length() - 1
    for i in range(n):
        bit = 1 << i
        for idx in range(len(c)):
            if idx & bit:
                c[idx] = hsub(c[idx], c[idx ^ bit])
    return c


def _ntt_limb(a, root):
    n = len(a)
    a = list(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= n:
        w = pow(root, n // length, P)
        half = length // 2
        ws = [1] * half
        for k in range(1, half):
            ws[k] = ws[k - 1] * w % P
        for start in range(0, n, length):
            lo = a[start:start + half]
            hi = [x * y % P for x, y in zip(a[start + half:start + length], ws)]
            a[start:start + half] = [(u + v) % P for u, v in zip(lo, hi)]
            a[start + half:start + length] = [(u - v) % P for u, v in zip(lo, hi)]
        length <<= 1
    return a


def root_of_unity(length):
    return pow(7, (P - 1) // length, P)


def rs_encode(coefficients, n):
    """RS codeword f(w^j), j < 2^(n+3), of the univariate with the given Fp3 coefficients."""
    length = RATE << n
    if len(coefficients) != 1 << n:
        raise ValueError("coefficient count")
    omega = root_of_unity(length)
    limbs = [_ntt_limb([c[k] for c in coefficients] + [0] * (length - len(coefficients)), omega)
             for k in range(3)]
    return list(zip(*limbs))


def fold4(values, x_inv, iota_inv, ra, rb):
    """Fused coset-4 fold (bit-exact with kernels/goldilocks_ntt.cu::fold4_fp3)."""
    v0, v1, v2, v3 = values
    a, b = hsub(v0, v2), hsub(v1, v3)
    tb = tuple(iota_inv * x % P for x in b)
    s = (hadd(hadd(v0, v1), hadd(v2, v3)), hadd(a, tb), hsub(hadd(v0, v2), hadd(v1, v3)), hsub(a, tb))
    f, scale = [], INV4
    for e in range(4):
        f.append(tuple(scale * x % P for x in s[e]))
        scale = scale * x_inv % P
    return hadd(hadd(f[0], hmul(ra, f[1])), hadd(hmul(rb, f[2]), hmul(hmul(ra, rb), f[3])))


def fold_codeword(code, t, n, ra, rb):
    length = len(code)
    quarter = length // 4
    omega_t = root_of_unity(RATE << n)
    omega_t = pow(omega_t, 4 ** t, P)
    inv = pow(omega_t, P - 2, P)
    iota_inv = pow(root_of_unity(4), P - 2, P)
    out, x_inv = [], 1
    for j in range(quarter):
        out.append(fold4([code[j + k * quarter] for k in range(4)], x_inv, iota_inv, ra, rb))
        x_inv = x_inv * inv % P
    return out


def coset_leaves(code):
    quarter = len(code) // 4
    return [leaf_hash([code[j + k * quarter] for k in range(4)]) for j in range(quarter)]


@dataclass(frozen=True)
class Commitment:
    values: tuple       # F on the hypercube (Fp3)
    code: tuple
    levels: tuple

    @property
    def root(self):
        return self.levels[-1][0]


def zk_commit(values):
    values = tuple(values)
    n = len(values).bit_length() - 1
    if len(values) != 1 << n or n % 2 or n < 2:
        raise ValueError("ZK-Basefold needs 2^n cells with n even")
    for v in values:
        enc3(v)
    code = tuple(rs_encode(mobius(values), n))
    return Commitment(values, code, tuple(map(tuple, merkle_levels(coset_leaves(code)))))


@dataclass(frozen=True)
class QueryOpening:
    values: tuple      # per layer: 4 Fp3
    paths: tuple       # per layer: Merkle siblings


@dataclass(frozen=True)
class ZkOpening:
    rounds: tuple      # n rounds of (h(0), h(1), h(2))
    roots: tuple       # roots of folded layers 1 .. n/2 - 1
    final: tuple       # F~(r)
    queries: tuple     # QueryOpening per query

    def wire(self):
        out = b""
        for t in range(len(self.rounds) // 2):
            out += enc3s(self.rounds[2 * t]) + enc3s(self.rounds[2 * t + 1])
            if t < len(self.roots):
                out += self.roots[t]
        out += enc3(self.final)
        for q in self.queries:
            for vals, path in zip(q.values, q.paths):
                out += enc3s(vals) + b"".join(path)
        return out


def _sc_round(fv, wv):
    out = []
    for t in (0, 1, 2):
        acc = ZERO
        tt = hf(t)
        for i in range(0, len(fv), 2):
            f = hadd(fv[i], hmul(tt, hsub(fv[i + 1], fv[i])))
            w = hadd(wv[i], hmul(tt, hsub(wv[i + 1], wv[i])))
            acc = hadd(acc, hmul(f, w))
        out.append(acc)
    return tuple(out)


def _halve(v, r):
    return [hadd(v[i], hmul(r, hsub(v[i + 1], v[i]))) for i in range(0, len(v), 2)]


def zk_open(com, weights, transcript, queries, hooks=None):
    """Prove claim = sum_b F(b) w(b) (claim already fixed by the caller's transcript)."""
    hooks = hooks or {}
    n = len(com.values).bit_length() - 1
    fv, wv = list(com.values), list(weights)
    code = list(com.code)
    layers = [(com.code, com.levels)]
    rounds, roots, rs = [], [], []
    for t in range(n // 2):
        for _ in range(2):
            h = _sc_round(fv, wv)
            if "round" in hooks:
                h = hooks["round"](len(rounds), h)
            transcript.absorb_bytes(enc3s(h))
            r = transcript.squeeze_fp3()
            rounds.append(h)
            rs.append(r)
            fv, wv = _halve(fv, r), _halve(wv, r)
        code = fold_codeword(code, t, n, rs[-2], rs[-1])
        if "code" in hooks:          # tests only: commit to a wrong (but well-formed) folded layer
            code = hooks["code"](t, code)
        if t < n // 2 - 1:
            levels = tuple(map(tuple, merkle_levels(coset_leaves(code))))
            layers.append((tuple(code), levels))
            roots.append(levels[-1][0])
            transcript.absorb_bytes(roots[-1])
    final = fv[0]
    if "code" not in hooks and any(v != final for v in code):
        raise AssertionError("final folded codeword is not the constant F~(r)")
    transcript.absorb_bytes(enc3(final))
    opened, positions = [], []
    for _ in range(queries):
        pos = int.from_bytes(transcript.squeeze_bytes(8), "little") % (len(com.code) // 4)
        positions.append(pos)
        vals, paths = [], []
        for lay_code, levels in layers:
            quarter = len(lay_code) // 4
            j = pos % quarter
            vals.append(tuple(lay_code[j + k * quarter] for k in range(4)))
            paths.append(tuple(merkle_path(levels, j)))
            pos = j
        opened.append(QueryOpening(tuple(vals), tuple(paths)))
    return ZkOpening(tuple(rounds), tuple(roots), final, tuple(opened)), tuple(rs), tuple(positions)


def zk_verify(root0, n, claim, w_eval, opening, transcript, queries):
    """Replay the fused folded opening; `w_eval(r)` is the verifier-side MLE of the weights."""
    if len(opening.rounds) != n or len(opening.roots) != n // 2 - 1 or len(opening.queries) != queries:
        raise PcsReject("E_LENGTH")
    rs = []
    for t in range(n // 2):
        for half in range(2):
            h = opening.rounds[2 * t + half]
            transcript.absorb_bytes(enc3s(h))
            if hadd(h[0], h[1]) != claim:
                raise PcsReject("E_SUMCHECK", f"pcs round {2 * t + half}")
            r = transcript.squeeze_fp3()
            rs.append(r)
            claim = lagrange_eval(h, r)
        if t < n // 2 - 1:
            transcript.absorb_bytes(opening.roots[t])
    transcript.absorb_bytes(enc3(opening.final))
    if claim != hmul(opening.final, w_eval(tuple(rs))):
        raise PcsReject("E_FINAL")
    roots = (root0, *opening.roots)
    length = RATE << n
    omega = root_of_unity(length)
    iota_inv = pow(root_of_unity(4), P - 2, P)
    for q in opening.queries:
        pos = int.from_bytes(transcript.squeeze_bytes(8), "little") % (length // 4)
        prev = None
        for t in range(n // 2):
            quarter = length >> (2 * t + 2)
            j, slot = pos % quarter, pos // quarter
            vals = q.values[t]
            if not merkle_verify(roots[t], leaf_hash(vals), j, q.paths[t]) or len(q.paths[t]) != quarter.bit_length() - 1:
                raise PcsReject("E_MERKLE", f"layer {t}")
            if t and vals[slot] != prev:
                raise PcsReject("E_FOLD", f"layer {t}")
            x_inv = pow(pow(omega, 4 ** t, P), (P - 1 - j) % (P - 1), P)
            prev = fold4(vals, x_inv, iota_inv, rs[2 * t], rs[2 * t + 1])
            pos = j
        if prev != opening.final:
            raise PcsReject("E_FOLD", "terminal")
    return tuple(rs)


# ----------------------------------------------------------------------------- ZK rank condition over Fp3
def rank_ext(matrix):
    rows = [list(row) for row in matrix]
    if not rows:
        return 0
    width = len(rows[0])
    rank_so_far = 0
    for column in range(width):
        pivot = next((i for i in range(rank_so_far, len(rows)) if rows[i][column] != ZERO), None)
        if pivot is None:
            continue
        rows[rank_so_far], rows[pivot] = rows[pivot], rows[rank_so_far]
        row = rows[rank_so_far]
        inverse = hinv(row[column])
        for other in rows[rank_so_far + 1:]:
            if other[column] == ZERO:
                continue
            factor = hmul(other[column], inverse)
            for j in range(column, width):
                other[j] = hsub(other[j], hmul(factor, row[j]))
        rank_so_far += 1
    return rank_so_far


def require_joint_mask_ext(secret, mask):
    """im(A) <= im(B) over Fp3: every A(x - x') is absorbed by a change of the reserved mask cells."""
    if len(secret) != len(mask):
        raise ValueError("observation count mismatch")
    if rank_ext([tuple(a) + tuple(b) for a, b in zip(secret, mask)]) != rank_ext(mask):
        raise ValueError("under-rank mask")


def observations(values, weights, rs, positions):
    """Every value a verifier sees in the opening, as a linear function of F for fixed challenges."""
    n = len(values).bit_length() - 1
    fv, wv, out = list(values), list(weights), []
    code = rs_encode(mobius(values), n)
    layers = [code]
    for t in range(n // 2):
        for half in range(2):
            out.extend(_sc_round(fv, wv))
            r = rs[2 * t + half]
            fv, wv = _halve(fv, r), _halve(wv, r)
        code = fold_codeword(code, t, n, rs[2 * t], rs[2 * t + 1])
        layers.append(code)
    out.append(fv[0])
    for pos in positions:
        for lay in layers[:-1]:
            quarter = len(lay) // 4
            j = pos % quarter
            out.extend(lay[j + k * quarter] for k in range(4))
            pos = j
    return out


def observation_matrix(size, weights, rs, positions, columns):
    """Columns of the observation map restricted to the given cell indices."""
    cols = []
    for idx in columns:
        e = [ZERO] * size
        e[idx] = ONE
        cols.append(observations(e, weights, rs, positions))
    return [tuple(col[i] for col in cols) for i in range(len(cols[0]))]


# ----------------------------------------------------------------------------- proof-size calculator
MIB = 1 << 20


def byte_budget(cells, cohorts=2, queries=256, field_bytes=8, digest=32):
    """Exact fixed-size proof bytes of the fused folded PCS (+ GKR) for `cells` committed cells.

    Each cohort commits ceil(cells / cohorts) base-field cells as a multilinear polynomial with an
    even number n of variables, RS-encoded at rate 1/8 (L = 2^(n+3)); layer-0 Merkle leaves hold a
    coset of 4 values per cohort; after a random batching combination a single Fp3 fold chain of
    n/2 coset-4 folds follows. The FracLogUp-GKR over 2^(n+1) leaves adds its masked layers.
    """
    per = -(-cells // cohorts)
    n = max(2, (per - 1).bit_length())
    n += n % 2
    folds = n // 2
    depth = [n + 1 - 2 * t for t in range(folds)]
    s = n + 1
    parts = {
        "header": 64,
        "statement": digest * cohorts + 32,
        "gkr": 48 + 96 + sum(24 + 96 * k + 120 for k in range(1, s + 1)),
        "pcs_sumcheck": n * 3 * 24,
        "pcs_roots": digest * (folds - 1),
        "pcs_final": 24,
        "query_layer0": queries * cohorts * (4 * field_bytes + digest * depth[0]),
        "query_folds": queries * sum(4 * 24 + digest * depth[t] for t in range(1, folds)),
    }
    total = sum(parts.values())
    return {"cells": cells, "cohorts": cohorts, "queries": queries, "n_vars": n, "rate": RATE,
            "codeword": RATE << n, "folds": folds, "parts": parts, "total": total,
            "fits_8mib": total <= 8 * MIB}


def _eq_table(point):
    out = [ONE]
    for r in point:
        out = [hmul(w, f) for f in (hsub(ONE, r), r) for w in out]
    return out


def observation_rows(n, weights, rs, positions):
    """Closed form of `observations`: one functional (row over all 2^n cells) per observed value.

    Codeword value of folded layer t at position p is F~(r_0..r_{2t-1}, x, x^2, x^4, ...) with
    x = w_t^p; the final value is F~(r); PCS round i at X = t has coefficient
    eq((r_<i, t), b_<=i) * W~(r_<i, t, b_>i) on cell b.
    """
    size = 1 << n
    rows = []
    wv = list(weights)
    for i in range(n):
        for t in (0, 1, 2):
            tt = hf(t)
            low = _eq_table((*rs[:i], tt))
            high = [hadd(wv[2 * j], hmul(tt, hsub(wv[2 * j + 1], wv[2 * j]))) for j in range(len(wv) // 2)]
            rows.append([hmul(low[b & ((2 << i) - 1)], high[b >> (i + 1)]) for b in range(size)])
        wv = _halve(wv, rs[i])
    rows.append(_eq_table(rs))
    length = RATE << n
    omega = root_of_unity(length)
    for pos in positions:
        for t in range(n // 2):
            quarter = length >> (2 * t + 2)
            j = pos % quarter
            omega_t = pow(omega, 4 ** t, P)
            for k in range(4):
                x = pow(omega_t, j + k * quarter, P)
                pt = list(rs[:2 * t])
                for _ in range(n - 2 * t):
                    pt.append(hf(x))
                    x = x * x % P
                rows.append(_eq_table(pt))
            pos = j
    return rows


def split_columns(rows, columns):
    return [tuple(row[c] for c in columns) for row in rows]
