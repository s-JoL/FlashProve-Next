# SPDX-License-Identifier: Apache-2.0
"""Fiat-Shamir transcript (BLAKE2b-512) and a safe proof codec (gzip'd JSON, no pickle)."""
import base64
import gzip
import hashlib
import json
import numpy as np
from .field import P


def _norm(x):
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in sorted(x.items())}
    return x


def canon(x):
    return json.dumps(_norm(x), separators=(",", ":")).encode()


class Transcript:
    def __init__(self, label=b"fpn-flat-v1"):
        self.h = hashlib.blake2b(label, digest_size=64).digest()

    def absorb(self, *parts):
        m = hashlib.blake2b(self.h, digest_size=64)
        for p in parts:
            if isinstance(p, (bytes, bytearray)):
                m.update(b"B" + len(p).to_bytes(8, "little") + bytes(p))
            elif isinstance(p, np.ndarray):
                b = np.ascontiguousarray(p).tobytes(); m.update(b"A" + len(b).to_bytes(8, "little") + b)
            else:
                b = canon(p); m.update(b"J" + len(b).to_bytes(8, "little") + b)
        self.h = m.digest()

    def _next(self):
        self.h = hashlib.blake2b(self.h + b"c", digest_size=64).digest()
        return self.h

    def ext(self):
        out = []
        while len(out) < 3:
            d = self._next()
            for i in range(0, 64, 8):
                v = int.from_bytes(d[i:i + 8], "little")
                if v < P:
                    out.append(v)
                    if len(out) == 3:
                        return (out[0], out[1], out[2])

    def exts(self, k):
        return [self.ext() for _ in range(k)]

    def indices(self, t, n):
        out = []
        while len(out) < t:
            d = self._next()
            for i in range(0, 64, 8):
                out.append(int.from_bytes(d[i:i + 8], "little") % n)
        return out[:t]


# ----------------------------------------------------------------------------- codec
_DT = {"uint64", "int64", "int32", "uint8", "uint16"}


def _enc(x):
    if isinstance(x, (bytes, bytearray)):
        return {"$b": base64.b64encode(bytes(x)).decode()}
    if isinstance(x, np.ndarray):
        assert str(x.dtype) in _DT, x.dtype
        return {"$a": str(x.dtype), "s": list(x.shape), "d": base64.b64encode(np.ascontiguousarray(x).tobytes()).decode()}
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, dict):
        return {str(k): _enc(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_enc(v) for v in x]
    if x is None or isinstance(x, (int, str, float, bool)):
        return x
    raise TypeError(type(x))


def _dec(x):
    if isinstance(x, dict):
        if set(x) == {"$b"}:
            return base64.b64decode(x["$b"], validate=True)
        if set(x) == {"$a", "s", "d"}:
            if x["$a"] not in _DT:
                raise ValueError("unsupported array dtype")
            sh = x["s"]
            if (not isinstance(sh, list) or len(sh) > 4
                    or any(isinstance(d, bool) or not isinstance(d, int) or d < 0 or d > (1 << 28) for d in sh)):
                raise ValueError("invalid array shape")
            raw_arr = base64.b64decode(x["d"], validate=True)
            return np.frombuffer(raw_arr, dtype=x["$a"]).reshape(sh).copy()
        return {k: _dec(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_dec(v) for v in x]
    return x


def dumps(obj):
    return gzip.compress(json.dumps(_enc(obj), separators=(",", ":")).encode(), 6)


def loads(b, max_bytes=512 << 20):
    import io
    with gzip.GzipFile(fileobj=io.BytesIO(bytes(b))) as gz:
        raw = gz.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("decompressed proof exceeds size limit")
    return _dec(json.loads(raw))
