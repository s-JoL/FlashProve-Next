# SPDX-License-Identifier: Apache-2.0
"""``ResourceBudget``: the ONE place in the codebase that reads environment variables / hardware limits.

Protocol, circuit and serving code receive a ``ResourceBudget`` (or plain numbers derived from it); they never touch
``os.environ``.  Hard-won host-memory and driver safety rules enforced here:

* Linux ``tmpfs`` / ``ramfs`` mounts are RAM-backed: GB-scale spill files must never go there (they exhaust host RAM).
  ``spill_dir`` therefore rejects ``/tmp`` (also its macOS realpath ``/private/tmp``), ``/var/tmp``, ``/dev/shm`` and
  *any* directory on a ``tmpfs`` / ``ramfs`` mount (tested against an injectable mount table).
* Host <-> device copies of big arrays are chunked at 64 MiB (``h2d_chunk_mib``) so drivers do not pin an oversized
  staging area indefinitely.
* Everything is validated at construction; a violated budget is a ``BudgetError``, not a warning.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional

__all__ = [
    "BudgetError",
    "ResourceBudget",
    "PRESETS",
    "validate_spill_dir",
    "detect_host_ram_gib",
    "detect_vram_gib",
    "DEFAULT_SPILL_DIR",
    "FORBIDDEN_SPILL_PREFIXES",
    "TMPFS_FSTYPES",
]

GIB = 1 << 30
MIB = 1 << 20

DEFAULT_SPILL_DIR = "~/.cache/fpn-spill"
FORBIDDEN_SPILL_PREFIXES: tuple[str, ...] = ("/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp", "/dev/shm", "/run/shm")
TMPFS_FSTYPES = frozenset({"tmpfs", "ramfs"})

# Environment variable names (read only by ResourceBudget.from_env).
ENV_PREFIX = "FPN_"
_ENV_FIELDS = {
    "FPN_PROFILE": "profile",
    "FPN_VRAM_LIMIT_GIB": "vram_limit_gib",
    "FPN_HOST_RAM_LIMIT_GIB": "host_ram_limit_gib",
    "FPN_SPILL_DIR": "spill_dir",
    "FPN_H2D_CHUNK_MIB": "h2d_chunk_mib",
    "FPN_STAGING_RING_MIB": "staging_ring_mib",
    "FPN_MAX_INFLIGHT_TOKENS": "max_inflight_tokens",
}


class BudgetError(ValueError):
    """The requested resource budget is invalid or unsafe."""


def _read_mounts(mounts_text: Optional[str]) -> list[tuple[str, str]]:
    """Return ``[(mount_point, fstype)]`` from ``/proc/mounts``-formatted text (or the real file when ``None``)."""
    if mounts_text is None:
        try:
            with open("/proc/mounts", "r", encoding="utf-8") as fh:
                mounts_text = fh.read()
        except OSError:
            return []          # macOS & friends: no /proc; the prefix blacklist still applies
    out = []
    for line in mounts_text.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            out.append((parts[1].replace("\\040", " "), parts[2]))
    return out


def _is_under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def validate_spill_dir(path: "str | os.PathLike[str]", *, mounts_text: Optional[str] = None) -> Path:
    """Expand and check a spill directory; returns the absolute, symlink-resolved path (not created)."""
    raw = os.path.abspath(os.path.expanduser(os.fspath(path)))
    resolved = os.path.realpath(raw)
    for cand in (raw, resolved):
        for pre in FORBIDDEN_SPILL_PREFIXES:
            if _is_under(cand, pre):
                raise BudgetError(f"spill_dir {raw!r} is under {pre!r} (RAM-backed on WSL): use {DEFAULT_SPILL_DIR}")
    best: Optional[tuple[str, str]] = None
    for mp, fs in _read_mounts(mounts_text):
        if _is_under(resolved, mp) and (best is None or len(mp) > len(best[0])):
            best = (mp, fs)
    if best is not None and best[1] in TMPFS_FSTYPES:
        raise BudgetError(f"spill_dir {resolved!r} is on a {best[1]} mount at {best[0]!r}: it would consume host RAM")
    return Path(resolved)


def detect_host_ram_gib() -> Optional[float]:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIB
    except (ValueError, OSError, AttributeError):
        return None


def detect_vram_gib() -> Optional[float]:
    """Total memory of device 0 if torch+CUDA are importable, else ``None`` (never required)."""
    try:
        import torch  # noqa: WPS433 - optional, lazy

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / GIB
    except Exception:  # pragma: no cover - environment dependent
        return None
    return None


@dataclass(frozen=True)
class ResourceBudget:
    """Immutable hardware/memory policy handed to every component that needs one."""

    profile: str = "custom"
    vram_limit_gib: float = 14.0
    host_ram_limit_gib: float = 20.0
    spill_dir: str = DEFAULT_SPILL_DIR
    h2d_chunk_mib: int = 64
    staging_ring_mib: int = 320
    max_inflight_tokens: int = 2048

    def __post_init__(self) -> None:
        if not (self.vram_limit_gib > 0 and self.host_ram_limit_gib > 0):
            raise BudgetError("memory limits must be > 0")
        if self.h2d_chunk_mib < 1 or self.h2d_chunk_mib > 64:
            raise BudgetError("h2d_chunk_mib must be in [1, 64] (WSL pins the largest single copy forever)")
        if self.staging_ring_mib < 1:
            raise BudgetError("staging_ring_mib must be >= 1")
        if self.staging_ring_mib > self.vram_limit_gib * 1024 // 4:
            raise BudgetError("staging ring may use at most 1/4 of the VRAM limit")
        if self.max_inflight_tokens < 1:
            raise BudgetError("max_inflight_tokens must be >= 1")
        validate_spill_dir(self.spill_dir)        # rejects /tmp, tmpfs, ...

    # ------------------------------------------------------------------------------------------------ derived values
    @property
    def spill_path(self) -> Path:
        return validate_spill_dir(self.spill_dir)

    @property
    def h2d_chunk_bytes(self) -> int:
        return self.h2d_chunk_mib * MIB

    @property
    def staging_ring_bytes(self) -> int:
        return self.staging_ring_mib * MIB

    @property
    def vram_limit_bytes(self) -> int:
        return int(self.vram_limit_gib * GIB)

    @property
    def host_ram_limit_bytes(self) -> int:
        return int(self.host_ram_limit_gib * GIB)

    def chunk_plan(self, nbytes: int) -> list[tuple[int, int]]:
        """``[(offset, length)]`` covering ``nbytes`` in chunks of at most ``h2d_chunk_mib`` MiB."""
        if nbytes < 0:
            raise BudgetError("negative size")
        step = self.h2d_chunk_bytes
        return [(o, min(step, nbytes - o)) for o in range(0, nbytes, step)]

    def ring_rows(self, row_bytes: int) -> int:
        """Largest row count whose ring fits in ``staging_ring_mib`` (the ring indexes rows modulo capacity)."""
        if row_bytes < 1:
            raise BudgetError("row_bytes must be >= 1")
        n = self.staging_ring_bytes // row_bytes
        if n < 1:
            raise BudgetError("a single row does not fit in the staging ring")
        return n

    def check_ring_fits(self, row_bytes: int, *, capacity_rows: int, tier_cap_mib: Optional[int] = None) -> int:
        """Return ring bytes; raise if it exceeds the staging budget (or the stricter per-tier cap)."""
        need = row_bytes * capacity_rows
        cap = self.staging_ring_bytes if tier_cap_mib is None else min(self.staging_ring_bytes, tier_cap_mib * MIB)
        if need > cap:
            raise BudgetError(f"ring needs {need / MIB:.1f} MiB > budget {cap / MIB:.1f} MiB")
        return need

    def with_overrides(self, **kw: object) -> "ResourceBudget":
        return replace(self, **kw)  # type: ignore[arg-type]

    # ------------------------------------------------------------------------------------------------ constructors
    @classmethod
    def preset(cls, name: str) -> "ResourceBudget":
        try:
            return PRESETS[name]
        except KeyError:
            raise BudgetError(f"unknown preset {name!r}; known: {sorted(PRESETS)}") from None

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None, *, base: Optional["ResourceBudget"] = None) -> "ResourceBudget":
        """Build from ``FPN_*`` variables (``environ`` defaults to ``os.environ`` -- the only such read in the repo).

        ``FPN_PROFILE`` picks a preset first; the remaining variables override individual fields."""
        env = os.environ if environ is None else environ
        prof = env.get("FPN_PROFILE")
        b = cls.preset(prof) if prof else (base or cls())
        kw: dict[str, object] = {}
        for var, fld in _ENV_FIELDS.items():
            if fld == "profile" or var not in env:
                continue
            raw = env[var]
            try:
                kw[fld] = raw if fld == "spill_dir" else (float(raw) if fld.endswith("_gib") else int(raw))
            except ValueError:
                raise BudgetError(f"{var}={raw!r} is not a valid number") from None
        return replace(b, **kw) if kw else b      # type: ignore[arg-type]

    @classmethod
    def from_hardware(cls, *, headroom_gib: float = 1.5, base: Optional["ResourceBudget"] = None) -> "ResourceBudget":
        """Clamp a base budget to what the machine reports (best effort; never raises on missing probes)."""
        b = base or cls()
        vram, ram = detect_vram_gib(), detect_host_ram_gib()
        kw: dict[str, object] = {}
        if vram is not None:
            kw["vram_limit_gib"] = min(b.vram_limit_gib, max(1.0, vram - headroom_gib))
        if ram is not None:
            kw["host_ram_limit_gib"] = min(b.host_ram_limit_gib, max(1.0, ram - 3.0))
        return replace(b, **kw) if kw else b      # type: ignore[arg-type]


PRESETS: dict[str, ResourceBudget] = {
    # 16 GB consumer workstation preset (e.g. RTX 4070 Ti SUPER 16 GB, 25 GB RAM).
    "4070ti_super_16gb": ResourceBudget(
        profile="4070ti_super_16gb", vram_limit_gib=14.5, host_ram_limit_gib=21.0, spill_dir=DEFAULT_SPILL_DIR,
        h2d_chunk_mib=64, staging_ring_mib=320, max_inflight_tokens=2048,
    ),
    "rtx6000d_84gb": ResourceBudget(
        profile="rtx6000d_84gb", vram_limit_gib=83.0, host_ram_limit_gib=896.0, spill_dir=DEFAULT_SPILL_DIR,
        h2d_chunk_mib=64, staging_ring_mib=320, max_inflight_tokens=16384,
    ),
    "h100_80gb": ResourceBudget(
        profile="h100_80gb", vram_limit_gib=80.0, host_ram_limit_gib=200.0, spill_dir=DEFAULT_SPILL_DIR,
        h2d_chunk_mib=64, staging_ring_mib=320, max_inflight_tokens=16384,
    ),
}
