# SPDX-License-Identifier: Apache-2.0
"""Host description for benchmark records: GPU name, CPU model, logical cores, RAM.

The GPU is named through ``torch.cuda.get_device_name(0)`` (or CuPy) when the calling process already uses CUDA, and
through ``nvidia-smi`` otherwise, so that describing the host never opens a CUDA context in a driver process whose
children are being measured.
"""
import os
import platform
import shutil
import subprocess
import sys


def nvidia_smi():
    """Path of the nvidia-smi binary: $FPN_NVIDIA_SMI if set (for driver directories that are not on PATH), else the
    first nvidia-smi on PATH, else None."""
    env = os.environ.get("FPN_NVIDIA_SMI")
    if env and os.path.exists(env):
        return env
    return shutil.which("nvidia-smi")


def _torch_gpu(import_ok):
    if "torch" not in sys.modules and not import_ok:
        return None
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return str(torch.cuda.get_device_name(0)), round(p.total_memory / (1 << 30), 1)
    except Exception:
        pass
    return None


def _cupy_gpu():
    if "cupy" not in sys.modules:
        return None
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props.get("name", b"")
        name = name.decode() if isinstance(name, bytes) else str(name)
        return name, round(int(props.get("totalGlobalMem", 0)) / (1 << 30), 1) or None
    except Exception:
        return None


def _smi_gpu():
    smi = nvidia_smi()
    if not smi:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        line = out.stdout.strip().splitlines()
        if line:
            name, mib = (s.strip() for s in line[0].split(",", 1))
            return name, round(float(mib) / 1024, 1)
    except Exception:
        pass
    return None


def gpu(allow_cuda_init=False):
    """(name, total GiB) of GPU 0, or None.  Importing torch to ask CUDA directly (which opens a CUDA context) happens
    only with allow_cuda_init=True."""
    return _torch_gpu(False) or _cupy_gpu() or _smi_gpu() or (_torch_gpu(True) if allow_cuda_init else None)


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def ram_gib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1 << 20), 1)
    except OSError:
        pass
    return None


def describe(allow_cuda_init=False):
    """One-line host description, e.g. 'NVIDIA H100 80GB HBM3 (79.6 GiB) + AMD EPYC 9454 (96 threads, 755.2 GiB RAM)'."""
    g = gpu(allow_cuda_init)
    ram = ram_gib()
    cpu = f"{cpu_name()} ({os.cpu_count()} threads" + (f", {ram} GiB RAM)" if ram else ")")
    if g is None:
        return f"CPU only: {cpu}"
    name, mem = g
    return name + (f" ({mem} GiB)" if mem else "") + f" + {cpu}"
