#!/usr/bin/env python3
"""Gate 0 environment check.

Reports what this machine can actually do. Every line is backed by a real
probe: a program that ran, an import that succeeded and did real work, or a
number read from the OS. If a probe cannot run, the status is FAIL or
NOT AVAILABLE. It is never PASS.

Statuses
  PASS           detected and working
  WARN           detected and working, but below the recommended level
  FAIL           a required tool is missing or broken
  NOT AVAILABLE  optional hardware is absent (e.g. no NVIDIA GPU)
  INFO           informational only, no verdict

Exit code: 0 if every REQUIRED item is PASS or WARN, otherwise 1.
GPU items are optional at Gate 0 (Gates 0-2 run on CPU), so a missing GPU
does not fail the check.

Standalone on purpose: it does not import the project package, so it still
works when the rest of the environment is broken.

Usage:  python scripts/check_env.py
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Heuristic thresholds. These only decide PASS vs WARN, never block Gate 0.
MIN_PYTHON = (3, 10)
RECOMMENDED_PYTHON_MAX = (3, 12)
VRAM_PASS_GB = 12.0
VRAM_WARN_GB = 8.0
DISK_PASS_GB = 50.0
DISK_WARN_GB = 10.0

PASS, WARN, FAIL, NA, INFO = "PASS", "WARN", "FAIL", "NOT AVAILABLE", "INFO"


@dataclass
class Result:
    name: str
    status: str
    detail: str = ""
    required: bool = False


def run(cmd: list[str], timeout: int = 15):
    """Run a command. Returns (returncode, stdout, stderr), or None if it could not run."""
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return p.returncode, p.stdout, p.stderr


# --------------------------------------------------------------------------- #
# System
# --------------------------------------------------------------------------- #
def check_os() -> Result:
    detail = f"{platform.platform()} ({platform.machine()}), {os.cpu_count()} logical CPUs"
    if platform.system() == "Linux" and "microsoft" in platform.release().lower():
        detail += " [WSL]"
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        detail += " [Apple Silicon: no NVIDIA/CUDA]"
    return Result("OS", INFO, detail)


def check_python() -> Result:
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < MIN_PYTHON:
        return Result("Python", FAIL, f"{ver}: need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}", True)
    if (v.major, v.minor) > RECOMMENDED_PYTHON_MAX:
        return Result(
            "Python",
            WARN,
            f"{ver}: works for Gate 0, but ML libraries may lag; 3.10-3.12 recommended",
            True,
        )
    return Result("Python", PASS, f"{ver} ({sys.executable})", True)


def check_venv() -> Result:
    in_env = sys.prefix != getattr(sys, "base_prefix", sys.prefix) or bool(
        os.environ.get("CONDA_PREFIX")
    )
    if in_env:
        return Result("Virtual env", PASS, sys.prefix)
    return Result("Virtual env", WARN, "not in a venv/conda env; use one for reproducibility")


# --------------------------------------------------------------------------- #
# FFmpeg / FFprobe
# --------------------------------------------------------------------------- #
def check_binary(exe: str, label: str) -> Result:
    path = shutil.which(exe)
    if not path:
        return Result(label, FAIL, f"'{exe}' not found on PATH", True)
    r = run([path, "-version"])
    if r is None or r[0] != 0:
        return Result(label, FAIL, f"found at {path} but '-version' failed", True)
    m = re.search(r"version\s+(\S+)", r[1].splitlines()[0] if r[1] else "")
    if not m:
        return Result(label, FAIL, f"found at {path} but version output unrecognized", True)
    return Result(label, PASS, f"{m.group(1)} ({path})", True)


def check_libx264() -> Result:
    label = "FFmpeg libx264"
    path = shutil.which("ffmpeg")
    if not path:
        return Result(label, NA, "requires a working ffmpeg")
    r = run([path, "-hide_banner", "-encoders"])
    if r is None or r[0] != 0:
        return Result(label, NA, "could not list ffmpeg encoders")
    if re.search(r"^\s*V\S*\s+libx264\s", r[1], re.M):
        return Result(label, PASS, "H.264 encoder present")
    return Result(label, WARN, "H.264 (libx264) encoder missing; install a full FFmpeg build")


# --------------------------------------------------------------------------- #
# Python packages (import AND do real work)
# --------------------------------------------------------------------------- #
def check_numpy() -> Result:
    try:
        import numpy as np
    except Exception as e:  # noqa: BLE001 - any import-time failure is a FAIL
        return Result("NumPy", FAIL, f"import failed: {type(e).__name__}: {e}", True)
    try:
        if int(np.arange(10).sum()) != 45:
            raise RuntimeError("arithmetic check returned a wrong result")
    except Exception as e:  # noqa: BLE001
        return Result("NumPy", FAIL, f"imported but not working: {e}", True)
    return Result("NumPy", PASS, np.__version__, True)


def check_opencv() -> Result:
    try:
        import cv2
        import numpy as np
    except Exception as e:  # noqa: BLE001
        return Result("OpenCV", FAIL, f"import failed: {type(e).__name__}: {e}", True)
    try:
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, _buf = cv2.imencode(".png", img)
        if gray.shape != (8, 8) or not ok:
            raise RuntimeError("basic operations returned unexpected results")
    except Exception as e:  # noqa: BLE001
        return Result("OpenCV", FAIL, f"imported but not working: {e}", True)
    return Result("OpenCV", PASS, f"{cv2.__version__} (color convert + PNG encode OK)", True)


def check_pytest() -> Result:
    try:
        import pytest
    except Exception as e:  # noqa: BLE001
        return Result("Pytest", FAIL, f"import failed: {type(e).__name__}: {e}", True)
    return Result("Pytest", PASS, pytest.__version__, True)


# --------------------------------------------------------------------------- #
# NVIDIA GPU / CUDA / VRAM (all optional at Gate 0)
# --------------------------------------------------------------------------- #
def check_gpu() -> list[Result]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        msg = "nvidia-smi not found (no NVIDIA driver installed)"
        return [Result("NVIDIA GPU", NA, msg), Result("CUDA", NA, msg), Result("VRAM", NA, msg)]

    q = run(
        [
            smi,
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if q is None or q[0] != 0 or not q[1].strip():
        err = ((q[2] or q[1]).strip().splitlines() or ["no output"])[0] if q else "could not execute"
        msg = f"nvidia-smi found but failed: {err}"
        return [
            Result("NVIDIA GPU", FAIL, msg),
            Result("CUDA", NA, "no working driver"),
            Result("VRAM", NA, "no working driver"),
        ]

    gpus = []
    for line in q[1].strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        try:
            name = ",".join(parts[:-3])
            total_gb = float(parts[-3]) / 1024
            free_gb = float(parts[-2]) / 1024
            driver = parts[-1]
        except (ValueError, IndexError):
            continue
        gpus.append((name, total_gb, free_gb, driver))
    if not gpus:
        return [
            Result("NVIDIA GPU", FAIL, f"could not parse nvidia-smi output: {q[1].strip()[:80]}"),
            Result("CUDA", NA, "GPU not parsed"),
            Result("VRAM", NA, "GPU not parsed"),
        ]

    results = []
    names = "; ".join(f"{n} (driver {d})" for n, _, _, d in gpus)
    results.append(Result("NVIDIA GPU", PASS, f"{len(gpus)}x: {names}"))

    # CUDA: the max CUDA version the *driver* supports. PyTorch wheels bundle their
    # own CUDA runtime, so this (not a local toolkit) decides which wheel we can use.
    header = run([smi])
    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", header[1]) if header else None
    nvcc = shutil.which("nvcc")
    nvcc_ver = None
    if nvcc:
        r = run([nvcc, "--version"])
        mm = re.search(r"release\s+([0-9]+\.[0-9]+)", r[1]) if r else None
        nvcc_ver = mm.group(1) if mm else None
    toolkit = f"; local toolkit nvcc {nvcc_ver}" if nvcc_ver else "; no local nvcc (not required)"
    if m:
        results.append(Result("CUDA", PASS, f"driver supports up to CUDA {m.group(1)}{toolkit}"))
    else:
        results.append(Result("CUDA", FAIL, "could not read CUDA version from nvidia-smi"))

    best = max(gpus, key=lambda g: g[1])
    detail = "; ".join(f"{n}: {t:.1f} GB total, {f:.1f} GB free" for n, t, f, _ in gpus)
    if best[1] >= VRAM_PASS_GB:
        status = PASS
    elif best[1] >= VRAM_WARN_GB:
        status = WARN
        detail += f" (below {VRAM_PASS_GB:.0f} GB: some later gates will need smaller models)"
    else:
        status = WARN
        detail += f" (below {VRAM_WARN_GB:.0f} GB: expect CPU/proxy-res fallbacks)"
    results.append(Result("VRAM", status, detail))
    return results


# --------------------------------------------------------------------------- #
# Disk
# --------------------------------------------------------------------------- #
def check_disk() -> Result:
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)
    except OSError as e:
        return Result("Disk space", FAIL, f"could not read disk usage: {e}")
    free_gb = usage.free / 1024**3
    total_gb = usage.total / 1024**3
    detail = f"{free_gb:.1f} GB free of {total_gb:.1f} GB (drive holding {PROJECT_ROOT})"
    if free_gb >= DISK_PASS_GB:
        return Result("Disk space", PASS, detail)
    if free_gb >= DISK_WARN_GB:
        return Result("Disk space", WARN, detail + "; lossless PNG frames are large")
    return Result("Disk space", FAIL, detail + "; need at least 10 GB")


# --------------------------------------------------------------------------- #
def main() -> int:
    try:
        sys.stdout.reconfigure(errors="replace")  # avoid crashes on odd console encodings
    except Exception:  # noqa: BLE001
        pass

    results: list[Result] = [
        check_os(),
        check_python(),
        check_binary("ffmpeg", "FFmpeg"),
        check_binary("ffprobe", "FFprobe"),
        check_opencv(),
        check_numpy(),
        check_pytest(),
        *check_gpu(),
        check_disk(),
        check_venv(),
        check_libx264(),
    ]

    print("Environment Check")
    print("=================")
    for r in results:
        print(f"{r.name:<16} {r.status:<14} {r.detail}")
    print()

    bad = [r.name for r in results if r.required and r.status in (FAIL, NA)]
    gpu_ok = any(r.name == "NVIDIA GPU" and r.status == PASS for r in results)
    if bad:
        print(f"REQUIRED ITEMS FAILING: {', '.join(bad)}")
    else:
        print("Required items: all OK (Gates 0-2 can run on this machine).")
    print(
        "GPU: available, AI gates can run locally."
        if gpu_ok
        else "GPU: no working NVIDIA GPU detected; Gates 0-2 are unaffected, but we must "
        "decide how to run the AI gates (Gate 3+) before choosing the AI stack."
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
