# Virtual Product Placement (working title)

## 1. What we are building

An AI system that takes a licensed movie or web-series video plus a brand/product asset, finds a suitable object in a scene (for example a generic soda bottle), and replaces it with the branded product (for example a Sprite bottle). The replacement must preserve the scene's motion, perspective, lighting, occlusion, shadows, and temporal consistency.

Approach: a hybrid pipeline. Geometric compositing is the controlled backbone. Generative models are used only where geometry can't help (background reconstruction). The unit of processing is one shot. Every stage reads and writes plain files under `runs/`, so each stage can be inspected and tested on its own.

Development is a sequence of independently verified gates. A gate is FROZEN only after it passes on a real machine.

## 2. Current gate

**Gate 0: Environment + project skeleton.** Status: awaiting results from your machine.

Scope: a pinned-minimum Python environment, an environment report, and one smoke test. There is no AI code yet.

## 3. Installation

You need Python 3.10-3.12 (3.11 recommended) and FFmpeg.

**Step 1: FFmpeg (includes ffprobe)**

| OS | Command |
|---|---|
| Windows | `winget install Gyan.FFmpeg` (then open a NEW terminal) |
| macOS | `brew install ffmpeg` |
| Ubuntu/Debian | `sudo apt update && sudo apt install -y ffmpeg` |

**Step 2: Python environment** (run from the project root)

Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then retry.

There is nothing to install for the GPU at this stage. The check only reads what your NVIDIA driver already reports.

## 4. Run the environment check

```bash
python scripts/check_env.py
```

Statuses:

| Status | Meaning |
|---|---|
| PASS | Detected and actually working (imports are exercised, programs are executed) |
| WARN | Works, but below the recommended level |
| FAIL | A required tool is missing or broken |
| NOT AVAILABLE | Optional hardware is absent (e.g. no NVIDIA GPU) |
| INFO | Information only |

Exit code is `0` if all required items (Python, FFmpeg, FFprobe, OpenCV, NumPy, Pytest) are OK, and `1` otherwise. A missing GPU does not fail Gate 0, because Gates 0-2 run on CPU. It does decide our AI stack later.

Notes on how to read it:
- **CUDA** is the highest CUDA version your *driver* supports. That is what decides which PyTorch build we can install. You don't need to install the CUDA toolkit.
- Thresholds for VRAM (12 GB) and disk (50 GB) are planning heuristics that only choose PASS vs WARN.

## 5. Run the tests

```bash
python -m pytest -v
```

Expected: `1 passed`.

## 6. What to send back

Paste these, as text, in one message:

1. The full output of `python scripts/check_env.py`.
2. The full output of `python -m pytest -v`.
3. The output of `pip freeze` (I'll freeze exact versions from it).
4. Anything that errored, with the exact command you ran.

Optional but saves time later: film **Clip A**, a locked-off phone clip (tripod or phone propped up), 5-10 s, 1080p, with a bottle standing on a table. Put it in `data/input/`. Don't send it yet.

## Project layout

```
src/            project package (pipeline code lands here, one module per gate)
scripts/        runnable entry points (check_env.py)
tests/          pytest tests
data/input/     source videos (git-ignored: licensed footage is never committed)
data/output/    final deliverables (git-ignored)
runs/           per-run intermediate artifacts (git-ignored)
```

`configs/` doesn't exist yet. It gets created when a gate first needs a config file.

## Freezing

`requirements.txt` has lower bounds only for now. After Gate 0 passes on your machine, the exact versions from your `pip freeze` are pinned and this gate is tagged FROZEN.
