# Virtual Product Placement (working title)

## 1. What we are building

An AI system that takes a licensed movie or web-series video plus a brand/product asset, finds a suitable object in a scene (for example a generic soda bottle), and replaces it with the branded product (for example a Sprite bottle). The replacement must preserve the scene's motion, perspective, lighting, occlusion, shadows, and temporal consistency.

Approach: a hybrid pipeline. Geometric compositing is the controlled backbone. Generative models are used only where geometry can't help (background reconstruction). The unit of processing is one shot. Every stage reads and writes plain files under `runs/`, so each stage can be inspected and tested on its own.

Development is a sequence of independently verified gates. A gate is FROZEN only after it passes on a real machine.

## 2. Current gate

**Gate 0: Environment + project skeleton — VERIFIED / FROZEN** (2026-09-22, Windows 11, Python 3.13.9, no NVIDIA GPU; see `pip freeze` output pinned in `requirements.txt`).

**Gate 1: Video ingestion and preprocessing — VERIFIED / FROZEN** (2026-09-23, verified against `clip_a.mp4`: 2160x3840, 260 frames, 30fps, 8.667s, no rotation tag needed since pixels are stored portrait-native. Exact frame count match, timestamps monotonic, round-trip PSNR 51.4 dB against a 35 dB threshold, manual frame inspection confirmed correct at start/middle/end with no corruption).

Known limitation, not yet exercised: rotation-tag handling (`side_data_list`/`tags.rotate`) is implemented but has never been tested against a clip that actually needs it (landscape-stored pixels with a rotate flag telling players to show it as portrait — common on some iPhones). If a future clip's frames come out sideways despite `rotation_deg: 0`, this is the first place to look.

**Gate 2: Shot / scene detection — awaiting your test results. Higher risk than previous gates, read this before running.**

This gate adds PySceneDetect, the first real pip dependency beyond numpy/opencv/pytest. **I could not install or run PySceneDetect in my dev sandbox at all** (no network access there — confirmed by trying). Every other file in this project was executed and observed by me before you got it; `src/shots.py` and `scripts/detect_shots.py` were not. They're written against PySceneDetect's documented stable API, but your run is the first real test. Expect a real chance of a first-try error, the way `-vsync` broke on ffmpeg 9 in Gate 1 — if something breaks, paste the exact error and we'll fix that one thing.

What it does: detects hard cuts (not fades/dissolves) using PySceneDetect's `ContentDetector`, writes `runs/<run_id>/shots.json` with each shot's frame range, and saves the frames on either side of each cut to `runs/<run_id>/boundaries/` so you can eyeball whether the cut looks right. It reuses Gate 1's frame extraction if it already ran for that clip.

### Running Gate 2

```bash
pip install -r requirements.txt
python scripts/detect_shots.py --input data/input/clip_a.mp4
python scripts/detect_shots.py --input data/input/clip_b.mp4
```

`clip_a.mp4` has no cuts (locked-off, continuous), so expect exactly **1 shot** spanning all 260 frames. `clip_b.mp4` is the clip you made specifically to have a cut — I don't know how many cuts it has or expect it to have, so just report what got detected and whether it matches what you actually filmed.

Expected terminal output: a probe line, a list of detected shots with frame ranges, a coverage check (shots should be contiguous with no gaps/overlaps), and `RESULT: PASS` if coverage is clean. A "PASS" here only means the frame accounting is internally consistent — it does **not** mean the cut was detected in the right place. That's what the boundary frames are for.

### What to send back for Gate 2
1. Output of `pip install -r requirements.txt` (the part relevant to `scenedetect`, in full if it errors).
2. Full terminal output of both `detect_shots.py` commands.
3. Contents of both `runs/clip_a/shots.json` and `runs/clip_b/shots.json`.
4. For `clip_b`: open the images in `runs/clip_b/boundaries/` and confirm the `shotXX_end` and `shotXX_start` pairs actually straddle a real cut, not a random frame in the middle of continuous footage.
5. How many real cuts you know `clip_b.mp4` has (so we can check the count matches).


### Hardware note (carried forward from Gate 0)
This machine has no NVIDIA GPU. Gates 0-2 run fine on CPU. Before Gate 3 (object detection) we need to decide how to run the GPU-heavy gates: rent a cloud GPU on demand, or develop against tiny inputs locally and validate on cloud periodically. Not decided yet, not urgent yet.

### Running Gate 1

```bash
python scripts/ingest_video.py --input data/input/clip_a.mp4 --roundtrip
```

This writes to `runs/clip_a/`:
- `frames/frame_000001.png`, `frame_000002.png`, ... — every decoded frame, lossless PNG
- `roundtrip.mp4` — the re-encoded video, used only for the fidelity check
- `meta.json` — everything below, in one file

Expected terminal output: a probe summary line, an extraction count, a monotonic-timestamps check, a round-trip summary line, and a final `RESULT: PASS` with exit code 0. If anything mismatches, it prints `RESULT: FAIL` and exits 1.

What `meta.json` contains:
- `probe`: width, height, exact frame count, fps, duration, pixel format, rotation, audio presence
- `frame_extraction`: how many PNGs were written and whether that matches ffprobe's count
- `timestamps_monotonic`: whether decode timestamps strictly increase
- `round_trip` (only with `--roundtrip`): frame count match, duration difference, PSNR average/min against a 35 dB threshold, and an overall pass/fail

Acceptance thresholds (in `src/ingest.py`, `DURATION_TOLERANCE_S = 0.05` and `PSNR_MIN_DB = 35.0`) are starting points, not fixed. If your real clip fails PSNR by a small margin, that's a discussion, not an automatic reject — send me the numbers.

### What to send back for Gate 1
1. The full terminal output of the command above.
2. The contents of `runs/clip_a/meta.json`.
3. Whether `rotation_deg` in that file matches what the video actually looks like (does it play right-side up?).
4. Open two or three PNGs from `runs/clip_a/frames/` and confirm they look correct (right orientation, no corruption).


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

Expected: `8 passed` (1 Gate 0, 4 Gate 1, 3 Gate 2) — **but only after `pip install -r requirements.txt` succeeds**, since the Gate 2 tests import `scenedetect`. If that install fails, the Gate 2 tests will be reported as skipped, not failed (they're set up to skip cleanly rather than error if the dependency isn't there) — everything else still runs.

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
