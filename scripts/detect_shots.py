#!/usr/bin/env python3
"""Gate 2 CLI: detect hard-cut shot boundaries and write shots.json.

Usage:
    python scripts/detect_shots.py --input data/input/clip_a.mp4
    python scripts/detect_shots.py --input data/input/clip_b.mp4 --threshold 27.0

Shares runs/<run-id>/ with Gate 1 (same default run-id: the input filename
stem), so if you already ran ingest_video.py on this clip, this reuses its
extracted frames instead of decoding again.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import IngestError, extract_frames, probe_video
from src.shots import (
    DEFAULT_THRESHOLD,
    ShotDetectionError,
    detect_shots,
    frame_index_to_filename,
    shots_to_dicts,
    validate_shot_coverage,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ensure_frames(input_path: Path, run_dir: Path, expected_count: int) -> Path:
    """Reuse Gate 1's frame extraction if it already ran for this clip and
    matches the expected count; otherwise (re-)extract.
    """
    frames_dir = run_dir / "frames"
    existing = list(frames_dir.glob("frame_*.png")) if frames_dir.exists() else []
    if len(existing) == expected_count:
        print(f"  Reusing {len(existing)} already-extracted frames from {frames_dir}")
        return frames_dir
    print("  No matching extracted frames found; extracting now...")
    extract_frames(input_path, frames_dir)
    return frames_dir


def save_boundary_frames(frames_dir: Path, shots, boundaries_dir: Path) -> None:
    if boundaries_dir.exists():
        shutil.rmtree(boundaries_dir)
    boundaries_dir.mkdir(parents=True)

    def copy_frame(frame_index_0based: int, label: str) -> None:
        fname = frame_index_to_filename(frame_index_0based)
        src = frames_dir / fname
        if not src.exists():
            print(f"  WARNING: expected frame file missing, skipping: {src}")
            return
        shutil.copy(src, boundaries_dir / f"{label}_{fname}")

    copy_frame(shots[0].start_frame, "video_start")
    copy_frame(shots[-1].end_frame, "video_end")
    for a, b in zip(shots, shots[1:]):
        copy_frame(a.end_frame, f"shot{a.shot_index:02d}_end")
        copy_frame(b.start_frame, f"shot{b.shot_index:02d}_start")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Path to the source video")
    ap.add_argument("--run-id", default=None, help="Defaults to the input filename stem")
    ap.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"PySceneDetect ContentDetector threshold (default {DEFAULT_THRESHOLD}). "
        "Lower = more sensitive to cuts.",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    run_id = args.run_id or input_path.stem
    run_dir = PROJECT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:     {input_path}")
    print(f"Run dir:   {run_dir}")
    print(f"Threshold: {args.threshold}")
    print()

    try:
        print("Probing source...")
        probe = probe_video(input_path)
        print(f"  {probe.nb_frames_exact} frames, {probe.duration_s:.3f}s, fps={probe.fps_avg}")

        print("Detecting shots (PySceneDetect ContentDetector)...")
        shots = detect_shots(input_path, threshold=args.threshold)
        print(f"  Detected {len(shots)} shot(s):")
        for s in shots:
            print(
                f"    shot {s.shot_index}: frames {s.start_frame}-{s.end_frame} "
                f"({s.num_frames} frames, {s.start_time_s:.2f}s-{s.end_time_s:.2f}s)"
            )

        problems = validate_shot_coverage(shots, probe.nb_frames_exact)
        if problems:
            print("  Coverage problems found:")
            for p in problems:
                print(f"    - {p}")
        else:
            print("  Coverage OK: shots are contiguous and cover the whole video.")

        print("Preparing frames for boundary review...")
        frames_dir = ensure_frames(input_path, run_dir, probe.nb_frames_exact)

        boundaries_dir = run_dir / "boundaries"
        save_boundary_frames(frames_dir, shots, boundaries_dir)
        print(f"  Wrote boundary frames to {boundaries_dir} for manual review")

        shots_json = {
            "created_at": datetime.datetime.now().astimezone().isoformat(),
            "source": str(input_path.resolve()),
            "threshold": args.threshold,
            "total_frames": probe.nb_frames_exact,
            "num_shots": len(shots),
            "shots": shots_to_dicts(shots),
            "coverage_problems": problems,
            "coverage_ok": not problems,
        }
        shots_path = run_dir / "shots.json"
        shots_path.write_text(json.dumps(shots_json, indent=2))
        print()
        print(f"Wrote {shots_path}")

        if problems:
            print("RESULT: FAIL (coverage problems above)")
            return 1
        print("RESULT: PASS (coverage clean; open runs/<run-id>/boundaries/ to sanity-check cuts)")
        return 0

    except (IngestError, ShotDetectionError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
