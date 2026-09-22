#!/usr/bin/env python3
"""Gate 1 CLI: probe a video, extract its frames, write meta.json, and
optionally verify a decode -> encode round trip.

Usage:
    python scripts/ingest_video.py --input data/input/clip_a.mp4
    python scripts/ingest_video.py --input data/input/clip_a.mp4 --roundtrip
    python scripts/ingest_video.py --input data/input/clip_a.mp4 --run-id my_test --roundtrip
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import (
    IngestError,
    extract_frames,
    frame_pts_monotonic,
    probe_to_dict,
    probe_video,
    report_to_dict,
    round_trip_test,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Path to the source video")
    ap.add_argument(
        "--run-id",
        default=None,
        help="Name for runs/<run-id>/. Defaults to the input filename stem.",
    )
    ap.add_argument(
        "--roundtrip",
        action="store_true",
        help="Also re-encode the extracted frames and verify frame count, "
        "duration, and PSNR against the source.",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    run_id = args.run_id or input_path.stem
    run_dir = PROJECT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:   {input_path}")
    print(f"Run dir: {run_dir}")
    print()

    try:
        print("Probing source with ffprobe (this decodes once, to count frames exactly)...")
        probe = probe_video(input_path)
        print(
            f"  {probe.width}x{probe.height}, {probe.nb_frames_exact} frames, "
            f"{probe.duration_s:.3f}s, fps_avg={probe.fps_avg}, "
            f"pix_fmt={probe.pix_fmt}, rotation={probe.rotation_deg}deg, "
            f"audio={'yes' if probe.has_audio else 'no'}"
        )

        print("Extracting frames...")
        frames_dir = run_dir / "frames"
        frame_count = extract_frames(input_path, frames_dir)
        print(f"  Wrote {frame_count} PNG frames to {frames_dir}")

        frame_count_matches_probe = frame_count == probe.nb_frames_exact
        print(
            f"  Extracted count matches ffprobe count: {frame_count_matches_probe} "
            f"({frame_count} vs {probe.nb_frames_exact})"
        )

        print("Checking timestamps are monotonic...")
        monotonic, n_pts = frame_pts_monotonic(input_path)
        print(f"  Monotonic: {monotonic} ({n_pts} timestamps read)")

        meta = {
            "created_at": datetime.datetime.now().astimezone().isoformat(),
            "source": str(input_path.resolve()),
            "probe": probe_to_dict(probe),
            "frame_extraction": {
                "output_dir": str(frames_dir),
                "frame_count": frame_count,
                "pattern": "frame_%06d.png",
                "matches_ffprobe_count": frame_count_matches_probe,
            },
            "timestamps_monotonic": monotonic,
            "timestamps_checked": n_pts,
        }

        if args.roundtrip:
            print("Running round-trip verification (decode -> encode -> compare)...")
            report = round_trip_test(input_path, run_dir)
            meta["round_trip"] = report_to_dict(report)
            print(
                f"  frame_count_match={report.frame_count_match} "
                f"duration_diff_s={report.duration_diff_s} "
                f"psnr_avg_db={report.psnr_average_db:.2f} "
                f"(threshold {report.psnr_threshold_db}) "
                f"overall_pass={report.overall_pass}"
            )

        meta_path = run_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print()
        print(f"Wrote {meta_path}")

        if not frame_count_matches_probe or not monotonic:
            print("RESULT: FAIL (see mismatches above)")
            return 1
        if args.roundtrip and not meta["round_trip"]["overall_pass"]:
            print("RESULT: FAIL (round-trip check did not pass)")
            return 1
        print("RESULT: PASS")
        return 0

    except IngestError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
