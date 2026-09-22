"""Gate 1: video ingestion and preprocessing.

Frame-accurate decode via FFmpeg, metadata via ffprobe, and a decode -> encode
-> compare round trip so we can prove no frames are silently lost or shifted.

Deliberately out of scope for Gate 1 (add only when a real clip needs it):
  - variable-frame-rate-aware re-encoding (we assume CFR in, CFR out)
  - proxy-resolution extraction
  - audio track handling beyond detecting its presence
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

FRAME_PATTERN = "frame_%06d.png"

# Round-trip acceptance thresholds. See README "Gate 1" for rationale.
DURATION_TOLERANCE_S = 0.05
PSNR_MIN_DB = 35.0


class IngestError(RuntimeError):
    """Raised when a required ffmpeg/ffprobe step fails or a tool is missing."""


def _require(exe: str) -> str:
    path = shutil.which(exe)
    if not path:
        raise IngestError(f"'{exe}' not found on PATH. Run scripts/check_env.py first.")
    return path


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        raise IngestError(f"Command timed out after {timeout}s: {' '.join(cmd)}") from e


def _parse_fraction(s: str | None) -> float | None:
    if not s or s in ("0/0", "N/A"):
        return None
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        return None


@dataclass
class ProbeResult:
    path: str
    width: int
    height: int
    fps_avg: float | None
    fps_r: float | None
    duration_s: float | None
    nb_frames_exact: int
    pix_fmt: str | None
    color_range: str | None
    color_space: str | None
    rotation_deg: int
    codec_name: str | None
    has_audio: bool


def probe_video(path: str | Path) -> ProbeResult:
    """Read container/stream metadata with ffprobe. Frame count is exact
    (uses -count_frames), which decodes the stream once, so it is slower
    than a header read but trustworthy for the round-trip comparison.
    """
    ffprobe = _require("ffprobe")
    path = str(path)
    if not Path(path).is_file():
        raise IngestError(f"Input video not found: {path}")

    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    r = _run(cmd)
    if r.returncode != 0:
        raise IngestError(f"ffprobe failed on {path}: {r.stderr.strip()}")
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise IngestError(f"ffprobe returned unparseable JSON for {path}") from e

    streams = data.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        raise IngestError(f"No video stream found in {path}")
    v = video_streams[0]
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    rotation_deg = 0
    tag_rotate = v.get("tags", {}).get("rotate")
    if tag_rotate is not None:
        try:
            rotation_deg = int(tag_rotate)
        except ValueError:
            pass
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            try:
                rotation_deg = int(sd["rotation"])
            except (ValueError, TypeError):
                pass

    duration_s = None
    fmt_duration = data.get("format", {}).get("duration")
    if fmt_duration is not None:
        try:
            duration_s = float(fmt_duration)
        except ValueError:
            duration_s = None

    exact_count_cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        path,
    ]
    rc = _run(exact_count_cmd)
    if rc.returncode != 0 or not rc.stdout.strip().isdigit():
        raise IngestError(
            f"Could not get an exact frame count for {path} "
            f"(ffprobe -count_frames): {rc.stderr.strip() or rc.stdout.strip()}"
        )
    nb_frames_exact = int(rc.stdout.strip())

    return ProbeResult(
        path=path,
        width=int(v.get("width", 0)),
        height=int(v.get("height", 0)),
        fps_avg=_parse_fraction(v.get("avg_frame_rate")),
        fps_r=_parse_fraction(v.get("r_frame_rate")),
        duration_s=duration_s,
        nb_frames_exact=nb_frames_exact,
        pix_fmt=v.get("pix_fmt"),
        color_range=v.get("color_range"),
        color_space=v.get("color_space"),
        rotation_deg=rotation_deg,
        codec_name=v.get("codec_name"),
        has_audio=has_audio,
    )


def frame_pts_monotonic(path: str | Path) -> tuple[bool, int]:
    """Read every video frame's presentation timestamp and check it strictly
    increases. Returns (is_monotonic, frame_count_seen). Decodes the whole
    stream, so cost scales with clip length.
    """
    ffprobe = _require("ffprobe")
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pts_time",
        "-of",
        "csv=p=0",
        str(path),
    ]
    r = _run(cmd)
    if r.returncode != 0:
        raise IngestError(f"ffprobe frame listing failed on {path}: {r.stderr.strip()}")
    values = []
    skipped = 0
    for line in r.stdout.strip().splitlines():
        line = line.strip().rstrip(",")  # ffprobe's csv writer can leave a trailing comma
        if not line or line == "N/A":
            continue
        try:
            values.append(float(line))
        except ValueError:
            skipped += 1
    if skipped:
        raise IngestError(
            f"{skipped} of {skipped + len(values)} frame timestamps from ffprobe "
            f"could not be parsed as numbers for {path}. Investigate before trusting "
            f"the monotonic-timestamps check."
        )
    monotonic = all(b > a for a, b in zip(values, values[1:]))
    return monotonic, len(values)


def extract_frames(input_path: str | Path, frames_dir: str | Path) -> int:
    """Decode every frame to lossless PNGs, in original decode order, with no
    frame drop/duplication (-fps_mode passthrough). Returns the number of
    PNG files written.
    """
    ffmpeg = _require("ffmpeg")
    frames_dir = Path(frames_dir)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-fps_mode",
        "passthrough",
        str(frames_dir / FRAME_PATTERN),
    ]
    r = _run(cmd)
    if r.returncode != 0:
        raise IngestError(f"ffmpeg frame extraction failed: {r.stderr.strip()[-2000:]}")

    count = len(list(frames_dir.glob("frame_*.png")))
    if count == 0:
        raise IngestError(f"ffmpeg reported success but wrote 0 frames to {frames_dir}")
    return count


def encode_frames(frames_dir: str | Path, output_path: str | Path, fps: float) -> None:
    """Re-encode a PNG frame sequence to a video, lossless x264, for the
    round-trip test. Assumes constant frame rate.
    """
    ffmpeg = _require("ffmpeg")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(Path(frames_dir) / FRAME_PATTERN),
        "-c:v",
        "libx264",
        "-crf",
        "0",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    r = _run(cmd)
    if r.returncode != 0:
        raise IngestError(f"ffmpeg re-encode failed: {r.stderr.strip()[-2000:]}")


def compute_psnr(reference_path: str | Path, distorted_path: str | Path) -> dict:
    """Use ffmpeg's own psnr filter to compare two videos frame-for-frame.
    Requires equal frame counts to align cleanly; the caller should verify
    that first. Returns the parsed summary line (y/u/v/average/min/max, dB).
    """
    ffmpeg = _require("ffmpeg")
    cmd = [
        ffmpeg,
        "-i",
        str(distorted_path),
        "-i",
        str(reference_path),
        "-lavfi",
        "psnr",
        "-f",
        "null",
        "-",
    ]
    r = _run(cmd)
    # ffmpeg's psnr summary is printed to stderr regardless of returncode conventions;
    # returncode 0 is still required to trust the run completed.
    if r.returncode != 0:
        raise IngestError(f"ffmpeg psnr filter failed: {r.stderr.strip()[-2000:]}")

    summary_line = None
    for line in r.stderr.splitlines():
        if "PSNR" in line and "average:" in line:
            summary_line = line
            break
    if summary_line is None:
        raise IngestError("Could not find a PSNR summary line in ffmpeg output.")

    result = {}
    for token in summary_line.split():
        if ":" in token:
            key, _, val = token.partition(":")
            try:
                result[key] = float(val)
            except ValueError:
                pass
    if "average" not in result:
        raise IngestError(f"PSNR summary line missing 'average': {summary_line}")
    return result


@dataclass
class RoundTripReport:
    original_frame_count: int
    roundtrip_frame_count: int
    frame_count_match: bool
    original_duration_s: float | None
    roundtrip_duration_s: float | None
    duration_diff_s: float | None
    duration_within_tolerance: bool | None
    psnr_average_db: float
    psnr_min_db: float
    psnr_threshold_db: float
    psnr_pass: bool
    overall_pass: bool


def round_trip_test(input_path: str | Path, run_dir: str | Path) -> RoundTripReport:
    """Decode -> re-encode -> compare. Proves the extract/encode path does
    not silently drop frames, shift duration, or destroy pixel fidelity.
    """
    run_dir = Path(run_dir)
    frames_dir = run_dir / "frames"
    roundtrip_video = run_dir / "roundtrip.mp4"

    original = probe_video(input_path)
    frame_count = extract_frames(input_path, frames_dir)

    if frame_count != original.nb_frames_exact:
        raise IngestError(
            f"Extracted {frame_count} frames but ffprobe counted "
            f"{original.nb_frames_exact} in the source. Investigate before continuing."
        )

    fps_for_encode = original.fps_avg or original.fps_r
    if not fps_for_encode:
        raise IngestError("Could not determine a frame rate to re-encode with.")

    encode_frames(frames_dir, roundtrip_video, fps_for_encode)
    roundtrip = probe_video(roundtrip_video)

    frame_count_match = original.nb_frames_exact == roundtrip.nb_frames_exact

    duration_diff = None
    duration_ok = None
    if original.duration_s is not None and roundtrip.duration_s is not None:
        duration_diff = abs(original.duration_s - roundtrip.duration_s)
        duration_ok = duration_diff <= DURATION_TOLERANCE_S

    psnr = compute_psnr(input_path, roundtrip_video)
    psnr_avg = psnr.get("average", 0.0)
    psnr_min = min(v for k, v in psnr.items() if k in ("y", "u", "v"))
    psnr_pass = psnr_avg >= PSNR_MIN_DB

    overall = frame_count_match and bool(duration_ok if duration_ok is not None else True) and psnr_pass

    return RoundTripReport(
        original_frame_count=original.nb_frames_exact,
        roundtrip_frame_count=roundtrip.nb_frames_exact,
        frame_count_match=frame_count_match,
        original_duration_s=original.duration_s,
        roundtrip_duration_s=roundtrip.duration_s,
        duration_diff_s=duration_diff,
        duration_within_tolerance=duration_ok,
        psnr_average_db=psnr_avg,
        psnr_min_db=psnr_min,
        psnr_threshold_db=PSNR_MIN_DB,
        psnr_pass=psnr_pass,
        overall_pass=overall,
    )


def probe_to_dict(p: ProbeResult) -> dict:
    return asdict(p)


def report_to_dict(r: RoundTripReport) -> dict:
    return asdict(r)
