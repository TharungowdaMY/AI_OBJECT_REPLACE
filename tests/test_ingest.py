"""Gate 1 tests. Uses a synthetic clip generated with ffmpeg's testsrc filter,
so these run on any machine without needing a real video fixture checked
into the repo (and never touch data/input/, which holds licensed footage).
"""

import shutil
import subprocess

import pytest

from src.ingest import (
    compute_psnr,
    extract_frames,
    frame_pts_monotonic,
    probe_video,
    round_trip_test,
)

FFMPEG_MISSING = shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None


@pytest.fixture(scope="module")
def synthetic_clip(tmp_path_factory):
    """A tiny, deterministic 2s CFR clip: 64x48 @ 10fps, 20 frames."""
    if FFMPEG_MISSING:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    out_dir = tmp_path_factory.mktemp("synthetic")
    clip_path = out_dir / "synthetic.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x48:rate=10:duration=2",
        "-pix_fmt",
        "yuv420p",
        str(clip_path),
    ]
    r = subprocess.run(cmd, capture_output=True, encoding="utf-8")
    assert r.returncode == 0, f"could not generate synthetic test clip: {r.stderr}"
    return clip_path


def test_probe_video_reads_expected_metadata(synthetic_clip):
    probe = probe_video(synthetic_clip)
    assert probe.width == 64
    assert probe.height == 48
    assert probe.nb_frames_exact == 20
    assert probe.duration_s == pytest.approx(2.0, abs=0.1)
    assert probe.fps_avg == pytest.approx(10.0, abs=0.01)


def test_extract_frames_matches_probe_count(tmp_path, synthetic_clip):
    frames_dir = tmp_path / "frames"
    count = extract_frames(synthetic_clip, frames_dir)
    probe = probe_video(synthetic_clip)
    assert count == probe.nb_frames_exact
    assert len(list(frames_dir.glob("frame_*.png"))) == count


def test_timestamps_are_monotonic(synthetic_clip):
    monotonic, n = frame_pts_monotonic(synthetic_clip)
    assert monotonic is True
    assert n == 20


def test_round_trip_preserves_frame_count_and_fidelity(tmp_path, synthetic_clip):
    report = round_trip_test(synthetic_clip, tmp_path)
    assert report.frame_count_match is True
    assert report.original_frame_count == report.roundtrip_frame_count == 20
    if report.duration_within_tolerance is not None:
        assert report.duration_within_tolerance is True
    assert report.psnr_average_db >= report.psnr_threshold_db
    assert report.overall_pass is True


def test_compute_psnr_is_perfect_against_itself(synthetic_clip):
    psnr = compute_psnr(synthetic_clip, synthetic_clip)
    # A file compared to itself should read as (near-)infinite/very high PSNR.
    assert psnr["average"] > 60
