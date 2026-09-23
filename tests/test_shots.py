"""Gate 2 tests. Uses synthetic ffmpeg-generated clips: one continuous (no
cut) and one built by concatenating two visually distinct segments (a real
hard cut). Never touches data/input/.

NOTE: these tests could not be run in the development sandbox (no network
access to install PySceneDetect there). They are the first real execution
of this code, same as the CLI - see src/shots.py's module docstring.
"""

import shutil
import subprocess

import pytest

from src.ingest import probe_video
from src.shots import detect_shots, validate_shot_coverage

FFMPEG_MISSING = shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None
try:
    import scenedetect  # noqa: F401

    SCENEDETECT_MISSING = False
except ImportError:
    SCENEDETECT_MISSING = True

pytestmark = pytest.mark.skipif(
    FFMPEG_MISSING or SCENEDETECT_MISSING,
    reason="requires ffmpeg/ffprobe and scenedetect",
)


@pytest.fixture(scope="module")
def continuous_clip(tmp_path_factory):
    """3s of one continuous pattern: should be detected as exactly one shot."""
    out_dir = tmp_path_factory.mktemp("continuous")
    clip_path = out_dir / "continuous.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=10:duration=3",
        "-pix_fmt",
        "yuv420p",
        str(clip_path),
    ]
    r = subprocess.run(cmd, capture_output=True, encoding="utf-8")
    assert r.returncode == 0, f"could not generate continuous test clip: {r.stderr}"
    return clip_path


@pytest.fixture(scope="module")
def cut_clip(tmp_path_factory):
    """1.5s of one solid color followed by 1.5s of a different solid color:
    a real, unambiguous hard cut in the middle.
    """
    out_dir = tmp_path_factory.mktemp("cut")
    part_a = out_dir / "a.mp4"
    part_b = out_dir / "b.mp4"
    clip_path = out_dir / "cut.mp4"
    concat_list = out_dir / "concat.txt"

    for path, color in [(part_a, "red"), (part_b, "blue")]:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:size=160x120:rate=10:duration=1.5",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        r = subprocess.run(cmd, capture_output=True, encoding="utf-8")
        assert r.returncode == 0, f"could not generate '{color}' segment: {r.stderr}"

    concat_list.write_text(f"file '{part_a.name}'\nfile '{part_b.name}'\n")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        str(clip_path),
    ]
    r = subprocess.run(cmd, capture_output=True, encoding="utf-8", cwd=out_dir)
    assert r.returncode == 0, f"could not concatenate segments: {r.stderr}"
    return clip_path


def test_continuous_clip_is_one_shot(continuous_clip):
    probe = probe_video(continuous_clip)
    shots = detect_shots(continuous_clip)
    assert len(shots) == 1, f"expected 1 shot on a cut-free clip, got {len(shots)}"
    assert shots[0].start_frame == 0
    assert shots[0].end_frame == probe.nb_frames_exact - 1
    assert validate_shot_coverage(shots, probe.nb_frames_exact) == []


def test_cut_clip_detects_the_cut(cut_clip):
    probe = probe_video(cut_clip)
    shots = detect_shots(cut_clip)
    assert len(shots) >= 2, f"expected at least 2 shots on a clip with a hard cut, got {len(shots)}"
    assert validate_shot_coverage(shots, probe.nb_frames_exact) == []


def test_shot_coverage_validator_catches_a_gap():
    from src.shots import Shot

    shots = [
        Shot(0, 0, 9, 0.0, 1.0, 10),
        Shot(1, 15, 29, 1.5, 3.0, 15),  # gap: should start at 10
    ]
    problems = validate_shot_coverage(shots, total_frames=30)
    assert problems, "expected the validator to flag a gap between shots"
