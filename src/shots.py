"""Gate 2: shot / scene (hard-cut) detection.

Splits a video into continuous shots using PySceneDetect's content-aware
detector. This detects hard cuts only. Gradual transitions (dissolves,
fades) are explicitly out of scope for Gate 2 - see README.

WARNING (disclosed on purpose): this module could not be executed even once
in the development sandbox, because PySceneDetect could not be installed
there (no network access). Every other module in this project was run and
observed before being handed over; this one was not. It is written against
PySceneDetect's documented stable API (open_video / SceneManager /
ContentDetector, stable since 0.6.0), but the first real test of this file
is your machine. Treat the first run as an experiment, not a known-good
tool.

Frame numbering note, a likely off-by-one trap: PySceneDetect frame indices
are 0-based (the first decoded frame is index 0). Gate 1's extracted PNGs
are 1-based (frame_000001.png is the first frame). Use
frame_index_to_filename() below whenever mapping a shot's frame index to a
Gate 1 PNG file - do not assume the numbers line up directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_THRESHOLD = 27.0  # PySceneDetect's own ContentDetector default


class ShotDetectionError(RuntimeError):
    """Raised when PySceneDetect fails, is missing, or returns something
    we don't trust (e.g. zero scenes)."""


@dataclass
class Shot:
    shot_index: int
    start_frame: int  # 0-based, inclusive
    end_frame: int  # 0-based, inclusive
    start_time_s: float
    end_time_s: float
    num_frames: int


def frame_index_to_filename(frame_index_0based: int) -> str:
    """Gate 1 PNGs are 1-indexed: frame_000001.png is decode index 0."""
    return f"frame_{frame_index_0based + 1:06d}.png"


def detect_shots(video_path: str | Path, threshold: float = DEFAULT_THRESHOLD) -> list[Shot]:
    """Run PySceneDetect's ContentDetector over the whole video and return
    one Shot per contiguous segment between hard cuts.
    """
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except ImportError as e:
        raise ShotDetectionError(
            "PySceneDetect is not installed. Run: pip install -r requirements.txt"
        ) from e

    video_path = str(video_path)
    if not Path(video_path).is_file():
        raise ShotDetectionError(f"Input video not found: {video_path}")

    try:
        video = open_video(video_path)
    except Exception as e:  # noqa: BLE001 - surface whatever PySceneDetect raises
        raise ShotDetectionError(f"Could not open {video_path} with PySceneDetect: {e}") from e

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    try:
        scene_manager.detect_scenes(video, show_progress=False)
    except Exception as e:  # noqa: BLE001
        raise ShotDetectionError(f"Scene detection failed on {video_path}: {e}") from e

    scene_list = scene_manager.get_scene_list()
    if not scene_list:
        # PySceneDetect's documented behavior: an empty scene list means no
        # cuts were found, i.e. the whole video is one shot - NOT an error.
        # (Earlier version of this code got this backwards and treated it
        # as a failure; fixed after seeing it reject a real single-shot clip.)
        total_frames = video.duration.get_frames()
        if total_frames <= 0:
            raise ShotDetectionError(
                f"PySceneDetect found no cuts in {video_path}, and the video's own "
                f"reported duration is {total_frames} frames, which is invalid. "
                f"Investigate the video file itself."
            )
        return [
            Shot(
                shot_index=0,
                start_frame=0,
                end_frame=total_frames - 1,
                start_time_s=0.0,
                end_time_s=video.duration.get_seconds(),
                num_frames=total_frames,
            )
        ]

    shots = []
    for i, (start_tc, end_tc) in enumerate(scene_list):
        start_frame = start_tc.get_frames()
        end_frame = end_tc.get_frames() - 1  # PySceneDetect's scene end is exclusive
        shots.append(
            Shot(
                shot_index=i,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time_s=start_tc.get_seconds(),
                end_time_s=end_tc.get_seconds(),
                num_frames=end_frame - start_frame + 1,
            )
        )
    return shots


def validate_shot_coverage(shots: list[Shot], total_frames: int) -> list[str]:
    """Check that shots are contiguous, gap-free, and exactly cover
    [0, total_frames) with no overlaps. Returns a list of problem
    descriptions; an empty list means coverage is clean.
    """
    problems = []
    if not shots:
        return ["No shots detected."]
    if shots[0].start_frame != 0:
        problems.append(f"First shot starts at frame {shots[0].start_frame}, expected 0.")
    for a, b in zip(shots, shots[1:]):
        if b.start_frame != a.end_frame + 1:
            problems.append(
                f"Gap or overlap between shot {a.shot_index} (ends frame {a.end_frame}) "
                f"and shot {b.shot_index} (starts frame {b.start_frame})."
            )
    last = shots[-1]
    if last.end_frame != total_frames - 1:
        problems.append(
            f"Last shot ends at frame {last.end_frame}, expected {total_frames - 1} "
            f"(probe reported {total_frames} total frames)."
        )
    return problems


def shots_to_dicts(shots: list[Shot]) -> list[dict]:
    return [asdict(s) for s in shots]
