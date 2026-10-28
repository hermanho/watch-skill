"""Scene detection and perceptual hashing.

Two detectors: PySceneDetect's ContentDetector by default, and an ffmpeg-only
path for environments where PySceneDetect's decoder conflicts with the
platform. Both return absolute times into the video and both use the same
empty-list contract for a static shot.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from watch_skill.config import get_settings
from watch_skill.errors import PerceptionError
from watch_skill.health.binaries import require_binary
from watch_skill.perceive.ffmpeg import hardware_decode_args, vaapi_scale_filter

_SCENE_TIME = re.compile(r"pts_time:([0-9.]+)")
_FFMPEG_TIMEOUT = 900.0
_STDERR_CAP = 4_000_000
"""Bound what we read back from ffmpeg. `showinfo` prints a line per selected
frame, and a pathological source could otherwise fill memory with log text."""


def _scene_filter(threshold: float) -> str:
    settings = get_settings()
    scene_filter = f"select=gt(scene\\,{threshold}),showinfo"
    if settings.ffmpeg_hwaccel == "vaapi":
        size = settings.ffmpeg_scene_max_dimension
        return (f"fps={settings.ffmpeg_scene_fps},"
                f"{vaapi_scale_filter(size, size)},{scene_filter}")
    return scene_filter


def detect_scenes_ffmpeg(
    video_path: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    threshold: float = 0.3,
) -> list[tuple[float, float]]:
    """Scene spans from ffmpeg's own cut score, without importing PyAV.

    A fallback for environments where PySceneDetect's decoder conflicts with
    the platform — the case that motivated it is macOS, where PyAV and
    AVFoundation can fight over the same frameworks.

    **Timestamps are normalised back to absolute.** ``-ss`` placed before
    ``-i`` seeks the *input*, which restarts the presentation clock at zero:
    a cut eight seconds into a window that began at 60 s is reported as
    ``pts_time:8``. Treating that as absolute would place every scene in a
    focused watch near the start of the video, and every citation with it. The
    window offset is added back explicitly, and a test with a known cut inside
    a non-zero window is what holds that.
    """
    # pylint: disable-next=import-outside-toplevel
    from watch_skill.perceive.media import probe  # noqa: PLC0415

    duration = probe(video_path).duration_seconds or 0.0
    lo = max(0.0, start_seconds or 0.0)
    hi = min(end_seconds if end_seconds is not None else duration, duration) \
        if duration else (end_seconds or 0.0)
    if hi <= lo:
        return []

    command = [str(require_binary("ffmpeg")), "-hide_banner", "-loglevel", "info"]
    if lo:
        command += ["-ss", f"{lo:.3f}"]
    command += [
        *hardware_decode_args(),
        "-i", str(video_path.resolve()),
        "-t", f"{hi - lo:.3f}",
        "-an", "-vf", _scene_filter(threshold),
        "-f", "null", "-",
    ]

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=_FFMPEG_TIMEOUT, stdin=subprocess.DEVNULL, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PerceptionError(
            f"ffmpeg scene detection timed out after {_FFMPEG_TIMEOUT:.0f}s",
            code="perceive.scene_detection_timeout",
            fix="narrow the window with --start/--end, or use --transcript-only",
            details={"path": str(video_path), "window": [lo, hi]},
        ) from exc
    except OSError as exc:
        raise PerceptionError(
            f"could not run ffmpeg for scene detection: {exc}",
            code="perceive.scene_detection_failed",
            fix="run `watch-skill doctor` to check the ffmpeg binary",
            details={"path": str(video_path)},
        ) from exc

    stderr = (result.stderr or "")[:_STDERR_CAP]
    if result.returncode != 0:
        raise PerceptionError(
            f"ffmpeg scene detection failed: {stderr.strip()[-300:]}",
            code="perceive.scene_detection_failed",
            fix="the media may be corrupt or zero-length — re-acquire it, "
            "or watch with --transcript-only",
            details={"path": str(video_path), "exit_code": result.returncode},
        )

    # `+ lo` is the normalisation. Everything downstream — citations, frame
    # extraction, the index — reads these as absolute times into the video.
    cuts = sorted({
        round(float(match.group(1)) + lo, 3)
        for match in _SCENE_TIME.finditer(stderr)
    })
    cuts = [cut for cut in cuts if lo < cut < hi]
    if not cuts:
        # The static-video contract: an empty list means "one continuous
        # shot", and callers fall back to uniform sampling. Returning a single
        # whole-window span instead would look like a detected scene and
        # silently disable that fallback.
        return []

    return [(round(start, 3), round(end, 3))
            for start, end in zip([lo, *cuts], [*cuts, hi], strict=True)
            if end > start]


def _import_scenedetect():
    try:
        # pylint: disable-next=import-outside-toplevel
        from scenedetect import ContentDetector, detect  # noqa: PLC0415
    except ImportError as exc:
        raise PerceptionError(
            "PySceneDetect is not installed",
            code="perceive.missing_dependency",
            fix='install the perception extras: `uv sync --extra perceive` or '
                '`pip install "watch-skill[perceive]"`',
        ) from exc
    return detect, ContentDetector


def detect_scenes(
    video_path: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> list[tuple[float, float]]:
    """Detect cuts; return (start_seconds, end_seconds) per scene.

    ``start/end`` bound the scan — a focused watch of one minute must not
    decode a whole hour of video (live finding: 10 min of decode for a 60 s
    window on a slow machine). An empty list means the video is effectively
    one static shot — callers fall back to uniform sampling.
    """
    detect, content_detector = _import_scenedetect()
    try:
        scene_list = detect(
            str(video_path),
            content_detector(),
            start_time=start_seconds,
            end_time=end_seconds,
        )
    except TypeError:
        # older scenedetect without start/end kwargs — full scan fallback
        try:
            scene_list = detect(str(video_path), content_detector())
        except Exception as exc:
            raise PerceptionError(
                f"scene detection failed: {exc}",
                code="perceive.scene_detection_failed",
                fix="the media may be corrupt or zero-length — re-acquire it, "
                "or watch with --transcript-only",
                details={"path": str(video_path)},
            ) from exc
    except Exception as exc:  # scenedetect raises plain Exceptions on bad media
        raise PerceptionError(
            f"scene detection failed: {exc}",
            code="perceive.scene_detection_failed",
            fix="the media may be corrupt or zero-length — re-acquire it, "
            "or watch with --transcript-only",
            details={"path": str(video_path)},
        ) from exc
    def _seconds(timecode) -> float:
        value = getattr(timecode, "seconds", None)  # scenedetect >= 0.7
        return float(value) if value is not None else timecode.get_seconds()

    return [(_seconds(start), _seconds(end)) for start, end in scene_list]


def compute_phash(image_path: Path) -> str:
    """Perceptual hash (phash) of an image, as a hex string."""
    try:
        # pylint: disable=import-outside-toplevel
        import imagehash  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        # pylint: enable=import-outside-toplevel
    except ImportError as exc:
        raise PerceptionError(
            "imagehash/Pillow are not installed",
            code="perceive.missing_dependency",
            fix='install the perception extras: `pip install "watch-skill[perceive]"` '
            '(from a checkout: `uv sync --extra perceive`)',
        ) from exc
    with Image.open(image_path) as img:
        return str(imagehash.phash(img))


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Hamming distance between two hex phash strings."""
    a = int(hash_a, 16)
    b = int(hash_b, 16)
    return (a ^ b).bit_count()
