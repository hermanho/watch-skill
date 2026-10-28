"""Shared FFmpeg input options for optional hardware decoding."""
from __future__ import annotations

from watch_skill.config import get_settings


def hardware_decode_args() -> list[str]:
    """Input arguments; leave decoded frames accessible to software filters."""
    settings = get_settings()
    if not settings.ffmpeg_hwaccel:
        return []
    args = ["-hwaccel", settings.ffmpeg_hwaccel]
    if settings.ffmpeg_hw_device:
        args += ["-hwaccel_device", settings.ffmpeg_hw_device]
    if settings.ffmpeg_hwaccel == "vaapi":
        args += ["-hwaccel_output_format", "vaapi"]
    return args


def vaapi_scale_filter(width: int, height: int) -> str:
    """Scale hardware frames before downloading NV12 frames to system memory."""
    return (
        f"scale_vaapi=w='min({width},iw)':h='min({height},ih)':"
        "force_original_aspect_ratio=decrease:force_divisible_by=2:format=nv12,"
        "hwdownload,format=nv12"
    )
