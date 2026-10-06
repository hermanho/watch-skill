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
    return args
