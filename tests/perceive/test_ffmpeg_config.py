"""Backend routing and FFmpeg hardware input options, without a physical GPU."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from watch_skill.config import Settings, get_settings, reset_settings
from watch_skill.errors import PerceptionError
from watch_skill.perceive import engine, media, scenes
from watch_skill.perceive.types import VideoMetadata


def test_default_backend_and_hardware_settings(monkeypatch) -> None:
    for key in ("WATCHSKILL_SCENE_BACKEND", "WATCHSKILL_FFMPEG_HWACCEL",
                "WATCHSKILL_FFMPEG_HW_DEVICE"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.scene_backend == "scenedetect"
    assert settings.ffmpeg_hwaccel == settings.ffmpeg_hw_device == ""


@pytest.mark.parametrize("source", ["env", "dotenv"])
def test_hardware_configuration_sources(source, tmp_path, monkeypatch) -> None:
    values = {
        "WATCHSKILL_SCENE_BACKEND": "ffmpeg",
        "WATCHSKILL_FFMPEG_HWACCEL": "vaapi",
        "WATCHSKILL_FFMPEG_HW_DEVICE": "/dev/dri/renderD128",
    }
    if source == "env":
        for key, value in values.items():
            monkeypatch.setenv(key, value)
    else:
        for key in values:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text(
            "\n".join(f"{key}={value}" for key, value in values.items()), encoding="utf-8"
        )
    reset_settings()
    settings = get_settings()
    assert settings.scene_backend == "ffmpeg"
    assert settings.ffmpeg_hwaccel == "vaapi"
    assert settings.ffmpeg_hw_device == "/dev/dri/renderD128"


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(scene_backend="unknown")


@pytest.mark.parametrize("backend", ["scenedetect", "ffmpeg"])
def test_backend_routing(backend, monkeypatch) -> None:
    if backend == "ffmpeg":
        monkeypatch.setenv("WATCHSKILL_SCENE_BACKEND", backend)
        reset_settings()
    expected = [(2.0, 4.0), (4.0, 8.0)]
    default = Mock(return_value=expected)
    ffmpeg = Mock(return_value=expected)
    monkeypatch.setattr(scenes, "detect_scenes", default)
    monkeypatch.setattr(scenes, "detect_scenes_ffmpeg", ffmpeg)
    path = Path("video with spaces.mp4")
    assert engine.detect_or_empty(path, 2.0, 8.0) == expected
    selected, other = (ffmpeg, default) if backend == "ffmpeg" else (default, ffmpeg)
    selected.assert_called_once_with(path, 2.0, 8.0)
    other.assert_not_called()


@pytest.mark.parametrize("backend", ["scenedetect", "ffmpeg"])
def test_detection_failure_keeps_uniform_fallback(backend, monkeypatch, capsys) -> None:
    monkeypatch.setenv("WATCHSKILL_SCENE_BACKEND", backend)
    reset_settings()
    selected = Mock(side_effect=PerceptionError(
        "device initialization failed", code="perceive.scene_detection_failed",
        fix="check the device",
    ))
    other = Mock()
    monkeypatch.setattr(scenes, "detect_scenes" if backend == "scenedetect"
                        else "detect_scenes_ffmpeg", selected)
    monkeypatch.setattr(scenes, "detect_scenes_ffmpeg" if backend == "scenedetect"
                        else "detect_scenes", other)
    assert engine.detect_or_empty(Path("video.mp4")) == []
    assert "uniform sampling" in capsys.readouterr().err
    selected.assert_called_once()
    other.assert_not_called()


@pytest.mark.parametrize(("backend", "device", "expected"), [
    ("", "", []),
    ("", "/dev/dri/renderD128", []),
    ("vaapi", "", ["-hwaccel", "vaapi"]),
    ("vaapi", "/dev/dri/renderD128", [
        "-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128",
    ]),
    ("cuda", "0", ["-hwaccel", "cuda", "-hwaccel_device", "0"]),
])
@pytest.mark.parametrize("operation", ["scenes", "frame"])
def test_shared_hardware_input_args(backend, device, expected, operation, tmp_path,
                                    monkeypatch) -> None:
    from watch_skill.perceive.ffmpeg import hardware_decode_args

    monkeypatch.setenv("WATCHSKILL_FFMPEG_HWACCEL", backend)
    monkeypatch.setenv("WATCHSKILL_FFMPEG_HW_DEVICE", device)
    reset_settings()
    assert hardware_decode_args() == expected
    assert scenes.hardware_decode_args is media.hardware_decode_args is hardware_decode_args
    monkeypatch.setattr(media, "probe", lambda path: VideoMetadata(10, 320, 180, 30, "h264", False))
    monkeypatch.setattr(media, "require_binary", lambda name: Path(name))
    monkeypatch.setattr(scenes, "require_binary", lambda name: Path(name))
    run = Mock(return_value=subprocess.CompletedProcess([], 0, stderr="pts_time:2.0"))
    monkeypatch.setattr(subprocess, "run", run)
    video = tmp_path / "video with spaces.mp4"
    if operation == "scenes":
        assert scenes.detect_scenes_ffmpeg(video, 1, 5) == [(1, 3), (3, 5)]
        prefix = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-ss", "1.000"]
        suffix = ["-t", "4.000", "-an", "-vf", "select=gt(scene\\,0.3),showinfo",
                  "-f", "null", "-"]
    else:
        dest = tmp_path / "frame.jpg"
        dest.write_bytes(b"jpeg")
        assert media.extract_frame_at(video, 1, dest) == dest
        prefix = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "1.000"]
        suffix = ["-frames:v", "1", "-vf", media._scale_filter(512), "-q:v", "4", str(dest)]
    command = run.call_args.args[0]
    assert command == [*prefix, *expected, "-i", str(video.resolve()), *suffix]
    assert "-hwaccel_output_format" not in command
    if expected:
        assert command.index("-hwaccel") < command.index("-i")
    run.assert_called_once()


@pytest.mark.parametrize("operation", ["scenes", "frame"])
def test_hardware_failure_is_structured_without_retry(operation, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WATCHSKILL_FFMPEG_HWACCEL", "vaapi")
    reset_settings()
    monkeypatch.setattr(media, "probe", lambda path: VideoMetadata(10, 320, 180, 30, "h264", False))
    monkeypatch.setattr(media, "require_binary", lambda name: Path(name))
    monkeypatch.setattr(scenes, "require_binary", lambda name: Path(name))
    run = Mock(return_value=subprocess.CompletedProcess(
        [], 1, stderr="Device creation failed: unavailable VAAPI device"
    ))
    monkeypatch.setattr(subprocess, "run", run)
    dest = tmp_path / "partial.jpg"
    dest.write_bytes(b"partial")
    with pytest.raises(PerceptionError) as raised:
        if operation == "scenes":
            scenes.detect_scenes_ffmpeg(tmp_path / "video.mp4")
        else:
            media.extract_frame_at(tmp_path / "video.mp4", 1, dest)
    assert "Device creation failed" in str(raised.value)
    assert raised.value.fix
    if operation == "frame":
        assert raised.value.code == "perceive.frame_extraction_failed"
        assert not dest.exists()
    run.assert_called_once()


@pytest.mark.parametrize(("hardware", "returncode"), [("", 1), ("", 0), ("vaapi", 0)])
def test_missing_frame_still_skipped(hardware, returncode, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WATCHSKILL_FFMPEG_HWACCEL", hardware)
    reset_settings()
    monkeypatch.setattr(media, "require_binary", lambda name: Path(name))
    monkeypatch.setattr(subprocess, "run", Mock(return_value=subprocess.CompletedProcess(
        [], returncode, stderr=""
    )))
    assert media.extract_frame_at(tmp_path / "video.mp4", 1000, tmp_path / "frame.jpg") is None
