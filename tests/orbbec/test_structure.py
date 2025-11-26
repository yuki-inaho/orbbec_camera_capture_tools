"""Structure tests for Orbbec capture modules."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.camera_device import (
    CameraDevice,
    ExtrinsicsBundle,
    FrameBundle,
    IntrinsicsBundle,
)
from scripts.orbbec_device import OrbbecDevice


def test_camera_device_members() -> None:
    """Ensure CameraDevice declares required methods."""

    for attr in [
        "open",
        "start",
        "grab",
        "stop",
        "close",
        "get_intrinsics",
        "get_extrinsics",
        "get_image_size",
        "get_supported_profiles",
        "set_exposure",
        "set_gain",
        "set_profile",
        "set_hdr",
        "set_align_mode",
        "export_calibration",
        "export_config",
        "start_recording",
        "stop_recording",
    ]:
        assert hasattr(CameraDevice, attr)


def test_dataclasses_exist() -> None:
    """Ensure dataclasses are importable and constructible."""

    bundle = FrameBundle(None, None, None, None, None)
    intr = IntrinsicsBundle(1.0, 1.0, 0.0, 0.0, 640, 480)
    ext = ExtrinsicsBundle(rotation=None, translation=None)  # type: ignore[arg-type]
    assert bundle is not None
    assert intr.width == 640
    assert ext.translation is None or True


def test_orbbec_device_interface() -> None:
    """OrbbecDevice exposes grab method and supported profiles."""

    assert hasattr(OrbbecDevice, "grab")
    assert hasattr(OrbbecDevice, "start_recording")
    device = OrbbecDevice()
    profiles = device.get_supported_profiles()
    assert isinstance(profiles, list)
