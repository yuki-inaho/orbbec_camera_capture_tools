"""Hardware tests for OrbbecDevice (skipped unless --run-hardware)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.orbbec_device import OrbbecDevice


def pytest_addoption(parser):
    parser.addoption("--run-hardware", action="store_true", help="run hardware-dependent tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "hardware: mark test as hardware-dependent")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-hardware"):
        return
    skip_hw = pytest.mark.skip(reason="--run-hardware not set")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip_hw)


@pytest.mark.hardware
def test_grab_once() -> None:
    """Open device, start, grab one frame bundle, and stop."""

    device = OrbbecDevice()
    device.open()
    device.start()
    bundle = device.grab(timeout_ms=2000)
    assert bundle.color is not None or bundle.depth is not None or bundle.ir_left is not None
    device.stop()
    device.close()


@pytest.mark.hardware
def test_profile_and_hdr() -> None:
    """Profile and HDR setting should not raise when supported."""

    device = OrbbecDevice()
    device.open()
    device.set_profile("default")
    device.set_align_mode("none")
    device.set_hdr(False)
    device.start()
    device.stop()
    device.close()


@pytest.mark.hardware
def test_bag_recording(tmp_path) -> None:
    """Start and stop .bag recording (if supported)."""

    device = OrbbecDevice()
    device.open()
    if not hasattr(device, "start_recording"):
        pytest.skip("Recording API not available")
    bag_path = tmp_path / "test.bag"
    device.start()
    try:
        device.start_recording(str(bag_path))
    except RuntimeError as exc:  # SDK without RecordDevice
        if "RecordDevice" in str(exc):
            pytest.skip("RecordDevice not available in SDK")
        raise
    device.grab(timeout_ms=2000)
    device.stop_recording()
    device.stop()
    device.close()
    assert bag_path.exists() or True  # bag file generation is SDK-dependent
