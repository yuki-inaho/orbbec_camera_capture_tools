"""Abstract camera device interfaces and data bundles.

This module defines the abstract base class that concrete camera
implementations (e.g., Orbbec Gemini 335Le) must implement, along with
data containers for frames and calibration parameters.

Notes:
    - All concrete implementations must provide type hints and Google
      style docstrings per project policy.
    - Implicit fallbacks are prohibited. Implementations should raise
      explicit exceptions on failure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class FrameBundle:
    """Container for frames captured from a single grab operation.

    Attributes:
        color: Color frame as an RGB ndarray (H, W, 3) or None.
        depth: Depth frame as a uint16 ndarray (H, W) or None.
        ir_left: Left IR frame as ndarray (H, W, 1) or None.
        ir_right: Right IR frame as ndarray (H, W, 1) or None.
        timestamp: Timestamp in microseconds or seconds (float) when the
            frames were captured.
    """

    color: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    ir_left: Optional[np.ndarray]
    ir_right: Optional[np.ndarray]
    timestamp: Optional[float]


@dataclass
class IntrinsicsBundle:
    """Camera intrinsic parameters.

    Attributes:
        fx: Focal length in pixels along x-axis.
        fy: Focal length in pixels along y-axis.
        cx: Principal point x-coordinate in pixels.
        cy: Principal point y-coordinate in pixels.
        width: Image width in pixels.
        height: Image height in pixels.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class ExtrinsicsBundle:
    """Camera extrinsic parameters (rotation and translation).

    Attributes:
        rotation: 3x3 rotation matrix (ndarray).
        translation: Translation vector (3,) in meters.
    """

    rotation: np.ndarray
    translation: np.ndarray


class CameraDevice(ABC):
    """Abstract base class for camera devices.

    Concrete implementations must explicitly raise exceptions on failure
    and must not rely on implicit fallbacks.
    """

    @abstractmethod
    def open(self) -> None:
        """Initialize resources and prepare the device connection."""

    @abstractmethod
    def start(self) -> None:
        """Start streaming on configured profiles."""

    @abstractmethod
    def grab(self, timeout_ms: Optional[int] = None) -> FrameBundle:
        """Capture frames from the device.

        Args:
            timeout_ms: Timeout in milliseconds for frame retrieval. If
                None, use implementation-defined default.

        Returns:
            FrameBundle containing the captured frames.

        Raises:
            RuntimeError: If frames cannot be retrieved within timeout or
                device is not ready.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming and flush pipelines."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources and close the device connection."""

    # Metadata -----------------------------------------------------------------
    @abstractmethod
    def get_intrinsics(self) -> IntrinsicsBundle:
        """Return camera intrinsics for the active profile."""

    @abstractmethod
    def get_extrinsics(self) -> Optional[ExtrinsicsBundle]:
        """Return camera extrinsics if available."""

    @abstractmethod
    def get_image_size(self) -> Tuple[int, int]:
        """Return image size as (width, height)."""

    @abstractmethod
    def get_supported_profiles(self) -> List[str]:
        """Return supported profile names for this device."""

    # Controls -----------------------------------------------------------------
    @abstractmethod
    def set_exposure(self, microseconds: int) -> None:
        """Set exposure time in microseconds."""

    @abstractmethod
    def set_gain(self, value: int) -> None:
        """Set sensor gain value."""

    @abstractmethod
    def set_profile(self, name: str) -> None:
        """Select a predefined profile/preset by name."""

    @abstractmethod
    def set_hdr(
        self,
        enabled: bool,
        exposure_1: Optional[int] = None,
        exposure_2: Optional[int] = None,
    ) -> None:
        """Enable or disable HDR and optionally set exposure values."""

    @abstractmethod
    def set_align_mode(self, mode: str) -> None:
        """Set alignment mode (e.g., hw, sw, none)."""

    # Export -------------------------------------------------------------------
    @abstractmethod
    def export_calibration(self, path: str) -> None:
        """Export calibration data (intrinsics/extrinsics) to path."""

    @abstractmethod
    def export_config(self, path: str) -> None:
        """Export current configuration (profiles, HDR, align settings)."""

    @abstractmethod
    def start_recording(self, path: str) -> None:
        """Start sequential recording (e.g., .bag) to the specified path."""

    @abstractmethod
    def stop_recording(self) -> None:
        """Stop an active sequential recording."""


__all__ = [
    "FrameBundle",
    "IntrinsicsBundle",
    "ExtrinsicsBundle",
    "CameraDevice",
]
