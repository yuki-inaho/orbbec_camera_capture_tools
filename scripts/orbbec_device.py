"""Orbbec Gemini 335Le implementation of CameraDevice.

This adapter wraps pyorbbecsdk to provide a unified interface defined by
`CameraDevice`. It avoids implicit fallbacks and raises explicit
exceptions on failure.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import pyorbbecsdk as sdk

    _SDK_AVAILABLE = True
    _SdkUnavailable = None  # type: ignore
except ImportError:

    class _SdkUnavailable:
        """Placeholder that raises on any access when SDK is missing."""

        def __getattr__(self, name):
            raise ImportError(
                "pyorbbecsdk is not available. Install the official SDK wheel for your "
                "Python version (e.g., pip install git+https://github.com/orbbec/pyorbbecsdk@v2-main)."
            )

    sdk = _SdkUnavailable()
    _SDK_AVAILABLE = False

from scripts.camera_device import (
    CameraDevice,
    ExtrinsicsBundle,
    FrameBundle,
    IntrinsicsBundle,
)


class OrbbecDevice(CameraDevice):
    """Orbbec Gemini 335Le device adapter using pyorbbecsdk.

    This class manages connection, streaming, frame conversion, parameter
    control (preset/HDR/exposure/gain/align), and calibration export.
    """

    def __init__(self, device_index: int = 0, serial_number: Optional[str] = None) -> None:
        """Initialize the adapter.

        Args:
            device_index: Index of the device to open when multiple devices are present.
            serial_number: Specific device serial to select. If provided, it overrides device_index.
        """

        self._device_index = device_index
        self._serial_number = serial_number
        self._ctx: Optional[sdk.Context] = None
        self._device: Optional[sdk.Device] = None
        self._pipeline: Optional[sdk.Pipeline] = None
        self._config: Optional[sdk.Config] = None
        self._frame_counters: Dict[str, int] = {}
        self._width: int = 640
        self._height: int = 480
        self._fps: int = 30
        self._profile_map: Dict[str, str] = {
            "default": "Default",
            "hq": "HighQuality",
            "hdr": "HDR",
        }
        self._recorder: Optional[Any] = None

    def has_imu(self) -> bool:
        """Return True if device exposes both ACCEL and GYRO sensors."""

        if not self._device:
            raise RuntimeError("Device not opened")
        sensor_list = self._device.get_sensor_list()
        has_accel = False
        has_gyro = False
        for idx in range(sensor_list.get_count()):
            sensor = sensor_list.get_sensor_by_index(idx)
            sensor_type = sensor.get_type()
            if sensor_type == sdk.OBSensorType.ACCEL_SENSOR:
                has_accel = True
            if sensor_type == sdk.OBSensorType.GYRO_SENSOR:
                has_gyro = True
        return has_accel and has_gyro

    def read_imu(self, timeout_ms: int = 1000) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """Read one accel/gyro sample if IMU is available.

        Args:
            timeout_ms: Timeout in milliseconds for IMU frame retrieval.

        Returns:
            Tuple of (accel_dict, gyro_dict). Each dict contains x, y, z, timestamp_us when available; otherwise None.

        Raises:
            RuntimeError: If device is not opened or IMU is not available.
        """

        if not _SDK_AVAILABLE or isinstance(sdk, _SdkUnavailable):
            raise RuntimeError("pyorbbecsdk is required but not available")
        if not self._device:
            raise RuntimeError("Device not opened")
        if not self.has_imu():
            raise RuntimeError("IMU sensors not available on this device")

        imu_pipeline = sdk.Pipeline(self._device)
        imu_config = sdk.Config()
        imu_config.enable_accel_stream()
        imu_config.enable_gyro_stream()
        imu_pipeline.start(imu_config)
        try:
            frames = imu_pipeline.wait_for_frames(timeout_ms)
            accel_data: Optional[Dict[str, float]] = None
            gyro_data: Optional[Dict[str, float]] = None
            if frames:
                accel_frame = frames.get_frame(sdk.OBFrameType.ACCEL_FRAME)
                gyro_frame = frames.get_frame(sdk.OBFrameType.GYRO_FRAME)
                if accel_frame:
                    a = accel_frame.as_accel_frame()
                    accel_data = {
                        "x": float(a.get_x()),
                        "y": float(a.get_y()),
                        "z": float(a.get_z()),
                        "timestamp_us": float(a.get_timestamp()),
                    }
                if gyro_frame:
                    g = gyro_frame.as_gyro_frame()
                    gyro_data = {
                        "x": float(g.get_x()),
                        "y": float(g.get_y()),
                        "z": float(g.get_z()),
                        "timestamp_us": float(g.get_timestamp()),
                    }
            return accel_data, gyro_data
        finally:
            imu_pipeline.stop()

    def open(self) -> None:
        """Initialize context, select device, and prepare pipeline/config."""

        global sdk
        if (not _SDK_AVAILABLE) or (_SdkUnavailable is not None and isinstance(sdk, _SdkUnavailable)):
            try:
                import pyorbbecsdk as sdk  # type: ignore

                globals()["sdk"] = sdk
                globals()["_SDK_AVAILABLE"] = True
                globals()["_SdkUnavailable"] = None
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("pyorbbecsdk is required but not available") from exc
        self._ctx = sdk.Context()
        device_list = self._ctx.query_devices()
        count = device_list.get_count()
        if count == 0:
            raise RuntimeError("No Orbbec device found")

        if self._serial_number:
            self._device = self._get_device_by_serial(device_list, self._serial_number)
        else:
            if self._device_index >= count:
                raise RuntimeError(f"Device index {self._device_index} out of range (found {count})")
            self._device = device_list.get_device_by_index(self._device_index)

        self._pipeline = sdk.Pipeline(self._device)
        self._config = sdk.Config()

    def start(self) -> None:
        """Start streaming with default color/depth/IR profiles."""

        if not _SDK_AVAILABLE:
            raise RuntimeError("pyorbbecsdk is required but not available")
        if not self._device or not self._pipeline or not self._config:
            raise RuntimeError("Device not opened")

        for sensor_type in [
            sdk.OBSensorType.COLOR_SENSOR,
            sdk.OBSensorType.DEPTH_SENSOR,
            sdk.OBSensorType.LEFT_IR_SENSOR,
            sdk.OBSensorType.RIGHT_IR_SENSOR,
        ]:
            self._enable_default_stream(sensor_type)

        self._pipeline.start(self._config)

    def grab(self, timeout_ms: Optional[int] = None) -> FrameBundle:
        """Capture frames from the device."""

        if not _SDK_AVAILABLE:
            raise RuntimeError("pyorbbecsdk is required but not available")
        if not self._pipeline:
            raise RuntimeError("Pipeline not started")

        timeout = 1000 if timeout_ms is None else timeout_ms
        frames = self._pipeline.wait_for_frames(timeout)
        if not frames:
            raise RuntimeError("Timed out waiting for frames")

        color = self._convert_color(frames.get_color_frame())
        depth = self._convert_depth(frames.get_depth_frame())
        ir_left = self._convert_ir(frames.get_frame(sdk.OBFrameType.LEFT_IR_FRAME))
        ir_right = self._convert_ir(frames.get_frame(sdk.OBFrameType.RIGHT_IR_FRAME))

        ts_us = 0.0
        if frames.get_color_frame() is not None:
            ts_us = float(frames.get_color_frame().get_timestamp_us())
        elif frames.get_depth_frame() is not None:
            ts_us = float(frames.get_depth_frame().get_timestamp_us())

        return FrameBundle(
            color=color,
            depth=depth,
            ir_left=ir_left,
            ir_right=ir_right,
            timestamp=ts_us,
        )

    def stop(self) -> None:
        """Stop streaming and reset counters."""

        if self._pipeline:
            self._pipeline.stop()
        self._frame_counters.clear()

    def close(self) -> None:
        """Release pipeline, config, and device resources."""

        try:
            if self._pipeline:
                self._pipeline.stop()
        finally:
            self._pipeline = None
            self._config = None
            self._device = None
            self._ctx = None

    # Metadata -----------------------------------------------------------------
    def get_intrinsics(self) -> IntrinsicsBundle:
        """Return camera intrinsics from the pipeline camera_param."""

        if not self._pipeline:
            raise RuntimeError("Pipeline not started")
        params = self._pipeline.get_camera_param()
        rgb = params.rgb_intrinsic
        return IntrinsicsBundle(
            fx=float(rgb.fx),
            fy=float(rgb.fy),
            cx=float(rgb.cx),
            cy=float(rgb.cy),
            width=int(rgb.width),
            height=int(rgb.height),
        )

    def get_extrinsics(self) -> Optional[ExtrinsicsBundle]:
        """Return depth-to-RGB extrinsics if available."""

        if not self._pipeline:
            raise RuntimeError("Pipeline not started")
        params = self._pipeline.get_camera_param()
        if not hasattr(params, "transform"):
            return None
        transform = params.transform
        rotation = np.array(transform.rot, dtype=float).reshape(3, 3)
        translation = np.array(transform.transform[:3], dtype=float)
        return ExtrinsicsBundle(rotation=rotation, translation=translation)

    def get_image_size(self) -> Tuple[int, int]:
        """Return configured image size (width, height)."""

        return self._width, self._height

    def get_supported_profiles(self) -> List[str]:
        """Return supported profile names."""

        return list(self._profile_map.keys())

    # Controls -----------------------------------------------------------------
    def set_exposure(self, microseconds: int) -> None:
        """Set color exposure time using device property."""

        if not self._device:
            raise RuntimeError("Device not opened")
        if not _SDK_AVAILABLE:
            raise RuntimeError("pyorbbecsdk is required but not available")
        prop = sdk.OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT
        if not self._device.is_property_supported(prop, sdk.OBPermissionType.PERMISSION_WRITE):
            raise RuntimeError("Color exposure property not supported")
        self._device.set_int_property(prop, microseconds)

    def set_gain(self, value: int) -> None:
        """Set color gain using device property."""

        if not self._device:
            raise RuntimeError("Device not opened")
        if not _SDK_AVAILABLE:
            raise RuntimeError("pyorbbecsdk is required but not available")
        prop = sdk.OBPropertyID.OB_PROP_COLOR_GAIN_INT
        if not self._device.is_property_supported(prop, sdk.OBPermissionType.PERMISSION_WRITE):
            raise RuntimeError("Color gain property not supported")
        self._device.set_int_property(prop, value)

    def set_profile(self, name: str) -> None:
        """Load a predefined device preset by name."""

        if not self._device:
            raise RuntimeError("Device not opened")
        if not _SDK_AVAILABLE:
            raise RuntimeError("pyorbbecsdk is required but not available")
        key = name.lower()
        if key not in self._profile_map:
            raise ValueError(f"Unsupported profile: {name}")
        self._device.load_preset(self._profile_map[key])

    def set_hdr(
        self,
        enabled: bool,
        exposure_1: Optional[int] = None,
        exposure_2: Optional[int] = None,
    ) -> None:
        """Configure HDR using OBHdrConfig."""

        if not self._device:
            raise RuntimeError("Device not opened")
        if not _SDK_AVAILABLE:
            raise RuntimeError("pyorbbecsdk is required but not available")
        hdr_config = sdk.OBHdrConfig()
        hdr_config.enable = enabled
        if exposure_1 is not None:
            hdr_config.exposure_1 = exposure_1
        if exposure_2 is not None:
            hdr_config.exposure_2 = exposure_2
        self._device.set_hdr_config(hdr_config)

    def set_align_mode(self, mode: str) -> None:
        """Set alignment mode on the config (hw, sw, none)."""

        if not self._config:
            raise RuntimeError("Config not initialized")
        if not _SDK_AVAILABLE:
            raise RuntimeError("pyorbbecsdk is required but not available")
        mode_key = mode.lower()
        if mode_key == "hw":
            self._config.set_align_mode(sdk.OBAlignMode.HW_MODE)
        elif mode_key == "sw":
            self._config.set_align_mode(sdk.OBAlignMode.SW_MODE)
        elif mode_key == "none":
            self._config.set_align_mode(sdk.OBAlignMode.DISABLE)
        else:
            raise ValueError(f"Unsupported align mode: {mode}")

    # Export -------------------------------------------------------------------
    def export_calibration(self, path: str) -> None:
        """Export intrinsics/extrinsics to JSON."""

        intrinsic = self.get_intrinsics()
        extrinsic = self.get_extrinsics()
        payload = {
            "intrinsic": asdict(intrinsic),
            "extrinsic": asdict(extrinsic) if extrinsic else None,
        }
        self._write_json(path, payload)

    def export_config(self, path: str) -> None:
        """Export current config (profiles, resolution, fps)."""

        payload = {
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
            "profiles": self.get_supported_profiles(),
        }
        self._write_json(path, payload)

    def start_recording(self, path: str) -> None:
        """Start .bag recording using pyorbbecsdk RecordDevice."""

        if not _SDK_AVAILABLE:
            raise RuntimeError("pyorbbecsdk is required but not available")
        if not hasattr(sdk, "RecordDevice"):
            raise RuntimeError("pyorbbecsdk.RecordDevice is not available in this SDK build")
        if not self._device:
            raise RuntimeError("Device not opened")
        if self._recorder is not None:
            raise RuntimeError("Recording already in progress")

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._recorder = sdk.RecordDevice(self._device, str(target))

    def stop_recording(self) -> None:
        """Stop active .bag recording."""

        if self._recorder is None:
            return
        try:
            if hasattr(self._recorder, "stop"):
                self._recorder.stop()
        finally:
            self._recorder = None

    # Internal helpers ---------------------------------------------------------
    def _write_json(self, path: str, payload: Dict) -> None:
        """Write payload to JSON file with safe directory creation."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _get_device_by_serial(self, device_list: sdk.DeviceList, serial: str) -> sdk.Device:
        """Select device matching the given serial."""

        for idx in range(device_list.get_count()):
            dev = device_list.get_device_by_index(idx)
            info = dev.get_device_info()
            if info.get_serial_number() == serial:
                return dev
        raise RuntimeError(f"Device with serial {serial} not found")

    def _enable_default_stream(self, sensor_type: sdk.OBSensorType) -> None:
        """Enable default stream profile for the given sensor type."""

        if not self._device or not self._config:
            raise RuntimeError("Device not opened")
        sensor_list = self._device.get_sensor_list()
        target_sensor = None
        for idx in range(sensor_list.get_count()):
            sensor = sensor_list.get_sensor_by_index(idx)
            if sensor.get_type() == sensor_type:
                target_sensor = sensor
                break
        if target_sensor is None:
            raise RuntimeError(f"Sensor not found: {sensor_type}")

        profiles = target_sensor.get_stream_profile_list()
        if profiles.get_count() == 0:
            raise RuntimeError(f"No profiles for sensor {sensor_type}")
        profile = profiles.get_default_video_stream_profile()
        self._width = profile.get_width()
        self._height = profile.get_height()
        self._fps = profile.get_fps()
        self._config.enable_stream(profile)

    def _convert_color(self, frame) -> Optional[np.ndarray]:
        """Convert color frame to RGB ndarray."""

        if frame is None:
            return None
        data = np.asanyarray(frame.get_data())
        fmt = frame.get_format()
        if fmt == sdk.OBFormat.MJPG:
            data = cv2.imdecode(data, cv2.IMREAD_COLOR)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        else:
            height, width = frame.get_height(), frame.get_width()
            expected = height * width * 3
            if data.size == expected:
                data = data.reshape((height, width, 3))
        self._frame_counters["color"] = self._frame_counters.get("color", 0) + 1
        return data

    def _convert_depth(self, frame) -> Optional[np.ndarray]:
        """Convert depth frame to uint16 ndarray."""

        if frame is None:
            return None
        data = np.asanyarray(frame.get_data())
        height, width = frame.get_height(), frame.get_width()
        expected_size = height * width
        depth = np.frombuffer(data, dtype=np.uint16)
        if depth.size < expected_size:
            raise RuntimeError("Depth frame size mismatch")
        depth = depth[:expected_size].reshape((height, width))
        self._frame_counters["depth"] = self._frame_counters.get("depth", 0) + 1
        return depth

    def _convert_ir(self, frame) -> Optional[np.ndarray]:
        """Convert IR frame to ndarray with channel dimension."""

        if frame is None:
            return None
        video = frame.as_video_frame()
        data = np.asanyarray(video.get_data())
        height, width = video.get_height(), video.get_width()
        fmt = video.get_format()
        if fmt == sdk.OBFormat.Y8:
            ir = np.resize(data, (height, width, 1))
        elif fmt == sdk.OBFormat.MJPG:
            decoded = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if decoded is None:
                raise RuntimeError("Failed to decode IR MJPG frame")
            ir = np.resize(decoded, (height, width, 1))
        else:
            raw = np.frombuffer(data, dtype=np.uint16)
            if raw.size < height * width:
                raise RuntimeError("IR frame size mismatch")
            ir = np.resize(raw, (height, width, 1))
        self._frame_counters["ir"] = self._frame_counters.get("ir", 0) + 1
        return ir


__all__ = ["OrbbecDevice"]
