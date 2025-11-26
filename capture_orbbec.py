"""CLI entry for Orbbec Gemini 335Le capture and recording.

This script uses `scripts.orbbec_device.OrbbecDevice` to start the
camera, configure profile/HDR/align, and save frames as raw files
(PNG/NPY+JSON) or record sequential .bag files via pyorbbecsdk. It
avoids implicit fallbacks and raises explicit exceptions on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from scripts.orbbec_device import OrbbecDevice


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Orbbec Gemini 335Le capture tool")
    parser.add_argument("--profile", default="default", help="Preset/profile name (default/hq/hdr)")
    parser.add_argument("--hdr", action="store_true", help="Enable HDR mode")
    parser.add_argument("--hdr-exposure1", type=int, default=None, help="HDR exposure 1 (us)")
    parser.add_argument("--hdr-exposure2", type=int, default=None, help="HDR exposure 2 (us)")
    parser.add_argument("--align", choices=["none", "hw", "sw"], default="none", help="Align mode")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("./output"),
        help="Directory to save frames/videos",
    )
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to capture")
    parser.add_argument(
        "--bag-path",
        type=Path,
        default=None,
        help="Record sequential .bag file using pyorbbecsdk RecordDevice",
    )
    parser.add_argument(
        "--save-mode",
        choices=["raw", "bag"],
        default="raw",
        help="raw: PNG/NPY+JSON を保存, bag: .bag でシーケンシャル記録",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save individual frame files")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""

    path.mkdir(parents=True, exist_ok=True)


def save_raw_bundle(base: Path, idx: int, bundle) -> None:
    """Save frames and metadata as raw files."""

    if bundle.color is not None:
        cv2.imwrite(str(base / f"color_{idx:06d}.png"), to_bgr(bundle.color))
    if bundle.depth is not None:
        np.save(base / f"depth_{idx:06d}.npy", bundle.depth)
    if bundle.ir_left is not None:
        np.save(base / f"ir_left_{idx:06d}.npy", bundle.ir_left)
    if bundle.ir_right is not None:
        np.save(base / f"ir_right_{idx:06d}.npy", bundle.ir_right)
    meta = {
        "timestamp": bundle.timestamp,
        "has_color": bundle.color is not None,
        "has_depth": bundle.depth is not None,
        "has_ir_left": bundle.ir_left is not None,
        "has_ir_right": bundle.ir_right is not None,
    }
    (base / f"meta_{idx:06d}.json").write_text(json.dumps(meta), encoding="utf-8")


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to BGR for OpenCV writing."""

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def visualize_depth(depth: np.ndarray) -> np.ndarray:
    """Visualize depth as BGR image."""

    depth_clipped = np.clip(depth, 0, 65535).astype(np.uint16)
    normalized = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return colored


def visualize_ir(ir: np.ndarray) -> np.ndarray:
    """Visualize IR as BGR image."""

    ir_8u = cv2.normalize(ir, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return cv2.cvtColor(ir_8u, cv2.COLOR_GRAY2BGR)


def main() -> int:
    """Run capture workflow."""

    args = parse_args()
    device = OrbbecDevice()
    device.open()
    device.set_profile(args.profile)
    device.set_align_mode(args.align)
    if args.hdr:
        device.set_hdr(True, exposure_1=args.hdr_exposure1, exposure_2=args.hdr_exposure2)
    device.start()

    ensure_dir(args.save_dir)

    if args.save_mode == "bag":
        bag_target = args.bag_path or (args.save_dir / "capture.bag")
        device.start_recording(str(bag_target))

    for idx in range(args.frames):
        bundle = device.grab()

        if not args.no_save and args.save_mode == "raw":
            save_raw_bundle(args.save_dir, idx, bundle)

        if idx % 10 == 0:
            print(f"Captured {idx + 1}/{args.frames}")

    device.stop_recording()
    device.stop()
    device.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
