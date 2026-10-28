#!/usr/bin/env python3
"""
scripts/calibrate_camera.py
============================
Interactive calibration wizard for the VoiceChess camera system.

Run this script once to define the four corners of the chess board in the
camera image.  The result is saved to camera_calibration.json and used by
the CameraSystem at runtime.

Usage
-----
    cd Oficinas/
    python scripts/calibrate_camera.py

    # Specify a different camera device or output file:
    python scripts/calibrate_camera.py --device 1 --output my_cal.json

    # Test the calibration on a static image (no camera needed):
    python scripts/calibrate_camera.py --image path/to/test.jpg --test-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sure we can import from the Oficinas root (vision's parent directory)
ROOT = Path(__file__).parent.parent.parent   # vision/scripts/ -> vision/ -> Oficinas/
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

import vision.config as cfg
from vision.camera      import CameraController
from vision.calibration import CalibrationData, InteractiveCalibrator, PerspectiveCalibrator
from vision.segmentation import BoardSegmenter


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VoiceChess Camera Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/calibrate_camera.py
  python scripts/calibrate_camera.py --device 1
  python scripts/calibrate_camera.py --image board.jpg --test-only
  python scripts/calibrate_camera.py --show-grid
        """,
    )
    p.add_argument(
        "--device", "-d",
        type=int, default=cfg.CAMERA_DEVICE_INDEX,
        help=f"OpenCV camera device index (default: {cfg.CAMERA_DEVICE_INDEX}).",
    )
    p.add_argument(
        "--output", "-o",
        default=cfg.CALIBRATION_FILE,
        help=f"Output JSON path (default: {cfg.CALIBRATION_FILE}).",
    )
    p.add_argument(
        "--image", "-i",
        help="Path to a static image for calibration (skips live capture).",
    )
    p.add_argument(
        "--test-only",
        action="store_true",
        help="Load existing calibration and display corrected result; no new calibration.",
    )
    p.add_argument(
        "--show-grid",
        action="store_true",
        help="Overlay the 8×8 board grid on the corrected output.",
    )
    p.add_argument(
        "--resolution",
        nargs=2, type=int, metavar=("W", "H"),
        default=list(cfg.CAMERA_RESOLUTION),
        help=f"Capture resolution (default: {cfg.CAMERA_RESOLUTION[0]} {cfg.CAMERA_RESOLUTION[1]}).",
    )
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def show_result(corrected: np.ndarray, show_grid: bool = False) -> None:
    """Display the corrected image (blocking until 'q' or window closed)."""
    segmenter = BoardSegmenter()
    display   = segmenter.draw_grid(corrected) if show_grid else corrected.copy()

    cv2.namedWindow("Corrected Board", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Corrected Board", 800, 600)
    cv2.imshow("Corrected Board", display)
    print("Press 'q' or close the window to exit.")
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q") or cv2.getWindowProperty("Corrected Board", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    resolution = tuple(args.resolution)

    # ── Test-only mode: load existing calibration ─────────────────────
    if args.test_only:
        cal_path = args.output
        if not Path(cal_path).exists():
            print(f"[ERROR] Calibration file not found: {cal_path}")
            sys.exit(1)

        calibrator = PerspectiveCalibrator.load(cal_path)
        print(f"[OK] Calibration loaded from {cal_path}")
        print(f"     Corners: {calibrator.calibration.board_corners}")

        # Source image
        if args.image:
            image = cv2.imread(args.image)
            if image is None:
                print(f"[ERROR] Cannot read image: {args.image}")
                sys.exit(1)
        else:
            print("Capturing live frame...")
            with CameraController(device_index=args.device, resolution=resolution) as cam:
                image = cam.capture_with_retry()
            if image is None:
                print("[ERROR] Failed to capture frame.")
                sys.exit(1)

        corrected = calibrator.correct_perspective(image)
        show_result(corrected, show_grid=args.show_grid)
        return

    # ── Calibration mode: select corners ─────────────────────────────
    if args.image:
        print(f"Using static image: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"[ERROR] Cannot read image: {args.image}")
            sys.exit(1)
    else:
        print(f"Opening camera device {args.device} ...")
        with CameraController(
            device_index=args.device,
            resolution=resolution,
            warmup_seconds=cfg.CAMERA_WARMUP_SECONDS,
        ) as cam:
            image = cam.capture_with_retry()

        if image is None:
            print("[ERROR] Failed to capture image from camera.")
            sys.exit(1)

    # Run the interactive corner-selection wizard
    ic  = InteractiveCalibrator()
    cal = ic.run(image, device_index=args.device, resolution=resolution)

    # Validate
    if not cal.is_valid():
        print("[ERROR] Fewer than 4 corners selected. Calibration aborted.")
        sys.exit(1)

    # Save
    out_path = args.output
    cal.save(out_path)
    print(f"[OK] Calibration saved to {out_path}")

    # Preview
    calibrator = PerspectiveCalibrator(cal)
    corrected  = calibrator.correct_perspective(image)

    # Save a reference diagnostic image
    diag_dir = Path(cfg.DIAGNOSTIC_DIR)
    diag_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(diag_dir / "calibration_corrected.jpg"), corrected)
    print(f"[OK] Preview saved to {diag_dir / 'calibration_corrected.jpg'}")

    show_result(corrected, show_grid=args.show_grid)


if __name__ == "__main__":
    main()
