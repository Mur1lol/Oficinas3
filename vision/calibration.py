"""
vision/calibration.py
=====================
Perspective calibration for the VoiceChess camera system.

Transforms a raw camera image (board seen at an angle) into a rectified
top-down view with the board and side cemeteries at fixed pixel positions.

Workflow
--------
1.  Run ``scripts/calibrate_camera.py`` once to select the four board corners
    and the two cemetery rectangles interactively.
2.  The result is saved to ``camera_calibration.json``.
3.  At runtime, ``PerspectiveCalibrator.load()`` reads that file and
    ``correct_perspective()`` applies the transform to every captured frame.

Coordinate conventions
-----------------------
Board corners are given in the *original* (raw) camera image, in order:
    [top-left, top-right, bottom-right, bottom-left]  (clock-wise)

After correction, the output image has fixed dimensions:
    width  = BOARD_OUTPUT_SIZE_PX + 2 * CEMETERY_REGION_WIDTH_PX
    height = BOARD_OUTPUT_SIZE_PX

    Left cemetery  : x = 0..CEMETERY_WIDTH
    Board          : x = CEMETERY_WIDTH .. CEMETERY_WIDTH + BOARD_SIZE
    Right cemetery : x = CEMETERY_WIDTH + BOARD_SIZE .. total_width
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config as cfg

log = logging.getLogger(__name__)

BOARD_SZ    = cfg.BOARD_OUTPUT_SIZE_PX
CEM_W       = cfg.CEMETERY_REGION_WIDTH_PX
TOTAL_W     = BOARD_SZ + 2 * CEM_W
TOTAL_H     = BOARD_SZ


# ---------------------------------------------------------------------------
# CalibrationData
# ---------------------------------------------------------------------------

@dataclass
class CalibrationData:
    """
    All parameters needed to rectify an image and locate the board/cemeteries.

    board_corners : list of 4 [x, y] pairs in the raw image (clock-wise).
    left_cemetery_raw  : [[x,y,w,h]] region in raw image for the left cemetery.
    right_cemetery_raw : [[x,y,w,h]] region in raw image for the right cemetery.
    board_orientation  : "normal" or "flipped" (A1 at bottom-left or top-left).
    camera_device      : device index used during calibration.
    capture_resolution : [width, height] used during calibration.
    """
    board_corners:        list[list[float]]        = field(default_factory=list)
    left_cemetery_raw:    Optional[list[float]]    = None   # [x, y, w, h]
    right_cemetery_raw:   Optional[list[float]]    = None   # [x, y, w, h]
    board_orientation:    str                      = "normal"
    camera_device:        int                      = 0
    capture_resolution:   list[int]                = field(default_factory=lambda: [1280, 960])

    def is_valid(self) -> bool:
        return len(self.board_corners) == 4

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationData":
        return cls(**d)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.info("[Calibration] Saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationData":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        data = cls.from_dict(d)
        log.info("[Calibration] Loaded from %s", path)
        return data


# ---------------------------------------------------------------------------
# PerspectiveCalibrator
# ---------------------------------------------------------------------------

class PerspectiveCalibrator:
    """
    Applies perspective correction to raw camera frames using stored calibration.

    Parameters
    ----------
    calibration : CalibrationData
        Loaded calibration data.
    """

    def __init__(self, calibration: CalibrationData) -> None:
        self._cal = calibration
        self._M: np.ndarray | None = None    # perspective transform matrix
        self._M_cem_left: np.ndarray | None  = None
        self._M_cem_right: np.ndarray | None = None
        if calibration.is_valid():
            self._build_transforms()

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path = cfg.CALIBRATION_FILE) -> "PerspectiveCalibrator":
        """Load calibration from JSON and build the warp matrices."""
        cal = CalibrationData.load(path)
        return cls(cal)

    @classmethod
    def default_uncalibrated(cls) -> "PerspectiveCalibrator":
        """Return an uncalibrated instance (identity transform)."""
        return cls(CalibrationData())

    # ------------------------------------------------------------------
    # Core transform
    # ------------------------------------------------------------------

    def correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """
        Warp *image* so the board fills a square at the centre of the output.

        Returns
        -------
        np.ndarray
            BGR image of size (TOTAL_H, TOTAL_W) = (board_size, board_size + 2*cem_w).
            If calibration is not valid, returns a centre-crop of the input resized
            to the output dimensions (passthrough for testing without calibration).
        """
        if self._M is None:
            log.warning("[Calibration] No valid calibration — returning resized crop.")
            return cv2.resize(image, (TOTAL_W, TOTAL_H))

        warped = cv2.warpPerspective(image, self._M, (TOTAL_W, TOTAL_H))
        return warped

    def extract_board_region(self, corrected: np.ndarray) -> np.ndarray:
        """Return the board-only region from a corrected image."""
        return corrected[:, CEM_W: CEM_W + BOARD_SZ]

    def extract_left_cemetery(self, corrected: np.ndarray) -> np.ndarray:
        """Return the left cemetery region from a corrected image."""
        return corrected[:, :CEM_W]

    def extract_right_cemetery(self, corrected: np.ndarray) -> np.ndarray:
        """Return the right cemetery region from a corrected image."""
        return corrected[:, CEM_W + BOARD_SZ:]

    # ------------------------------------------------------------------
    # Calibration update
    # ------------------------------------------------------------------

    def update(self, calibration: CalibrationData) -> None:
        """Replace the active calibration and rebuild warp matrices."""
        self._cal = calibration
        self._M  = None
        if calibration.is_valid():
            self._build_transforms()

    @property
    def is_calibrated(self) -> bool:
        return self._M is not None

    @property
    def calibration(self) -> CalibrationData:
        return self._cal

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_transforms(self) -> None:
        """Pre-compute the perspective transform matrix."""
        src = np.array(self._cal.board_corners, dtype=np.float32)

        # Destination: board occupies the centre band [CEM_W .. CEM_W+BOARD_SZ]
        dst = np.array([
            [CEM_W,           0],           # top-left
            [CEM_W + BOARD_SZ, 0],          # top-right
            [CEM_W + BOARD_SZ, BOARD_SZ],   # bottom-right
            [CEM_W,           BOARD_SZ],    # bottom-left
        ], dtype=np.float32)

        self._M = cv2.getPerspectiveTransform(src, dst)
        log.debug("[Calibration] Perspective matrix built.")


# ---------------------------------------------------------------------------
# Interactive calibration helpers (used by scripts/calibrate_camera.py)
# ---------------------------------------------------------------------------

class InteractiveCalibrator:
    """
    Guides the user through clicking the four board corners and optionally
    the two cemetery bounding boxes on a live or static image.

    Typical usage (from the calibration script):
        ic = InteractiveCalibrator()
        cal = ic.run(image)
        cal.save(cfg.CALIBRATION_FILE)
    """

    WINDOW = "VoiceChess Calibration"

    def __init__(self) -> None:
        self._points: list[tuple[int, int]] = []
        self._done = False
        self._image: np.ndarray | None = None

    def run(
        self,
        image: np.ndarray,
        device_index: int = cfg.CAMERA_DEVICE_INDEX,
        resolution: tuple[int, int] = cfg.CAMERA_RESOLUTION,
    ) -> CalibrationData:
        """
        Display *image* and ask the user to click the 4 board corners
        in clock-wise order starting from the top-left (A8 side).

        Returns a populated CalibrationData (without cemetery info).
        """
        self._image  = image.copy()
        self._points = []
        self._done   = False

        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 1024, 768)
        cv2.setMouseCallback(self.WINDOW, self._mouse_callback)

        print("\n--- VoiceChess Camera Calibration ---")
        print("Click the 4 BOARD CORNERS in clock-wise order:")
        print("  1. Top-Left  (corner near A8)")
        print("  2. Top-Right (corner near H8)")
        print("  3. Bottom-Right (corner near H1)")
        print("  4. Bottom-Left  (corner near A1)")
        print("Press ENTER to confirm, ESC to restart.\n")

        while True:
            display = self._draw_overlay()
            cv2.imshow(self.WINDOW, display)
            key = cv2.waitKey(20) & 0xFF

            if key == 27:   # ESC — restart
                self._points = []
                print("Restarting corner selection...")

            elif key == 13 and len(self._points) == 4:  # ENTER — confirm
                break

        cv2.destroyWindow(self.WINDOW)

        cal = CalibrationData(
            board_corners=[list(p) for p in self._points],
            board_orientation="normal",
            camera_device=device_index,
            capture_resolution=list(resolution),
        )
        print(f"Calibration complete. Corners: {cal.board_corners}")
        return cal

    def _mouse_callback(
        self, event: int, x: int, y: int, flags: int, param
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self._points) < 4:
            self._points.append((x, y))
            print(f"  Corner {len(self._points)}: ({x}, {y})")

    def _draw_overlay(self) -> np.ndarray:
        img = self._image.copy()
        colors = [(0, 255, 0), (0, 200, 255), (0, 0, 255), (255, 0, 0)]
        labels = ["1:TL", "2:TR", "3:BR", "4:BL"]

        for i, (px, py) in enumerate(self._points):
            cv2.circle(img, (px, py), 8, colors[i], -1)
            cv2.putText(img, labels[i], (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)

        if len(self._points) == 4:
            pts = np.array(self._points, dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 255), 2)
            cv2.putText(img, "Press ENTER to confirm", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        else:
            remaining = 4 - len(self._points)
            cv2.putText(img, f"Click {remaining} more corner(s)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img
