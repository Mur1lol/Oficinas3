"""
vision/segmentation.py
======================
Divides a perspective-corrected image into individual square ROIs for each
of the 64 chess squares (A1–H8) and for the cemetery slots on each side.

After PerspectiveCalibrator.correct_perspective(), the output image has:

    +----------+--------------------+----------+
    | Left Cem |     Board 8×8      | Right Cem|
    | (160 px) |    (640×640 px)    | (160 px) |
    +----------+--------------------+----------+

The board is divided into 8 columns (A–H, left→right) and 8 rows
(8→1, top→bottom) following standard chess board orientation when
A1 is at the bottom-left and the camera looks straight down.

Cemetery slots are arranged vertically, 8 per side.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from . import config as cfg

log = logging.getLogger(__name__)

BOARD_SZ  = cfg.BOARD_OUTPUT_SIZE_PX        # 640
CEM_W     = cfg.CEMETERY_REGION_WIDTH_PX    # 160
SQUARE_SZ = BOARD_SZ // 8                  # 80 px per square
SLOT_H    = BOARD_SZ // cfg.CEMETERY_SLOTS_PER_SIDE   # cemetery slot height

COLUMNS = "ABCDEFGH"
ROWS    = "87654321"  # top→bottom in the corrected image → row 8 first


# ---------------------------------------------------------------------------
# BoardSegmenter
# ---------------------------------------------------------------------------

class BoardSegmenter:
    """
    Extracts individual ROI (Region of Interest) arrays from a corrected image.

    Parameters
    ----------
    board_orientation : str
        "normal" → A1 is at bottom-left (default).
        "flipped" → A1 is at top-right (rotated 180°).
    """

    def __init__(self, board_orientation: str = "normal") -> None:
        self._orientation = board_orientation

    # ------------------------------------------------------------------
    # Board segmentation
    # ------------------------------------------------------------------

    def segment_board(self, corrected: np.ndarray) -> dict[str, np.ndarray]:
        """
        Split the board region into 64 square ROIs.

        Parameters
        ----------
        corrected : np.ndarray
            Full corrected image (TOTAL_H × TOTAL_W) from PerspectiveCalibrator.

        Returns
        -------
        dict mapping square name (e.g. "A1") to its ROI (np.ndarray BGR).
        """
        board = corrected[:, CEM_W: CEM_W + BOARD_SZ]

        if self._orientation == "flipped":
            board = cv2.rotate(board, cv2.ROTATE_180)

        squares: dict[str, np.ndarray] = {}
        for row_idx, row_label in enumerate(ROWS):   # 0=row8, 7=row1
            for col_idx, col_label in enumerate(COLUMNS):   # 0=A, 7=H
                y0 = row_idx * SQUARE_SZ
                y1 = y0 + SQUARE_SZ
                x0 = col_idx * SQUARE_SZ
                x1 = x0 + SQUARE_SZ
                roi = board[y0:y1, x0:x1]
                name = f"{col_label}{row_label}"
                squares[name] = roi

        log.debug("[Segmenter] Board split into %d squares.", len(squares))
        return squares

    # ------------------------------------------------------------------
    # Cemetery segmentation
    # ------------------------------------------------------------------

    def segment_cemeteries(
        self, corrected: np.ndarray
    ) -> dict[str, list[np.ndarray]]:
        """
        Split the two cemetery side regions into individual slot ROIs.

        By convention, the left cemetery holds white captured pieces and
        the right cemetery holds black captured pieces (configurable).

        Returns
        -------
        dict with keys "white" and "black", each a list of slot ROIs.
        """
        left_region  = corrected[:, :CEM_W]
        right_region = corrected[:, CEM_W + BOARD_SZ:]

        white_slots = self._split_cemetery(left_region)
        black_slots = self._split_cemetery(right_region)

        log.debug(
            "[Segmenter] Cemeteries: %d white slots, %d black slots.",
            len(white_slots), len(black_slots),
        )
        return {"white": white_slots, "black": black_slots}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_cemetery(region: np.ndarray) -> list[np.ndarray]:
        """Split a cemetery region vertically into equal-height slots."""
        h = region.shape[0]
        slot_h = h // cfg.CEMETERY_SLOTS_PER_SIDE
        slots = []
        for i in range(cfg.CEMETERY_SLOTS_PER_SIDE):
            y0 = i * slot_h
            y1 = y0 + slot_h
            slots.append(region[y0:y1, :])
        return slots

    def draw_grid(self, corrected: np.ndarray) -> np.ndarray:
        """
        Draw the segmentation grid on a copy of *corrected* for debugging.

        Returns the annotated image.
        """
        img = corrected.copy()

        # Board grid
        for i in range(9):
            x = CEM_W + i * SQUARE_SZ
            cv2.line(img, (x, 0), (x, BOARD_SZ), (0, 255, 0), 1)
            y = i * SQUARE_SZ
            cv2.line(img, (CEM_W, y), (CEM_W + BOARD_SZ, y), (0, 255, 0), 1)

        # Square labels (centre of each)
        for row_idx, row_label in enumerate(ROWS):
            for col_idx, col_label in enumerate(COLUMNS):
                cx = CEM_W + col_idx * SQUARE_SZ + SQUARE_SZ // 2
                cy = row_idx * SQUARE_SZ + SQUARE_SZ // 2
                cv2.putText(
                    img, f"{col_label}{row_label}",
                    (cx - 16, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
                )

        # Cemetery dividers
        for i in range(1, cfg.CEMETERY_SLOTS_PER_SIDE):
            y = i * SLOT_H
            cv2.line(img, (0, y), (CEM_W, y), (255, 165, 0), 1)
            cv2.line(img, (CEM_W + BOARD_SZ, y),
                     (CEM_W + BOARD_SZ + CEM_W, y), (255, 165, 0), 1)

        return img
