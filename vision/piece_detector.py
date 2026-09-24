"""
vision/piece_detector.py
========================
Detects whether a board square is occupied and identifies the piece colour.

Detection pipeline (per square)
--------------------------------
1. **Occupancy** — Compare the square ROI to its corresponding region in the
   reference (empty board) image using mean absolute difference (MAD) and
   pixel variance.  A high MAD or high variance relative to the reference
   indicates a piece is present.

2. **Colour** — For occupied squares, analyse the HSV histogram of the top
   80% of the ROI (avoiding the board surface colour) to distinguish
   white pieces (high V, low S) from black pieces (low V, any S).

3. **Type** (future — Opção B) — A lightweight CNN (MobileNetV3 or similar)
   loaded once and reused for every call.  Currently returns None.

All thresholds are defined in vision_config.py.

Usage
-----
    detector = PieceDetector(reference_image)
    analysis = detector.analyze_square("E2", roi)
    print(analysis.occupied, analysis.color, analysis.confidence)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config as cfg

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SquareAnalysis
# ---------------------------------------------------------------------------

@dataclass
class SquareAnalysis:
    """
    Result of analysing one board square.

    Attributes
    ----------
    square : str
        Square name, e.g. "E2".
    occupied : bool
        True if a piece was detected.
    color : str | None
        "white" or "black" if occupied; else None.
    piece : str | None
        Piece type (future); always None in this build.
    confidence : float
        Combined detection confidence [0.0–1.0].
    mad : float
        Mean absolute difference vs. reference (occupancy signal).
    variance : float
        Pixel variance of the ROI (secondary occupancy signal).
    """
    square:     str
    occupied:   bool
    color:      Optional[str]
    piece:      Optional[str]
    confidence: float
    mad:        float = 0.0
    variance:   float = 0.0

    def to_dict(self) -> dict:
        return {
            "square":     self.square,
            "occupied":   self.occupied,
            "color":      self.color,
            "piece":      self.piece,
            "confidence": round(self.confidence, 4),
        }


# ---------------------------------------------------------------------------
# PieceDetector
# ---------------------------------------------------------------------------

class PieceDetector:
    """
    Stateless detector that classifies one square ROI at a time.

    Parameters
    ----------
    reference_image : np.ndarray | None
        The corrected empty-board image captured before the game starts.
        If None, occupancy detection falls back to variance-only mode
        (less reliable but usable for testing without a real board).
    occupied_diff_threshold : float
        Minimum MAD [0–1] to consider a square occupied.
    occupied_variance_threshold : float
        Minimum pixel variance to consider a square occupied (fallback).
    brightness_threshold : int
        HSV V-channel threshold separating white (high) from black (low).
    min_confidence : float
        Minimum confidence to consider a result valid.
    """

    def __init__(
        self,
        reference_image: np.ndarray | None = None,
        occupied_diff_threshold: float = cfg.OCCUPIED_DIFF_THRESHOLD,
        occupied_variance_threshold: float = cfg.OCCUPIED_VARIANCE_THRESHOLD,
        brightness_threshold: int = cfg.PIECE_BRIGHTNESS_THRESHOLD,
        min_confidence: float = cfg.MIN_CONFIDENCE,
    ) -> None:
        self._ref = reference_image
        self._diff_thresh = occupied_diff_threshold
        self._var_thresh  = occupied_variance_threshold
        self._bright_thr  = brightness_threshold
        self._min_conf    = min_confidence

        # Extract reference square ROIs once (avoid re-slicing on every call)
        self._ref_squares: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_reference(self, reference_image: np.ndarray) -> None:
        """Update the reference (empty board) image and clear the ROI cache."""
        self._ref = reference_image
        self._ref_squares = {}
        log.info("[Detector] Reference image updated.")

    def analyze_square(
        self,
        square: str,
        roi: np.ndarray,
        ref_roi: np.ndarray | None = None,
    ) -> SquareAnalysis:
        """
        Analyse a single square ROI.

        Parameters
        ----------
        square : str
            Square label, e.g. "A1".
        roi : np.ndarray
            Current BGR image crop of the square.
        ref_roi : np.ndarray | None
            Reference (empty) crop of the same square.  If None and a
            reference image is stored, uses it automatically.

        Returns
        -------
        SquareAnalysis
        """
        # ── Occupancy detection ───────────────────────────────────────
        occupied, occ_confidence, mad, variance = self._detect_occupancy(
            roi, ref_roi
        )

        # ── Colour detection ──────────────────────────────────────────
        color: Optional[str] = None
        color_confidence = 1.0
        if occupied:
            color, color_confidence = self._detect_color(roi)

        # ── Combined confidence ───────────────────────────────────────
        confidence = occ_confidence * color_confidence if occupied else occ_confidence

        return SquareAnalysis(
            square=square,
            occupied=occupied,
            color=color,
            piece=None,    # Opção B — future
            confidence=confidence,
            mad=mad,
            variance=variance,
        )

    def analyze_all(
        self,
        squares: dict[str, np.ndarray],
        ref_squares: dict[str, np.ndarray] | None = None,
    ) -> dict[str, SquareAnalysis]:
        """
        Analyse all 64 board squares.

        Parameters
        ----------
        squares : dict[str, np.ndarray]
            From BoardSegmenter.segment_board().
        ref_squares : dict[str, np.ndarray] | None
            Reference ROIs.  If None, extracted from self._ref automatically.

        Returns
        -------
        dict mapping square name to SquareAnalysis.
        """
        results: dict[str, SquareAnalysis] = {}
        for sq, roi in squares.items():
            ref_roi = (ref_squares or {}).get(sq)
            results[sq] = self.analyze_square(sq, roi, ref_roi=ref_roi)
        return results

    def analyze_cemetery_slot(
        self, index: int, roi: np.ndarray, side: str = "white"
    ) -> SquareAnalysis:
        """
        Analyse a single cemetery slot.

        Parameters
        ----------
        index : int
            Slot index (0-based).
        roi : np.ndarray
            Cemetery slot image crop.
        side : str
            "white" or "black" (helps set expected colour for confidence).
        """
        occupied, occ_confidence, mad, variance = self._detect_occupancy(roi, None)
        color: Optional[str] = None
        color_confidence = 1.0
        if occupied:
            color, color_confidence = self._detect_color(roi)
        confidence = occ_confidence * color_confidence if occupied else occ_confidence

        return SquareAnalysis(
            square=f"CEM_{side.upper()}_{index}",
            occupied=occupied,
            color=color,
            piece=None,
            confidence=confidence,
            mad=mad,
            variance=variance,
        )

    # ------------------------------------------------------------------
    # Occupancy detection
    # ------------------------------------------------------------------

    def _detect_occupancy(
        self,
        roi: np.ndarray,
        ref_roi: np.ndarray | None,
    ) -> tuple[bool, float, float, float]:
        """
        Returns (occupied, confidence, mad, variance).

        Strategy
        --------
        * If a reference ROI is available: compute Mean Absolute Difference.
          High MAD → occupied.
        * Fallback (no reference): use pixel variance of the ROI itself.
          High variance → occupied (a piece creates texture; empty board is
          relatively uniform within a single square).
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        variance = float(np.var(gray))

        if ref_roi is not None:
            ref_gray = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
            diff = np.abs(gray - ref_gray)
            mad = float(np.mean(diff)) / 255.0    # normalise to [0, 1]

            occupied = mad >= self._diff_thresh
            # Confidence: how far above/below the threshold we are
            if occupied:
                raw_conf = min(1.0, mad / (self._diff_thresh * 3))
            else:
                raw_conf = min(1.0, (self._diff_thresh - mad) / self._diff_thresh)
            confidence = 0.5 + raw_conf * 0.5    # map to [0.5, 1.0]
            return occupied, confidence, mad, variance

        else:
            # Variance-only fallback
            mad = 0.0
            occupied = variance >= self._var_thresh
            raw_conf = min(1.0, variance / (self._var_thresh * 3))
            if not occupied:
                raw_conf = min(1.0, (self._var_thresh - variance) / self._var_thresh)
            confidence = 0.4 + raw_conf * 0.4   # slightly lower max (less reliable)
            return occupied, confidence, mad, variance

    # ------------------------------------------------------------------
    # Colour detection
    # ------------------------------------------------------------------

    def _detect_color(self, roi: np.ndarray) -> tuple[str, float]:
        """
        Returns (color, confidence) where color is "white" or "black".

        Method
        ------
        Convert to HSV. Focus on the centre 60% of the ROI to avoid board
        edge bleed. Compute the mean V (brightness) channel.  Pieces brighter
        than PIECE_BRIGHTNESS_THRESHOLD are white; darker are black.
        """
        h, w = roi.shape[:2]
        # Centre crop (60% of each dimension)
        margin_y = int(h * 0.2)
        margin_x = int(w * 0.2)
        centre = roi[margin_y: h - margin_y, margin_x: w - margin_x]

        hsv = cv2.cvtColor(centre, cv2.COLOR_BGR2HSV)
        mean_v = float(np.mean(hsv[:, :, 2]))    # brightness channel

        if mean_v >= self._bright_thr:
            color = "white"
            confidence = min(1.0, (mean_v - self._bright_thr) / (255 - self._bright_thr))
        else:
            color = "black"
            confidence = min(1.0, (self._bright_thr - mean_v) / self._bright_thr)

        # Ensure at least 0.5 confidence even near the threshold
        confidence = 0.5 + confidence * 0.5
        return color, confidence
