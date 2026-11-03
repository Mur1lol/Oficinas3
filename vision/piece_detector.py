"""
vision/piece_detector.py
========================
Detects whether a board square is occupied and identifies the piece colour
and type.

Detection pipeline (per square)
--------------------------------
1. **Dynamic Center Localization** — Locate the physical circular puck center
   ``(cx, cy)`` inside the square ROI using 4-zone concentric gradient and
   variance filters (``locate_puck_center``), immune to cast shadows, off-center
   placement, and printed target dots on empty squares.

2. **Occupancy** — Compare the square ROI to its corresponding region in the
   reference (empty board) image after local outer-border illumination
   alignment (cancelling global brightness changes and shadows) inside the
   located puck disk, reinforced by the concentric structural puck score.
   Without a reference image, uses the 4-quadrant structural puck score.

3. **Colour** — For occupied squares, analyse the relative concentric contrast
   around the located center ``(cx, cy)`` (icon percentiles vs. puck body ring
   vs. outer dark rim) via ``detect_puck_color``.

4. **Type** — Classified by ``PieceClassifier`` centered on ``(cx, cy)``.

All thresholds are defined in vision/config.py.

Usage
-----
    detector = PieceDetector(reference_image)
    analysis = detector.analyze_square("E2", roi)
    print(analysis.occupied, analysis.color, analysis.piece, analysis.confidence)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config as cfg
from .piece_classifier import (
    PieceClassifier,
    detect_puck_color,
    locate_puck_center,
)
from .segmentation import BoardSegmenter

log = logging.getLogger(__name__)

_YG80, _XG80 = np.ogrid[:80, :80]


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
        Piece type ("pawn", "rook", "knight", "bishop", "queen", "king") or None.
    confidence : float
        Combined detection confidence [0.0–1.0].
    mad : float
        Illumination-normalized mean absolute difference vs. reference.
    variance : float
        Pixel variance of the ROI (diagnostic metric).
    center : tuple[int, int]
        Located puck center (cx, cy) in 80x80 ROI coordinates.
    """
    square:     str
    occupied:   bool
    color:      Optional[str]
    piece:      Optional[str]
    confidence: float
    mad:        float = 0.0
    variance:   float = 0.0
    center:     tuple[int, int] = (40, 40)

    def to_dict(self) -> dict:
        return {
            "square":     self.square,
            "occupied":   self.occupied,
            "color":      self.color,
            "piece":      self.piece,
            "confidence": round(self.confidence, 4),
            "center":     list(self.center),
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
        If None, occupancy detection uses structural puck detection mode.
    occupied_diff_threshold : float
        Minimum illumination-normalized MAD [0–1] to consider a square occupied.
    occupied_variance_threshold : float
        Minimum pixel variance threshold (legacy fallback parameter).
    brightness_threshold : int
        HSV V-channel threshold (legacy parameter).
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
        board_orientation: str = "normal",
    ) -> None:
        self._diff_thresh = occupied_diff_threshold
        self._var_thresh  = occupied_variance_threshold
        self._bright_thr  = brightness_threshold
        self._min_conf    = min_confidence
        self._orientation = board_orientation
        self._struct_thresh = float(getattr(cfg, "PUCK_STRUCTURAL_SCORE_THRESHOLD", 5500.0))

        self._ref = None
        self._ref_squares: dict[str, np.ndarray] = {}
        if reference_image is not None:
            self.set_reference(reference_image, orientation=board_orientation)

        # Piece type classifier (template matching on pieces/*.png or real samples)
        self._classifier: PieceClassifier | None = None
        if cfg.PIECE_TYPE_ENABLED:
            clf = PieceClassifier()
            self._classifier = clf if clf.is_ready else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_reference(
        self, reference_image: np.ndarray, orientation: str | None = None
    ) -> None:
        """Update the reference (empty board) image and pre-slice all 64 square ROIs."""
        self._ref = reference_image
        if orientation is not None:
            self._orientation = orientation
        try:
            segmenter = BoardSegmenter(board_orientation=self._orientation)
            self._ref_squares = segmenter.segment_board(reference_image)
            log.info("[Detector] Reference image updated (%d squares).", len(self._ref_squares))
        except Exception as e:
            self._ref_squares = {}
            log.warning("[Detector] Could not auto-segment reference image: %s", e)

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
            Reference (empty) crop of the same square. If None and a
            reference image is stored, uses it automatically.

        Returns
        -------
        SquareAnalysis
        """
        if ref_roi is None and square in self._ref_squares:
            ref_roi = self._ref_squares[square]

        # ── Occupancy & dynamic center detection ──────────────────────
        occupied, occ_confidence, mad, variance, center = self._detect_occupancy(
            roi, ref_roi
        )

        # ── Colour detection (centered on located puck) ───────────────
        color: Optional[str] = None
        color_confidence = 1.0
        if occupied:
            color, color_confidence = self._detect_color(roi, center=center)

        # ── Combined confidence ───────────────────────────────────────
        confidence = occ_confidence * color_confidence if occupied else occ_confidence

        # ── Piece type (centered template matching) ───────────────────
        piece: Optional[str] = None
        if occupied and self._classifier is not None:
            piece, _ = self._classifier.classify(roi, color, center=center)

        return SquareAnalysis(
            square=square,
            occupied=occupied,
            color=color,
            piece=piece,
            confidence=confidence,
            mad=mad,
            variance=variance,
            center=center,
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
            Reference ROIs. If None, extracted from self._ref automatically.

        Returns
        -------
        dict mapping square name to SquareAnalysis.
        """
        results: dict[str, SquareAnalysis] = {}
        ref_map = ref_squares if ref_squares is not None else self._ref_squares
        for sq, roi in squares.items():
            ref_roi = ref_map.get(sq)
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
        occupied, occ_confidence, mad, variance, center = self._detect_occupancy(roi, None)
        color: Optional[str] = None
        color_confidence = 1.0
        if occupied:
            color, color_confidence = self._detect_color(roi, center=center)
        confidence = occ_confidence * color_confidence if occupied else occ_confidence

        return SquareAnalysis(
            square=f"CEM_{side.upper()}_{index}",
            occupied=occupied,
            color=color,
            piece=None,
            confidence=confidence,
            mad=mad,
            variance=variance,
            center=center,
        )

    # ------------------------------------------------------------------
    # Occupancy detection
    # ------------------------------------------------------------------

    def _detect_occupancy(
        self,
        roi: np.ndarray,
        ref_roi: np.ndarray | None,
    ) -> tuple[bool, float, float, float, tuple[int, int]]:
        """
        Returns ``(occupied, confidence, mad, variance, center)``.

        Strategy
        --------
        1. Locate candidate puck center ``(cx, cy)`` and compute its 4-quadrant
           concentric ``puck_score`` and ``icon_std``.
        2. If ``ref_roi`` is available:
           * Align current ROI illumination to ``ref_roi`` using the exposed
             outer square border (r >= 22 px around ``(cx, cy)``), cancelling
             global brightness changes and cast shadows.
           * Compute illumination-normalized MAD inside the puck disk (r <= 16.5 px).
           * Require ``mad >= diff_thresh`` (with structural confirmation to
             reject strong non-uniform shadow edges on empty squares).
        3. Fallback (no ``ref_roi``):
           * Use ``puck_score >= PUCK_STRUCTURAL_SCORE_THRESHOLD``, which cleanly
             separates real 3D pieces with icons from empty squares (even when
             empty squares have printed target dots).
        """
        gray_raw = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        if gray_raw.shape[0] != 80 or gray_raw.shape[1] != 80:
            gray = cv2.resize(gray_raw, (80, 80), interpolation=cv2.INTER_AREA).astype(np.float32)
        else:
            gray = gray_raw.astype(np.float32)

        variance = float(np.var(gray))
        cx, cy, puck_score, icon_std = locate_puck_center(gray)
        center = (cx, cy)

        if ref_roi is not None:
            ref_raw = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY) if ref_roi.ndim == 3 else ref_roi
            if ref_raw.shape[0] != 80 or ref_raw.shape[1] != 80:
                ref_gray = cv2.resize(ref_raw, (80, 80), interpolation=cv2.INTER_AREA).astype(np.float32)
            else:
                ref_gray = ref_raw.astype(np.float32)

            d = np.sqrt((_XG80 - cx) ** 2 + (_YG80 - cy) ** 2)
            # Outer square border outside the puck (r >= 22) and inside square edges (4..76)
            border = (d >= 22.0) & (_XG80 >= 4) & (_XG80 < 76) & (_YG80 >= 4) & (_YG80 < 76)
            if int(border.sum()) > 20:
                med_cur = float(np.median(gray[border]))
                med_ref = float(np.median(ref_gray[border]))
                gain = float(np.clip(med_cur / max(20.0, med_ref), 0.4, 2.5))
                aligned_mult = gray / gain
                aligned_add  = gray - (med_cur - med_ref)
                aligned = 0.5 * aligned_mult + 0.5 * aligned_add
            else:
                aligned = gray

            disk = d <= 16.5
            mad = float(np.mean(np.abs(aligned[disk] - ref_gray[disk]))) / 255.0

            # Structural gate prevents sharp shadows on empty squares from triggering occupancy
            has_structure = (
                puck_score >= self._struct_thresh * 0.55
                or icon_std >= 12.0
                or mad >= self._diff_thresh * 1.85
            )
            occupied = (mad >= self._diff_thresh) and has_structure

            if occupied:
                mad_conf = min(1.0, mad / (self._diff_thresh * 2.5))
                struct_conf = min(1.0, puck_score / (self._struct_thresh * 2.0))
                raw_conf = max(mad_conf, struct_conf)
            else:
                if mad < self._diff_thresh:
                    raw_conf = min(1.0, (self._diff_thresh - mad) / max(0.01, self._diff_thresh))
                else:
                    raw_conf = min(1.0, (self._struct_thresh - puck_score) / self._struct_thresh)
            confidence = 0.5 + raw_conf * 0.5
            return occupied, float(confidence), mad, variance, center

        else:
            mad = 0.0
            occupied = puck_score >= self._struct_thresh
            if occupied:
                raw_conf = min(1.0, puck_score / (self._struct_thresh * 2.5))
            else:
                raw_conf = min(1.0, (self._struct_thresh - puck_score) / max(1.0, self._struct_thresh))
            confidence = 0.5 + raw_conf * 0.45
            return occupied, float(confidence), mad, variance, center

    # ------------------------------------------------------------------
    # Colour detection
    # ------------------------------------------------------------------

    def _detect_color(
        self,
        roi: np.ndarray,
        center: Optional[tuple[int, int]] = None,
    ) -> tuple[str, float]:
        """
        Returns ``(color, confidence)`` where color is ``"white"`` or ``"black"``.

        Delegates to ``detect_puck_color`` around the dynamically located puck
        center ``(cx, cy)``, using illumination-invariant relative contrast
        between the central icon, the puck body ring, and the outer dark rim.
        """
        split_thresh = float(getattr(cfg, "PIECE_PUCK_SPLIT_THRESHOLD", 112.0))
        return detect_puck_color(roi, center=center, split_thresh=split_thresh)
