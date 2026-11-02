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
from .piece_classifier import PieceClassifier
from .segmentation import BoardSegmenter

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
    is_piece:   bool = False
    is_hand:    bool = False

    def to_dict(self) -> dict:
        return {
            "square":     self.square,
            "occupied":   self.occupied,
            "color":      self.color,
            "piece":      self.piece,
            "confidence": round(self.confidence, 4),
            "is_piece":   self.is_piece,
            "is_hand":    self.is_hand,
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
        board_orientation: str = "normal",
    ) -> None:
        self._diff_thresh = occupied_diff_threshold
        self._var_thresh  = occupied_variance_threshold
        self._bright_thr  = brightness_threshold
        self._min_conf    = min_confidence
        self._orientation = board_orientation

        self._ref = None
        self._ref_squares: dict[str, np.ndarray] = {}
        if reference_image is not None:
            self.set_reference(reference_image, orientation=board_orientation)

        # Piece type classifier (template matching on pieces/*.png)
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
        Analyse a single square ROI:
        1. Checks raw occupancy (difference against reference empty board).
        2. If occupied, detects color and runs piece classification.
        3. Verifies if the object is an actual chess piece (has valid piece icon & confidence).
           If not a piece (hand, arm, shadow, table disturbance), it is rejected as empty.
        """
        if ref_roi is None and square in self._ref_squares:
            ref_roi = self._ref_squares[square]

        # ── Occupancy detection ───────────────────────────────────────
        occupied_raw, occ_confidence, mad, variance = self._detect_occupancy(
            roi, ref_roi
        )

        if not occupied_raw:
            return SquareAnalysis(
                square=square,
                occupied=False,
                color=None,
                piece=None,
                confidence=occ_confidence,
                mad=mad,
                variance=variance,
                is_piece=False,
                is_hand=False,
            )

        # ── Colour detection ──────────────────────────────────────────
        color, color_confidence = self._detect_color(roi)

        # ── Piece type & verification (is it an actual piece?) ────────
        piece: Optional[str] = None
        piece_confidence = 0.0
        if self._classifier is not None:
            piece, piece_confidence = self._classifier.classify(roi, color)

        is_piece = (piece is not None)
        confidence = piece_confidence if is_piece else (occ_confidence * color_confidence)

        return SquareAnalysis(
            square=square,
            occupied=occupied_raw,
            color=color,
            piece=piece,
            confidence=confidence,
            mad=mad,
            variance=variance,
            is_piece=is_piece,
            is_hand=False,
        )

    def analyze_all(
        self,
        squares: dict[str, np.ndarray],
        ref_squares: dict[str, np.ndarray] | None = None,
    ) -> dict[str, SquareAnalysis]:
        """
        Analyse all 64 board squares.
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
            is_piece=occupied,
            is_hand=False,
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
            if ref_gray.shape != gray.shape:
                ref_gray = cv2.resize(ref_gray, (gray.shape[1], gray.shape[0]))
            diff = np.abs(gray - ref_gray)
            h, w = diff.shape
            my, mx = int(h * 0.15), int(w * 0.15)
            center_diff = diff[my : h - my, mx : w - mx]

            full_mad = float(np.mean(diff)) / 255.0
            center_mad = float(np.mean(center_diff)) / 255.0
            mad = max(full_mad, center_mad)

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
        Detecta a cor da peca ("white" ou "black").
        As pecas brancas possuem verniz claro com tom frio/azulado (B - G >= 9.0).
        As pecas pretas possuem corpo de madeira neutro/escuro (B - G <= 8.2).
        Fallback para anel de brilho quando imagem em tons de cinza.
        """
        from .piece_classifier import find_puck_center
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        gray = cv2.resize(gray, (80, 80))
        cx, cy = find_puck_center(gray)

        if roi.ndim == 3:
            h, w = roi.shape[:2]
            scale_y, scale_x = h / 80.0, w / 80.0
            rx, ry = int(cx * scale_x), int(cy * scale_y)
            patch = roi[max(0, ry - 8): min(h, ry + 8), max(0, rx - 8): min(w, rx + 8)].astype(np.float32)
            bg_diff = float(patch[:, :, 0].mean() - patch[:, :, 1].mean())
            color = "white" if bg_diff >= 9.0 else "black"
            conf = min(1.0, 0.70 + abs(bg_diff - 9.0) / 20.0)
            return color, conf

        Y, X = np.ogrid[:80, :80]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        ring_val = float(gray[(dist >= 11) & (dist <= 15)].mean())
        split_thresh = getattr(cfg, "PIECE_PUCK_SPLIT_THRESHOLD", 112.0)
        color = "white" if ring_val < split_thresh else "black"
        conf = min(1.0, 0.6 + abs(split_thresh - ring_val) / 50.0)
        return color, conf
