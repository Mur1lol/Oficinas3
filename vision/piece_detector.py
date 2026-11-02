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

    def _verify_piece(
        self,
        roi: np.ndarray,
        ref_roi: np.ndarray | None,
        occupied_raw: bool,
        mad: float,
        variance: float,
        color: Optional[str],
        piece: Optional[str],
    ) -> tuple[bool, bool, float, float]:
        """
        Verifica se o objeto no ROI e uma peca de xadrez fisica real ou uma mao/obstrucao/ruido.

        Retorna:
            (is_piece, is_hand, confidence, radial_edge)
        """
        if not occupied_raw:
            return False, False, max(0.5, 1.0 - mad), 0.0

        if not getattr(cfg, "PIECE_VERIFICATION_ENABLED", True):
            return True, False, 0.85, 50.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        gray = cv2.resize(gray, (80, 80))

        # 1. Medicao de centro vs borda (para detectar se o objeto transborda a casa)
        center_mad = 0.0
        border_mad = 0.0
        ratio = 1.0
        if ref_roi is not None:
            ref_gray = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY) if ref_roi.ndim == 3 else ref_roi
            ref_gray = cv2.resize(ref_gray, (80, 80))
            diff = cv2.absdiff(gray, ref_gray)
            Y, X = np.ogrid[:80, :80]
            dist_c = np.sqrt((X - 40) ** 2 + (Y - 40) ** 2)
            border_mask = np.ones((80, 80), dtype=bool)
            border_mask[8:72, 8:72] = False
            center_mad = float(np.mean(diff[dist_c <= 20])) / 255.0
            border_mad = float(np.mean(diff[border_mask])) / 255.0
            ratio = center_mad / max(0.001, border_mad)

        # 2. Contraste do icone central vs anel do disco
        h, w = gray.shape
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - 40) ** 2 + (Y - 40) ** 2)
        ring_val = float(gray[(dist >= 12) & (dist <= 16)].mean())
        split_thresh = getattr(cfg, "PIECE_PUCK_SPLIT_THRESHOLD", 112.0)
        det_color = color if color in ("white", "black") else ("white" if ring_val < split_thresh else "black")

        if det_color == "black":
            norm = ring_val - gray.astype(np.float32)
        else:
            norm = gray.astype(np.float32) - ring_val
        norm = np.maximum(0.0, norm) * (dist <= 12.0)
        max_contrast = float(norm.max())
        min_contrast = getattr(cfg, "PIECE_MIN_ICON_CONTRAST", 35.0)

        # 3. Gradiente radial circular (disco da peca)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        thetas = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        best_radial_edge = 0.0
        for r in range(13, 18):
            xs = np.clip(np.round(40 + r * cos_t).astype(int), 0, 79)
            ys = np.clip(np.round(40 + r * sin_t).astype(int), 0, 79)
            g_rad = np.abs(gx[ys, xs] * cos_t + gy[ys, xs] * sin_t)
            best_radial_edge = max(best_radial_edge, float(np.mean(g_rad)))

        # ── Decisoes de classificacao da estrutura ──
        # Regra A: Peca confirmada por template match e centralizada / bom contraste
        if piece is not None and (best_radial_edge >= 35.0 or ratio >= 0.95 or max_contrast >= 60.0):
            return True, False, 0.90, best_radial_edge

        # Regra B: Formato de disco e contraste forte de icone confirmados
        if max_contrast >= 60.0 and best_radial_edge >= 35.0 and (ref_roi is None or ratio >= 0.90):
            return True, False, 0.75, best_radial_edge

        # Regra C: Transbordo de bordas (mao/braco atravessando divisorias das casas)
        if ref_roi is not None and border_mad >= 0.18 and ratio < 0.98 and best_radial_edge < 35.0:
            return False, True, 0.90, best_radial_edge

        # Regra D: Diferenca de brilho sem icone gravado (pele/madeira/sombra)
        if max_contrast < min_contrast and mad >= 0.14:
            return False, True, 0.85, best_radial_edge

        # Regra E: Obstrucao generica sem evidencia de disco
        if mad >= 0.14:
            return False, True, 0.70, best_radial_edge

        return False, False, 0.60, best_radial_edge

    def analyze_square(
        self,
        square: str,
        roi: np.ndarray,
        ref_roi: np.ndarray | None = None,
    ) -> SquareAnalysis:
        """
        Analyse a single square ROI with piece-vs-hand discrimination.
        """
        if ref_roi is None and square in self._ref_squares:
            ref_roi = self._ref_squares[square]

        # ── Occupancy detection ───────────────────────────────────────
        occupied_raw, occ_confidence, mad, variance = self._detect_occupancy(
            roi, ref_roi
        )

        # ── Colour detection ──────────────────────────────────────────
        color: Optional[str] = None
        color_confidence = 1.0
        if occupied_raw:
            color, color_confidence = self._detect_color(roi)

        # ── Piece type (template matching) ────────────────────────────
        piece: Optional[str] = None
        piece_confidence = 0.0
        if occupied_raw and self._classifier is not None:
            piece, piece_confidence = self._classifier.classify(roi, color)

        # ── Piece structure verification (piece vs hand/obstruction) ──
        is_piece, is_hand, verif_conf, _ = self._verify_piece(
            roi, ref_roi, occupied_raw, mad, variance, color, piece
        )

        occupied = is_piece
        if not is_piece:
            color = None
            piece = None

        confidence = piece_confidence if (is_piece and piece is not None) else verif_conf

        return SquareAnalysis(
            square=square,
            occupied=occupied,
            color=color,
            piece=piece,
            confidence=confidence,
            mad=mad,
            variance=variance,
            is_piece=is_piece,
            is_hand=is_hand,
        )

    @staticmethod
    def _get_neighbors(square: str) -> list[str]:
        cols = "ABCDEFGH"
        rows = "87654321"
        if len(square) != 2 or square[0] not in cols or square[1] not in rows:
            return []
        c = cols.index(square[0])
        r = rows.index(square[1])
        nbrs = []
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if dc == 0 and dr == 0:
                    continue
                nc, nr = c + dc, r + dr
                if 0 <= nc < 8 and 0 <= nr < 8:
                    nbrs.append(f"{cols[nc]}{rows[nr]}")
        return nbrs

    def analyze_all(
        self,
        squares: dict[str, np.ndarray],
        ref_squares: dict[str, np.ndarray] | None = None,
    ) -> dict[str, SquareAnalysis]:
        """
        Analyse all 64 board squares with multi-square obstruction clustering.
        """
        results: dict[str, SquareAnalysis] = {}
        ref_map = ref_squares if ref_squares is not None else self._ref_squares
        for sq, roi in squares.items():
            ref_roi = ref_map.get(sq)
            results[sq] = self.analyze_square(sq, roi, ref_roi=ref_roi)

        # ── Pass 2: Connected component cluster check (obstruction filtering) ──
        initial_hands = {sq for sq, a in results.items() if a.is_hand}
        if len(initial_hands) >= 2:
            for sq, a in results.items():
                if a.is_piece:
                    nbrs = self._get_neighbors(sq)
                    adj_hands = [n for n in nbrs if n in initial_hands]
                    if len(adj_hands) >= 2:
                        ref_roi = ref_map.get(sq)
                        if ref_roi is not None:
                            roi = squares[sq]
                            diff = cv2.absdiff(
                                cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
                                cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY),
                            )
                            border_mask = np.ones((80, 80), dtype=bool)
                            border_mask[8:72, 8:72] = False
                            Y, X = np.ogrid[:80, :80]
                            dist_c = np.sqrt((X - 40) ** 2 + (Y - 40) ** 2)
                            c_mad = float(np.mean(diff[dist_c <= 20]))
                            b_mad = float(np.mean(diff[border_mask]))
                            sq_ratio = c_mad / max(0.001, b_mad)
                            # Se estiver cercado por maos e nao tiver confianca alta nem for claramente centrado:
                            if a.confidence < 0.60 or sq_ratio < 1.10:
                                a.is_piece = False
                                a.occupied = False
                                a.is_hand = True
                                a.color = None
                                a.piece = None

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
        Returns (color, confidence) where color is "white" or "black".

        Method
        ------
        For circular puck pieces:
        Evaluates the puck body ring (radius 12 to 16 px) and center icon (radius <= 8 px).
        In the physical set:
        - White player: Dark body puck (mean ~85) with bright icon (contrast > 0).
        - Black player: Light body puck (mean ~138) with dark icon (contrast < 0).
        Threshold at PIECE_PUCK_SPLIT_THRESHOLD (112) cleanly separates white from
        black pieces without being distorted by the underlying square color.
        """
        if roi.ndim == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        gray = cv2.resize(gray, (80, 80))
        h, w = gray.shape
        cx, cy = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

        ring_val = float(gray[(dist >= 12) & (dist <= 16)].mean())
        icon_val = float(gray[dist <= 8].mean())
        contrast = icon_val - ring_val

        split_thresh = getattr(cfg, "PIECE_PUCK_SPLIT_THRESHOLD", 112.0)
        if ring_val < split_thresh:
            color = "white"
            conf = min(1.0, 0.6 + abs(split_thresh - ring_val) / 50.0)
            if contrast > 0:
                conf = min(1.0, conf + 0.1)
        else:
            color = "black"
            conf = min(1.0, 0.6 + abs(ring_val - split_thresh) / 50.0)
            if contrast < 0:
                conf = min(1.0, conf + 0.1)

        return color, float(conf)
