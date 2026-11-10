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
    puck_score: float = 0.0
    occ_score:  float = 0.0

    def to_dict(self) -> dict:
        return {
            "square":     self.square,
            "occupied":   self.occupied,
            "color":      self.color,
            "piece":      self.piece,
            "confidence": round(self.confidence, 4),
            "center":     list(self.center),
            "occ_score":  round(self.occ_score, 4),
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
        expected_color: Optional[str] = None,
        expected_piece: Optional[str] = None,
        classify_type: bool = True,
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
        expected_color : str | None
            Optional prior colour ("white" or "black") from logical board.
        expected_piece : str | None
            Optional prior piece type ("pawn", "rook", ...) from logical board.
        classify_type : bool
            When False, skips expensive template matching during the initial pass
            because ``_apply_board_prior`` will assign known piece types directly
            from ``prior_board`` and only classify ambiguous/promoted squares.

        Returns
        -------
        SquareAnalysis
        """
        if ref_roi is None and square in self._ref_squares:
            ref_roi = self._ref_squares[square]

        # ── Occupancy & dynamic center detection ──────────────────────
        occupied, occ_confidence, mad, variance, center, puck_score, occ_score = (
            self._detect_occupancy(roi, ref_roi)
        )

        # ── Colour detection (centered on located puck) ───────────────
        color: Optional[str] = None
        color_confidence = 1.0
        if occupied:
            color, color_confidence = self._detect_color(roi, center=center)
            if (
                expected_color in ("white", "black")
                and color != expected_color
                and color_confidence < 0.53
            ):
                color = expected_color
                color_confidence = 0.55

        # ── Combined confidence ───────────────────────────────────────
        confidence = occ_confidence * color_confidence if occupied else occ_confidence

        # ── Piece type (centered template matching + prior bias) ──────
        piece: Optional[str] = None
        if classify_type:
            if occupied and self._classifier is not None:
                prior_p = expected_piece if (expected_color is None or color == expected_color) else None
                piece, _ = self._classifier.classify(
                    roi, color, center=center, expected_piece=prior_p
                )
            elif occupied and expected_piece and (expected_color is None or color == expected_color):
                piece = expected_piece

        return SquareAnalysis(
            square=square,
            occupied=occupied,
            color=color,
            piece=piece,
            confidence=confidence,
            mad=mad,
            variance=variance,
            center=center,
            puck_score=puck_score,
            occ_score=occ_score,
        )

    def analyze_all(
        self,
        squares: dict[str, np.ndarray],
        ref_squares: dict[str, np.ndarray] | None = None,
        prior_board: Optional[object] = None,
        max_pieces: int = 32,
        prior_mode: str = "auto",
        unbiased_squares: Optional[set[str] | list[str] | tuple[str, ...]] = None,
    ) -> dict[str, SquareAnalysis]:
        """
        Analyse all 64 board squares, using the logical ``prior_board`` (when
        available) and the finite piece-count limit (``max_pieces <= 32``) to
        suppress false positives on empty squares and guide piece classification.

        Parameters
        ----------
        squares : dict[str, np.ndarray]
            From BoardSegmenter.segment_board().
        ref_squares : dict[str, np.ndarray] | None
            Reference ROIs. If None, extracted from self._ref automatically.
        prior_board : chess.Board | None
            Logical board state used as a Bayesian prior for piece locations,
            piece count, colours, and types. If None and ``prior_mode == "auto"``,
            loads ``logical_board_state.json`` if available.
        max_pieces : int
            Hard upper bound on total occupied squares on the board (default 32).
        prior_mode : str
            "auto" (default), "pre_move" (before manual move), "post_move" (after Gantry),
            or "off" (disable prior board, apply only max_pieces cap).
        unbiased_squares : collection[str] | None
            Squares (e.g. Gantry move origin & destination) that must be evaluated
            without occupancy prior hysteresis.

        Returns
        -------
        dict mapping square name to SquareAnalysis.
        """
        if prior_board is None and prior_mode == "auto":
            try:
                import chess
                from .board_verifier import load_logical_board_snapshot
                snap = load_logical_board_snapshot()
                if snap and snap.get("fen"):
                    prior_board = chess.Board(snap["fen"])
            except Exception:
                prior_board = None

        use_fast_prior_pass = (
            prior_mode != "off"
            and prior_board is not None
            and hasattr(prior_board, "piece_at")
        )
        results: dict[str, SquareAnalysis] = {}
        ref_map = ref_squares if ref_squares is not None else self._ref_squares
        for sq, roi in squares.items():
            ref_roi = ref_map.get(sq)
            results[sq] = self.analyze_square(
                sq, roi, ref_roi=ref_roi, classify_type=not use_fast_prior_pass
            )

        if prior_mode != "off":
            self._apply_board_prior(
                results,
                squares,
                prior_board=prior_board,
                max_pieces=max_pieces,
                prior_mode=prior_mode,
                unbiased_squares=unbiased_squares,
            )
        return results

    def _apply_board_prior(
        self,
        results: dict[str, SquareAnalysis],
        squares: dict[str, np.ndarray],
        prior_board: Optional[object] = None,
        max_pieces: int = 32,
        prior_mode: str = "auto",
        unbiased_squares: Optional[set[str] | list[str] | tuple[str, ...]] = None,
    ) -> None:
        """
        In-place refinement of ``results`` using:
        1. Expected occupancy hysteresis & empty-square noise suppression.
        2. Single-turn empty-square entry constraint (at most 1 new square, or 2 in castling).
        3. Global piece-count cap (``Top-N`` by occupancy score, where ``N <= 32``).
        4. Prior-guided piece type & colour propagation from ``prior_board``.
        """
        import chess

        type_names = {
            chess.PAWN: "pawn",
            chess.KNIGHT: "knight",
            chess.BISHOP: "bishop",
            chess.ROOK: "rook",
            chess.QUEEN: "queen",
            chess.KING: "king",
        }
        unbiased = {s.upper() for s in (unbiased_squares or ())}

        prior_map: dict[str, tuple[str, str] | None] = {}
        if prior_board is not None and hasattr(prior_board, "piece_at"):
            for sq_idx in chess.SQUARES:
                sq_name = chess.square_name(sq_idx).upper()
                p = prior_board.piece_at(sq_idx)
                if p is not None:
                    c_str = "white" if p.color == chess.WHITE else "black"
                    t_str = type_names.get(p.piece_type, "pawn")
                    prior_map[sq_name] = (c_str, t_str)
                else:
                    prior_map[sq_name] = None

        has_prior = bool(prior_map) and any(v is not None for v in prior_map.values())
        n_expected = (
            sum(1 for v in prior_map.values() if v is not None)
            if has_prior
            else max_pieces
        )
        cap_n = max(1, min(int(max_pieces), int(n_expected)))

        def _clear_sq(a: SquareAnalysis) -> None:
            a.occupied = False
            a.color = None
            a.piece = None
            a.confidence = max(0.65, min(0.98, 1.0 - 0.25 * a.occ_score))

        if has_prior:
            # 1. Measure baseline occupancy score of real stationary pieces on the board
            stationary_scores = [
                a.occ_score
                for sq, a in results.items()
                if prior_map.get(sq) is not None and a.occ_score >= 0.85
            ]
            median_occ = (
                float(np.median(stationary_scores))
                if len(stationary_scores) >= 4
                else 1.45
            )

            # Check if the expected destination piece in post_move is missing (piece dropped mid-way)
            dest_missing = (
                prior_mode == "post_move"
                and any(
                    prior_map.get(u) is not None and not results[u].occupied
                    for u in unbiased
                    if u in results
                )
            )

            # 2. Apply per-square prior hysteresis & stricter empty-square gate
            for sq, a in results.items():
                if sq in unbiased:
                    continue
                expected = prior_map.get(sq)
                if expected is not None:
                    # Square had a piece: protect against borderline dim lighting
                    if (
                        not a.occupied
                        and a.occ_score >= 0.82
                        and a.puck_score >= self._struct_thresh * 0.45
                    ):
                        a.occupied = True
                        exp_color, _ = expected
                        roi = squares.get(sq)
                        if roi is not None:
                            col, col_conf = self._detect_color(roi, center=a.center)
                            a.color = exp_color if col_conf < 0.53 else col
                        else:
                            a.color = exp_color
                        a.confidence = max(a.confidence, 0.74)
                else:
                    # Square was empty in prior_board: reject weak reflections/shadows
                    if a.occupied:
                        if prior_mode == "post_move" and not dest_missing:
                            min_req = max(1.40, min(1.85, 0.60 * median_occ))
                            min_puck = self._struct_thresh * 0.85
                        else:
                            min_req = max(1.10, min(1.45, 0.46 * median_occ))
                            min_puck = self._struct_thresh * 0.62
                        if a.occ_score < min_req or a.puck_score < min_puck:
                            _clear_sq(a)

            # 3. In a single move from prior_board, at most 1 previously-empty square
            #    becomes occupied (or 2 on rank 1/8 during castling). Only suppress a 2nd
            #    empty square if it is significantly weaker than the primary destination
            #    AND at most 1 origin square was vacated (so moving 2 real pieces is not hidden).
            if prior_mode in ("pre_move", "auto") or dest_missing:
                n_vacated = sum(
                    1 for sq, exp in prior_map.items()
                    if exp is not None and not results[sq].occupied
                )
                new_occupied = [
                    a for sq, a in results.items()
                    if prior_map.get(sq) is None and a.occupied and sq not in unbiased
                ]
                if len(new_occupied) > 1 and (n_vacated <= 1 or dest_missing):
                    new_occupied.sort(key=lambda x: x.occ_score, reverse=True)
                    best_new = new_occupied[0]
                    for extra in new_occupied[1:]:
                        if extra.occ_score < 0.75 * best_new.occ_score:
                            _clear_sq(extra)

        # 4. Global Piece-Count Limit (Top-N by prior-weighted occupancy score)
        occupied_list = [a for a in results.values() if a.occupied]
        if len(occupied_list) > cap_n:
            occupied_list.sort(
                key=lambda a: a.occ_score + (0.18 if prior_map.get(a.square) is not None else 0.0),
                reverse=True,
            )
            for extra in occupied_list[cap_n:]:
                _clear_sq(extra)

        # 5. Classify occupied squares visually (using prior only as a light tie-breaker)
        moved_into: dict[str, tuple[str, str]] = {}
        if has_prior:
            vacated = [
                sq for sq, exp in prior_map.items()
                if exp is not None and not results[sq].occupied
            ]
            entered = [
                sq for sq, exp in prior_map.items()
                if exp is None and results[sq].occupied
            ]

            if len(vacated) == 1 and len(entered) == 1:
                orig_info = prior_map.get(vacated[0])
                if orig_info is not None:
                    moved_into[entered[0]] = orig_info
            elif len(vacated) == 1 and len(entered) == 0:
                # Standard capture: 1 origin vacated, 1 opponent square changed colour
                orig_info = prior_map.get(vacated[0])
                if orig_info is not None:
                    captured_sqs = [
                        sq for sq, exp in prior_map.items()
                        if exp is not None
                        and results[sq].occupied
                        and exp[0] != orig_info[0]
                        and results[sq].color == orig_info[0]
                    ]
                    if len(captured_sqs) == 1:
                        moved_into[captured_sqs[0]] = orig_info
            elif len(vacated) == 2 and len(entered) == 1:
                # En passant capture: 1 own pawn + 1 opponent pawn vacated, 1 empty square entered
                dst_sq = entered[0]
                dst_col = results[dst_sq].color
                same_col_src = [
                    prior_map[s] for s in vacated
                    if prior_map.get(s) is not None and prior_map[s][0] == dst_col
                ]
                if len(same_col_src) == 1 and same_col_src[0] is not None:
                    moved_into[dst_sq] = same_col_src[0]
            elif len(vacated) == 2 and len(entered) == 2:
                for dst in entered:
                    for src in vacated:
                        s_info = prior_map.get(src)
                        if s_info is None:
                            continue
                        if dst[0] in ("C", "G") and s_info[1] == "king":
                            moved_into[dst] = s_info
                        elif dst[0] in ("D", "F") and s_info[1] == "rook":
                            moved_into[dst] = s_info

        for sq, a in results.items():
            if not a.occupied:
                continue
            roi = squares.get(sq)
            exp = prior_map.get(sq)

            # Determine weak tie-breaker hint ONLY if detected colour matches expected colour
            prior_piece_hint: Optional[str] = None
            if sq in moved_into:
                mv_color, mv_piece = moved_into[sq]
                is_promo = (mv_piece == "pawn" and sq[1] in ("1", "8"))
                if not is_promo and a.color == mv_color:
                    prior_piece_hint = mv_piece
            elif exp is not None and a.color == exp[0]:
                prior_piece_hint = exp[1]

            # Always run visual classification on occupied squares so swapped pieces are detected
            if a.piece is None and self._classifier is not None and roi is not None:
                a.piece, _ = self._classifier.classify(
                    roi, a.color, center=a.center, expected_piece=prior_piece_hint
                )

            # Fallback only when visual classifier returned None
            if a.piece is None and prior_piece_hint is not None:
                a.piece = prior_piece_hint

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
        occupied, occ_confidence, mad, variance, center, puck_score, occ_score = (
            self._detect_occupancy(roi, None)
        )
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
            puck_score=puck_score,
            occ_score=occ_score,
        )

    # ------------------------------------------------------------------
    # Occupancy detection
    # ------------------------------------------------------------------

    def _detect_occupancy(
        self,
        roi: np.ndarray,
        ref_roi: np.ndarray | None,
    ) -> tuple[bool, float, float, float, tuple[int, int], float, float]:
        """
        Returns ``(occupied, confidence, mad, variance, center, puck_score, occ_score)``.

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

            # Structural gate prevents sharp shadows on empty squares from triggering occupancy,
            # while strong 3D puck structure confirms occupancy even when lighting changes
            # make a piece's brightness match the old reference (e.g. moving the board to a new room).
            has_structure = (
                puck_score >= self._struct_thresh * 0.55
                or (icon_std >= 11.5 and puck_score >= self._struct_thresh * 0.30)
                or (mad >= self._diff_thresh * 2.10 and puck_score >= self._struct_thresh * 0.35)
            )
            strong_puck = puck_score >= self._struct_thresh * 2.5
            occupied = ((mad >= self._diff_thresh) and has_structure) or strong_puck

            mad_ratio = mad / max(0.01, self._diff_thresh)
            struct_ratio = puck_score / max(1.0, self._struct_thresh)
            if has_structure or strong_puck:
                occ_score = max(mad_ratio, struct_ratio / 2.2)
            else:
                occ_score = min(0.75, mad_ratio * 0.45)

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
            return occupied, float(confidence), mad, variance, center, float(puck_score), float(occ_score)

        else:
            mad = 0.0
            occ_score = puck_score / max(1.0, self._struct_thresh)
            occupied = puck_score >= self._struct_thresh
            if occupied:
                raw_conf = min(1.0, puck_score / (self._struct_thresh * 2.5))
            else:
                raw_conf = min(1.0, (self._struct_thresh - puck_score) / max(1.0, self._struct_thresh))
            confidence = 0.5 + raw_conf * 0.45
            return occupied, float(confidence), mad, variance, center, float(puck_score), float(occ_score)

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
