"""
vision/state_comparator.py
==========================
Compares two BoardState snapshots to validate that a chess move was
physically executed correctly.

Given:
    before  : BoardState captured just before the gantry moves
    after   : BoardState captured just after the gantry stops
    move    : dict with keys "from" and "to" (e.g. {"from": "E2", "to": "E4"})

The comparator checks:
    1. The origin square (from) is now empty.
    2. The destination square (to) is now occupied.
    3. The piece colour at the destination matches the piece colour at the origin.
    4. If a capture occurred (to was occupied before), the captured piece
       colour appears in the corresponding cemetery.

Returns a CompareResult with success flag and detailed diagnostics.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .board_state import BoardState, SquareState

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CompareResult
# ---------------------------------------------------------------------------

@dataclass
class CompareResult:
    """
    Outcome of comparing two board states after a move.

    Attributes
    ----------
    success : bool
        True only if all expected changes are confirmed.
    origin_empty : bool
        True if the origin square is empty in *after*.
    dest_occupied : bool
        True if the destination square is occupied in *after*.
    color_match : bool
        True if the piece colour at the destination matches what was at origin.
    capture_confirmed : bool | None
        True if a captured piece appeared in the correct cemetery.
        None if no capture was expected.
    error : str | None
        Error code (see BOARD_STATE_UNCERTAIN, ORIGIN_NOT_EMPTY, etc.).
    message : str
        Human-readable explanation.
    details : dict
        Raw comparison metrics for debugging.
    attempts : int
        Number of capture attempts made before this result.
    """
    success:           bool
    origin_empty:      bool              = False
    dest_occupied:     bool              = False
    color_match:       bool              = False
    capture_confirmed: Optional[bool]    = None
    error:             Optional[str]     = None
    message:           str               = ""
    details:           dict              = field(default_factory=dict)
    attempts:          int               = 1

    def to_dict(self) -> dict:
        return {
            "success":           self.success,
            "origin_empty":      self.origin_empty,
            "dest_occupied":     self.dest_occupied,
            "color_match":       self.color_match,
            "capture_confirmed": self.capture_confirmed,
            "error":             self.error,
            "message":           self.message,
            "attempts":          self.attempts,
        }

    @classmethod
    def uncertain(cls, attempts: int = 1) -> "CompareResult":
        return cls(
            success=False,
            error="BOARD_STATE_UNCERTAIN",
            message="Não foi possível confirmar a posição das peças.",
            attempts=attempts,
        )

    @classmethod
    def low_confidence(cls, confidence: float, attempts: int = 1) -> "CompareResult":
        return cls(
            success=False,
            error="LOW_CONFIDENCE",
            message=f"Confiança da leitura abaixo do mínimo: {confidence:.2f}",
            attempts=attempts,
        )


# ---------------------------------------------------------------------------
# StateComparator
# ---------------------------------------------------------------------------

class StateComparator:
    """
    Validates that a move was physically executed by comparing before/after states.

    Parameters
    ----------
    min_confidence : float
        Minimum confidence required in the *after* state for validation.
    """

    def __init__(self, min_confidence: float = 0.70) -> None:
        self._min_conf = min_confidence

    def compare(
        self,
        before: BoardState,
        after: BoardState,
        move: dict,
        attempts: int = 1,
    ) -> CompareResult:
        """
        Compare *before* and *after* states for a given move.

        Parameters
        ----------
        before : BoardState
            Snapshot captured before the move.
        after : BoardState
            Snapshot captured after the move.
        move : dict
            {"from": "E2", "to": "E4"} — uses uppercase square names.
        attempts : int
            How many capture attempts were made (logged in result).

        Returns
        -------
        CompareResult
        """
        from_sq = move.get("from", "").upper()
        to_sq   = move.get("to",   "").upper()

        if not from_sq or not to_sq:
            return CompareResult(
                success=False,
                error="INVALID_MOVE_DATA",
                message=f"Dados de movimento inválidos: {move}",
                attempts=attempts,
            )

        # ── Confidence gate ───────────────────────────────────────────
        if not after.is_confident(self._min_conf):
            log.warning(
                "[Comparator] After-state confidence too low: %.2f",
                after.overall_confidence,
            )
            return CompareResult.low_confidence(after.overall_confidence, attempts)

        # ── Snapshot pieces ──────────────────────────────────────────
        origin_before   = before.board.get(from_sq)
        dest_before     = before.board.get(to_sq)
        origin_after    = after.board.get(from_sq)
        dest_after      = after.board.get(to_sq)

        moving_color    = origin_before.color if origin_before else None
        was_capture     = dest_before.occupied if dest_before else False
        captured_color  = dest_before.color if (dest_before and was_capture) else None

        # ── Check origin is empty ─────────────────────────────────────
        origin_empty = not (origin_after.occupied if origin_after else True)

        # ── Check destination is occupied ─────────────────────────────
        dest_occupied = dest_after.occupied if dest_after else False

        # ── Check colour match at destination ─────────────────────────
        color_match = (
            dest_after is not None
            and dest_after.occupied
            and dest_after.color == moving_color
        )

        # ── Capture confirmation ──────────────────────────────────────
        capture_confirmed: Optional[bool] = None
        if was_capture and captured_color:
            capture_confirmed = self._check_cemetery(after, captured_color)

        # ── Assemble result ───────────────────────────────────────────
        success = origin_empty and dest_occupied and color_match
        if was_capture:
            success = success and (capture_confirmed is True)

        if success:
            message = f"Movimento {from_sq}→{to_sq} confirmado visualmente."
            error   = None
        else:
            parts = []
            if not origin_empty:
                parts.append(f"origem {from_sq} não esvaziou")
            if not dest_occupied:
                parts.append(f"destino {to_sq} permanece vazio")
            if not color_match:
                parts.append("cor da peça não confere")
            if was_capture and capture_confirmed is False:
                parts.append("peça capturada não apareceu no cemitério")
            message = "Validação falhou: " + ", ".join(parts) + "."
            error   = "MOVE_NOT_CONFIRMED"

        details = {
            "from_sq":          from_sq,
            "to_sq":            to_sq,
            "moving_color":     moving_color,
            "was_capture":      was_capture,
            "captured_color":   captured_color,
            "after_confidence": after.overall_confidence,
        }

        result = CompareResult(
            success=success,
            origin_empty=origin_empty,
            dest_occupied=dest_occupied,
            color_match=color_match,
            capture_confirmed=capture_confirmed,
            error=error,
            message=message,
            details=details,
            attempts=attempts,
        )

        if success:
            log.info("[Comparator] %s", message)
        else:
            log.warning("[Comparator] %s", message)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_cemetery(after: BoardState, captured_color: str) -> bool:
        """
        Return True if at least one occupied slot in the captured piece's
        cemetery side matches the expected color.
        """
        expected_side = captured_color   # white piece → white cemetery
        slots = after.cemeteries.get(expected_side, [])
        return any(s.occupied and s.color == captured_color for s in slots)

    def diff_squares(
        self, before: BoardState, after: BoardState
    ) -> dict[str, tuple[SquareState, SquareState]]:
        """
        Return a dict of squares that changed between *before* and *after*.

        Keys are square names; values are (before_state, after_state).
        Useful for debugging unexpected board changes.
        """
        changed: dict[str, tuple[SquareState, SquareState]] = {}
        for sq in before.board:
            b = before.board[sq]
            a = after.board.get(sq)
            if a is None:
                continue
            if b.occupied != a.occupied or b.color != a.color:
                changed[sq] = (b, a)
        return changed
