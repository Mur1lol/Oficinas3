"""
vision/board_state.py
=====================
Data structures representing the complete physical state of the chess board.

BoardState captures a snapshot of all 64 squares and both cemetery regions
at a given point in time.  It can be serialised to/from dict (JSON-compatible).

Piece type classification (pawn, rook, knight, …) is reserved for a future
Opção-B update using a lightweight CNN trained on board photos.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# SquareState
# ---------------------------------------------------------------------------

@dataclass
class SquareState:
    """
    Detected state of one board square or cemetery slot.

    Attributes
    ----------
    occupied : bool
        True if a piece was detected in this region.
    color : str | None
        "white" or "black" if occupied and color was detected; else None.
    piece : str | None
        Piece type (future: "pawn", "rook", …); always None in current build.
    confidence : float
        Detection confidence [0.0–1.0]. Below MIN_CONFIDENCE the result
        should be treated as uncertain.
    """
    occupied:   bool
    color:      Optional[str]  = None   # "white" | "black" | None
    piece:      Optional[str]  = None   # future: "pawn" | "rook" | ...
    confidence: float          = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def empty(cls, confidence: float = 1.0) -> "SquareState":
        return cls(occupied=False, color=None, piece=None, confidence=confidence)

    @classmethod
    def occupied_white(cls, confidence: float = 1.0) -> "SquareState":
        return cls(occupied=True, color="white", piece=None, confidence=confidence)

    @classmethod
    def occupied_black(cls, confidence: float = 1.0) -> "SquareState":
        return cls(occupied=True, color="black", piece=None, confidence=confidence)


# ---------------------------------------------------------------------------
# BoardState
# ---------------------------------------------------------------------------

@dataclass
class BoardState:
    """
    Snapshot of the entire board at one point in time.

    Attributes
    ----------
    board : dict[str, SquareState]
        Maps square name ("A1"…"H8") to its detected state.
    cemeteries : dict[str, list[SquareState]]
        Maps "white" and "black" to a list of cemetery slot states.
    timestamp : float
        Unix timestamp (time.monotonic()-based for relative comparisons).
    overall_confidence : float
        Mean confidence across all squares.
    """
    board:              dict[str, SquareState]        = field(default_factory=dict)
    cemeteries:         dict[str, list[SquareState]]  = field(default_factory=dict)
    timestamp:          float                         = field(default_factory=time.monotonic)
    overall_confidence: float                         = 0.0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dict."""
        return {
            "board": {sq: state.to_dict() for sq, state in self.board.items()},
            "cemeteries": {
                side: [s.to_dict() for s in slots]
                for side, slots in self.cemeteries.items()
            },
            "timestamp": self.timestamp,
            "overall_confidence": self.overall_confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BoardState":
        board = {sq: SquareState(**s) for sq, s in d["board"].items()}
        cemeteries = {
            side: [SquareState(**s) for s in slots]
            for side, slots in d["cemeteries"].items()
        }
        return cls(
            board=board,
            cemeteries=cemeteries,
            timestamp=d.get("timestamp", time.monotonic()),
            overall_confidence=d.get("overall_confidence", 0.0),
        )

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def is_occupied(self, square: str) -> bool:
        state = self.board.get(square.upper())
        return state.occupied if state else False

    def get_color(self, square: str) -> str | None:
        state = self.board.get(square.upper())
        return state.color if state else None

    def occupied_squares(self) -> list[str]:
        return [sq for sq, s in self.board.items() if s.occupied]

    def empty_squares(self) -> list[str]:
        return [sq for sq, s in self.board.items() if not s.occupied]

    def is_confident(self, threshold: float = 0.70) -> bool:
        return self.overall_confidence >= threshold

    @classmethod
    def empty_board(cls) -> "BoardState":
        """Return a BoardState with all squares empty (useful for testing)."""
        squares = "ABCDEFGH"
        rows    = "12345678"
        board = {
            f"{c}{r}": SquareState.empty()
            for c in squares for r in rows
        }
        cemeteries = {"white": [], "black": []}
        return cls(board=board, cemeteries=cemeteries, overall_confidence=1.0)

    def __repr__(self) -> str:
        occ = len(self.occupied_squares())
        return (
            f"<BoardState occupied={occ}/64 "
            f"confidence={self.overall_confidence:.2f} "
            f"t={self.timestamp:.1f}>"
        )
