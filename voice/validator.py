"""
validator.py
============
Validates parsed chess commands from parser.py.

Returns a standardised result dict consumed by main.py.

Public API
----------
    result = validate(command, reason)
    # Always returns a dict with at least {"valid": bool}
"""

from __future__ import annotations

import logging

import config

log = logging.getLogger(__name__)

_VALID_COLS: frozenset[str] = frozenset("ABCDEFGH")
_VALID_ROWS: frozenset[str] = frozenset("12345678")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_valid_square(square: str) -> tuple[bool, str]:
    """
    Return (True, "") if *square* is a legal chess coordinate,
    or (False, reason) if it is not.
    """
    sq = square.upper().strip()

    if len(sq) != 2:
        return False, (
            f"Square '{square}' must be exactly 2 characters (e.g. E4)."
        )

    col, row = sq[0], sq[1]

    if col not in _VALID_COLS:
        return False, (
            f"Invalid column '{col}' in '{square}'. Valid columns: A-H."
        )

    if row not in _VALID_ROWS:
        return False, (
            f"Invalid row '{row}' in '{square}'. Valid rows: 1-8."
        )

    return True, ""


# ---------------------------------------------------------------------------
# Public validate function
# ---------------------------------------------------------------------------

def validate(
    command: "MoveCommand | None",
    reason: str | None = None,
) -> dict:
    """
    Validate a parsed MoveCommand.

    Parameters
    ----------
    command : MoveCommand | None
        The parsed command returned by parser.parse().
        Pass None when parsing itself failed.
    reason : str | None
        Optional pre-populated failure reason (e.g. from the parser).

    Returns
    -------
    dict
        Success::

            {
                "wake_word": "MAGNUS",
                "command": "MOVE",
                "from": "E2",
                "to": "E4",
                "valid": True,
            }

        Failure::

            {
                "valid": False,
                "reason": "<human-readable explanation>",
            }
    """
    if command is None:
        msg = reason or "Command could not be parsed from the recognised text."
        log.warning("Validation failed (no command): %s", msg)
        return {"valid": False, "reason": msg}

    from_sq = command.from_square.upper()
    to_sq   = command.to_square.upper()

    ok, err = is_valid_square(from_sq)
    if not ok:
        log.warning("Invalid source square: %s", err)
        return {"valid": False, "reason": f"Invalid source square — {err}"}

    ok, err = is_valid_square(to_sq)
    if not ok:
        log.warning("Invalid destination square: %s", err)
        return {"valid": False, "reason": f"Invalid destination square — {err}"}

    if from_sq == to_sq:
        msg = f"Source and destination are the same square ({from_sq})."
        log.warning("Null move rejected: %s", msg)
        return {"valid": False, "reason": msg}

    result = {
        "wake_word": config.WAKE_WORD,
        "command": "MOVE",
        "from": from_sq,
        "to": to_sq,
        "valid": True,
    }
    log.info("Valid move command: %s -> %s", from_sq, to_sq)
    return result
