"""
parser.py
=========
Converts raw STT transcript text into structured MoveCommand objects.

Handles all common spoken variations:

  Input                         -> Normalised
  "move e two e four"           -> MoveCommand(from_square="E2", to_square="E4")
  "move e2 e4"                  -> MoveCommand(from_square="E2", to_square="E4")
  "MOVE E2 E4"                  -> MoveCommand(from_square="E2", to_square="E4")
  "move g one f three"          -> MoveCommand(from_square="G1", to_square="F3")
  "move from a seven to a eight"-> MoveCommand(from_square="A7", to_square="A8")

Public API
----------
    result = parse(text)
    # Returns (MoveCommand, None) on success
    # Returns (None, reason_str)  on failure
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MoveCommand dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MoveCommand:
    """Immutable, validated chess move command."""
    from_square: str   # always 2 chars, uppercase, e.g. "E2"
    to_square:   str   # always 2 chars, uppercase, e.g. "E4"

    def __str__(self) -> str:
        return f"MOVE {self.from_square} {self.to_square}"


# ---------------------------------------------------------------------------
# Normalisation tables
# ---------------------------------------------------------------------------

# Spoken digit words -> digit character
_DIGIT_WORDS: dict[str, str] = config.SPOKEN_DIGITS

# Spoken column phonetics (Vosk sometimes hears these or user uses NATO alphabet)
_COLUMN_PHONETICS: dict[str, str] = {
    **getattr(config, "NATO_PHONETICS", {}),
    "ay": "a",
    "bee": "b",
    "see": "c",
    "dee": "d",
    "ee": "e",
    "eff": "f",
    "gee": "g",
    "aitch": "h",
    "age": "h",
    "hey": "h",
}

# Filler / connector words to strip before parsing squares
_FILLER_WORDS: set[str] = {"from", "to", "the", "at", "on", "square"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> str:
    """
    Lower-case, strip punctuation, expand spoken digits/columns.

    Steps
    -----
    1. Lowercase and strip leading/trailing whitespace.
    2. Replace spoken digit words with digit characters.
    3. Replace spoken column phonetics with letters.
    4. Collapse multiple spaces.
    """
    text = text.lower().strip()

    # Remove punctuation except alphanumeric and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Replace spoken digit words (e.g. "two" -> "2")
    tokens = text.split()
    tokens = [_DIGIT_WORDS.get(t, t) for t in tokens]

    # Replace column phonetics (e.g. "ee" -> "e", "echo" -> "e")
    tokens = [_COLUMN_PHONETICS.get(t, t) for t in tokens]

    return " ".join(tokens)


def _extract_squares(tokens: list[str]) -> list[str]:
    """
    Extract chess square tokens from a token list.

    Accepts formats:
      - Compact   : "e2"  (single token, letter + digit)
      - Expanded  : "e" "2"  (two adjacent tokens)
      - Heuristic : "8" "4" -> "h4" (Vosk acoustic confusion between "aitch" and "eight")

    Returns a list of normalised square strings (e.g. ["e2", "e4"]).
    """
    squares: list[str] = []
    i = 0
    valid_cols = set("abcdefgh")
    valid_rows = set("12345678")

    while i < len(tokens):
        tok = tokens[i]

        # Compact format: letter+digit in one token (e.g. "e2")
        if len(tok) == 2 and tok[0] in valid_cols and tok[1] in valid_rows:
            squares.append(tok)
            i += 1
            continue

        # Expanded format: separate letter and digit tokens (e.g. "e" "2")
        if (
            len(tok) == 1
            and tok in valid_cols
            and i + 1 < len(tokens)
            and len(tokens[i + 1]) == 1
            and tokens[i + 1] in valid_rows
        ):
            squares.append(tok + tokens[i + 1])
            i += 2
            continue

        # Acoustic heuristic: "8" misheard as "H" ("eight" vs "aitch").
        # If token is "8" and followed by a valid row (1-8), it is in column position
        # and represents the letter "H" (e.g. "eight four" -> "h4").
        if (
            tok == "8"
            and i + 1 < len(tokens)
            and len(tokens[i + 1]) == 1
            and tokens[i + 1] in valid_rows
        ):
            squares.append("h" + tokens[i + 1])
            i += 2
            continue

        i += 1

    return squares


# ---------------------------------------------------------------------------
# Public parse function
# ---------------------------------------------------------------------------

def parse(text: str) -> tuple[MoveCommand | None, str | None]:
    """
    Parse a raw STT transcript into a MoveCommand.

    Parameters
    ----------
    text : str
        Raw transcript from the STT engine, e.g. "move e two e four".

    Returns
    -------
    (MoveCommand, None)
        On successful parse.
    (None, reason: str)
        On failure, with a human-readable explanation.
    """
    if not text or not text.strip():
        return None, "Received empty transcript."

    log.debug("[Parser] Raw text: '%s'", text)
    normalised = _normalise_text(text)
    log.debug("[Parser] Normalised: '%s'", normalised)

    tokens = normalised.split()

    # 1. Require "move" keyword
    if "move" not in tokens:
        return None, (
            f"No 'move' keyword found in recognised text: '{text}'. "
            "Try saying 'Move E2 E4'."
        )

    # 2. Discard everything up to and including "move"
    move_idx = tokens.index("move")
    tokens = tokens[move_idx + 1 :]

    # 3. Strip filler words
    tokens = [t for t in tokens if t not in _FILLER_WORDS]

    log.debug("[Parser] Tokens after stripping: %s", tokens)

    # 4. Extract chess squares
    squares = _extract_squares(tokens)
    log.debug("[Parser] Extracted squares: %s", squares)

    if len(squares) < 2:
        return None, (
            f"Could not extract two chess squares from: '{text}'. "
            f"Found: {squares}. Expected format: 'Move E2 E4'."
        )

    from_sq = squares[0].upper()
    to_sq   = squares[1].upper()

    command = MoveCommand(from_square=from_sq, to_square=to_sq)
    log.info("[Parser] Parsed: %s", command)
    return command, None
