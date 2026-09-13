"""
parser.py
=========
Converts raw STT transcript text into structured Command objects:
  - MoveCommand(from_square="E2", to_square="E4")
  - ActionCommand(action="NEW_GAME" | "RESUME_GAME" | "RESIGN_GAME")

Supports both English (en-US) and Portuguese (pt-BR).

Examples:
  "play new game"             -> ActionCommand("NEW_GAME")
  "jogar novo jogo"           -> ActionCommand("NEW_GAME")
  "resume game"               -> ActionCommand("RESUME_GAME")
  "continuar jogo"            -> ActionCommand("RESUME_GAME")
  "resign game"               -> ActionCommand("RESIGN_GAME")
  "desistir do jogo"          -> ActionCommand("RESIGN_GAME")
  "move e2 e4"                -> MoveCommand("E2", "E4")
  "mova de e dois para e quatro" -> MoveCommand("E2", "E4")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Union

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Command Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MoveCommand:
    """Immutable, validated chess move command."""
    from_square: str   # 2 chars, uppercase, e.g. "E2"
    to_square:   str   # 2 chars, uppercase, e.g. "E4"

    def __str__(self) -> str:
        return f"MOVE {self.from_square} {self.to_square}"


@dataclass(frozen=True)
class ActionCommand:
    """Game control actions: NEW_GAME, RESUME_GAME, RESIGN_GAME."""
    action: str        # "NEW_GAME", "RESUME_GAME", "RESIGN_GAME"

    def __str__(self) -> str:
        return self.action


Command = Union[MoveCommand, ActionCommand]


# ---------------------------------------------------------------------------
# Normalisation tables (Combined EN + PT for maximum flexibility)
# ---------------------------------------------------------------------------

# Spoken digit words -> digit character (supports both languages)
_DIGIT_WORDS: dict[str, str] = {
    **config.SPOKEN_DIGITS_EN,
    **config.SPOKEN_DIGITS_PT,
}

# Spoken column phonetics (NATO, English letters, Portuguese letters)
_COLUMN_PHONETICS: dict[str, str] = {
    **config.SPOKEN_COLUMNS_EN,
    **config.SPOKEN_COLUMNS_PT,
}

# Verbs that introduce a move command
_MOVE_KEYWORDS: set[str] = {"move", "mova", "mover", "jogar", "mexe", "mexer"}

# Filler / connector words to strip before parsing squares
# NOTE: 'a' is intentionally NOT included here because 'A' is chess column A.
_FILLER_WORDS: set[str] = {
    "from", "to", "the", "at", "on", "square",
    "de", "para", "pra", "ao", "casa", "o", "do", "da",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> str:
    """Lower-case, strip punctuation, expand spoken digits and column phonetics."""
    text = text.lower().strip()
    text = re.sub(r"[^a-záàâãéêíóôõúç0-9\s]", " ", text)

    tokens = text.split()
    tokens = [_DIGIT_WORDS.get(t, t) for t in tokens]
    tokens = [_COLUMN_PHONETICS.get(t, t) for t in tokens]

    return " ".join(tokens)


def _extract_squares(tokens: list[str]) -> list[str]:
    """
    Extract chess square tokens from a token list.
    Accepts compact ('e2'), expanded ('e' '2'), and acoustic recovery ('8' '4' -> 'h4').
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
        # If token is "8" and followed by a valid row (1-8), it is in column position.
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

def parse(text: str, language: str | None = None) -> tuple[Command | None, str | None]:
    """
    Parse a raw STT transcript into a MoveCommand or ActionCommand.

    Parameters
    ----------
    text : str
        Raw transcript from the STT engine.
    language : str | None
        Target language ("pt-BR" or "en-US"), defaults to config.LANGUAGE.

    Returns
    -------
    (Command, None) on successful parse.
    (None, reason: str) on failure.
    """
    if not text or not text.strip():
        return None, "Received empty transcript."

    log.debug("[Parser] Raw text: '%s'", text)
    normalised = _normalise_text(text)
    log.debug("[Parser] Normalised: '%s'", normalised)
    tokens = normalised.split()

    if not tokens:
        return None, "Received empty transcript."

    # 1. Action Commands (Game state control)
    norm_str = " ".join(tokens)

    # PLAY NEW GAME
    if (
        "new game" in norm_str
        or "play new game" in norm_str
        or "start game" in norm_str
        or "novo jogo" in norm_str
        or "jogar novo jogo" in norm_str
        or "iniciar jogo" in norm_str
        or "comecar jogo" in norm_str
        or "começar jogo" in norm_str
    ):
        cmd = ActionCommand("NEW_GAME")
        log.info("[Parser] Parsed ActionCommand: %s", cmd)
        return cmd, None

    # RESUME GAME
    if (
        "resume game" in norm_str
        or "continue game" in norm_str
        or "continuar jogo" in norm_str
        or "retomar jogo" in norm_str
        or norm_str == "resume"
        or norm_str == "continuar"
        or norm_str == "retomar"
    ):
        cmd = ActionCommand("RESUME_GAME")
        log.info("[Parser] Parsed ActionCommand: %s", cmd)
        return cmd, None

    # RESIGN GAME
    if (
        "resign game" in norm_str
        or "desistir do jogo" in norm_str
        or "abandonar jogo" in norm_str
        or norm_str == "resign"
        or norm_str == "desistir"
        or norm_str == "abandonar"
        or norm_str == "surrender"
    ):
        cmd = ActionCommand("RESIGN_GAME")
        log.info("[Parser] Parsed ActionCommand: %s", cmd)
        return cmd, None

    # 2. Move Command
    # Find any move introduction keyword
    move_idx = None
    for idx, tok in enumerate(tokens):
        if tok in _MOVE_KEYWORDS:
            move_idx = idx
            break

    if move_idx is None:
        return None, (
            f"No recognised command keyword found in: '{text}'. "
            "Try 'Play new game', 'Resume game', 'Resign game' or 'Move E2 E4' (ou 'Mova E2 E4')."
        )

    tokens_after_move = tokens[move_idx + 1 :]

    # Strip filler words
    cleaned_tokens = [t for t in tokens_after_move if t not in _FILLER_WORDS]
    log.debug("[Parser] Cleaned tokens for move: %s", cleaned_tokens)

    squares = _extract_squares(cleaned_tokens)
    log.debug("[Parser] Extracted squares: %s", squares)

    if len(squares) < 2:
        return None, (
            f"Could not extract two chess squares from: '{text}'. "
            f"Found: {squares}. Expected format: 'Move E2 E4' (ou 'Mova E2 E4')."
        )

    from_sq = squares[0].upper()
    to_sq   = squares[1].upper()

    command = MoveCommand(from_square=from_sq, to_square=to_sq)
    log.info("[Parser] Parsed MoveCommand: %s", command)
    return command, None


# ---------------------------------------------------------------------------
# Confirmation Parsing (YES / NO, SIM / NÃO)
# ---------------------------------------------------------------------------

def parse_confirmation(text: str, language: str | None = None) -> bool | None:
    """
    Parse a user confirmation response.

    Returns:
      True  -> affirmative (YES / SIM)
      False -> negative (NO / NÃO)
      None  -> unrecognized / inconclusive
    """
    if not text:
        return None

    norm = text.lower().strip()
    tokens = set(re.sub(r"[^a-záàâãéêíóôõúç0-9\s]", " ", norm).split())

    yes_tokens = set(config.CONFIRM_YES_EN + config.CONFIRM_YES_PT)
    no_tokens  = set(config.CONFIRM_NO_EN  + config.CONFIRM_NO_PT)

    # Check for negative first (safe default)
    if any(t in tokens for t in no_tokens) or "não" in norm or "nao" in norm:
        return False

    # Check for affirmative
    if any(t in tokens for t in yes_tokens) or "sim" in norm:
        return True

    return None
