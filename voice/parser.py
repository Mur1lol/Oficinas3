"""
voice/parser.py
===============
Gramatica ESTRITA e deterministica dos comandos de voz.

Regras:
* Palavras fora do vocabulario chegam do Vosk como ``[unk]`` e sao descartadas.
* O que sobra precisa formar EXATAMENTE um dos comandos abaixo (nada a mais, nada a menos):

    START [NEW] GAME              -> NEW_GAME   (primeira e ultima palavra obrigatorias)
    OPTION ONE | OPTION TWO       -> MODE_PVP / MODE_PVE
    LEVEL ONE .. LEVEL TEN        -> LevelCommand(1..10)
    WHITE | BLACK                 -> COLOR_WHITE / COLOR_BLACK
    VALIDATE                      -> VALIDATE_MOVE
    REPEAT                        -> REPEAT_MOVE
    RESIGN [GAME]                 -> RESIGN_GAME
    CANCEL                        -> CANCEL
    QUEEN | ROOK | BISHOP | KNIGHT-> PROMOTE_*
    MOVE <col> <row> [TO] <col> <row> [QUEEN|ROOK|BISHOP|KNIGHT] -> MoveCommand

* ``match_status(text)`` informa se o texto e um comando COMPLETO, um PREFIXO valido
  (ex.: "option", "level", "move e two") que precisa esperar a proxima palavra, ou INVALIDO.
"""

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class MoveCommand:
    from_square: str
    to_square: str
    promotion: str | None = None

    def __str__(self):
        if self.promotion:
            return f"MOVE {self.from_square} {self.to_square} {self.promotion.upper()}"
        return f"MOVE {self.from_square} {self.to_square}"


@dataclass(frozen=True)
class ActionCommand:
    action: str

    def __str__(self):
        return self.action


@dataclass(frozen=True)
class LevelCommand:
    """Difficulty selection (1-10), e.g. "level three"."""

    level: int

    def __str__(self):
        return f"SET_DIFFICULTY {self.level}"


Command = MoveCommand | ActionCommand | LevelCommand

COMPLETE, PREFIX, INVALID = "complete", "prefix", "invalid"

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
DIGITS = {w: str(n) for w, n in NUMBER_WORDS.items() if n <= 8}   # linhas do tabuleiro
LEVEL_DIGITS = {w: str(n) for w, n in NUMBER_WORDS.items()}
VALID_LEVEL_STRS = {str(i) for i in range(1, 11)}
_DIGIT_TO_WORD = {str(n): w for w, n in NUMBER_WORDS.items()}

# Colunas: letras e alfabeto NATO
COLUMNS = {
    "a": "a", "ay": "a", "alpha": "a",
    "b": "b", "bee": "b", "bravo": "b",
    "c": "c", "see": "c", "charlie": "c",
    "d": "d", "dee": "d", "delta": "d",
    "e": "e", "ee": "e", "echo": "e",
    "f": "f", "eff": "f", "foxtrot": "f",
    "g": "g", "gee": "g", "golf": "g",
    "h": "h", "hotel": "h",
}
PROMO_PIECES = {"queen": "q", "rook": "r", "bishop": "b", "knight": "n"}

# Frases canonicas (usadas pelo validator e pela documentacao)
ACTIONS = {
    "NEW_GAME": ("start game", "start new game"),
    "RESIGN_GAME": ("resign", "resign game"),
    "VALIDATE_MOVE": ("validate",),
    "REPEAT_MOVE": ("repeat",),
    "MODE_PVP": ("option one",),
    "MODE_PVE": ("option two",),
    "COLOR_WHITE": ("white",),
    "COLOR_BLACK": ("black",),
    "PROMOTE_QUEEN": ("queen",),
    "PROMOTE_ROOK": ("rook",),
    "PROMOTE_BISHOP": ("bishop",),
    "PROMOTE_KNIGHT": ("knight",),
    "CANCEL": ("cancel",),
}
_SINGLE_WORD = {
    "white": "COLOR_WHITE",
    "black": "COLOR_BLACK",
    "validate": "VALIDATE_MOVE",
    "repeat": "REPEAT_MOVE",
    "cancel": "CANCEL",
    "queen": "PROMOTE_QUEEN",
    "rook": "PROMOTE_ROOK",
    "bishop": "PROMOTE_BISHOP",
    "knight": "PROMOTE_KNIGHT",
}

# Comandos que executam direto sem pedir confirmacao YES/NO (vazio = todos pedem confirmacao)
IMMEDIATE_ACTIONS: set[str] = set()
IMMEDIATE_LEVEL: bool = False

MOVE_WORDS = {"move"}
LEVEL_WORDS = {"level"}
OPTION_WORDS = {"option"}
FILLERS = {"to"}

YES = {"yes", "yeah", "yep", "confirm", "correct"}
NO = {"no", "nope", "cancel", "negative"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text).split())


def _tokens(text: str) -> list[str]:
    """
    Normaliza, descarta [unk]/'magnus', expande atalhos digitados (e2 -> e two, 3 -> three)
    e remove palavras identicas repetidas em sequencia (ex.: 'level level two' -> 'level two').
    """
    raw: list[str] = []
    for tok in normalize(text).split():
        if tok in ("unk", "magnus"):
            continue
        if re.fullmatch(r"[a-h][1-8]", tok):
            raw += [tok[0], _DIGIT_TO_WORD[tok[1]]]
        elif tok in _DIGIT_TO_WORD:
            raw.append(_DIGIT_TO_WORD[tok])
        else:
            raw.append(tok)

    out: list[str] = []
    for tok in raw:
        if out and out[-1] == tok:
            continue
        out.append(tok)
    return out


def _match_move(rest: list[str]):
    i = 0
    squares = []
    for k in range(2):
        if k == 1 and i < len(rest) and rest[i] in ("to", "two"):
            i += 1                                   # "to" (ou "two" ouvido no lugar de "to")
        if i >= len(rest):
            return PREFIX, None, "Waiting for the square."
        col = COLUMNS.get(rest[i])
        if col is None:
            return INVALID, None, f"Expected a column (A-H), heard {rest[i]!r}."
        i += 1
        if i >= len(rest):
            return PREFIX, None, "Waiting for the row."
        row = DIGITS.get(rest[i])
        if row is None:
            return INVALID, None, f"Expected a row (one-eight), heard {rest[i]!r}."
        i += 1
        squares.append(col.upper() + row)
    promo = None
    if i < len(rest):
        promo = PROMO_PIECES.get(rest[i])
        if promo is None or i + 1 != len(rest):
            return INVALID, None, f"Unexpected words after the move: {' '.join(rest[i:])!r}."
    if squares[0] == squares[1]:
        return INVALID, None, "Source and destination are the same square."
    return COMPLETE, MoveCommand(squares[0], squares[1], promotion=promo), None


_COMMAND_HEADS = {"start", "option", "level", "resign", "move"} | set(_SINGLE_WORD)


def _match_once(tokens: list[str]):
    if not tokens:
        return PREFIX, None, "Empty transcript."
    head, rest = tokens[0], tokens[1:]

    if head == "start":
        if rest in ([], ["new"]):
            return PREFIX, None, "Waiting for 'game'."
        if rest in (["game"], ["new", "game"]):
            return COMPLETE, ActionCommand("NEW_GAME"), None
        return INVALID, None, "Expected 'start new game'."

    if head == "option":
        if not rest:
            return PREFIX, None, "Waiting for 'one' or 'two'."
        if rest == ["one"]:
            return COMPLETE, ActionCommand("MODE_PVP"), None
        if rest == ["two"]:
            return COMPLETE, ActionCommand("MODE_PVE"), None
        return INVALID, None, "Expected 'option one' or 'option two'."

    if head == "level":
        if not rest:
            return PREFIX, None, "Waiting for the level (one-ten)."
        if len(rest) == 1 and rest[0] in NUMBER_WORDS:
            return COMPLETE, LevelCommand(NUMBER_WORDS[rest[0]]), None
        return INVALID, None, "Expected a difficulty level from one to ten."

    if head == "resign":
        if rest in ([], ["game"]):
            return COMPLETE, ActionCommand("RESIGN_GAME"), None
        return INVALID, None, "Expected 'resign'."

    if head in _SINGLE_WORD:
        if not rest:
            return COMPLETE, ActionCommand(_SINGLE_WORD[head]), None
        return INVALID, None, f"Unexpected words after {head!r}."

    if head == "move":
        return _match_move(rest)

    return INVALID, None, f"Unknown command word {head!r}."


def _match(tokens: list[str]):
    status, cmd, reason = _match_once(tokens)
    if status != INVALID:
        return status, cmd, reason
    # Se o usuario recomeçou o comando do inicio (ex.: "move e ... move e two e four"),
    # avalia a partir da ultima palavra-chave de inicio de comando.
    for i in range(len(tokens) - 1, 0, -1):
        if tokens[i] in _COMMAND_HEADS:
            # Evita confundir peca de promocao no final de um lance com novo comando
            if tokens[0] == "move" and tokens[i] in PROMO_PIECES:
                continue
            s2, c2, r2 = _match_once(tokens[i:])
            if s2 in (COMPLETE, PREFIX):
                return s2, c2, r2
            break
    return status, cmd, reason


def match_status(text: str) -> str:
    """COMPLETE, PREFIX (esperar proxima palavra) ou INVALID."""
    return _match(_tokens(text))[0]


def parse(text: str, language: str | None = None) -> tuple[Command | None, str | None]:
    tokens = _tokens(text)
    if not tokens:
        return None, "Received empty transcript."
    status, command, reason = _match(tokens)
    if status == COMPLETE:
        return command, None
    if status == PREFIX:
        return None, f"Incomplete command: {reason}"
    return None, reason


def parse_confirmation(text: str | None, language: str | None = None) -> bool | None:
    tokens = {t for t in normalize(text or "").split() if t != "unk"}
    if tokens and tokens <= YES:
        return True
    if tokens and tokens <= NO:
        return False
    return None


def command_vocabulary() -> set[str]:
    words = {"start", "new", "game", "option", "level", "resign", "move", "to"}
    words |= set(_SINGLE_WORD) | set(NUMBER_WORDS) | set(COLUMNS)
    return words


def grammar(stage: str, language: str = "en-US") -> list[str]:
    if stage == "confirmation":
        return sorted(YES | NO | {"[unk]"})
    words = command_vocabulary() | {"[unk]"}
    if stage == "wake":
        words.add("magnus")
    return sorted(words)
