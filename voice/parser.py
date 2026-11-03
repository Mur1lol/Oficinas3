import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class MoveCommand:
    from_square: str
    to_square: str

    def __str__(self):
        return f"MOVE {self.from_square} {self.to_square}"


@dataclass(frozen=True)
class ActionCommand:
    action: str

    def __str__(self):
        return self.action


@dataclass(frozen=True)
class LevelCommand:
    """Difficulty selection (1-10), e.g. "level three" or "level 10"."""

    level: int

    def __str__(self):
        return f"SET_DIFFICULTY {self.level}"


Command = MoveCommand | ActionCommand | LevelCommand

DIGITS = dict(zip("one two three four five six seven eight".split(), "12345678", strict=True))
LEVEL_DIGITS = {
    **DIGITS,
    "nine": "9",
    "ten": "10",
}
VALID_LEVEL_STRS = {str(i) for i in range(1, 11)}

# English letters and NATO phonetic alphabet
COLUMNS = {
    # Letters
    "a": "a",
    "ay": "a",
    "b": "b",
    "bee": "b",
    "c": "c",
    "see": "c",
    "d": "d",
    "dee": "d",
    "e": "e",
    "ee": "e",
    "f": "f",
    "eff": "f",
    "g": "g",
    "gee": "g",
    "h": "h",
    # NATO phonetic alphabet
    "alpha": "a",
    "bravo": "b",
    "charlie": "c",
    "delta": "d",
    "echo": "e",
    "foxtrot": "f",
    "golf": "g",
    "hotel": "h",
}

ACTIONS = {
    "NEW_GAME": (
        "start new game",
        "start game",
        "new game",
        "play new game",
        "start new",
        "start",
        "play",
    ),
    "RESUME_GAME": (
        "resume game",
        "resume",
        "continue game",
        "continue",
    ),
    "RESIGN_GAME": (
        "resign game",
        "resign",
        "surrender",
    ),
    "ORGANIZE_BOARD": (
        "organize board",
        "organize pieces",
        "organize",
        "reset board",
        "reset pieces",
        "arrange board",
        "arrange pieces",
        "setup board",
    ),
    "VALIDATE_MOVE": (
        "validate",
        "validate move",
        "check move",
        "verify move",
        "verify",
    ),
    "MODE_PVE": (
        "pve",
        "p v e",
        "player versus engine",
        "player vs engine",
        "versus engine",
        "player versus computer",
        "player versus machine",
        "against computer",
        " against engine",
        "single player",
        "engine",
        "machine",
    ),
    "MODE_PVP": (
        "pvp",
        "p v p",
        "player versus player",
        "player vs player",
        "two players",
        "versus player",
        "multiplayer",
    ),
    "MODE_MANUAL": (
        "manual",
        "manual mode",
        "hand mode",
    ),
    "MODE_VOICE": (
        "voice",
        "voice mode",
        "automatic",
        "gantry mode",
    ),
    "COLOR_WHITE": (
        "white",
        "play white",
        "white pieces",
        "color white",
    ),
    "COLOR_BLACK": (
        "black",
        "play black",
        "black pieces",
        "color black",
    ),
}

# Comandos que executam imediatamente apos "MAGNUS <COMANDO>" sem exigir confirmacao YES/NO.
# Caso mude de ideia e queira exigir confirmacao (YES/NO) nos comandos de menu,
# basta deixar apenas {"VALIDATE_MOVE"} abaixo e mudar IMMEDIATE_LEVEL para False:
# IMMEDIATE_ACTIONS = {"VALIDATE_MOVE"}
IMMEDIATE_ACTIONS = {
    "VALIDATE_MOVE",
    "NEW_GAME",
    "MODE_PVP",
    "MODE_PVE",
    "MODE_MANUAL",
    "MODE_VOICE",
    "COLOR_WHITE",
    "COLOR_BLACK",
}
IMMEDIATE_LEVEL = True

MOVE_WORDS = {"move", "play"}
LEVEL_WORDS = {"level", "difficulty"}
FILLERS = {
    "from",
    "to",
    "the",
    "at",
    "on",
    "square",
}

YES = {"yes", "yeah", "yep", "confirm", "correct"}
NO = {"no", "nope", "cancel", "negative"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text).split())


def parse(text: str, language: str | None = None) -> tuple[Command | None, str | None]:
    text = normalize(text)
    if not text:
        return None, "Received empty transcript."
    for action, phrases in ACTIONS.items():
        if text in phrases:
            return ActionCommand(action), None
    tokens = text.split()
    if tokens[0] in LEVEL_WORDS:
        rest = [LEVEL_DIGITS.get(t, t) for t in tokens[1:] if t not in FILLERS]
        if len(rest) == 1 and rest[0] in VALID_LEVEL_STRS:
            return LevelCommand(int(rest[0])), None
        return None, "Expected a difficulty level from 1 to 10."
    if tokens[0] not in MOVE_WORDS:
        return None, "No recognised command keyword."
    tokens = [DIGITS.get(t, COLUMNS.get(t, t)) for t in tokens[1:] if t not in FILLERS]
    squares = []
    while tokens:
        token = tokens.pop(0)
        if re.fullmatch(r"[a-h][1-8]", token):
            squares.append(token.upper())
        elif token in {*"abcdefgh", "8"} and tokens and tokens[0] in set("12345678"):
            squares.append(("h" if token == "8" else token).upper() + tokens.pop(0))
        else:
            return None, "Expected exactly two chess squares without ambiguous words."
    if len(squares) != 2:
        return None, "Expected exactly two chess squares."
    return MoveCommand(*squares), None


def parse_confirmation(text: str | None, language: str | None = None) -> bool | None:
    tokens = set(normalize(text or "").split())
    if tokens and tokens <= YES:
        return True
    if tokens and tokens <= NO:
        return False
    return None


def grammar(stage: str, language: str = "en-US") -> list[str]:
    if stage == "wake":
        return ["magnus", "[unk]"]
    if stage == "confirmation":
        return sorted(YES | NO | {"[unk]"})
    words = (
        MOVE_WORDS
        | LEVEL_WORDS
        | FILLERS
        | set(LEVEL_DIGITS)
        | VALID_LEVEL_STRS
        | set(COLUMNS)
        | set("abcdefgh")
    )
    words |= {word for phrases in ACTIONS.values() for phrase in phrases for word in phrase.split()}
    return sorted(words | {"[unk]"})
