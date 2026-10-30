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
    """Difficulty selection (1-8), e.g. "level four" / "nivel quatro"."""

    level: int

    def __str__(self):
        return f"SET_DIFFICULTY {self.level}"


Command = MoveCommand | ActionCommand | LevelCommand

DIGITS = dict(zip("one two three four five six seven eight".split(), "12345678", strict=True))
DIGITS.update(zip("um dois tres quatro cinco seis sete oito".split(), "12345678", strict=True))
COLUMNS = {
    "alpha": "a",
    "alfa": "a",
    "ay": "a",
    "bravo": "b",
    "bee": "b",
    "be": "b",
    "charlie": "c",
    "see": "c",
    "ce": "c",
    "delta": "d",
    "dee": "d",
    "de": "d",
    "echo": "e",
    "eco": "e",
    "ee": "e",
    "foxtrot": "f",
    "foxtrote": "f",
    "eff": "f",
    "golf": "g",
    "gee": "g",
    "hotel": "h",
    "aitch": "h",
    "aga": "h",
}
ACTIONS = {
    "NEW_GAME": (
        "play new game",
        "new game",
        "start game",
        "start new game",
        "start new",
        "start",
        "play",
        "jogar novo jogo",
        "novo jogo",
        "iniciar jogo",
        "comecar jogo",
        "iniciar",
        "comecar",
    ),
    "RESUME_GAME": (
        "resume game",
        "resume",
        "continue game",
        "continuar jogo",
        "retomar jogo",
        "continuar",
        "retomar",
    ),
    "RESIGN_GAME": (
        "resign game",
        "resign",
        "surrender",
        "desistir do jogo",
        "desistir",
        "abandonar jogo",
        "abandonar",
    ),
}
MOVE_WORDS = {"move", "mova", "mover", "jogar", "mexe", "mexer"}
LEVEL_WORDS = {"level", "difficulty", "nivel", "dificuldade"}
FILLERS = {
    "from",
    "to",
    "the",
    "at",
    "on",
    "square",
    "de",
    "para",
    "pra",
    "ao",
    "casa",
    "o",
    "do",
    "da",
}
YES = {"yes", "yeah", "yep", "confirm", "sim", "confirma", "confirmo", "positivo"}
NO = {"no", "nope", "cancel", "nao", "cancela", "cancelar", "negativo"}


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
        rest = [DIGITS.get(t, t) for t in tokens[1:] if t not in FILLERS]
        if len(rest) == 1 and rest[0] in set("12345678"):
            return LevelCommand(int(rest[0])), None
        return None, "Expected a difficulty level from 1 to 8."
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


def grammar(stage: str, language: str) -> list[str]:
    if stage == "wake":
        return ["magnus", "[unk]"]
    if stage == "confirmation":
        return sorted(YES | NO | {"não", "[unk]"})
    words = MOVE_WORDS | LEVEL_WORDS | FILLERS | set(DIGITS) | set(COLUMNS) | set("abcdefgh")
    words |= {word for phrases in ACTIONS.values() for phrase in phrases for word in phrase.split()}
    if language == "pt-BR":
        words |= {"três", "agá", "bê", "cê", "dê", "é", "começar"}
    return sorted(words | {"[unk]"})
