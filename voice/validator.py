from .config import WAKE_WORD
from .parser import ACTIONS, ActionCommand, Command, MoveCommand


def is_valid_square(square: str) -> tuple[bool, str]:
    square = square.upper().strip()
    if len(square) != 2:
        return False, "Square must contain 2 characters."
    if square[0] not in "ABCDEFGH":
        return False, "Invalid column; expected A-H."
    if square[1] not in "12345678":
        return False, "Invalid row; expected 1-8."
    return True, ""


def validate(command: Command | None, reason: str | None = None) -> dict:
    """Check command syntax only. Chess legality belongs to the game subsystem."""
    if command is None:
        return {"valid": False, "reason": reason or "Command could not be parsed."}
    result = {"wake_word": WAKE_WORD, "valid": True}
    if isinstance(command, ActionCommand) and command.action in ACTIONS:
        return {**result, "command": command.action}
    if isinstance(command, MoveCommand):
        source, destination = command.from_square.strip().upper(), command.to_square.strip().upper()
        for label, square in (("source", source), ("destination", destination)):
            ok, error = is_valid_square(square)
            if not ok:
                return {"valid": False, "reason": f"Invalid {label}: {error}"}
        if source == destination:
            return {"valid": False, "reason": "Source and destination are the same square."}
        return {**result, "command": "MOVE", "from": source, "to": destination}
    return {"valid": False, "reason": "Unknown command type or action."}
