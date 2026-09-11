"""
tests/test_parser.py
====================
Parametrised unit tests for voice/parser.py.

Run with:
    cd voice/
    python -m pytest tests/test_parser.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from parser import parse, MoveCommand


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------

SUCCESS_CASES = [
    # (input_text, expected_from, expected_to)
    # Standard compact notation
    ("move e2 e4",               "E2", "E4"),
    ("MOVE E2 E4",               "E2", "E4"),
    ("Move G1 F3",               "G1", "F3"),
    ("move a7 a8",               "A7", "A8"),
    ("move h1 h8",               "H1", "H8"),
    # Spoken digit words
    ("move e two e four",        "E2", "E4"),
    ("move g one f three",       "G1", "F3"),
    ("move a seven a eight",     "A7", "A8"),
    ("move c one c three",       "C1", "C3"),
    # Mixed compact + spoken
    ("move e2 e four",           "E2", "E4"),
    ("move e two e4",            "E2", "E4"),
    # With connectors
    ("move from e2 to e4",       "E2", "E4"),
    ("move from a7 to a8",       "A7", "A8"),
    # Uppercase and mixed case
    ("MOVE FROM E2 TO E4",       "E2", "E4"),
    ("Move From E2 To E4",       "E2", "E4"),
    # Extra whitespace
    ("  move   e2   e4  ",       "E2", "E4"),
    # All corners
    ("move a1 h8",               "A1", "H8"),
    ("move h8 a1",               "H8", "A1"),
    ("move a8 h1",               "A8", "H1"),
    ("move h1 a8",               "H1", "A8"),
    # Adjacent column letters
    ("move b2 c3",               "B2", "C3"),
    ("move d4 d5",               "D4", "D5"),
    ("move f6 g7",               "F6", "G7"),
]


@pytest.mark.parametrize("text, expected_from, expected_to", SUCCESS_CASES)
def test_parse_success(text, expected_from, expected_to):
    command, error = parse(text)
    assert error is None, f"Expected success but got error: {error}"
    assert command is not None
    assert command.from_square == expected_from, (
        f"from_square mismatch: got {command.from_square!r}, "
        f"expected {expected_from!r}"
    )
    assert command.to_square == expected_to, (
        f"to_square mismatch: got {command.to_square!r}, "
        f"expected {expected_to!r}"
    )


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

FAILURE_CASES = [
    # (input_text, expected_reason_fragment)
    # No move keyword
    ("e2 e4",                    "move"),
    ("go e2 e4",                 "move"),
    ("",                         "empty"),
    # Not enough squares
    ("move e2",                  "two chess squares"),
    ("move to e4",               "two chess squares"),
    ("move something",           "two chess squares"),
    # Gibberish
    ("move xyz abc",             "two chess squares"),
    ("hello world",              "move"),
]


@pytest.mark.parametrize("text, reason_fragment", FAILURE_CASES)
def test_parse_failure(text, reason_fragment):
    command, error = parse(text)
    assert command is None, f"Expected failure for input '{text}'"
    assert error is not None
    assert reason_fragment.lower() in error.lower(), (
        f"Expected '{reason_fragment}' in error message, got: {error!r}"
    )


# ---------------------------------------------------------------------------
# MoveCommand dataclass
# ---------------------------------------------------------------------------

def test_move_command_str():
    cmd = MoveCommand(from_square="E2", to_square="E4")
    assert str(cmd) == "MOVE E2 E4"


def test_move_command_immutable():
    cmd = MoveCommand(from_square="E2", to_square="E4")
    with pytest.raises((AttributeError, TypeError)):
        cmd.from_square = "A1"  # type: ignore


def test_move_command_equality():
    a = MoveCommand("E2", "E4")
    b = MoveCommand("E2", "E4")
    c = MoveCommand("G1", "F3")
    assert a == b
    assert a != c
