"""
tests/test_validator.py
=======================
Unit tests for voice/validator.py.

Run with:
    cd voice/
    python -m pytest tests/test_validator.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from parser import MoveCommand
from validator import validate, is_valid_square
import config


# ---------------------------------------------------------------------------
# is_valid_square
# ---------------------------------------------------------------------------

VALID_SQUARES = [
    "A1", "A8", "H1", "H8",   # corners
    "E2", "E4", "G1", "F3",   # common moves
    "a1", "h8", "e2", "d4",   # lowercase
    "B2", "C3", "D4", "F6",   # mid-board
]

@pytest.mark.parametrize("sq", VALID_SQUARES)
def test_is_valid_square_ok(sq):
    ok, err = is_valid_square(sq)
    assert ok is True, f"Expected {sq!r} to be valid, got error: {err}"
    assert err == ""


INVALID_SQUARES = [
    ("I1",  "column"),      # column I doesn't exist
    ("A9",  "row"),         # row 9 doesn't exist
    ("A0",  "row"),         # row 0 doesn't exist
    ("Z5",  "column"),      # column Z doesn't exist
    ("E",   "2 characters"),# too short
    ("E2E4","2 characters"),# too long
    ("",    "2 characters"),# empty
    ("11",  "column"),      # not a letter for col
    ("EE",  "row"),         # not a digit for row
]

@pytest.mark.parametrize("sq, fragment", INVALID_SQUARES)
def test_is_valid_square_fail(sq, fragment):
    ok, err = is_valid_square(sq)
    assert ok is False, f"Expected {sq!r} to be invalid"
    assert fragment.lower() in err.lower(), (
        f"Expected '{fragment}' in error: {err!r}"
    )


# ---------------------------------------------------------------------------
# validate() — success path
# ---------------------------------------------------------------------------

def test_validate_success_basic():
    cmd = MoveCommand("E2", "E4")
    result = validate(cmd)
    assert result["valid"] is True
    assert result["wake_word"] == config.WAKE_WORD
    assert result["command"] == "MOVE"
    assert result["from"] == "E2"
    assert result["to"] == "E4"


@pytest.mark.parametrize("from_sq, to_sq", [
    ("A1", "H8"),
    ("H8", "A1"),
    ("G1", "F3"),
    ("A7", "A8"),
    ("E1", "G1"),
    ("D4", "D5"),
])
def test_validate_success_various_moves(from_sq, to_sq):
    cmd = MoveCommand(from_sq, to_sq)
    result = validate(cmd)
    assert result["valid"] is True
    assert result["from"] == from_sq
    assert result["to"] == to_sq


# ---------------------------------------------------------------------------
# validate() — failure paths
# ---------------------------------------------------------------------------

def test_validate_no_command():
    result = validate(None)
    assert result["valid"] is False
    assert "reason" in result


def test_validate_no_command_with_reason():
    result = validate(None, reason="Custom parse error message")
    assert result["valid"] is False
    assert "Custom parse error message" in result["reason"]


def test_validate_invalid_source_column():
    cmd = MoveCommand("I2", "E4")  # I is not a valid column
    result = validate(cmd)
    assert result["valid"] is False
    assert "source" in result["reason"].lower()


def test_validate_invalid_dest_row():
    cmd = MoveCommand("E2", "E9")  # row 9 doesn't exist
    result = validate(cmd)
    assert result["valid"] is False
    assert "destination" in result["reason"].lower()


def test_validate_null_move():
    cmd = MoveCommand("E4", "E4")  # same square
    result = validate(cmd)
    assert result["valid"] is False
    assert "same" in result["reason"].lower()


def test_validate_lowercase_squares_normalised():
    """validate() should uppercase squares from parser output."""
    cmd = MoveCommand("e2", "e4")
    result = validate(cmd)
    assert result["valid"] is True
    assert result["from"] == "E2"
    assert result["to"] == "E4"


# ---------------------------------------------------------------------------
# All 64 valid squares round-trip
# ---------------------------------------------------------------------------

ALL_SQUARES = [
    f"{col}{row}"
    for col in "ABCDEFGH"
    for row in "12345678"
]

@pytest.mark.parametrize("sq", ALL_SQUARES)
def test_all_64_squares_valid(sq):
    ok, _ = is_valid_square(sq)
    assert ok is True, f"Expected {sq!r} to be a valid chess square"
