"""
tests/test_parser.py
====================
Unit tests for voice/parser.py covering:
  - English & Portuguese move commands
  - English & Portuguese action commands (NEW_GAME, RESUME_GAME, RESIGN_GAME)
  - YES/NO and SIM/NÃO confirmation parsing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from parser import parse, parse_confirmation, MoveCommand, ActionCommand


# ---------------------------------------------------------------------------
# Move Command Success Cases (EN and PT)
# ---------------------------------------------------------------------------

MOVE_SUCCESS_CASES = [
    # (input_text, expected_from, expected_to)
    # Standard English compact
    ("move e2 e4",                   "E2", "E4"),
    ("MOVE E2 E4",                   "E2", "E4"),
    ("Move G1 F3",                   "G1", "F3"),
    ("move a7 a8",                   "A7", "A8"),
    ("move h1 h8",                   "H1", "H8"),
    # Spoken digit words (EN)
    ("move e two e four",            "E2", "E4"),
    ("move g one f three",           "G1", "F3"),
    ("move a seven a eight",         "A7", "A8"),
    ("move c one c three",           "C1", "C3"),
    # With English connectors
    ("move from e2 to e4",           "E2", "E4"),
    ("move from a7 to a8",           "A7", "A8"),
    # NATO phonetic alphabet
    ("move echo two echo four",       "E2", "E4"),
    ("move alpha two alpha four",     "A2", "A4"),
    ("move hotel one hotel four",     "H1", "H4"),
    ("move charlie one foxtrot four", "C1", "F4"),
    ("move bravo one delta two",      "B1", "D2"),
    ("move golf one hotel eight",     "G1", "H8"),
    # Acoustic misrecognition recovery ('8' misheard for 'h' / 'aitch')
    ("move h one eight four",         "H1", "H4"),
    ("move eight two eight four",     "H2", "H4"),
    ("move eight one eight eight",    "H1", "H8"),

    # Portuguese move commands
    ("mova e2 e4",                   "E2", "E4"),
    ("mover e dois e quatro",        "E2", "E4"),
    ("jogar g um f três",            "G1", "F3"),
    ("mova de a um para a dois",     "A1", "A2"),
    ("mover de agá um para agá quatro", "H1", "H4"),
    ("mova de b um pra c três",      "B1", "C3"),
    ("mova echo dois echo quatro",   "E2", "E4"),
    ("jogar de e dois para e quatro", "E2", "E4"),
]


@pytest.mark.parametrize("text, expected_from, expected_to", MOVE_SUCCESS_CASES)
def test_parse_move_success(text, expected_from, expected_to):
    command, error = parse(text)
    assert error is None, f"Expected success but got error: {error}"
    assert isinstance(command, MoveCommand), f"Expected MoveCommand, got {type(command)}"
    assert command.from_square == expected_from
    assert command.to_square == expected_to


# ---------------------------------------------------------------------------
# Action Commands Success Cases (PLAY NEW GAME, RESUME GAME, RESIGN GAME)
# ---------------------------------------------------------------------------

ACTION_SUCCESS_CASES = [
    # PLAY NEW GAME (EN & PT)
    ("play new game",       "NEW_GAME"),
    ("new game",            "NEW_GAME"),
    ("start game",          "NEW_GAME"),
    ("jogar novo jogo",     "NEW_GAME"),
    ("novo jogo",           "NEW_GAME"),
    ("iniciar jogo",        "NEW_GAME"),
    ("começar jogo",        "NEW_GAME"),

    # RESUME GAME (EN & PT)
    ("resume game",         "RESUME_GAME"),
    ("resume",              "RESUME_GAME"),
    ("continue game",       "RESUME_GAME"),
    ("continuar jogo",      "RESUME_GAME"),
    ("retomar jogo",        "RESUME_GAME"),
    ("continuar",           "RESUME_GAME"),
    ("retomar",             "RESUME_GAME"),

    # RESIGN GAME (EN & PT)
    ("resign game",         "RESIGN_GAME"),
    ("resign",              "RESIGN_GAME"),
    ("surrender",           "RESIGN_GAME"),
    ("desistir do jogo",    "RESIGN_GAME"),
    ("abandonar jogo",      "RESIGN_GAME"),
    ("desistir",            "RESIGN_GAME"),
    ("abandonar",           "RESIGN_GAME"),
]


@pytest.mark.parametrize("text, expected_action", ACTION_SUCCESS_CASES)
def test_parse_action_success(text, expected_action):
    command, error = parse(text)
    assert error is None, f"Expected success but got error: {error}"
    assert isinstance(command, ActionCommand), f"Expected ActionCommand, got {type(command)}"
    assert command.action == expected_action


# ---------------------------------------------------------------------------
# Confirmation Parsing Tests (YES / NO, SIM / NÃO)
# ---------------------------------------------------------------------------

CONFIRMATION_CASES = [
    # Affirmative
    ("yes",         True),
    ("yeah",        True),
    ("yep",         True),
    ("confirm",     True),
    ("sim",         True),
    ("confirma",    True),
    ("confirmo",    True),
    ("positivo",    True),

    # Negative
    ("no",          False),
    ("nope",        False),
    ("cancel",      False),
    ("não",         False),
    ("nao",         False),
    ("cancela",     False),
    ("cancelar",    False),
    ("negativo",    False),

    # Inconclusive / Noise
    ("",            None),
    ("banana",      None),
    ("e4",          None),
]


@pytest.mark.parametrize("text, expected_bool", CONFIRMATION_CASES)
def test_parse_confirmation(text, expected_bool):
    result = parse_confirmation(text)
    assert result == expected_bool


# ---------------------------------------------------------------------------
# Failure Cases
# ---------------------------------------------------------------------------

FAILURE_CASES = [
    ("e2 e4",          "recognised command keyword"),
    ("go e2 e4",       "recognised command keyword"),
    ("",               "empty"),
    ("move e2",        "two chess squares"),
    ("mova e2",        "two chess squares"),
    ("move xyz abc",   "two chess squares"),
    ("hello world",    "recognised command keyword"),
]


@pytest.mark.parametrize("text, reason_fragment", FAILURE_CASES)
def test_parse_failure(text, reason_fragment):
    command, error = parse(text)
    assert command is None
    assert error is not None
    assert reason_fragment.lower() in error.lower()


# ---------------------------------------------------------------------------
# Dataclasses tests
# ---------------------------------------------------------------------------

def test_move_command_str():
    cmd = MoveCommand(from_square="E2", to_square="E4")
    assert str(cmd) == "MOVE E2 E4"


def test_action_command_str():
    cmd = ActionCommand("NEW_GAME")
    assert str(cmd) == "NEW_GAME"
