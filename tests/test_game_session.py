import chess
import time
from unittest.mock import MagicMock

from game.listener import ListenResult
from game.peripherals import StatusLeds
from game.session import COMPUTER, PLAYER, PLAYER_COLOR, VoiceChessGame
from voice.parser import ActionCommand, MoveCommand


class DummyLeds:
    def __init__(self):
        self.history = []
        self._state = {PLAYER: "off", COMPUTER: "off"}

    def state(self, side):
        return self._state.get(side, "off")

    def set(self, side, color):
        self._state[side] = color
        self.history.append(("set", side, color))

    def off(self, side=None):
        if side:
            self._state[side] = "off"
            self.history.append(("off", side))
        else:
            self._state = {PLAYER: "off", COMPUTER: "off"}
            self.history.append(("off", "all"))

    def blink(self, side, color, seconds=3.0, hz=3.0):
        self.history.append(("blink", side, color, seconds))

    def celebrate(self, winner, loser, seconds=5.0, hz=3.0):
        self.history.append(("celebrate", winner, loser, seconds))

    def blink_both(self, color, seconds=5.0, hz=3.0):
        self.history.append(("blink_both", color, seconds))

    def close(self):
        pass


class DummyLcd:
    def __init__(self):
        self.lines = ()

    def show(self, *lines):
        self.lines = lines

    def close(self):
        pass


class DummySpeaker:
    def __init__(self):
        self.spoken = []

    def say(self, text, wait=False):
        self.spoken.append(text)

    def wait_idle(self):
        pass

    def close(self):
        pass


class DummyGantry:
    def __init__(self):
        self.executed = []

    def connect(self):
        pass

    def home(self):
        pass

    def execute(self, x, y, ima):
        self.executed.append((list(x), list(y), list(ima)))

    def close(self):
        pass


class DummyListener:
    def __init__(self, responses):
        self.responses = list(responses)

    def listen(self):
        if self.responses:
            return self.responses.pop(0)
        return ListenResult(ActionCommand("NEW_GAME"), confirmed=True)

    def close(self):
        pass


def test_first_run_welcome_and_subsequent_prompt():
    leds = DummyLeds()
    lcd = DummyLcd()
    speaker = DummySpeaker()
    gantry = DummyGantry()

    listener = DummyListener([
        ListenResult(ActionCommand("NEW_GAME"), confirmed=True),
        ListenResult(ActionCommand("NEW_GAME"), confirmed=True),
    ])
    game = VoiceChessGame(leds, lcd, speaker, gantry, lambda *args: listener)

    assert game._first_run is True
    # First execution of _idle
    game._idle()
    assert game._first_run is False
    assert any("Welcome to Voice Chess" in s for s in speaker.spoken)

    # Second execution of _idle
    speaker.spoken.clear()
    game._idle()
    assert not any("Welcome to Voice Chess" in s for s in speaker.spoken)
    assert any("Would you like to start a new game" in s for s in speaker.spoken)


def test_check_notification_during_execute():
    leds = DummyLeds()
    lcd = DummyLcd()
    speaker = DummySpeaker()
    gantry = DummyGantry()
    listener = DummyListener([])
    game = VoiceChessGame(leds, lcd, speaker, gantry, lambda *args: listener)

    # Setup a move that gives check (e.g. Scholar's Mate check attempt or Qf7+)
    # 1. e4 e5 2. Qh5 g6 3. Qxe5+ (Check!)
    moves = [
        ("e2e4", PLAYER),
        ("e7e5", COMPUTER),
        ("d1h5", PLAYER),
        ("g7g6", COMPUTER),
    ]
    for m, side in moves:
        game._execute(chess.Move.from_uci(m), side)

    speaker.spoken.clear()
    leds.history.clear()

    # Move 3. Qxe5+ gives check to Black King on E8
    game._execute(chess.Move.from_uci("h5e5"), PLAYER)

    # Check announced on speaker and yellow LED on COMPUTER (in check)
    assert "Check!" in speaker.spoken
    assert ("set", COMPUTER, "yellow") in leds.history
    assert ("off", COMPUTER) in leds.history


def test_organize_board_command_availability():
    leds = DummyLeds()
    lcd = DummyLcd()
    speaker = DummySpeaker()
    gantry = DummyGantry()

    # Case 1: Organize requested BEFORE any game is completed
    listener = DummyListener([
        ListenResult(ActionCommand("ORGANIZE_BOARD"), confirmed=True),
        ListenResult(ActionCommand("NEW_GAME"), confirmed=True),
    ])
    game = VoiceChessGame(leds, lcd, speaker, gantry, lambda *args: listener)

    assert game._has_completed_game is False
    game._idle()
    assert any("only available after finishing a game" in s for s in speaker.spoken)
    assert len(gantry.executed) == 0

    # Case 2: Organize requested AFTER a game is completed
    game._has_completed_game = True
    speaker.spoken.clear()
    listener.responses = [
        ListenResult(ActionCommand("ORGANIZE_BOARD"), confirmed=True),
        ListenResult(ActionCommand("NEW_GAME"), confirmed=True),
    ]
    game._idle()
    assert any("Organizing board" in s for s in speaker.spoken)
    assert any("Board organized" in s for s in speaker.spoken)
    assert len(gantry.executed) > 0


def test_led_celebrate_method():
    leds = StatusLeds(simulate=True)
    calls = []

    def record_set(side, color):
        calls.append((side, color))

    leds.set = record_set
    # Run celebrate for 0.4 seconds at 5 Hz
    leds.celebrate(PLAYER, COMPUTER, seconds=0.4, hz=5.0)

    # Winner never receives "red"
    winner_colors = [c for s, c in calls if s == PLAYER and c != "off"]
    assert len(winner_colors) > 0
    assert "red" not in winner_colors

    # Loser receives "red"
    loser_colors = [c for s, c in calls if s == COMPUTER and c != "off"]
    assert len(loser_colors) > 0
    assert all(c == "red" for c in loser_colors)


def test_draw_blinks_both_yellow():
    leds = StatusLeds(simulate=True)
    calls = []

    def record_set(side, color):
        calls.append((side, color))

    leds.set = record_set
    leds.blink_both("yellow", seconds=0.4, hz=5.0)

    # Both player and computer received "yellow"
    player_colors = [c for s, c in calls if s == PLAYER and c != "off"]
    computer_colors = [c for s, c in calls if s == COMPUTER and c != "off"]
    assert len(player_colors) > 0
    assert all(c == "yellow" for c in player_colors)
    assert len(computer_colors) > 0
    assert all(c == "yellow" for c in computer_colors)

