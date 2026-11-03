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
    def __init__(self, responses, yes_no_responses=None):
        self.responses = list(responses)
        self.yes_no_responses = list(yes_no_responses or [])
        self._filter = None

    def set_command_filter(self, fn):
        self._filter = fn

    def listen(self):
        if self.responses:
            return self.responses.pop(0)
        return ListenResult(ActionCommand("NEW_GAME"), confirmed=True)

    def listen_yes_no(self):
        if self.yes_no_responses:
            return self.yes_no_responses.pop(0)
        return False

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

    # Second execution of _idle (after a game)
    speaker.spoken.clear()
    game._idle()
    assert not any("Welcome to Voice Chess" in s for s in speaker.spoken)
    assert any(
        "If you want to start a new game, just say the wake word Magnus, followed by the words Start new game." in s
        for s in speaker.spoken
    )


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


def test_post_game_organize_flow_and_command_locking():
    leds = DummyLeds()
    lcd = DummyLcd()
    speaker = DummySpeaker()
    gantry = DummyGantry()

    # 1. Test command locking in _idle: MOVE, OPTION, RESIGN are rejected; only NEW_GAME succeeds
    listener = DummyListener(
        responses=[
            ListenResult(MoveCommand("E2", "E4"), confirmed=True),
            ListenResult(ActionCommand("MODE_PVE"), confirmed=True),
            ListenResult(ActionCommand("RESIGN_GAME"), confirmed=True),
            ListenResult(ActionCommand("NEW_GAME"), confirmed=True),
        ],
        yes_no_responses=[True],
    )
    game = VoiceChessGame(leds, lcd, speaker, gantry, lambda *args: listener)
    # Prepare a board with moves so _organize_board has work to do
    game.planner.plan(chess.Move.from_uci("e2e4"))
    gantry.executed.clear()

    game._idle()
    assert speaker.spoken.count("Command not recognized. Please try again.") == 3

    # 2. Test post-game flow asking to organize board (YES -> organizes board)
    speaker.spoken.clear()
    game._post_game_flow()
    assert any(
        "It was a good game, would you like me to organize the board?" in s
        for s in speaker.spoken
    )
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



# ---------------------------------------------------------------------------
# Vision Board Verifier & PvE / PvP Game Flow Tests
# ---------------------------------------------------------------------------

from vision.board_state import BoardState, SquareState
from vision.board_verifier import (
    ManualValidationResult,
    PositionVerificationResult,
    detect_and_validate_manual_move,
    verify_board_position,
)
from voice.parser import parse


def _board_state_from_chess(board: chess.Board) -> BoardState:
    """Helper: build a physical BoardState matching a python-chess Board."""
    name_map = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king",
    }
    squares = {}
    for sq_idx in chess.SQUARES:
        sq_name = chess.square_name(sq_idx).upper()
        p = board.piece_at(sq_idx)
        if p is None:
            squares[sq_name] = SquareState(occupied=False, color=None, piece=None, confidence=0.95)
        else:
            c_str = "white" if p.color == chess.WHITE else "black"
            squares[sq_name] = SquareState(
                occupied=True, color=c_str, piece=name_map[p.piece_type], confidence=0.95
            )
    return BoardState(
        board=squares,
        cemeteries={"white": [], "black": []},
        timestamp=0.0,
        overall_confidence=0.95,
    )


def test_parse_validate_and_mode_commands():
    assert parse("validate") == (ActionCommand("VALIDATE_MOVE"), None)
    assert parse("validate move") == (ActionCommand("VALIDATE_MOVE"), None)
    assert parse("player versus player") == (ActionCommand("MODE_PVP"), None)
    assert parse("pvp") == (ActionCommand("MODE_PVP"), None)
    assert parse("player versus computer") == (ActionCommand("MODE_PVE"), None)
    assert parse("player versus engine") == (ActionCommand("MODE_PVE"), None)
    assert parse("pve") == (ActionCommand("MODE_PVE"), None)
    assert parse("manual mode") == (ActionCommand("MODE_MANUAL"), None)
    assert parse("voice mode") == (ActionCommand("MODE_VOICE"), None)
    assert parse("white") == (ActionCommand("COLOR_WHITE"), None)
    assert parse("black") == (ActionCommand("COLOR_BLACK"), None)
    from voice.parser import LevelCommand
    assert parse("level ten") == (LevelCommand(10), None)
    assert parse("level 10") == (LevelCommand(10), None)


def test_verify_board_position_post_gantry():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    board.push(move)

    # 1. Physical board matches logical board after e2e4
    state_ok = _board_state_from_chess(board)
    res_ok = verify_board_position(board, state_ok, last_move=move)
    assert res_ok.success is True
    assert res_ok.origin_ok is True
    assert res_ok.dest_ok is True
    assert len(res_ok.mismatches) == 0

    # 2. Physical piece failed to reach E4 (still on E2)
    state_bad = _board_state_from_chess(chess.Board())
    res_bad = verify_board_position(board, state_bad, last_move=move)
    assert res_bad.success is False
    assert res_bad.origin_ok is False
    assert res_bad.dest_ok is False
    assert len(res_bad.mismatches) >= 2


def test_detect_and_validate_manual_move_cases():
    board = chess.Board()

    # Case 1: Valid quiet move e2 -> e4
    after_e4 = board.copy()
    after_e4.push_uci("e2e4")
    res = detect_and_validate_manual_move(board, _board_state_from_chess(after_e4))
    assert res.valid is True
    assert res.move_uci.lower() == "e2e4"
    assert "Valid move" in res.speech or "Move valid" in res.speech
    assert "pawn" in res.speech.lower()
    assert "E 2 to E 4" in res.speech

    # Case 2: Valid capture (1. e4 d5 2. exd5)
    b_cap = chess.Board()
    b_cap.push_uci("e2e4")
    b_cap.push_uci("d7d5")
    after_cap = b_cap.copy()
    after_cap.push_uci("e4d5")
    res_cap = detect_and_validate_manual_move(b_cap, _board_state_from_chess(after_cap))
    assert res_cap.valid is True
    assert res_cap.move_uci.lower() == "e4d5"
    assert "captur" in res_cap.speech.lower()

    # Case 3: Illegal move (e2 -> e5 on initial board)
    state_illegal = _board_state_from_chess(board)
    state_illegal.board["E2"] = SquareState(occupied=False, color=None, piece=None, confidence=0.95)
    state_illegal.board["E5"] = SquareState(occupied=True, color="white", piece="pawn", confidence=0.95)
    res_ill = detect_and_validate_manual_move(board, state_illegal)
    assert res_ill.valid is False
    assert res_ill.error_code == "ILLEGAL_MOVE"
    assert "Invalid move" in res_ill.speech

    # Case 4: Wrong color moved (Black moves on White's turn)
    state_wrong_turn = _board_state_from_chess(board)
    state_wrong_turn.board["E7"] = SquareState(occupied=False, color=None, piece=None, confidence=0.95)
    state_wrong_turn.board["E5"] = SquareState(occupied=True, color="black", piece="pawn", confidence=0.95)
    res_wt = detect_and_validate_manual_move(board, state_wrong_turn)
    assert res_wt.valid is False
    assert res_wt.error_code == "WRONG_SIDE_MOVED"

    # Case 5: No move detected
    res_none = detect_and_validate_manual_move(board, _board_state_from_chess(board))
    assert res_none.valid is False
    assert res_none.error_code in ("NO_MOVE", "NO_MOVE_DETECTED")


def test_pve_post_gantry_camera_verification_in_session():
    leds = DummyLeds()
    lcd = DummyLcd()
    speaker = DummySpeaker()
    gantry = DummyGantry()
    listener = DummyListener([])

    mock_camera = MagicMock()
    mock_camera.is_ready = True
    mock_camera.verify_gantry_position.return_value = PositionVerificationResult(
        success=False,
        move_uci="e2e4",
        origin_ok=True,
        dest_ok=False,
        mismatches=[{"square": "E4", "expected": "white", "detected": "empty", "piece": None}],
        message="Piece did not arrive at E4.",
        speech="Warning: piece did not arrive at E 4. Please check the board.",
        lcd_lines=("CAM ERR: E4", "CHECK PIECE"),
        confidence=0.95,
    )

    game = VoiceChessGame(
        leds, lcd, speaker, gantry, lambda *args: listener,
        camera=mock_camera, game_mode="pve",
    )
    game._execute(chess.Move.from_uci("e2e4"), PLAYER)

    assert mock_camera.verify_gantry_position.called
    assert any("Warning: piece did not arrive" in s for s in speaker.spoken)


def test_pvp_manual_player_turn_with_magnus_validate():
    leds = DummyLeds()
    lcd = DummyLcd()
    speaker = DummySpeaker()
    gantry = DummyGantry()

    # Player first says VALIDATE when move is invalid, then says VALIDATE when move is valid (e2e4)
    listener = DummyListener([
        ListenResult(ActionCommand("VALIDATE_MOVE"), confirmed=True),
        ListenResult(ActionCommand("VALIDATE_MOVE"), confirmed=True),
    ])

    invalid_res = ManualValidationResult(
        valid=False,
        move=None,
        move_uci="e2e5",
        san=None,
        from_square="E2",
        to_square="E5",
        piece_name="pawn",
        captured_piece=None,
        error_code="ILLEGAL_MOVE",
        message="Invalid move.",
        speech="Invalid move. White Pawn cannot move from E 2 to E 5.",
        lcd_lines=("INVALID MOVE", "E2->E5 ILLEGAL"),
    )
    valid_move = chess.Move.from_uci("e2e4")
    valid_res = ManualValidationResult(
        valid=True,
        move=valid_move,
        move_uci="e2e4",
        san="e4",
        from_square="E2",
        to_square="E4",
        piece_name="pawn",
        captured_piece=None,
        error_code=None,
        message="Valid move: e4",
        speech="Move valid. White Pawn from E 2 to E 4.",
        lcd_lines=("VALID: e4", "PAWN E2->E4"),
    )

    mock_camera = MagicMock()
    mock_camera.is_ready = True
    mock_camera.validate_manual_turn.side_effect = [invalid_res, valid_res]

    game = VoiceChessGame(
        leds, lcd, speaker, gantry, lambda *args: listener,
        camera=mock_camera, game_mode="pvp",
        white_control="manual", black_control="manual",
    )

    # Execute White's manual turn
    cont = game._manual_player_turn(chess.WHITE)
    assert cont is True
    assert mock_camera.validate_manual_turn.call_count == 2
    assert any("Invalid move" in s for s in speaker.spoken)
    assert any("Move valid. White Pawn from E 2 to E 4." in s for s in speaker.spoken)
    # Logical board and physical grid must be updated with e2e4 without running the Gantry
    assert game.planner.board.peek() == valid_move
    assert len(gantry.executed) == 0


def test_configure_pve_menu_flow(monkeypatch):
    from voice.parser import LevelCommand

    leds = DummyLeds()
    lcd = DummyLcd()
    speaker = DummySpeaker()
    gantry = DummyGantry()
    listener = DummyListener([
        ListenResult(ActionCommand("MODE_PVE"), confirmed=True),
        ListenResult(LevelCommand(10), confirmed=True),
        ListenResult(ActionCommand("COLOR_BLACK"), confirmed=True),
    ])

    game = VoiceChessGame(
        leds, lcd, speaker, gantry, lambda *args: listener,
    )
    monkeypatch.setattr(game, "_sync_settings_from_disk", lambda: None)
    monkeypatch.setattr(game, "_persist_settings", lambda: None)

    assert game._configure() is True

    assert game.game_mode == "pve"
    assert game.level == 10
    assert game.player_color == chess.BLACK
    assert game.black_control == "voice"
    assert game.white_control == "computer"
    assert any("Select the game mode" in s for s in speaker.spoken)
    assert any("Select the difficulty level from 1 to 10" in s for s in speaker.spoken)
    assert any("Select the piece color" in s for s in speaker.spoken)


def test_verify_initial_board_setup_requires_magnus_validate():
    leds = DummyLeds()
    lcd = DummyLcd()
    speaker = DummySpeaker()
    gantry = DummyGantry()

    listener = DummyListener([
        ListenResult(ActionCommand("VALIDATE_MOVE"), confirmed=True),
    ])

    bad_check = PositionVerificationResult(
        success=False,
        move_uci=None,
        origin_ok=True,
        dest_ok=True,
        mismatches=[{"square": "E2", "expected": "white", "detected": "empty", "piece": None}],
        message="Mismatch on E2",
        speech="Mismatch on E2",
        lcd_lines=("ERR E2", "CHECK BOARD"),
        confidence=0.95,
    )
    ok_check = PositionVerificationResult(
        success=True,
        move_uci=None,
        origin_ok=True,
        dest_ok=True,
        mismatches=[],
        message="Board ok",
        speech="",
        lcd_lines=("BOARD VERIFIED", "POSITION OK"),
        confidence=0.98,
    )

    mock_camera = MagicMock()
    mock_camera.is_ready = True
    mock_camera.verify_gantry_position.side_effect = [bad_check, ok_check]

    game = VoiceChessGame(
        leds, lcd, speaker, gantry, lambda *args: listener,
        camera=mock_camera, game_mode="pve",
    )
    assert game._verify_initial_board_setup() is True

    assert mock_camera.verify_gantry_position.call_count == 2
    assert any("The pieces are not in the correct positions" in s for s in speaker.spoken)
    assert any("Magnus Validate" in s for s in speaker.spoken)
    assert ("blink_both", "red", 3.0) in leds.history
    assert lcd.lines[0] == "BOARD VERIFIED"

