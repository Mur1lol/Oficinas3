"""
game/session.py
===============
Fluxo da partida Jogador (brancas) x Maquina (pretas) com o Gantry.

    IDLE        espera "MAGNUS START NEW GAME"
    CONFIGURE   escolhe a dificuldade por voz ("MAGNUS LEVEL <1-8>")
    CONNECT     conecta no ESP32 e faz o homing
    PLAY        alterna: jogador (voz) / maquina (Stockfish); todo lance vai ao Gantry
    END         resultado no speaker + LCD, volta ao IDLE

Regras de LED
    - AZUL    : sempre que o usuario diz MAGNUS, enquanto escuta comando/confirmacao
                (LED do lado do jogador).
    - VERDE   : lance valido; no LED do lado de quem joga, ate o ESP32 responder "ok".
    - VERMELHO: lance invalido; pisca 3 s no LED do lado do jogador.
"""

from __future__ import annotations

import logging
import time

import chess
import chess.engine

from voice.parser import ActionCommand, LevelCommand, MoveCommand

from . import config as cfg
from . import phrases
from .gantry_link import GantryError
from .planner import MovePlanner, PlanningError

log = logging.getLogger(__name__)

PLAYER, COMPUTER = "player", "computer"      # lados / LEDs
PLAYER_COLOR = chess.WHITE                   # por enquanto o jogador joga de brancas


class GameAbort(RuntimeError):
    """Erro que interrompe a partida atual (ex.: falha no Gantry)."""


class VoiceChessGame:
    def __init__(self, leds, lcd, speaker, gantry, listener_factory,
                 stockfish_path: str | None = None) -> None:
        self.leds, self.lcd, self.speaker, self.gantry = leds, lcd, speaker, gantry
        self.planner = MovePlanner()
        self.level = cfg.DEFAULT_LEVEL
        self._stockfish_path = stockfish_path or cfg.find_stockfish()
        self._engine: chess.engine.SimpleEngine | None = None
        self._gantry_connected = False
        self._last_confirm_lines: tuple[str, ...] = ()
        self.listener = listener_factory(self._on_listening, self._on_confirm,
                                         self.speaker.wait_idle)

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            while True:
                self._idle()
                try:
                    if not self._configure():
                        continue
                    self._connect_gantry()
                    self._play()
                except GameAbort as exc:
                    self._show_abort(str(exc))
                finally:
                    self._stop_engine()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        self._stop_engine()
        for obj in (self.listener, self.gantry, self.leds, self.lcd, self.speaker):
            try:
                obj.close()
            except Exception as exc:                          # noqa: BLE001
                log.warning("Erro ao encerrar %s: %s", type(obj).__name__, exc)

    # ------------------------------------------------------------------
    # Callbacks do listener
    # ------------------------------------------------------------------

    def _on_listening(self, active: bool, context: str = "") -> None:
        if active:
            self.leds.set(PLAYER, "blue")
            if context == "command":
                self.lcd.show(
                    "=== VOICE CHESS ===",
                    "OUVINDO COMANDO...",
                    "DIGA SEU LANCE",
                    "ex: Move E2 to E4",
                )
            elif context == "confirmation":
                if self._last_confirm_lines:
                    c0 = self._last_confirm_lines[0] if len(self._last_confirm_lines) > 0 else "CONFIRM MOVE?"
                    c1 = self._last_confirm_lines[1] if len(self._last_confirm_lines) > 1 else ""
                    self.lcd.show(c0, c1, "SAY: YES OR NO", "OUVINDO: YES / NO")
                else:
                    self.lcd.show("CONFIRM?", "SAY: YES OR NO", "OUVINDO: YES / NO", "")
        elif self.leds.state(PLAYER) == "blue":
            self.leds.off(PLAYER)

    def _on_confirm(self, command) -> None:
        """Speaker repete o comando e o LCD o mostra, enquanto espera SIM/NAO."""
        if isinstance(command, MoveCommand):
            speech, lines = phrases.confirm_move(self.planner.board,
                                                 command.from_square, command.to_square)
        elif isinstance(command, LevelCommand):
            speech = f"level {command.level}. Say yes or no."
            lines = ("CONFIRM LEVEL?", f"LEVEL {command.level}", "SAY: YES OR NO", "")
        elif isinstance(command, ActionCommand) and command.action == "NEW_GAME":
            speech = "start new game. Say yes or no."
            lines = ("START NEW GAME?", "=== VOICE CHESS ===", "SAY: YES OR NO", "")
        elif isinstance(command, ActionCommand) and command.action == "RESIGN_GAME":
            speech = "resign game. Say yes or no."
            lines = ("RESIGN GAME?", "BLACK WINS", "SAY: YES OR NO", "")
        else:
            speech = f"{command}. Say yes or no."
            lines = ("CONFIRM?", f"{command}", "SAY: YES OR NO", "")
        self._last_confirm_lines = lines
        self.lcd.show(*lines)
        self.speaker.say(speech, wait=True)

    # ------------------------------------------------------------------
    # Etapas
    # ------------------------------------------------------------------

    def _idle(self) -> None:
        self.leds.off()
        self.lcd.show(
            "=== VOICE CHESS ===",
            "SAY: MAGNUS",
            "START NEW GAME",
            "TO BEGIN PLAY",
        )
        while True:
            result = self.listener.listen()
            if (result.confirmed and isinstance(result.command, ActionCommand)
                    and result.command.action == "NEW_GAME"):
                return
            if result.command and not result.confirmed:
                self.lcd.show(
                    "=== VOICE CHESS ===",
                    "CANCELLED",
                    "SAY: MAGNUS",
                    "START NEW GAME",
                )

    def _configure(self) -> bool:
        """Escolhe a dificuldade por voz. Pode dizer o nível ou apenas iniciar com o padrão."""
        self.lcd.show(
            "CHOOSE LEVEL",
            f"DEFAULT: LEVEL {self.level}",
            "SAY: MAGNUS LEVEL 1-8",
            "OR MAGNUS START",
        )
        self.speaker.say(f"New game. Default difficulty is level {self.level}. "
                         f"Say Magnus level, or say Magnus start.")
        while True:
            result = self.listener.listen()
            if not result.confirmed:
                continue
            cmd = result.command
            if isinstance(cmd, LevelCommand):
                self.level = cmd.level
                self.lcd.show("LEVEL SELECTED", f"LEVEL {self.level}", "STARTING...", "")
                self.speaker.say(f"Level {self.level} selected.")
                return True
            if isinstance(cmd, ActionCommand) and cmd.action == "NEW_GAME":
                self.lcd.show("LEVEL SELECTED", f"LEVEL {self.level}", "STARTING...", "")
                self.speaker.say(f"Starting with level {self.level}.")
                return True
            if isinstance(cmd, ActionCommand) and cmd.action == "RESIGN_GAME":
                return False
            self.speaker.say("Please choose a level from one to eight, or say start.")

    def _connect_gantry(self) -> None:
        try:
            self.lcd.show("CONNECTING", "ESP32...")
            if not self._gantry_connected:
                self.gantry.connect()
                self._gantry_connected = True
            self.lcd.show("HOMING", "PLEASE WAIT")
            self.gantry.home()
        except GantryError as exc:
            self._gantry_connected = False
            self.leds.blink(PLAYER, "red")
            raise GameAbort(f"Gantry: {exc}") from exc

    def _start_engine(self) -> None:
        if not self._stockfish_path:
            raise GameAbort("Stockfish nao encontrado! No Linux/Raspberry Pi instale com: sudo apt install stockfish")
        self._engine = chess.engine.SimpleEngine.popen_uci(self._stockfish_path)
        skill, elo = cfg.SKILL_MAP[self.level]
        try:
            self._engine.configure({"Skill Level": skill})
            if elo:
                self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            else:
                self._engine.configure({"UCI_LimitStrength": False})
        except Exception as exc:                              # noqa: BLE001
            log.warning("Nao foi possivel ajustar o nivel do Stockfish: %s", exc)

    def _stop_engine(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:                                 # noqa: BLE001
                pass
            self._engine = None

    # ------------------------------------------------------------------
    # Partida
    # ------------------------------------------------------------------

    def _play(self) -> None:
        self.planner.reset()
        self._start_engine()
        self.lcd.show("GAME STARTED", "YOU: WHITE")
        self.speaker.say("Game started. You play white.")

        board = self.planner.board
        while not board.is_game_over():
            if board.turn == PLAYER_COLOR:
                move = self._player_move()
                if move is None:                               # desistiu
                    self.lcd.show("YOU RESIGNED", "BLACK WINS")
                    self.speaker.say("You resigned. Black wins.")
                    return
                self._execute(move, PLAYER)
            else:
                self._execute(self._computer_move(), COMPUTER)
            board = self.planner.board

        speech, lines = phrases.game_result(board)
        self.leds.off()
        self.lcd.show(*lines)
        self.speaker.say(speech)
        time.sleep(3)

    def _player_move(self) -> chess.Move | None:
        """Espera MAGNUS MOVE [FROM] [TO] confirmado. None = jogador desistiu."""
        while True:
            self.lcd.show(
                "YOUR TURN (WHITE)",
                "SAY: MAGNUS",
                "THEN YOUR MOVE",
                "(ex: Move E2 to E4)"
            )
            result = self.listener.listen()
            if result.command and not result.confirmed:
                self.lcd.show(
                    "MOVE CANCELLED",
                    "YOUR TURN (WHITE)",
                    "SAY: MAGNUS MOVE",
                    "TRY AGAIN"
                )
                self.speaker.say("Move cancelled.", wait=False)
                time.sleep(1.5)
                continue
            if not result.confirmed:
                continue
            cmd = result.command
            if isinstance(cmd, ActionCommand) and cmd.action == "RESIGN_GAME":
                return None
            if not isinstance(cmd, MoveCommand):
                continue
            move = self._resolve_move(cmd)
            if move is None:
                self._invalid_move()
                continue
            return move

    def _resolve_move(self, cmd: MoveCommand) -> chess.Move | None:
        """Converte o comando de voz num lance legal e fisicamente executavel."""
        board = self.planner.board
        try:
            base = chess.Move.from_uci(f"{cmd.from_square}{cmd.to_square}".lower())
        except ValueError:
            return None
        piece = board.piece_at(base.from_square)
        if piece and piece.piece_type == chess.PAWN and chess.square_rank(base.to_square) in (0, 7):
            candidates = [chess.Move(base.from_square, base.to_square, promotion=p)
                          for p in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)]
        else:
            candidates = [base]
        for move in candidates:
            if move in board.legal_moves and self.planner.is_phys_valid(move):
                return move
        return None

    def _invalid_move(self) -> None:
        self.lcd.show(
            "INVALID MOVE!",
            "NOT LEGAL ON BOARD",
            "TRY AGAIN",
            "SAY: MAGNUS MOVE"
        )
        self.speaker.say("Invalid move.", wait=False)
        self.leds.blink(PLAYER, "red", cfg.BLINK_ERROR_SECONDS)
        self.speaker.wait_idle()

    def _computer_move(self) -> chess.Move:
        self.lcd.show("COMPUTER", "THINKING...")
        legal = self.planner.legal_physical_moves()
        result = self._engine.analyse(self.planner.board,
                                      chess.engine.Limit(time=cfg.ENGINE_THINK_SECONDS),
                                      root_moves=legal)
        pv = result.get("pv")
        return pv[0] if pv else legal[0]

    def _execute(self, move: chess.Move, side: str) -> None:
        """Anuncia o lance, envia os passos ao ESP32 e espera a conclusao."""
        speech, lines = phrases.announce(self.planner.board, move)
        self.lcd.show(*lines)
        self.speaker.say(speech, wait=False)
        try:
            plan = self.planner.plan(move)
        except PlanningError as exc:
            raise GameAbort(f"Planejamento: {exc}") from exc

        self.leds.set(side, "green")                           # verde ate o ESP32 responder
        try:
            self.gantry.execute(plan.x, plan.y, plan.eletroima)
        except GantryError as exc:
            self.planner.rollback()
            self.leds.off(side)
            self.leds.blink(side, "red", cfg.BLINK_ERROR_SECONDS)
            raise GameAbort(f"Gantry: {exc}") from exc
        self.leds.off(side)

    def _show_abort(self, message: str) -> None:
        log.error("Partida interrompida: %s", message)
        print(f"\n[ERRO] {message}")
        self.lcd.show("GAME ABORTED", "SEE TERMINAL")
        self.speaker.say("The game was aborted.")
        time.sleep(2)
