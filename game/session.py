"""
game/session.py
===============
Fluxo da partida do VoiceChess (PvE: Jogador x Maquina | PvP: Jogador x Jogador)
com integracao do Gantry e do Sistema de Visao (CameraSystem).

Fluxo de Menu e Inicio de Partida
---------------------------------
1. IDLE:
   - Speaker: "Welcome to Voice Chess. To start a new game, just say the wake word
     Magnus, followed by the words Start new game."
   - LCD: inalterado ("=== VOICE CHESS ===", "SAY: MAGNUS", "START NEW GAME", "TO BEGIN PLAY").
2. CONFIGURE:
   - Seleciona o modo de jogo: "Select the game mode. Say Magnus P V P, or Magnus P V E."
   - Se escolher PvE:
     a) Seleciona dificuldade (1 a 10): "Select the difficulty level from 1 to 10. Say Magnus Level."
     b) Seleciona cor das pecas: "Select the piece color. Say Magnus White, or Magnus Black."
3. HOMING & VERIFICACAO INICIAL DA CAMERA:
   - O Gantry executa o Homing.
   - A camera captura uma foto do estado atual do tabuleiro.
   - Se as pecas estiverem nas posicoes iniciais corretas, a partida inicia;
     caso contrario, Speaker/LCD/LED avisam que as pecas nao estao nas posicoes
     corretas e instruem a dizer "MAGNUS VALIDATE" quando estiverem prontas.
"""

from __future__ import annotations

import logging
import time

import chess
import chess.engine

from vision.board_verifier import GameSettings, save_logical_board_snapshot
from voice.parser import ActionCommand, LevelCommand, MoveCommand

from . import config as cfg
from . import phrases
from .gantry_link import GantryError
from .planner import MovePlanner, PlanningError

log = logging.getLogger(__name__)

PLAYER, COMPUTER = "player", "computer"      # lados / LEDs (player=painel Brancas, computer=painel Pretas)
PLAYER_COLOR = chess.WHITE                   # cor padrao do painel "player"


class GameAbort(RuntimeError):
    """Erro que interrompe a partida atual (ex.: falha no Gantry)."""


class VoiceChessGame:
    def __init__(
        self,
        leds,
        lcd,
        speaker,
        gantry,
        listener_factory,
        stockfish_path: str | None = None,
        return_home: bool = True,
        camera=None,
        game_mode: str | None = None,
        white_control: str | None = None,
        black_control: str | None = None,
    ) -> None:
        self.leds, self.lcd, self.speaker, self.gantry = leds, lcd, speaker, gantry
        self.camera = camera
        self.planner = MovePlanner(return_home=return_home)

        settings = GameSettings.load()
        self.level = settings.level if settings.level else cfg.DEFAULT_LEVEL
        self.game_mode = (game_mode or settings.game_mode or "pve").lower()
        self.white_control = (white_control or settings.white_control or "voice").lower()
        self.black_control = (
            black_control
            or (settings.black_control if self.game_mode == "pvp" else "computer")
        ).lower()
        self.verify_gantry_moves = settings.verify_gantry_moves
        self.player_color = chess.BLACK if self.white_control == "computer" else chess.WHITE

        self._explicit_mode_overrides = (
            game_mode is not None or white_control is not None or black_control is not None
        )
        self._stockfish_path = stockfish_path or cfg.find_stockfish()
        self._engine: chess.engine.SimpleEngine | None = None
        self._gantry_connected = False
        self._last_confirm_lines: tuple[str, ...] = ()
        self._first_run = True
        self._has_completed_game = False
        self._active_side_led = PLAYER
        self.listener = listener_factory(
            self._on_listening, self._on_confirm, self.speaker.wait_idle
        )

    def _side_led(self, color: bool) -> str:
        """Retorna o identificador do LED ('player' p/ Brancas, 'computer' p/ Pretas)."""
        return PLAYER if color == chess.WHITE else COMPUTER

    def _control_for(self, color: bool) -> str:
        """Retorna 'voice', 'manual' ou 'computer' para o lado especificado."""
        if color == chess.WHITE:
            return self.white_control
        return self.black_control

    def _sync_settings_from_disk(self) -> None:
        """Recarrega as configuracoes salvas pela interface web."""
        if self._explicit_mode_overrides:
            return
        s = GameSettings.load()
        self.game_mode = s.game_mode
        self.level = s.level
        self.white_control = s.white_control
        self.black_control = s.black_control
        self.verify_gantry_moves = s.verify_gantry_moves

    def _persist_settings(self) -> None:
        s = GameSettings.load()
        s.game_mode = self.game_mode
        s.level = self.level
        s.white_control = self.white_control
        s.black_control = self.black_control
        s.verify_gantry_moves = self.verify_gantry_moves
        s.save()

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
                    if not self._verify_initial_board_setup():
                        continue
                    self._play()
                except GameAbort as exc:
                    self._show_abort(str(exc))
                finally:
                    self._stop_engine()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def _needs_gantry(self) -> bool:
        """Retorna True se pelo menos um lado usa o Gantry (voice ou computer)."""
        return (
            self._control_for(chess.WHITE) in ("voice", "computer")
            or self._control_for(chess.BLACK) in ("voice", "computer")
        )

    def close(self) -> None:
        self._stop_engine()
        if self.camera is not None:
            try:
                self.camera.shutdown()
            except Exception as exc:                          # noqa: BLE001
                log.warning("Erro ao encerrar camera: %s", exc)
        for obj in (self.listener, self.gantry, self.leds, self.lcd, self.speaker):
            try:
                obj.close()
            except Exception as exc:                          # noqa: BLE001
                log.warning("Erro ao encerrar %s: %s", type(obj).__name__, exc)

    # ------------------------------------------------------------------
    # Callbacks do listener
    # ------------------------------------------------------------------

    def _set_allowed_commands(self, predicate) -> None:
        """Define quais comandos de voz sao permitidos no estado atual."""
        self._allowed_predicate = predicate
        if hasattr(self.listener, "set_command_filter"):
            self.listener.set_command_filter(predicate)

    def _is_command_allowed(self, cmd) -> bool:
        pred = getattr(self, "_allowed_predicate", None)
        if pred is None:
            return True
        return bool(pred(cmd))

    def _on_listening(self, active: bool, context: str = "") -> None:
        led_side = self._active_side_led
        if active:
            self.leds.set(led_side, "blue")
            if context == "command":
                self.lcd.show(
                    "=== VOICE CHESS ===",
                    "LISTENING...",
                    "SPEAK YOUR COMMAND",
                    "",
                )
            elif context == "yes_no":
                self.lcd.show(
                    "GOOD GAME!",
                    "ORGANIZE BOARD?",
                    "SAY: YES OR NO",
                    "LISTENING: YES/NO",
                )
            elif context == "confirmation":
                if self._last_confirm_lines:
                    c0 = self._last_confirm_lines[0] if len(self._last_confirm_lines) > 0 else "CONFIRM MOVE?"
                    c1 = self._last_confirm_lines[1] if len(self._last_confirm_lines) > 1 else ""
                    self.lcd.show(c0, c1, "SAY: YES OR NO", "LISTENING: YES/NO")
                else:
                    self.lcd.show("CONFIRM?", "SAY: YES OR NO", "LISTENING: YES/NO", "")
        elif self.leds.state(led_side) == "blue":
            self.leds.off(led_side)

    def _command_not_recognized(self, side: str = PLAYER) -> None:
        """Chamado quando escuta a wake word mas nao reconhece o comando (ou comando bloqueado no estado)."""
        self.lcd.show(
            "NOT RECOGNIZED",
            "PLEASE TRY AGAIN",
            "SAY: MAGNUS",
            "THEN YOUR COMMAND",
        )
        self.leds.set(side, "red")
        self.speaker.say("Command not recognized. Please try again.", wait=True)
        self.leds.off(side)

    def _on_confirm(self, command) -> None:
        """
        Speaker repete o comando e o LCD o mostra, enquanto espera SIM/NAO.
        (Nota: os comandos de menu sao imediatos por padrao em parser.IMMEDIATE_ACTIONS,
        mas as frases de confirmacao abaixo permanecem prontas caso queira reativar).
        """
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
            lines = ("RESIGN GAME?", "END CURRENT GAME", "SAY: YES OR NO", "")
        elif isinstance(command, ActionCommand) and command.action == "MODE_PVE":
            speech = "Option 2, Player versus engine mode. Say yes or no."
            lines = ("CONFIRM MODE?", "2 - PVE (ENGINE)", "SAY: YES OR NO", "")
        elif isinstance(command, ActionCommand) and command.action == "MODE_PVP":
            speech = "Option 1, Player versus player mode. Say yes or no."
            lines = ("CONFIRM MODE?", "1 - PVP (2P)", "SAY: YES OR NO", "")
        elif isinstance(command, ActionCommand) and command.action == "COLOR_WHITE":
            speech = "White pieces. Say yes or no."
            lines = ("CONFIRM COLOR?", "COLOR: WHITE", "SAY: YES OR NO", "")
        elif isinstance(command, ActionCommand) and command.action == "COLOR_BLACK":
            speech = "Black pieces. Say yes or no."
            lines = ("CONFIRM COLOR?", "COLOR: BLACK", "SAY: YES OR NO", "")
        else:
            speech = f"{command}. Say yes or no."
            lines = ("CONFIRM?", f"{command}", "SAY: YES OR NO", "")
        self._last_confirm_lines = lines
        self.lcd.show(*lines)
        self.speaker.say(speech, wait=True)

    # ------------------------------------------------------------------
    # Etapas de Menu (IDLE -> CONFIGURE -> HOMING -> VERIFY BOARD)
    # ------------------------------------------------------------------

    def _idle(self) -> None:
        self._active_side_led = PLAYER
        self.leds.off()
        # Bloqueio: em IDLE, apenas START NEW GAME e permitido
        self._set_allowed_commands(
            lambda cmd: isinstance(cmd, ActionCommand) and cmd.action == "NEW_GAME"
        )
        save_logical_board_snapshot(self.planner.board, status_text="IDLE - Aguardando início")
        idle_lcd = (
            "=== VOICE CHESS ===",
            "SAY: MAGNUS",
            "START NEW GAME",
            "TO BEGIN PLAY",
        )
        self.lcd.show(*idle_lcd)
        if self._first_run:
            self.speaker.say(
                "Welcome to Voice Chess. To start a new game, just say the wake word Magnus, "
                "followed by the words Start new game.",
                wait=True,
            )
            self._first_run = False
        else:
            self.speaker.say(
                "If you want to start a new game, just say the wake word Magnus, "
                "followed by the words Start new game.",
                wait=True,
            )

        while True:
            result = self.listener.listen()
            if (
                result.confirmed
                and isinstance(result.command, ActionCommand)
                and result.command.action == "NEW_GAME"
            ):
                return
            # Se tentou usar comando bloqueado fora de hora (ex: MOVE, OPTION, LEVEL, RESIGN)
            if result.command is not None and not self._is_command_allowed(result.command):
                self._command_not_recognized()
                self.lcd.show(*idle_lcd)
                continue
            # --- Caso reative a confirmacao (YES/NO) para o comando START NEW GAME: ---
            if result.command and not result.confirmed:
                self.lcd.show(
                    "=== VOICE CHESS ===",
                    "CANCELLED",
                    "SAY: MAGNUS",
                    "START NEW GAME",
                )
                self.speaker.say("Cancelled.", wait=True)
            elif result.wake_detected and not result.command:
                self._command_not_recognized()
                self.lcd.show(*idle_lcd)

    def _configure(self) -> bool:
        """
        Fluxo de configuracao do menu:
        1. Seleciona modo de jogo (Magnus Option 1 = PvP ou Magnus Option 2 = PvE).
        2. Se PvE:
           - Seleciona nivel de dificuldade (1 a 10): "Magnus Level <1-10>"
           - Seleciona cor das pecas: "Magnus White" ou "Magnus Black"
        """
        self._sync_settings_from_disk()

        # ── Passo 1: Escolher Modo de Jogo (Option 1 = PvP ou Option 2 = PvE) ──
        mode = self._prompt_game_mode()
        if mode is None:
            return False

        self.game_mode = mode
        if self.game_mode == "pvp":
            if self.white_control == "computer":
                self.white_control = "voice"
            if self.black_control == "computer":
                self.black_control = "manual"
            self._persist_settings()
            return True

        # ── Passo 2 (PvE): Escolher Nivel de Dificuldade (1 a 10) ────────────
        level = self._prompt_difficulty_level()
        if level is None:
            return False
        self.level = level

        # ── Passo 3 (PvE): Escolher Cor das Pecas (White ou Black) ───────────
        color = self._prompt_piece_color()
        if color is None:
            return False

        self.player_color = color
        if color == chess.WHITE:
            self.white_control = "voice"
            self.black_control = "computer"
        else:
            self.white_control = "computer"
            self.black_control = "voice"

        self._persist_settings()
        return True

    def _prompt_game_mode(self) -> str | None:
        """Pergunta ao jogador o modo de jogo: Magnus Option 1 (PvP) ou Magnus Option 2 (PvE)."""
        # Bloqueio: apenas OPTION 1 (MODE_PVP) e OPTION 2 (MODE_PVE) sao permitidos
        self._set_allowed_commands(
            lambda cmd: isinstance(cmd, ActionCommand) and cmd.action in ("MODE_PVP", "MODE_PVE")
        )
        mode_lcd = (
            "SELECT GAME MODE",
            "1 - Player vs Player",
            "2 - Player vs Engine",
            "Ex: Magnus Option 2",
        )
        self.lcd.show(*mode_lcd)
        self.speaker.say("Select the game mode. Say Magnus Option 1, or Magnus Option 2.", wait=True)

        while True:
            result = self.listener.listen()
            # --- Caso reative confirmacao YES/NO no futuro: ---
            # if result.command and not result.confirmed:
            #     self.lcd.show(*mode_lcd)
            #     self.speaker.say("Cancelled.", wait=True)
            #     continue
            if result.command is not None and not self._is_command_allowed(result.command):
                self._command_not_recognized()
                self.lcd.show(*mode_lcd)
                continue
            if result.wake_detected and not result.command:
                self._command_not_recognized()
                self.lcd.show(*mode_lcd)
                continue
            if not result.confirmed:
                continue

            cmd = result.command
            if isinstance(cmd, ActionCommand):
                if cmd.action == "MODE_PVP":
                    return "pvp"
                if cmd.action == "MODE_PVE":
                    return "pve"

            self.lcd.show(*mode_lcd)
            self.speaker.say("Select the game mode. Say Magnus Option 1, or Magnus Option 2.", wait=True)

    def _prompt_difficulty_level(self) -> int | None:
        """Pergunta o nivel de dificuldade entre 1 e 10 (Magnus Level <1-10>)."""
        # Bloqueio: apenas LEVEL (1 a 10) e permitido nesta etapa
        self._set_allowed_commands(
            lambda cmd: isinstance(cmd, LevelCommand) and 1 <= cmd.level <= 10
        )
        lvl_lcd = (
            "SELECT DIFFICULTY",
            "Level 1 - Very Easy",
            "Level 10 -Super Hard",
            "Ex: Magnus Level 3",
        )
        self.lcd.show(*lvl_lcd)
        self.speaker.say(
            "Select the difficulty level from 1 to 10. Say Magnus Level.",
            wait=True,
        )

        while True:
            result = self.listener.listen()
            # --- Caso reative confirmacao YES/NO no futuro: ---
            # if result.command and not result.confirmed:
            #     self.lcd.show(*lvl_lcd)
            #     self.speaker.say("Cancelled.", wait=True)
            #     continue
            if result.command is not None and not self._is_command_allowed(result.command):
                self._command_not_recognized()
                self.lcd.show(*lvl_lcd)
                continue
            if result.wake_detected and not result.command:
                self._command_not_recognized()
                self.lcd.show(*lvl_lcd)
                continue
            if not result.confirmed:
                continue

            cmd = result.command
            if isinstance(cmd, LevelCommand) and 1 <= cmd.level <= 10:
                return cmd.level

            self.lcd.show(*lvl_lcd)
            self.speaker.say(
                "Select the difficulty level from 1 to 10. Say Magnus Level.",
                wait=True,
            )

    def _prompt_piece_color(self) -> bool | None:
        """Pergunta a cor das pecas do jogador no modo PvE (Magnus White ou Magnus Black)."""
        # Bloqueio: apenas WHITE e BLACK sao permitidos nesta etapa
        self._set_allowed_commands(
            lambda cmd: isinstance(cmd, ActionCommand) and cmd.action in ("COLOR_WHITE", "COLOR_BLACK")
        )
        color_lcd = (
            "SELECT PIECE COLOR",
            "White",
            "Black",
            "Ex: Magnus White",
        )
        self.lcd.show(*color_lcd)
        self.speaker.say(
            "Select the piece color. Say Magnus White, or Magnus Black.",
            wait=True,
        )

        while True:
            result = self.listener.listen()
            # --- Caso reative confirmacao YES/NO no futuro: ---
            # if result.command and not result.confirmed:
            #     self.lcd.show(*color_lcd)
            #     self.speaker.say("Cancelled.", wait=True)
            #     continue
            if result.command is not None and not self._is_command_allowed(result.command):
                self._command_not_recognized()
                self.lcd.show(*color_lcd)
                continue
            if result.wake_detected and not result.command:
                self._command_not_recognized()
                self.lcd.show(*color_lcd)
                continue
            if not result.confirmed:
                continue

            cmd = result.command
            if isinstance(cmd, ActionCommand):
                if cmd.action == "COLOR_WHITE":
                    return chess.WHITE
                if cmd.action == "COLOR_BLACK":
                    return chess.BLACK

            self.lcd.show(*color_lcd)
            self.speaker.say(
                "Select the piece color. Say Magnus White, or Magnus Black.",
                wait=True,
            )

    def _verify_initial_board_setup(self) -> bool:
        """
        Apos o Homing do Gantry, captura uma foto do tabuleiro para verificar se
        todas as pecas estao nas posicoes iniciais corretas.
        Se nao estiverem, avisa no Speaker/LCD/LED e aguarda ``MAGNUS VALIDATE``
        para checar novamente.
        """
        self.planner.reset()
        if self.camera is None or not self.camera.is_ready:
            return True

        # Bloqueio: durante a checagem inicial da camera, apenas MAGNUS VALIDATE e permitido
        self._set_allowed_commands(
            lambda cmd: isinstance(cmd, ActionCommand) and cmd.action == "VALIDATE_MOVE"
        )

        while True:
            self.lcd.show(
                "CHECKING BOARD...",
                "CAMERA CAPTURING",
                "PLEASE WAIT",
                "",
            )
            ver = self.camera.verify_gantry_position(
                self.planner.board,
                last_move=None,
                settle_seconds=0.5,
            )
            if ver.success and len(ver.mismatches) == 0:
                self.lcd.show(
                    "BOARD VERIFIED",
                    "ALL PIECES IN PLACE",
                    "STARTING GAME...",
                    "",
                )
                return True

            # Pecas fora do lugar -> avisa no speaker, LCD e LED
            err_lcd = (
                "PIECES OUT OF PLACE",
                "Check board setup",
                "When ready, say:",
                "MAGNUS VALIDATE",
            )
            self.lcd.show(*err_lcd)
            self.speaker.say(
                "The pieces are not in the correct positions. "
                "When they are in the correct positions, use the command Magnus Validate to check again.",
                wait=False,
            )
            self.leds.blink_both("red", cfg.BLINK_ERROR_SECONDS)
            self.speaker.wait_idle()

            # Aguarda o jogador dizer MAGNUS VALIDATE para verificar novamente
            while True:
                result = self.listener.listen()
                if result.command is not None and not self._is_command_allowed(result.command):
                    self._command_not_recognized()
                    self.lcd.show(*err_lcd)
                    continue
                if result.wake_detected and not result.command:
                    self._command_not_recognized()
                    self.lcd.show(*err_lcd)
                    continue
                if not result.confirmed:
                    continue
                cmd = result.command
                if isinstance(cmd, ActionCommand) and cmd.action == "VALIDATE_MOVE":
                    break
                self.lcd.show(*err_lcd)
                self.speaker.say(
                    "Please arrange the pieces in the starting positions and say Magnus Validate.",
                    wait=True,
                )

    def _connect_gantry(self) -> None:
        try:
            self.lcd.show("CONNECTING", "ESP32...")
            if not self._gantry_connected:
                self.gantry.connect()
                self._gantry_connected = True
            self.lcd.show("HOMING", "PLEASE WAIT")
            self.gantry.home()
            self.planner.robot_phys = self.planner.home_phys()
            self.planner.carrying = False
        except GantryError as exc:
            self._gantry_connected = False
            self.leds.blink(PLAYER, "red")
            raise GameAbort(f"Gantry: {exc}") from exc

    def _start_engine(self) -> None:
        if not self._stockfish_path:
            raise GameAbort("Stockfish nao encontrado! No Linux/Raspberry Pi instale com: sudo apt install stockfish")
        self._engine = chess.engine.SimpleEngine.popen_uci(self._stockfish_path)
        skill, elo = cfg.SKILL_MAP.get(self.level, cfg.SKILL_MAP[cfg.DEFAULT_LEVEL])
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
        if (
            self.game_mode == "pve"
            or self._control_for(chess.WHITE) == "computer"
            or self._control_for(chess.BLACK) == "computer"
        ):
            self._start_engine()

        if self.game_mode == "pvp":
            self.lcd.show(
                "PVP GAME STARTED",
                "PLAYER VS PLAYER",
                "WHITE MOVES FIRST",
                "",
            )
            self.speaker.say("Player versus Player game started.", wait=True)
        else:
            you_color = "BLACK" if self.player_color == chess.BLACK else "WHITE"
            self.lcd.show("GAME STARTED", f"YOU: {you_color}")
            self.speaker.say(f"Game started. You play {you_color.lower()}.", wait=True)

        save_logical_board_snapshot(self.planner.board, status_text=f"Em jogo ({self.game_mode.upper()})")

        board = self.planner.board
        while not board.is_game_over():
            turn_color = board.turn
            side_led = self._side_led(turn_color)
            self._active_side_led = side_led
            control = self._control_for(turn_color)

            if control == "computer":
                turn_label = "White" if turn_color == chess.WHITE else "Black"
                self.speaker.say(f"{turn_label}'s turn.", wait=True)
                move = self._computer_move()
                self._execute(move, side_led)
            elif control == "manual":
                ok = self._manual_player_turn(turn_color)
                if not ok:                                     # desistiu
                    self._handle_resignation(turn_color)
                    return
            else:
                move = self._player_move(turn_color)
                if move is None:                               # desistiu
                    self._handle_resignation(turn_color)
                    return
                # Se _player_move ja aplicou via MAGNUS VALIDATE, move vem com flag _already_applied
                if not getattr(move, "_already_applied", False):
                    self._execute(move, side_led)

            board = self.planner.board
            save_logical_board_snapshot(
                board,
                last_move_uci=board.peek().uci().upper() if board.move_stack else None,
                status_text=f"Turno: {'Brancas' if board.turn == chess.WHITE else 'Pretas'}",
            )

        self._has_completed_game = True
        speech, lines = phrases.game_result(board)
        self.lcd.show(*lines)
        save_logical_board_snapshot(board, status_text=f"Fim de jogo: {lines[0]}")
        celebrate_sec = getattr(cfg, "CELEBRATION_SECONDS", 8.0)
        if board.is_checkmate():
            winner_color = chess.BLACK if board.turn == chess.WHITE else chess.WHITE
            winner_side = PLAYER if winner_color == PLAYER_COLOR else COMPUTER
            loser_side = COMPUTER if winner_side == PLAYER else PLAYER
            self.speaker.say(speech, wait=False)
            self.leds.celebrate(winner_side, loser_side, seconds=celebrate_sec)
            self.speaker.wait_idle()
        else:
            # Empate / afogamento: speaker avisa e leds dos dois lados piscam em amarelo
            self.speaker.say(speech, wait=False)
            self.leds.blink_both("yellow", seconds=celebrate_sec)
            self.speaker.wait_idle()

        self._post_game_flow()

    def _handle_resignation(self, resigning_color: bool) -> None:
        self._has_completed_game = True
        if resigning_color == chess.WHITE:
            self.lcd.show("YOU RESIGNED" if self.game_mode == "pve" else "WHITE RESIGNED", "BLACK WINS")
            msg = "You resigned. Black wins." if self.game_mode == "pve" else "White resigned. Black wins."
            winner_side, loser_side = COMPUTER, PLAYER
        else:
            self.lcd.show("BLACK RESIGNED", "WHITE WINS")
            msg = "Black resigned. White wins."
            winner_side, loser_side = PLAYER, COMPUTER
        celebrate_sec = getattr(cfg, "CELEBRATION_SECONDS", 8.0)
        self.speaker.say(msg, wait=False)
        self.leds.celebrate(winner_side, loser_side, seconds=celebrate_sec)
        self.speaker.wait_idle()
        self._post_game_flow()

    def _post_game_flow(self) -> None:
        """
        Ao final da partida (apos a comemoracao):
        Pergunta se o jogador gostaria que o sistema organizasse o tabuleiro e aguarda YES/NO.
        """
        self._active_side_led = PLAYER
        prompt_lcd = (
            "GOOD GAME!",
            "ORGANIZE BOARD?",
            "SAY: YES OR NO",
            "",
        )
        self.lcd.show(*prompt_lcd)
        self.speaker.say(
            "It was a good game, would you like me to organize the board?",
            wait=True,
        )

        while True:
            if hasattr(self.listener, "listen_yes_no"):
                answer = self.listener.listen_yes_no()
            else:
                answer = False
            if answer is True:
                self._organize_board()
                return
            if answer is False:
                return
            self.lcd.show(*prompt_lcd)
            self.speaker.say(
                "Please say yes or no. Would you like me to organize the board?",
                wait=True,
            )

    def _is_active_game_command(self, cmd) -> bool:
        """Durante a partida ativa, apenas MOVE, VALIDATE e RESIGN sao permitidos."""
        return isinstance(cmd, MoveCommand) or (
            isinstance(cmd, ActionCommand) and cmd.action in ("RESIGN_GAME", "VALIDATE_MOVE")
        )

    def _manual_player_turn(self, turn_color: bool) -> bool:
        """
        Turno de um jogador em modo MANUAL:
        O jogador move a peca fisicamente no tabuleiro e diz ``MAGNUS VALIDATE``
        (ou fala o lance via ``MAGNUS MOVE``).
        Retorna True quando um lance valido foi confirmado e aplicado, ou False se desistiu.
        """
        self._set_allowed_commands(self._is_active_game_command)
        side_led = self._side_led(turn_color)
        color_name = "WHITE" if turn_color == chess.WHITE else "BLACK"
        turn_label = "White" if turn_color == chess.WHITE else "Black"
        self.lcd.show(
            f"{color_name} TURN",
            "SAY: MAGNUS MOVE",
            "OR MAGNUS VALIDATE",
            "",
        )
        self.speaker.say(f"{turn_label}'s turn.", wait=True)
        while True:
            self.lcd.show(
                f"{color_name} TURN",
                "SAY: MAGNUS MOVE",
                "OR MAGNUS VALIDATE",
                "",
            )
            result = self.listener.listen()
            if result.command is not None and not self._is_command_allowed(result.command):
                self._command_not_recognized(side=side_led)
                continue
            if result.command and not result.confirmed:
                self.lcd.show(
                    "=== VOICE CHESS ===",
                    "CANCELLED",
                    f"{color_name} TURN",
                    "SAY: MAGNUS MOVE",
                )
                self.speaker.say("Cancelled.", wait=False)
                time.sleep(1.0)
                continue
            if result.wake_detected and not result.command:
                self._command_not_recognized(side=side_led)
                continue
            if not result.confirmed:
                continue

            cmd = result.command
            if isinstance(cmd, ActionCommand) and cmd.action == "RESIGN_GAME":
                return False

            if isinstance(cmd, ActionCommand) and cmd.action == "VALIDATE_MOVE":
                if self.camera is None or not self.camera.is_ready:
                    self.lcd.show("CAMERA OFFLINE", "USE VOICE MOVE", "SAY: MAGNUS MOVE", "EX: E2 TO E4")
                    self.speaker.say("Camera is offline. Please announce your move by voice.", wait=True)
                    continue

                self.lcd.show("VALIDATING MOVE", "CAMERA CAPTURING...", "PLEASE WAIT", "")
                val = self.camera.validate_manual_turn(
                    self.planner.board,
                    is_phys_valid_fn=self.planner.is_phys_valid,
                )
                if val.valid and val.move is not None:
                    self.leds.set(side_led, "green")
                    self.lcd.show(*val.lcd_lines)
                    self.speaker.say(val.speech, wait=True)
                    self.planner.apply_manual_move(val.move)
                    self.leds.off(side_led)
                    self._check_alert_after_move()
                    return True
                else:
                    self.lcd.show(*val.lcd_lines)
                    self.speaker.say(val.speech, wait=False)
                    self.leds.blink(side_led, "red", cfg.BLINK_ERROR_SECONDS)
                    self.speaker.wait_idle()
                    continue

            # Tambem aceita se o jogador decidir falar o lance (MAGNUS MOVE E2 E4)
            if isinstance(cmd, MoveCommand):
                move = self._resolve_move(cmd)
                if move is None:
                    self._invalid_move(side=side_led)
                    continue
                if self._gantry_connected:
                    self._execute(move, side_led)
                else:
                    speech, lines = phrases.announce(self.planner.board, move)
                    self.lcd.show(*lines)
                    self.speaker.say(speech, wait=True)
                    self.planner.apply_manual_move(move)
                    self._check_alert_after_move()
                return True

            self._command_not_recognized(side=side_led)

    def _player_move(self, turn_color: bool = chess.WHITE) -> chess.Move | None:
        self._set_allowed_commands(self._is_active_game_command)
        side_led = self._side_led(turn_color)
        color_name = "WHITE" if turn_color == chess.WHITE else "BLACK"
        turn_label = "White" if turn_color == chess.WHITE else "Black"
        self.lcd.show(
            f"{color_name} TURN",
            "SAY: MAGNUS MOVE",
            "OR MAGNUS VALIDATE",
            "",
        )
        self.speaker.say(f"{turn_label}'s turn.", wait=True)
        while True:
            self.lcd.show(
                f"{color_name} TURN",
                "SAY: MAGNUS MOVE",
                "OR MAGNUS VALIDATE",
                "",
            )
            result = self.listener.listen()
            if result.command is not None and not self._is_command_allowed(result.command):
                self._command_not_recognized(side=side_led)
                continue
            if result.command and not result.confirmed:
                self.lcd.show(
                    "=== VOICE CHESS ===",
                    "CANCELLED",
                    f"{color_name} TURN",
                    "SAY: MAGNUS MOVE",
                )
                self.speaker.say("Cancelled.", wait=False)
                time.sleep(1.0)
                continue
            if result.wake_detected and not result.command:
                self._command_not_recognized(side=side_led)
                continue
            if not result.confirmed:
                continue
            cmd = result.command
            if isinstance(cmd, ActionCommand) and cmd.action == "RESIGN_GAME":
                return None

            # Mesmo no modo de voz, se o jogador mover na mao e disser MAGNUS VALIDATE, aceita!
            if isinstance(cmd, ActionCommand) and cmd.action == "VALIDATE_MOVE":
                if self.camera is not None and self.camera.is_ready:
                    self.lcd.show("VALIDATING MOVE", "CAMERA CAPTURING...", "PLEASE WAIT", "")
                    val = self.camera.validate_manual_turn(
                        self.planner.board,
                        is_phys_valid_fn=self.planner.is_phys_valid,
                    )
                    if val.valid and val.move is not None:
                        self.leds.set(side_led, "green")
                        self.lcd.show(*val.lcd_lines)
                        self.speaker.say(val.speech, wait=True)
                        self.planner.apply_manual_move(val.move)
                        self.leds.off(side_led)
                        self._check_alert_after_move()
                        setattr(val.move, "_already_applied", True)
                        return val.move
                    else:
                        self.lcd.show(*val.lcd_lines)
                        self.speaker.say(val.speech, wait=False)
                        self.leds.blink(side_led, "red", cfg.BLINK_ERROR_SECONDS)
                        self.speaker.wait_idle()
                        continue

            if not isinstance(cmd, MoveCommand):
                self._command_not_recognized(side=side_led)
                continue
            move = self._resolve_move(cmd)
            if move is None:
                self._invalid_move(side=side_led)
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

    def _invalid_move(self, side: str = PLAYER) -> None:
        self.lcd.show(
            "INVALID MOVE!",
            "NOT LEGAL ON BOARD",
            "TRY AGAIN",
            "SAY: MAGNUS MOVE",
        )
        self.speaker.say("Invalid move.", wait=False)
        self.leds.blink(side, "red", cfg.BLINK_ERROR_SECONDS)
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
        """
        Anuncia o lance, envia os passos ao ESP32, espera a conclusao e
        (quando a camera estiver ativa) captura uma foto para verificar a posicao.
        """
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

        # Verificacao visual pos-movimento do Gantry (Modo PvE e lances Gantry em PvP)
        if self.camera is not None and self.camera.is_ready and self.verify_gantry_moves:
            self.lcd.show("CAMERA CHECKING", "BOARD POSITION...", "PLEASE WAIT", "")
            ver = self.camera.verify_gantry_position(self.planner.board, last_move=move)
            if ver.success:
                self.lcd.show(*ver.lcd_lines)
                log.info("[Vision] %s", ver.message)
            else:
                log.warning("[Vision] %s", ver.message)
                self.lcd.show(*ver.lcd_lines)
                if ver.speech:
                    self.speaker.say(ver.speech, wait=False)
                self.leds.blink(side, "yellow", 2.0)
                self.speaker.wait_idle()

        self._check_alert_after_move()

    def _check_alert_after_move(self) -> None:
        """
        Cheque: se o movimento gerou uma posicao de cheque (e a partida nao acabou),
        o speaker fala 'Check!' e o LED do lado em cheque fica amarelo por 1,5 segundos.
        """
        board = self.planner.board
        if board.is_check() and not board.is_game_over():
            in_check_side = PLAYER if board.turn == PLAYER_COLOR else COMPUTER
            self.lcd.show("CHECK!", f"{in_check_side.upper()} IN CHECK")
            self.leds.set(in_check_side, "yellow")
            self.speaker.say("Check!", wait=False)
            time.sleep(1.5)
            self.leds.off(in_check_side)

    def _organize_board(self) -> None:
        """
        Organiza as pecas de ambos os jogadores de volta as casas iniciais,
        dividindo em partes (lotes de ate 2 pecas por envio) para manter a
        comunicacao ativa entre a Raspberry Pi e o ESP32 sem estourar o timeout.
        """
        self.lcd.show("ORGANIZING BOARD", "PLEASE WAIT...", "RESETTING ALL", "PIECES...")
        self.speaker.say("Organizing board. Please wait.", wait=False)
        try:
            self._connect_gantry()
            batches = self.planner.plan_reorganize_batches(max_carries_per_batch=2)
            self.leds.set(PLAYER, "green")
            self.leds.set(COMPUTER, "green")
            total = len(batches)
            for idx, batch in enumerate(batches, start=1):
                if total > 1:
                    self.lcd.show(
                        "ORGANIZING BOARD",
                        f"PART {idx} OF {total}",
                        "PLEASE WAIT...",
                        "",
                    )
                self.gantry.execute(batch.x, batch.y, batch.eletroima)
            self.leds.off()
            self.lcd.show("BOARD ORGANIZED", "ALL PIECES READY", "SAY: MAGNUS", "START NEW GAME")
            self.speaker.say("Board organized.", wait=True)
            time.sleep(1.5)
        except (GantryError, PlanningError) as exc:
            self.planner.rollback()
            self.leds.off()
            self.leds.blink(PLAYER, "red", cfg.BLINK_ERROR_SECONDS)
            self.leds.blink(COMPUTER, "red", cfg.BLINK_ERROR_SECONDS)
            self.lcd.show("ORGANIZATION", "FAILED", "SEE TERMINAL", "")
            self.speaker.say("Failed to organize board.", wait=True)
            log.error("Erro ao organizar tabuleiro: %s", exc)

    def _show_abort(self, message: str) -> None:
        log.error("Partida interrompida: %s", message)
        print(f"\n[ERRO] {message}")
        self.lcd.show("GAME ABORTED", "SEE TERMINAL")
        self.speaker.say("The game was aborted.")
        time.sleep(2)
