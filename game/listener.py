"""
game/listener.py
================
Fonte de comandos do jogador, sempre no fluxo:

    MAGNUS (wake word) -> comando -> [speaker repete + LCD] -> SIM/NAO

Callbacks fornecidos pelo jogo:
    on_listening(bool) : True enquanto escuta o comando/confirmacao (LED AZUL)
    on_confirm(cmd)    : fala o comando no speaker e mostra no LCD (bloqueante)
    before_listen()    : espera o speaker terminar antes de abrir o microfone

``VoiceListener`` usa o microfone (voice.VoiceCommandPipeline).
``TextListener``  digita os comandos no terminal (teste sem microfone).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from voice.parser import normalize, parse, parse_confirmation
from voice.validator import validate


@dataclass(frozen=True)
class ListenResult:
    command: object | None = None      # MoveCommand | ActionCommand | LevelCommand
    confirmed: bool = False
    reason: str | None = None
    wake_detected: bool = field(default=False, compare=False)


class VoiceListener:
    def __init__(self, voice_config, on_listening: Callable[[bool], None],
                 on_confirm: Callable[[object], None],
                 before_listen: Callable[[], None] = lambda: None) -> None:
        from voice.audio_capture import AudioStream
        from voice.config import load_settings
        from voice.pipeline import PipelineState, VoiceCommandPipeline

        self._State = PipelineState
        self._on_listening = on_listening
        self._on_confirm = on_confirm
        self._before_listen = before_listen
        from pathlib import Path
        settings = load_settings(Path(voice_config) if voice_config else None)
        self._pipeline = VoiceCommandPipeline(settings, on_state=self._on_state)
        self._ctx = AudioStream(settings)
        self._stream = self._ctx.__enter__()

    def _on_state(self, state, command) -> None:
        S = self._State
        if state == S.COMMAND_LISTEN:
            self._on_listening(True, "command")          # ouviu MAGNUS -> LED azul + LCD escutando
        elif state == S.CONFIRMATION_LISTEN:
            self._on_listening(False, "")
            self._on_confirm(command)                    # speaker repete lance + LCD
            self._stream.discard_pending()               # descarta o eco do speaker
            self._on_listening(True, "confirmation")     # LED azul volta a ligar + LCD aguarda yes/no
        elif state in (S.WAKE_WORD_LISTEN, S.EMIT, S.ERROR, S.IDLE):
            self._on_listening(False, "")

    def listen(self) -> ListenResult:
        self._before_listen()
        result = self._pipeline.listen_once(self._stream)
        return ListenResult(result.command, result.confirmed, result.reason, result.wake_detected)

    def close(self) -> None:
        self._ctx.__exit__(None, None, None)


class TextListener:
    """Mesmo fluxo, digitado: 'magnus move e2 e4' e depois 'yes'/'no'."""

    def __init__(self, on_listening, on_confirm, before_listen=lambda: None) -> None:
        self._on_listening = on_listening
        self._on_confirm = on_confirm
        self._before_listen = before_listen

    def listen(self) -> ListenResult:
        self._before_listen()
        try:
            text = input("\n[voz] diga o comando (ex.: 'magnus move e2 e4'): ")
        except EOFError:
            raise KeyboardInterrupt
        text = normalize(text)
        if not text.startswith("magnus"):
            return ListenResult(reason="Wake word not detected", wake_detected=False)
        self._on_listening(True, "command")
        command, reason = parse(text[len("magnus"):], "en-US")
        checked = validate(command, reason)
        if not checked["valid"]:
            self._on_listening(False, "")
            return ListenResult(command, reason=checked["reason"], wake_detected=True)
        self._on_listening(False, "")
        self._on_confirm(command)
        self._on_listening(True, "confirmation")
        answer = input("[voz] confirma? (yes/no): ")
        self._on_listening(False, "")
        confirmed = parse_confirmation(answer, "en-US")
        if confirmed is True:
            return ListenResult(command, confirmed=True, wake_detected=True)
        return ListenResult(command, reason="Command cancelled" if confirmed is False
                            else "Confirmation missing or ambiguous", wake_detected=True)

    def close(self) -> None:
        pass
