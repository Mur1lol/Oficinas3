from dataclasses import dataclass, field
from enum import StrEnum, auto

from loguru import logger

from .audio_capture import AudioStream
from .config import Settings
from .parser import (
    COMPLETE,
    INVALID,
    PREFIX,
    match_status,
    ActionCommand,
    Command,
    IMMEDIATE_ACTIONS,
    IMMEDIATE_LEVEL,
    LevelCommand,
    parse,
    parse_confirmation,
)
from .validator import validate


class PipelineState(StrEnum):
    IDLE = auto()
    WAKE_WORD_LISTEN = auto()
    COMMAND_LISTEN = auto()
    PARSE = auto()
    VALIDATE = auto()
    CONFIRMATION_LISTEN = auto()
    EMIT = auto()
    ERROR = auto()


@dataclass(frozen=True)
class VoiceResult:
    command: Command | None = None
    confirmed: bool = False
    reason: str | None = None
    wake_detected: bool = field(default=False, compare=False)

    def to_dict(self):
        result = validate(self.command, self.reason)
        result.update(
            confirmed=self.confirmed,
            valid=self.confirmed and self.reason is None and result["valid"],
        )
        if self.reason:
            result["reason"] = self.reason
        return result


class VoiceCommandPipeline:
    def __init__(self, settings: Settings, *, recognizer=None, wake=None, on_state=None, command_filter=None):
        self.settings = settings
        self.on_state = on_state
        self.command_filter = command_filter
        self.state = PipelineState.IDLE
        if recognizer is None:
            from .stt_engine import VoskSTTEngine

            recognizer = VoskSTTEngine(settings)
        self.recognizer = recognizer
        if wake is None and settings.wake_backend == "openwakeword":
            from .wake_word import WakeWordDetector

            wake = WakeWordDetector(settings)
        self.wake = wake

    def _set_state(self, state, command=None):
        logger.debug("Voice state: {} -> {}", self.state, state)
        self.state = state
        if self.on_state is not None:
            self.on_state(state, command)

    def listen_once(self, stream=None) -> VoiceResult:
        if stream is None:
            with AudioStream(self.settings) as opened:
                return self.listen_once(opened)
        try:
            return self._cycle(stream)
        except Exception:
            self._set_state(PipelineState.ERROR)
            raise

    def listen_yes_no(self, stream=None) -> bool | None:
        """Escuta diretamente uma resposta YES/NO (sem exigir a wake word MAGNUS)."""
        if stream is None:
            with AudioStream(self.settings) as opened:
                return self.listen_yes_no(opened)
        stream.discard_pending()
        answer = self.recognizer.listen(stream, "confirmation", self.settings.confirmation_timeout)
        return parse_confirmation(answer, self.settings.language)

    def _cycle(self, stream):
        stream.discard_pending()
        self._set_state(PipelineState.WAKE_WORD_LISTEN)
        rest = ""
        if self.wake is not None:
            self.wake.wait(stream)
        else:
            heard = self.recognizer.listen(stream, "wake", None)
            if heard is None:
                return self._failure(None, "Wake word not detected", wake_detected=False)
            rest = heard.split(" ", 1)[1] if " " in heard else ""
        self._set_state(PipelineState.COMMAND_LISTEN)
        status = match_status(rest) if rest else INVALID
        if status == COMPLETE:
            text = rest                                   # "magnus <comando>" dito de uma vez
        elif status == PREFIX:
            # Ja veio um prefixo junto com "magnus" (ex.: "magnus option");
            # continua ouvindo para completar as palavras restantes.
            text = self.recognizer.listen(stream, "command", self.settings.command_timeout, initial=rest)
        else:
            text = self.recognizer.listen(stream, "command", self.settings.command_timeout)
        if not text:
            logger.warning("[VOZ] Nenhum comando ouvido (timeout de {}s)", self.settings.command_timeout)
            return self._failure(None, "Command timed out", wake_detected=True)
        logger.info("[VOZ] Ouvi: {!r}", text)
        self._set_state(PipelineState.PARSE)
        command, reason = parse(text, self.settings.language)
        self._set_state(PipelineState.VALIDATE, command)
        checked = validate(command, reason)
        if not checked["valid"]:
            logger.warning("[VOZ] NAO ENTENDI {!r}: {}", text, checked["reason"])
            return self._failure(command, checked["reason"], wake_detected=True)

        # Bloqueio de comandos por estado (impede usar comandos fora do momento permitido)
        if self.command_filter is not None and not self.command_filter(command):
            logger.warning("[VOZ] Entendi {} mas NAO e permitido neste momento do jogo", command)
            return self._failure(command, "Command not allowed in current state", wake_detected=True)

        # Comandos imediatos (caso algum seja configurado em IMMEDIATE_ACTIONS)
        if (isinstance(command, ActionCommand) and command.action in IMMEDIATE_ACTIONS) or (
            isinstance(command, LevelCommand) and IMMEDIATE_LEVEL
        ):
            logger.success("[VOZ] ENTENDI: {} -> EXECUTANDO IMEDIATAMENTE", command)
            stream.check()
            self._set_state(PipelineState.EMIT, command)
            return VoiceResult(command, confirmed=True, wake_detected=True)

        logger.info("[VOZ] ENTENDI: {} -> Aguardando confirmacao (YES/NO)...", command)
        stream.discard_pending()
        self._set_state(PipelineState.CONFIRMATION_LISTEN, command)
        answer = self.recognizer.listen(stream, "confirmation", self.settings.confirmation_timeout)
        confirmed = parse_confirmation(answer, self.settings.language)
        if confirmed is not True:
            reason = (
                "Command cancelled" if confirmed is False else "Confirmation missing or ambiguous"
            )
            logger.warning("[VOZ] Confirmacao recusada ou ausente (ouvi {!r}): {}", answer, reason)
            return self._failure(command, reason, wake_detected=True)

        logger.success("[VOZ] CONFIRMADO (YES): {} -> EXECUTANDO", command)
        stream.check()
        self._set_state(PipelineState.EMIT, command)
        return VoiceResult(command, confirmed=True, wake_detected=True)

    def _failure(self, command, reason, wake_detected=False):
        self._set_state(PipelineState.ERROR, command)
        return VoiceResult(command, reason=reason, wake_detected=wake_detected)
