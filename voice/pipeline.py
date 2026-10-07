from dataclasses import dataclass
from enum import StrEnum, auto

from loguru import logger

from .audio_capture import AudioStream
from .config import Settings
from .parser import Command, parse, parse_confirmation
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

    def to_dict(self):
        result = validate(self.command, self.reason)
        result.update(confirmed=self.confirmed,
                      valid=self.confirmed and self.reason is None and result["valid"])
        if self.reason:
            result["reason"] = self.reason
        return result


class VoiceCommandPipeline:
    def __init__(self, settings: Settings, *, recognizer=None, wake=None, on_state=None):
        self.settings = settings
        self.on_state = on_state
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

    def _cycle(self, stream):
        stream.discard_pending()
        self._set_state(PipelineState.WAKE_WORD_LISTEN)
        if self.wake is not None:
            self.wake.wait(stream)
        elif self.recognizer.listen(stream, "wake", None) is None:
            return self._failure(None, "Wake word not detected")
        stream.discard_pending()
        self._set_state(PipelineState.COMMAND_LISTEN)
        text = self.recognizer.listen(stream, "command", self.settings.command_timeout)
        if not text:
            return self._failure(None, "Command timed out")
        self._set_state(PipelineState.PARSE)
        command, reason = parse(text, self.settings.language)
        self._set_state(PipelineState.VALIDATE, command)
        checked = validate(command, reason)
        if not checked["valid"]:
            return self._failure(command, checked["reason"])
        stream.discard_pending()
        self._set_state(PipelineState.CONFIRMATION_LISTEN, command)
        answer = self.recognizer.listen(stream, "confirmation", self.settings.confirmation_timeout)
        confirmed = parse_confirmation(answer, self.settings.language)
        if confirmed is not True:
            reason = "Command cancelled" if confirmed is False else "Confirmation missing or ambiguous"
            return self._failure(command, reason)
        stream.check()
        self._set_state(PipelineState.EMIT, command)
        return VoiceResult(command, confirmed=True)

    def _failure(self, command, reason):
        self._set_state(PipelineState.ERROR, command)
        return VoiceResult(command, reason=reason)
