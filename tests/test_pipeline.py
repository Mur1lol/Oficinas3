from collections import deque

import pytest

from voice.audio_capture import AudioError
from voice.config import Settings
from voice.parser import MoveCommand, parse, parse_confirmation
from voice.pipeline import PipelineState, VoiceCommandPipeline, VoiceResult


class Stream:
    def __init__(self):
        self.discards = 0
        self.error = None

    def discard_pending(self):
        self.check()
        self.discards += 1

    def check(self):
        if self.error:
            raise self.error


class Recognizer:
    def __init__(self, *answers):
        self.answers = deque(answers)
        self.stages = []

    def listen(self, stream, stage, timeout):
        self.stages.append(stage)
        answer = self.answers.popleft()
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_complete_cycle_and_led_callback():
    states = []
    stream = Stream()
    engine = Recognizer("magnus", "move echo two echo four", "yes")
    pipeline = VoiceCommandPipeline(Settings(), recognizer=engine,
                                   on_state=lambda state, cmd: states.append((state, cmd)))
    result = pipeline.listen_once(stream)
    assert result == VoiceResult(MoveCommand("E2", "E4"), True)
    assert result.to_dict() == {"wake_word": "MAGNUS", "command": "MOVE", "from": "E2",
                                "to": "E4", "valid": True, "confirmed": True}
    assert stream.discards == 3
    assert engine.stages == ["wake", "command", "confirmation"]
    assert [s for s, _ in states] == [PipelineState.WAKE_WORD_LISTEN, PipelineState.COMMAND_LISTEN,
                                     PipelineState.PARSE, PipelineState.VALIDATE,
                                     PipelineState.CONFIRMATION_LISTEN, PipelineState.EMIT]
    assert states[-2][1] == result.command


@pytest.mark.parametrize("text,answer", [("move e2 e4", "no"), ("move e2 e4", None),
    ("move e2 e4", "yes no"), ("move e2 e4", "yes [unk]"), (None, None),
    ("move e2 e4 a1", "yes"), ("move e2 e2", "yes"), ("move i2 e4", "yes")])
def test_failures_never_emit(text, answer):
    pipeline = VoiceCommandPipeline(Settings(), recognizer=Recognizer("magnus", text, answer))
    result = pipeline.listen_once(Stream())
    assert not result.confirmed
    assert not result.to_dict()["valid"]
    assert pipeline.state == PipelineState.ERROR


def test_audio_error_aborts_confirmation():
    engine = Recognizer("magnus", "move e2 e4", AudioError("disconnected"))
    pipeline = VoiceCommandPipeline(Settings(), recognizer=engine)
    with pytest.raises(AudioError, match="disconnected"):
        pipeline.listen_once(Stream())
    assert pipeline.state == PipelineState.ERROR


def test_pending_capture_error_prevents_emit():
    stream = Stream()
    engine = Recognizer("magnus", "move e2 e4", "yes")
    original = engine.listen

    def listen(stream, stage, timeout):
        value = original(stream, stage, timeout)
        if stage == "confirmation":
            stream.error = AudioError("overflow")
        return value

    engine.listen = listen
    with pytest.raises(AudioError):
        VoiceCommandPipeline(Settings(), recognizer=engine).listen_once(stream)


def test_cycles_are_independent_of_game_phase():
    engine = Recognizer("magnus", "resign", "yes", "magnus", "move a2 a4", "yes")
    pipeline = VoiceCommandPipeline(Settings(), recognizer=engine)
    assert pipeline.listen_once(Stream()).confirmed
    assert pipeline.listen_once(Stream()).confirmed
    assert not hasattr(pipeline, "game_started")


@pytest.mark.parametrize("text", ["yes no", "yesterday", "similarly", "noise yes", "don't confirm", "yes [unk]"])
def test_ambiguous_confirmation(text):
    assert parse_confirmation(text) is None


@pytest.mark.parametrize("text", ["move e2 e4 a1", "move e2 e4 maybe", "do not move e2 e4",
    "new game then resign", "move i2 e2 e4", "move e2 e4 and move d2 d4"])
def test_ambiguous_command(text):
    assert parse(text)[0] is None


def test_unconfirmed_result_is_not_executable():
    assert not VoiceResult(MoveCommand("E2", "E4")).to_dict()["valid"]
