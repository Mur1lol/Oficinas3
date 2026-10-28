import json
import wave
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from voice.audio_capture import AudioError
from voice.config import FRAME_BYTES, Settings
from voice.stt_engine import VoskSTTEngine


def fake_engine(monkeypatch, voices, finals, partials=()):
    clock = [0.0]
    monkeypatch.setattr("voice.stt_engine.time.monotonic", lambda: clock[0])
    frames = deque(voices)
    vad = Mock()
    vad.is_speech.side_effect = lambda frame, rate: frame[0] == 1
    rec = Mock()
    accepted = deque(partials)
    rec.AcceptWaveform.side_effect = lambda frame: accepted.popleft() if accepted else False
    rec.Result.return_value = json.dumps({"text": "move echo"})
    rec.FinalResult.side_effect = [json.dumps({"text": t}) for t in finals]
    engine = VoskSTTEngine.__new__(VoskSTTEngine)
    engine.settings = Settings(silence_ms=90, audio_timeout=0.2)
    engine.vad = vad
    engine.recognizers = dict.fromkeys(("wake", "command", "confirmation"), rec)

    def read(timeout):
        clock[0] += min(timeout, 0.03)
        if frames:
            return bytes([frames.popleft()]) * FRAME_BYTES
        return None

    return engine, SimpleNamespace(read_frame=read, check=Mock()), rec


def test_stt_preserves_partial_and_final_results(monkeypatch):
    engine, stream, rec = fake_engine(monkeypatch, [1, 1, 0, 0, 0], ["two echo four"], [True])
    assert engine.listen(stream, "command", 1) == "move echo two echo four"
    assert rec.AcceptWaveform.call_count == 5


def test_wake_word_requires_whole_utterance(monkeypatch):
    engine, stream, _ = fake_engine(
        monkeypatch, [1, 0, 0, 0, 1, 0, 0, 0], ["magnus move e2", "magnus"]
    )
    assert engine.listen(stream, "wake", None) == "magnus"


def test_silence_expires_without_guessing(monkeypatch):
    engine, stream, rec = fake_engine(monkeypatch, [0] * 20, [])
    assert engine.listen(stream, "command", 0.3) is None
    rec.FinalResult.assert_not_called()


def test_missing_frames_raise_even_during_wake(monkeypatch):
    engine, stream, _ = fake_engine(monkeypatch, [], [])
    with pytest.raises(AudioError, match="No audio"):
        engine.listen(stream, "wake", None)


def test_incomplete_speech_at_deadline_is_not_confirmed(monkeypatch):
    engine, stream, rec = fake_engine(monkeypatch, [1] * 20, [])
    assert engine.listen(stream, "confirmation", 0.3) is None
    rec.FinalResult.assert_not_called()


def test_wrong_frame_size_fails(monkeypatch):
    engine, stream, _ = fake_engine(monkeypatch, [], [])
    stream.read_frame = lambda timeout: b"bad"
    with pytest.raises(AudioError):
        engine.listen(stream, "command", 1)


def write_wav(path, rate=16000, channels=1):
    with wave.open(str(path), "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(bytes(rate * channels * 2))


def test_wav_results_and_format_validation(tmp_path, monkeypatch):
    engine, _, _ = fake_engine(monkeypatch, [], ["two echo four"], [True])
    path = tmp_path / "input.wav"
    write_wav(path)
    assert engine.transcribe_file(path) == "move echo two echo four"
    write_wav(path, 48000)
    with pytest.raises(ValueError, match="16000"):
        engine.transcribe_file(path)


@pytest.mark.model
def test_real_model_silence(tmp_path):
    settings = Settings()
    if not settings.model_path.exists():
        pytest.skip("Vosk model not installed on this machine")
    engine = VoskSTTEngine(settings)
    path = tmp_path / "silence.wav"
    write_wav(path)
    assert engine.transcribe_file(path) is None
