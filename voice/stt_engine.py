import json
import time
import wave
from collections import deque

from loguru import logger

from .audio_capture import AudioError
from .config import FRAME_BYTES, FRAME_MS, SAMPLE_RATE, Settings
from .parser import grammar, normalize


class VoskSTTEngine:
    def __init__(self, settings: Settings):
        import webrtcvad
        from vosk import KaldiRecognizer, Model, SetLogLevel

        self.settings = settings
        if not (settings.model_path / "am" / "final.mdl").is_file():
            raise FileNotFoundError(f"Vosk model missing: {settings.model_path}")
        SetLogLevel(-1)
        logger.info("Loading Vosk model: {}", settings.model_path)
        self.model = Model(str(settings.model_path))
        self.recognizers = {
            stage: KaldiRecognizer(
                self.model,
                SAMPLE_RATE,
                json.dumps(grammar(stage, settings.language), ensure_ascii=False),
            )
            for stage in ("wake", "command", "confirmation")
        }
        self.vad = webrtcvad.Vad(2)

    def listen(self, stream, stage: str, timeout: float | None) -> str | None:
        recognizer = self.recognizers[stage]
        recognizer.Reset()
        deadline = time.monotonic() + timeout if timeout is not None else float("inf")
        last_frame = time.monotonic()
        speech = False
        silence = 0
        parts = []
        preroll = deque(maxlen=5)
        while time.monotonic() < deadline:
            frame = stream.read_frame(min(0.1, max(0, deadline - time.monotonic())))
            now = time.monotonic()
            if frame is None:
                if now - last_frame >= self.settings.audio_timeout:
                    raise AudioError("No audio frames received from microphone")
                continue
            last_frame = now
            if len(frame) != FRAME_BYTES:
                raise AudioError("Expected 30 ms PCM frames at 16 kHz")
            voiced = self.vad.is_speech(frame, SAMPLE_RATE)
            if not speech:
                preroll.append(frame)
                if not voiced:
                    continue
                speech = True
                for previous in preroll:
                    if recognizer.AcceptWaveform(previous):
                        parts.append(json.loads(recognizer.Result()).get("text", ""))
                preroll.clear()
            elif recognizer.AcceptWaveform(frame):
                parts.append(json.loads(recognizer.Result()).get("text", ""))
            silence = 0 if voiced else silence + FRAME_MS
            if silence < self.settings.silence_ms:
                continue
            parts.append(json.loads(recognizer.FinalResult()).get("text", ""))
            text = " ".join(part for part in parts if part).strip()
            logger.debug("{} transcript: {!r}", stage, text)
            if text and (stage != "wake" or normalize(text) == "magnus"):
                stream.check()
                return text
            recognizer.Reset()
            speech, silence, parts = False, 0, []
        return None

    def transcribe_file(self, path) -> str | None:
        recognizer = self.recognizers["command"]
        recognizer.Reset()
        parts = []
        with wave.open(str(path), "rb") as source:
            if (
                source.getnchannels(),
                source.getsampwidth(),
                source.getframerate(),
                source.getcomptype(),
            ) != (1, 2, SAMPLE_RATE, "NONE"):
                raise ValueError("WAV must be uncompressed mono, 16-bit, 16000 Hz")
            while data := source.readframes(SAMPLE_RATE * FRAME_MS // 1000):
                if recognizer.AcceptWaveform(data):
                    parts.append(json.loads(recognizer.Result()).get("text", ""))
        parts.append(json.loads(recognizer.FinalResult()).get("text", ""))
        return " ".join(part for part in parts if part).strip() or None
