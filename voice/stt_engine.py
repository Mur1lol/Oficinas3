import json
import math
import time
import wave
from array import array
from collections import deque

from loguru import logger

from .audio_capture import AudioError
from .config import FRAME_BYTES, FRAME_MS, SAMPLE_RATE, Settings
from .parser import grammar, normalize, parse

# Palavras que sozinhas sao ambiguas (ex.: "play" = NEW_GAME ou inicio de "play e2 e4").
AMBIGUOUS_PREFIXES = {"play"}
# Frames de 30 ms consecutivos com o mesmo comando parcial antes de executar.
EARLY_STABLE_FRAMES = 3
# Voz precisa ser NOISE_MARGIN x mais forte que o ruido de fundo estimado.
NOISE_MARGIN = 2.5


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
        self.vad = webrtcvad.Vad(settings.vad_aggressiveness)
        self._noise_floor = None

    @staticmethod
    def _is_wake_word(text: str) -> bool:
        return "magnus" in normalize(text).split()

    @staticmethod
    def _early_match(text: str) -> bool:
        """True se o texto (parcial ou final) ja forma um comando valido completo."""
        norm = normalize(text)
        if not norm or norm in AMBIGUOUS_PREFIXES:
            return False
        command, _ = parse(norm)
        return command is not None

    def _is_voiced(self, frame: bytes) -> bool:
        """VAD + gate de energia adaptativo para rejeitar ruido de fundo."""
        samples = array("h")
        samples.frombytes(frame)
        rms = math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0
        vad_voiced = self.vad.is_speech(frame, SAMPLE_RATE)
        if not vad_voiced:
            floor = self._noise_floor
            self._noise_floor = rms if floor is None else 0.97 * floor + 0.03 * rms
        threshold = self.settings.min_rms
        if self._noise_floor is not None:
            threshold = max(threshold, NOISE_MARGIN * self._noise_floor)
        return vad_voiced and rms >= threshold

    def listen(self, stream, stage: str, timeout: float | None) -> str | None:
        recognizer = self.recognizers[stage]
        recognizer.Reset()
        deadline = time.monotonic() + timeout if timeout is not None else float("inf")
        last_frame = time.monotonic()
        min_frames = max(1, self.settings.min_speech_ms // FRAME_MS)
        speech = False
        silence = 0
        voiced_run = 0
        parts = []
        preroll = deque(maxlen=min_frames + 6)
        last_text, stable = "", 0
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
            voiced = self._is_voiced(frame)
            if not speech:
                preroll.append(frame)
                voiced_run = voiced_run + 1 if voiced else 0
                if voiced_run < min_frames:
                    continue
                speech = True
                to_feed = list(preroll)
                preroll.clear()
            else:
                to_feed = [frame]

            for chunk in to_feed:
                if recognizer.AcceptWaveform(chunk):
                    res = json.loads(recognizer.Result()).get("text", "")
                    if res:
                        parts.append(res)
                    current = " ".join(parts).strip()
                else:
                    partial = json.loads(recognizer.PartialResult()).get("partial", "")
                    current = " ".join(parts + [partial]).strip()

                if stage == "wake" and self._is_wake_word(current):
                    logger.info("[VOZ] Wake word detectada: {!r}", current)
                    stream.check()
                    return "magnus"
                if stage == "command":
                    if current and self._early_match(current):
                        stable = stable + 1 if current == last_text else 1
                        last_text = current
                        if stable >= EARLY_STABLE_FRAMES:
                            logger.info("[VOZ] Comando valido detectado: {!r} -> executando", current)
                            stream.check()
                            return current
                    else:
                        stable, last_text = 0, ""

            silence = 0 if voiced else silence + FRAME_MS
            if silence < self.settings.silence_ms:
                continue
            parts.append(json.loads(recognizer.FinalResult()).get("text", ""))
            text = " ".join(part for part in parts if part).strip()
            logger.debug("{} transcript: {!r}", stage, text)
            if text and (stage != "wake" or self._is_wake_word(text)):
                stream.check()
                return "magnus" if stage == "wake" else text
            recognizer.Reset()
            speech, silence, voiced_run, parts = False, 0, 0, []
            last_text, stable = "", 0
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
