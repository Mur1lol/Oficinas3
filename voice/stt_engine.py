import json
import math
import time
import wave
from array import array
from collections import deque

from loguru import logger

from .audio_capture import AudioError
from .config import FRAME_BYTES, FRAME_MS, SAMPLE_RATE, Settings
from .parser import COMPLETE, PREFIX, grammar, match_status, normalize

# Voz precisa ser NOISE_MARGIN x mais forte que o ruido de fundo estimado.
NOISE_MARGIN = 2.5


def _clean(text: str) -> str:
    """Remove [unk] e normaliza."""
    return " ".join(w for w in normalize(text).split() if w != "unk")


class VoskSTTEngine:
    """
    Reconhecimento deterministico:
    * Nunca usa resultado PARCIAL para decidir (nada de "chutar").
    * Uma frase termina apos ``silence_ms`` de silencio (pausa entre palavras).
    * Se o texto e um PREFIXO de comando (ex.: "option", "level", "move e two"),
      continua ouvindo por ate ``incomplete_timeout`` segundos esperando o resto.
    """

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
        for rec in self.recognizers.values():
            rec.SetWords(True)
        self.vad = webrtcvad.Vad(settings.vad_aggressiveness)
        self._noise_floor = None

    # ── wake word ────────────────────────────────────────────────────────────

    def _wake_split(self, res: dict) -> str | None:
        """
        Aceita "[unk] magnus [unk] ..." somente no resultado FINAL, com confianca e duracao
        minimas na palavra "magnus". Retorna "magnus <resto>" (o resto pode ja ser o comando).
        """
        heard = _clean(res.get("text", ""))
        if not heard:
            return None
        words = res.get("result") or []
        saw_magnus = False
        for idx, w in enumerate(words):
            if w.get("word") != "magnus":
                continue
            saw_magnus = True
            conf = float(w.get("conf", 0.0))
            dur = float(w.get("end", 0.0)) - float(w.get("start", 0.0))
            if conf < self.settings.wake_confidence or dur < self.settings.wake_min_duration:
                logger.info(
                    "[VOZ] Wake word REJEITADA: ouvi {!r} | 'magnus' conf={:.2f} (min {:.2f}) "
                    "dur={:.2f}s (min {:.2f}s)",
                    heard, conf, self.settings.wake_confidence, dur, self.settings.wake_min_duration,
                )
                continue
            rest = " ".join(x["word"] for x in words[idx + 1:] if x.get("word") not in ("[unk]", "magnus"))
            logger.success("[VOZ] Wake word CONFIRMADA: ouvi {!r} | conf={:.2f} dur={:.2f}s", heard, conf, dur)
            return f"magnus {rest}".strip()
        if not saw_magnus:
            logger.debug("[VOZ] Ignorado (sem 'magnus'): {!r}", heard)
        return None

    # ── ruido ────────────────────────────────────────────────────────────────

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

    # ── escuta ───────────────────────────────────────────────────────────────

    def listen(self, stream, stage: str, timeout: float | None, initial: str = "") -> str | None:
        recognizer = self.recognizers[stage]
        recognizer.Reset()
        deadline = time.monotonic() + timeout if timeout is not None else float("inf")
        last_frame = time.monotonic()
        min_frames = max(1, self.settings.min_speech_ms // FRAME_MS)
        speech = False
        silence = 0
        voiced_run = 0
        parts: list[str] = [initial] if initial else []
        waiting_since = time.monotonic() if initial else None
        preroll = deque(maxlen=min_frames + 6)

        def text_so_far() -> str:
            return " ".join(p for p in parts if p).strip()

        while time.monotonic() < deadline:
            frame = stream.read_frame(min(0.1, max(0, deadline - time.monotonic())))
            now = time.monotonic()

            # Comando incompleto: desiste se a proxima palavra nao vier a tempo
            if not speech and waiting_since is not None and now - waiting_since >= self.settings.incomplete_timeout:
                logger.warning("[VOZ] Comando incompleto, proxima palavra nao veio: {!r}", text_so_far())
                return text_so_far()

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
                waiting_since = None
                if timeout is not None:
                    deadline = max(deadline, now + self.settings.incomplete_timeout)
                to_feed = list(preroll)
                preroll.clear()
            else:
                to_feed = [frame]

            for chunk in to_feed:
                if recognizer.AcceptWaveform(chunk):
                    res = json.loads(recognizer.Result())
                    if stage == "wake":
                        hit = self._wake_split(res)
                        if hit:
                            stream.check()
                            return hit
                    else:
                        parts.append(_clean(res.get("text", "")))

            silence = 0 if voiced else silence + FRAME_MS
            if silence < self.settings.silence_ms:
                continue

            # Fim da frase (pausa entre palavras atingida)
            res = json.loads(recognizer.FinalResult())
            recognizer.Reset()
            speech, silence, voiced_run = False, 0, 0

            if stage == "wake":
                hit = self._wake_split(res)
                if hit:
                    stream.check()
                    return hit
                continue

            parts.append(_clean(res.get("text", "")))
            text = text_so_far()
            if not text:
                continue                                   # so ruido / [unk]
            if stage != "command":
                stream.check()
                return text
            status = match_status(text)
            if status == COMPLETE:
                stream.check()
                return text
            if status == PREFIX:
                logger.info("[VOZ] Ouvi {!r} -> aguardando proxima palavra...", text)
                now = time.monotonic()
                waiting_since = now
                if timeout is not None:
                    deadline = max(deadline, now + self.settings.incomplete_timeout)
                continue
            stream.check()
            return text                                    # invalido: pipeline informa o erro
        return text_so_far() or None

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
                    parts.append(_clean(json.loads(recognizer.Result()).get("text", "")))
        parts.append(_clean(json.loads(recognizer.FinalResult()).get("text", "")))
        return " ".join(part for part in parts if part).strip() or None
