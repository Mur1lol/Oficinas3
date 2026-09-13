"""
wake_word.py
============
Wake-word detection for ChessAI 2.0.

Primary engine : openWakeWord (ONNX) — runs the custom MAGNUS model.
Fallback engine: Vosk keyword scan — works immediately with no ONNX file.

The detector exposes a single blocking call:

    detector = WakeWordDetector()
    detector.wait_for_wake_word(stream)  # blocks until MAGNUS is heard
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod

import config
from audio_capture import AudioStream

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class _BaseWakeWordEngine(ABC):
    """Common interface for wake-word backends."""

    @abstractmethod
    def process_frame(self, frame: bytes) -> bool:
        """
        Feed one audio frame.
        Returns True if the wake word was detected in this frame.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (call after a detection)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""


# ---------------------------------------------------------------------------
# openWakeWord engine
# ---------------------------------------------------------------------------

class _OWWEngine(_BaseWakeWordEngine):
    """
    openWakeWord ONNX backend.

    Loads the custom magnus.onnx model and scores every 80-ms chunk.
    """

    _OWW_CHUNK_SAMPLES = 1280
    _OWW_CHUNK_BYTES   = _OWW_CHUNK_SAMPLES * 2  # 16-bit

    def __init__(self, model_path: str) -> None:
        import numpy as np
        from openwakeword.model import Model as OWWModel  # type: ignore

        self._np = np
        self._model = OWWModel(
            wakeword_models=[str(model_path)],
            inference_framework="onnx",
        )
        self._buffer = b""
        self._consecutive = 0
        log.info("[OWW] Loaded MAGNUS model from %s", model_path)

    @property
    def name(self) -> str:
        return "openWakeWord (ONNX)"

    def process_frame(self, frame: bytes) -> bool:
        self._buffer += frame

        detected = False
        while len(self._buffer) >= self._OWW_CHUNK_BYTES:
            chunk = self._buffer[: self._OWW_CHUNK_BYTES]
            self._buffer = self._buffer[self._OWW_CHUNK_BYTES :]

            audio_np = self._np.frombuffer(chunk, dtype=self._np.int16)
            predictions = self._model.predict(audio_np)

            score = max(predictions.values(), default=0.0)
            log.debug("[OWW] score=%.3f", score)

            if score >= config.OWW_THRESHOLD:
                self._consecutive += 1
                if self._consecutive >= config.OWW_TRIGGER_LEVEL:
                    log.info("[OWW] Wake word detected! score=%.3f", score)
                    detected = True
                    self._consecutive = 0
            else:
                self._consecutive = 0

        return detected

    def reset(self) -> None:
        self._buffer = b""
        self._consecutive = 0


# ---------------------------------------------------------------------------
# Vosk keyword engine (fallback)
# ---------------------------------------------------------------------------

class _VoskKeywordEngine(_BaseWakeWordEngine):
    """
    Vosk KaldiRecognizer configured with grammar restricted to wake-word phrases.
    Supports both Portuguese and English models.
    """

    def __init__(
        self,
        language: str | None = None,
        model_path: str | None = None,
    ) -> None:
        from vosk import KaldiRecognizer, Model, SetLogLevel  # type: ignore

        SetLogLevel(-1)

        self._language = language or config.LANGUAGE
        path = str(model_path or config.get_model_path(self._language))
        log.info("[VoskKW] Loading Vosk model (%s) from %s", self._language, path)
        model = Model(path)

        self._phrases = config.get_wake_word_phrases(self._language)
        grammar = json.dumps(self._phrases + ["[unk]"], ensure_ascii=False)
        self._rec = KaldiRecognizer(model, config.SAMPLE_RATE, grammar)
        self._rec.SetWords(False)
        log.info("[VoskKW] Ready. Listening for: %s", self._phrases)

    @property
    def name(self) -> str:
        return "Vosk keyword fallback"

    def process_frame(self, frame: bytes) -> bool:
        if self._rec.AcceptWaveform(frame):
            result = json.loads(self._rec.Result())
            text = result.get("text", "").lower().strip()
            if text:
                log.debug("[VoskKW] heard: '%s'", text)
            if any(phrase in text for phrase in self._phrases):
                log.info("[VoskKW] Wake word detected in: '%s'", text)
                return True
        else:
            partial = json.loads(self._rec.PartialResult())
            partial_text = partial.get("partial", "").lower()
            if any(phrase in partial_text for phrase in self._phrases):
                log.info("[VoskKW] Wake word in partial: '%s'", partial_text)
                self._rec.Reset()
                return True
        return False

    def reset(self) -> None:
        self._rec.Reset()


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

class WakeWordDetector:
    """
    High-level wake-word detector with bilingual support.

    Automatically selects the best available backend:
    1. openWakeWord ONNX  (if config.OWW_MODEL_PATH exists)
    2. Vosk keyword scan  (fallback)
    """

    def __init__(self, language: str | None = None) -> None:
        self.language = language or config.LANGUAGE
        self._engine = self._build_engine()

    def _build_engine(self) -> _BaseWakeWordEngine:
        oww_path = config.OWW_MODEL_PATH
        if oww_path.exists():
            try:
                engine = _OWWEngine(str(oww_path))
                log.info("Wake-word engine: %s", engine.name)
                return engine
            except Exception as exc:
                log.warning(
                    "openWakeWord failed to load (%s). Falling back to Vosk keyword engine.",
                    exc,
                )

        engine = _VoskKeywordEngine(language=self.language)
        log.info("Wake-word engine: %s", engine.name)
        return engine

    def wait_for_wake_word(self, stream: AudioStream) -> None:
        """Block until the wake word MAGNUS is detected in the audio stream."""
        log.info("Listening for wake word '%s'...", config.WAKE_WORD)
        self._engine.reset()

        for frame in stream:
            if self._engine.process_frame(frame):
                print(f"\n[{config.WAKE_WORD}] Wake word detected!")
                return

    @property
    def engine_name(self) -> str:
        return self._engine.name
