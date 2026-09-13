"""
stt_engine.py
=============
Speech-to-Text engine for ChessAI 2.0.

Uses Vosk (KaldiRecognizer) with:
  - Grammar restriction  : limits vocabulary to chess commands and actions.
  - Confirmation grammar : dedicated ultra-restricted recognizer for YES/NO responses.
  - Bilingual support    : pt-BR and en-US.
  - WebRTC VAD filtering : skips silent frames to save CPU cycles.
  - Timeout safety       : returns None if no speech within the deadline.
"""

from __future__ import annotations

import json
import logging
import time

import webrtcvad  # type: ignore

import config
from audio_capture import AudioStream

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VoskSTTEngine
# ---------------------------------------------------------------------------

class VoskSTTEngine:
    """
    Streaming Vosk STT with VAD gating and grammar restriction.

    Loads the acoustic model once. Provides separate recognition pipelines
    for game commands and YES/NO confirmations.

    Parameters
    ----------
    model_path : str | None
        Path to the Vosk model directory. Defaults to config.get_model_path(language).
    language : str | None
        Language code ("pt-BR" or "en-US"). Defaults to config.LANGUAGE.
    """

    def __init__(
        self,
        model_path: str | None = None,
        language: str | None = None,
    ) -> None:
        from vosk import KaldiRecognizer, Model, SetLogLevel  # type: ignore

        SetLogLevel(-1)  # suppress verbose Kaldi output

        self.language = language or config.LANGUAGE
        path = str(model_path or config.get_model_path(self.language))

        log.info("[STT] Loading Vosk model (%s) from %s ...", self.language, path)
        self._model = Model(path)
        log.info("[STT] Vosk model loaded.")

        # 1. Grammar-restricted recognizer for game commands
        cmd_grammar = config.get_grammar_words(self.language)
        grammar_json = json.dumps(cmd_grammar, ensure_ascii=False)
        self._rec = KaldiRecognizer(self._model, config.SAMPLE_RATE, grammar_json)
        self._rec.SetWords(True)

        # 2. Ultra-restricted recognizer for confirmation responses (YES/NO, SIM/NÃO)
        confirm_grammar = config.get_confirmation_grammar(self.language)
        confirm_json = json.dumps(confirm_grammar, ensure_ascii=False)
        self._confirm_rec = KaldiRecognizer(self._model, config.SAMPLE_RATE, confirm_json)
        self._confirm_rec.SetWords(False)

        # WebRTC VAD
        self._vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)

        log.info(
            "[STT] Ready. Lang=%s VAD aggressiveness=%d  Command words=%d  Confirm words=%d",
            self.language,
            config.VAD_AGGRESSIVENESS,
            len(cmd_grammar),
            len(confirm_grammar),
        )

    # ------------------------------------------------------------------
    # Command transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        stream: AudioStream,
        timeout: float = config.COMMAND_TIMEOUT_S,
    ) -> str | None:
        """
        Listen to *stream* until command speech is recognised or *timeout* expires.
        """
        return self._stream_recognize(
            stream=stream,
            recognizer=self._rec,
            timeout=timeout,
            label="Command",
        )

    # ------------------------------------------------------------------
    # Confirmation transcription (YES / NO)
    # ------------------------------------------------------------------

    def transcribe_confirmation(
        self,
        stream: AudioStream,
        timeout: float = config.CONFIRM_TIMEOUT_S,
    ) -> str | None:
        """
        Listen to *stream* specifically for confirmation words (YES/NO, SIM/NÃO).
        """
        return self._stream_recognize(
            stream=stream,
            recognizer=self._confirm_rec,
            timeout=timeout,
            label="Confirmation",
        )

    # ------------------------------------------------------------------
    # Internal generic streaming recognizer with VAD
    # ------------------------------------------------------------------

    def _stream_recognize(
        self,
        stream: AudioStream,
        recognizer,
        timeout: float,
        label: str = "Audio",
    ) -> str | None:
        recognizer.Reset()

        deadline = time.monotonic() + timeout
        speech_started = False
        silence_count = 0
        frames_captured = 0

        log.info("[STT] Listening for %s (timeout=%.1fs)...", label.lower(), timeout)

        for frame in stream:
            now = time.monotonic()
            if now >= deadline:
                log.warning("[STT] Timeout — no %s recognised.", label.lower())
                break

            frames_captured += 1

            # VAD gate
            try:
                is_speech = self._vad.is_speech(frame, config.SAMPLE_RATE)
            except Exception:
                is_speech = True

            if is_speech:
                speech_started = True
                silence_count = 0
                recognizer.AcceptWaveform(frame)
            else:
                if speech_started:
                    silence_count += 1

            # After enough silence post-speech, flush the recognizer
            if (
                speech_started
                and silence_count >= config.VAD_SILENCE_THRESHOLD
                and frames_captured >= config.COMMAND_MIN_FRAMES
            ):
                log.debug("[STT] End of %s speech detected.", label.lower())
                break

        # Retrieve final result
        result_json = json.loads(recognizer.FinalResult())
        text = result_json.get("text", "").strip().lower()

        if text:
            log.info("[STT] %s recognised: '%s'", label, text)
        else:
            log.warning("[STT] Empty %s transcript.", label.lower())
            return None

        return text

    # ------------------------------------------------------------------
    # WAV file transcription
    # ------------------------------------------------------------------

    def transcribe_file(self, wav_path: str) -> str | None:
        """Transcribe a 16 kHz / 16-bit / mono WAV file."""
        import wave

        self._rec.Reset()
        log.info("[STT] Transcribing file: %s", wav_path)

        with wave.open(wav_path, "rb") as wf:
            while True:
                data = wf.readframes(config.VAD_FRAME_SAMPLES)
                if not data:
                    break
                self._rec.AcceptWaveform(data)

        result = json.loads(self._rec.FinalResult())
        text = result.get("text", "").strip().lower()
        return text if text else None
