"""
stt_engine.py
=============
Speech-to-Text engine for ChessAI 2.0.

Uses Vosk (KaldiRecognizer) with:
  - Grammar restriction  : limits vocabulary to chess commands only,
                           dramatically improving accuracy on Pi 3B+.
  - WebRTC VAD filtering : skips silent frames to save CPU cycles.
  - Timeout safety       : returns None if no speech within the deadline.

Public API
----------
    engine = VoskSTTEngine()
    with AudioStream() as stream:
        text = engine.transcribe(stream, timeout=4.0)
        # e.g. "move e two e four" or None on timeout
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

    The recognizer is loaded once at construction time (~100 MB RAM).
    Subsequent calls to transcribe() reuse the same model — no reload cost.

    Parameters
    ----------
    model_path : str | None
        Path to the Vosk model directory.  Defaults to config.VOSK_MODEL_PATH.
    """

    def __init__(self, model_path: str | None = None) -> None:
        from vosk import KaldiRecognizer, Model, SetLogLevel  # type: ignore

        SetLogLevel(-1)  # suppress verbose Kaldi output

        path = str(model_path or config.VOSK_MODEL_PATH)
        log.info("[STT] Loading Vosk model from %s ...", path)
        self._model = Model(path)
        log.info("[STT] Vosk model loaded.")

        # Grammar-restricted recognizer
        grammar = json.dumps(config.VOSK_GRAMMAR_WORDS)
        self._rec = KaldiRecognizer(
            self._model, config.SAMPLE_RATE, grammar
        )
        self._rec.SetWords(True)   # include word-level timestamps in result

        # WebRTC VAD
        self._vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)

        log.info(
            "[STT] Ready. VAD aggressiveness=%d  grammar=%d words",
            config.VAD_AGGRESSIVENESS,
            len(config.VOSK_GRAMMAR_WORDS),
        )

    # ------------------------------------------------------------------
    # Main transcription method
    # ------------------------------------------------------------------

    def transcribe(
        self,
        stream: AudioStream,
        timeout: float = config.COMMAND_TIMEOUT_S,
    ) -> str | None:
        """
        Listen to *stream* until speech is fully recognised or *timeout* expires.

        Behaviour
        ---------
        1. Frames are fed to WebRTC VAD.
        2. Only speech frames are passed to KaldiRecognizer.
        3. When VAD detects *VAD_SILENCE_THRESHOLD* consecutive silent frames
           after speech has been heard, we flush Vosk and return the result.
        4. Hard timeout: if *timeout* seconds elapse before any result,
           return None.

        Returns
        -------
        str | None
            Normalised lowercase transcript, or None on timeout / empty result.
        """
        self._rec.Reset()

        deadline = time.monotonic() + timeout
        speech_started = False
        silence_count = 0
        frames_captured = 0

        log.info("[STT] Listening for command (timeout=%.1fs)...", timeout)

        for frame in stream:
            now = time.monotonic()
            if now >= deadline:
                log.warning("[STT] Timeout — no command recognised.")
                break

            frames_captured += 1

            # VAD gate
            try:
                is_speech = self._vad.is_speech(frame, config.SAMPLE_RATE)
            except Exception:
                is_speech = True  # if VAD fails, pass the frame through

            if is_speech:
                speech_started = True
                silence_count = 0
                self._rec.AcceptWaveform(frame)
            else:
                if speech_started:
                    silence_count += 1

            # After enough silence post-speech, flush the recognizer
            if (
                speech_started
                and silence_count >= config.VAD_SILENCE_THRESHOLD
                and frames_captured >= config.COMMAND_MIN_FRAMES
            ):
                log.debug("[STT] End of speech detected — flushing recognizer.")
                break

        # Retrieve final result
        result_json = json.loads(self._rec.FinalResult())
        text = result_json.get("text", "").strip().lower()

        if text:
            log.info("[STT] Recognised: '%s'", text)
        else:
            log.warning("[STT] Empty transcript.")
            return None

        return text

    # ------------------------------------------------------------------
    # Convenience: transcribe from a WAV file (for testing)
    # ------------------------------------------------------------------

    def transcribe_file(self, wav_path: str) -> str | None:
        """
        Transcribe a WAV file instead of a live stream.

        The file must be 16 kHz / 16-bit / mono PCM.
        Useful for offline testing without a microphone.

        Parameters
        ----------
        wav_path : str
            Path to the WAV file.

        Returns
        -------
        str | None
            Recognised text or None if nothing was recognised.
        """
        import wave
        import struct

        self._rec.Reset()
        log.info("[STT] Transcribing file: %s", wav_path)

        with wave.open(wav_path, "rb") as wf:
            if wf.getnchannels() != 1:
                raise ValueError("WAV must be mono (1 channel).")
            if wf.getsampwidth() != 2:
                raise ValueError("WAV must be 16-bit.")
            if wf.getframerate() != config.SAMPLE_RATE:
                raise ValueError(
                    f"WAV must be {config.SAMPLE_RATE} Hz, "
                    f"got {wf.getframerate()} Hz."
                )

            chunk_size = config.VAD_FRAME_BYTES
            while True:
                data = wf.readframes(config.VAD_FRAME_SAMPLES)
                if not data:
                    break
                if len(data) == chunk_size:
                    self._rec.AcceptWaveform(data)

        result_json = json.loads(self._rec.FinalResult())
        text = result_json.get("text", "").strip().lower()
        log.info("[STT] File transcript: '%s'", text)
        return text if text else None
