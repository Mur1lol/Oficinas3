"""
tests/test_stt_mock.py
======================
Integration tests for voice/stt_engine.py using pre-recorded WAV files.

These tests simulate real speech by feeding synthetic or pre-recorded
WAV audio through the Vosk STT engine without requiring a live microphone.

Run with:
    cd voice/
    python -m pytest tests/test_stt_mock.py -v

Prerequisites
-------------
* Vosk model downloaded to voice/models/vosk-model-small-en-us-0.15/
* WAV test fixtures in tests/audio/ (see generate_test_audio() below).

Generating test audio
---------------------
Run the helper at the bottom of this file to generate synthetic WAV files
using the pyttsx3 TTS engine (offline):

    python tests/test_stt_mock.py --generate

This creates tests/audio/move_e2_e4.wav, move_g1_f3.wav, etc.
"""

from __future__ import annotations

import os
import struct
import sys
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

# Skip this entire module if Vosk model is not present
import config

VOSK_AVAILABLE = config.VOSK_MODEL_PATH.exists()
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")

pytestmark = pytest.mark.skipif(
    not VOSK_AVAILABLE,
    reason=(
        f"Vosk model not found at {config.VOSK_MODEL_PATH}. "
        "Run scripts/install_pi.sh first."
    ),
)


# ---------------------------------------------------------------------------
# Helper: create a minimal silent WAV (for smoke-testing the STT plumbing)
# ---------------------------------------------------------------------------

def _create_silent_wav(path: str, duration_s: float = 1.0) -> None:
    """Generate a silent 16 kHz / 16-bit / mono WAV file."""
    num_samples = int(config.SAMPLE_RATE * duration_s)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(struct.pack("<" + "h" * num_samples, *([0] * num_samples)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSTTWithSilentAudio:
    """Smoke tests: verify the STT engine handles silence gracefully."""

    def setup_method(self):
        from stt_engine import VoskSTTEngine
        self.stt = VoskSTTEngine()
        os.makedirs(AUDIO_DIR, exist_ok=True)
        self.silent_wav = os.path.join(AUDIO_DIR, "_silent_test.wav")
        _create_silent_wav(self.silent_wav, duration_s=0.5)

    def teardown_method(self):
        if os.path.exists(self.silent_wav):
            os.remove(self.silent_wav)

    def test_silent_wav_returns_none(self):
        """Silent audio should return None (nothing recognised)."""
        result = self.stt.transcribe_file(self.silent_wav)
        assert result is None or result.strip() == ""

    def test_engine_loads_without_error(self):
        """Vosk model should load successfully."""
        from stt_engine import VoskSTTEngine
        engine = VoskSTTEngine()
        assert engine is not None


class TestSTTWithRealAudio:
    """
    Integration tests using real WAV recordings.

    These tests only run if the WAV files exist in tests/audio/.
    Generate them with: python tests/test_stt_mock.py --generate
    """

    def setup_method(self):
        from stt_engine import VoskSTTEngine
        self.stt = VoskSTTEngine()

    EXPECTED_TRANSCRIPTS = [
        ("move_e2_e4.wav",  "move e2 e4"),
        ("move_g1_f3.wav",  "move g1 f3"),
        ("move_a7_a8.wav",  "move a7 a8"),
    ]

    @pytest.mark.parametrize("filename, expected_contains", EXPECTED_TRANSCRIPTS)
    def test_transcribe_wav_file(self, filename, expected_contains):
        wav_path = os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(wav_path):
            pytest.skip(f"Test WAV not found: {wav_path}. Run --generate first.")

        text = self.stt.transcribe_file(wav_path)
        assert text is not None, f"STT returned None for {filename}"

        # Check that key tokens are present (not exact match — STT may vary)
        for token in expected_contains.split():
            assert token in text.lower(), (
                f"Expected '{token}' in transcript '{text}' for {filename}"
            )


class TestEndToEndParseValidate:
    """
    End-to-end: STT transcript -> parse -> validate pipeline.

    Uses pre-defined transcript strings to bypass the STT model,
    testing parser + validator integration only.
    """

    PIPELINE_CASES = [
        # (transcript,                     valid, from,  to)
        ("move e2 e4",                     True,  "E2", "E4"),
        ("move g1 f3",                     True,  "G1", "F3"),
        ("move a seven a eight",           True,  "A7", "A8"),
        ("move from e two to e four",      True,  "E2", "E4"),
        ("",                               False, None, None),
        ("move xyz xyz",                   False, None, None),
        ("hello world",                    False, None, None),
    ]

    @pytest.mark.parametrize(
        "transcript, expected_valid, expected_from, expected_to",
        PIPELINE_CASES,
    )
    def test_parse_validate_pipeline(
        self, transcript, expected_valid, expected_from, expected_to
    ):
        from parser import parse
        from validator import validate

        command, error = parse(transcript)
        result = validate(command, reason=error)

        assert result["valid"] == expected_valid

        if expected_valid:
            assert result["from"] == expected_from
            assert result["to"] == expected_to
        else:
            assert "reason" in result


# ---------------------------------------------------------------------------
# Audio generation helper  (run directly, not via pytest)
# ---------------------------------------------------------------------------

def generate_test_audio() -> None:
    """
    Generate test WAV files using pyttsx3 (offline TTS).

    Install: pip install pyttsx3
    Usage:   python tests/test_stt_mock.py --generate
    """
    try:
        import pyttsx3  # type: ignore
    except ImportError:
        print("pyttsx3 not installed. Run: pip install pyttsx3")
        sys.exit(1)

    import tempfile
    import shutil

    os.makedirs(AUDIO_DIR, exist_ok=True)
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)

    phrases = {
        "move_e2_e4.wav": "move e two e four",
        "move_g1_f3.wav": "move g one f three",
        "move_a7_a8.wav": "move a seven a eight",
        "move_h1_h8.wav": "move h one h eight",
    }

    for filename, phrase in phrases.items():
        out_path = os.path.join(AUDIO_DIR, filename)
        print(f"Generating: {filename} <- '{phrase}'")
        engine.save_to_file(phrase, out_path)
        engine.runAndWait()
        print(f"  Saved: {out_path}")

    print(f"\nGenerated {len(phrases)} test WAV files in {AUDIO_DIR}/")
    print("NOTE: These are at the TTS engine's native sample rate.")
    print("Convert to 16 kHz mono if needed:")
    print("  ffmpeg -i input.wav -ar 16000 -ac 1 output.wav")


if __name__ == "__main__":
    if "--generate" in sys.argv:
        generate_test_audio()
    else:
        print("Usage: python tests/test_stt_mock.py --generate")
        print("Or run via pytest: python -m pytest tests/test_stt_mock.py -v")
