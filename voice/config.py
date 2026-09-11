"""
config.py
=========
Central configuration for the ChessAI 2.0 voice command system.

All tuneable constants live here. Edit this file to adapt the system
to your hardware (mic index, model paths, thresholds, timeouts).

Optimised for Raspberry Pi 3B+ running Raspberry Pi OS Bookworm (64-bit).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root directory of this voice module (wherever config.py lives)
VOICE_DIR = Path(__file__).parent.resolve()

# Vosk model directory  (download separately — see README.md)
# Expected: voice/models/vosk-model-small-en-us-0.15/
VOSK_MODEL_PATH = VOICE_DIR / "models" / "vosk-model-small-en-us-0.15"

# openWakeWord ONNX model for "MAGNUS"  (train via Colab — see README.md)
# If this file does not exist the system falls back to Vosk-keyword mode.
OWW_MODEL_PATH = VOICE_DIR / "models" / "magnus.onnx"

# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------

# Sample rate required by Vosk, openWakeWord, and WebRTC VAD
SAMPLE_RATE: int = 16_000          # Hz

# Bits per sample (must be 16 for Vosk / VAD)
SAMPLE_WIDTH: int = 2              # bytes  (16-bit)

# Mono audio
CHANNELS: int = 1

# Frame duration for WebRTC VAD  (10 | 20 | 30 ms)
VAD_FRAME_MS: int = 30            # ms

# Derived: samples per VAD frame
VAD_FRAME_SAMPLES: int = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)  # 480

# Derived: bytes per VAD frame
VAD_FRAME_BYTES: int = VAD_FRAME_SAMPLES * SAMPLE_WIDTH           # 960

# PyAudio device index for the USB microphone.
# Set to None for auto-detection (first USB mic found).
# Override with an integer if auto-detection picks the wrong device.
MIC_DEVICE_INDEX: int | None = None

# PyAudio internal ring-buffer size (in frames)
PYAUDIO_BUFFER_FRAMES: int = VAD_FRAME_SAMPLES

# ---------------------------------------------------------------------------
# Wake-word detection
# ---------------------------------------------------------------------------

# Canonical wake word string (used in log messages and output dicts)
WAKE_WORD: str = "MAGNUS"

# openWakeWord confidence threshold  [0.0 - 1.0]
# Lower = more sensitive (more false positives)
# Higher = more strict (may miss some utterances)
OWW_THRESHOLD: float = 0.5

# Number of consecutive frames above threshold before triggering
OWW_TRIGGER_LEVEL: int = 1

# Vosk-keyword fallback: phrases that count as the wake word
# (used when magnus.onnx is absent)
WAKE_WORD_PHRASES: list[str] = [
    "magnus",
    "magnets",   # common mis-hear
    "magnus chess",
]

# ---------------------------------------------------------------------------
# WebRTC VAD
# ---------------------------------------------------------------------------

# Aggressiveness mode  0 (lenient) - 3 (aggressive silence filtering)
# Mode 2 works well for a quiet room; use 1 if the mic picks up more noise.
VAD_AGGRESSIVENESS: int = 2

# How many consecutive silent frames before we consider speech ended
VAD_SILENCE_THRESHOLD: int = 20   # frames  (~600 ms at 30 ms/frame)

# ---------------------------------------------------------------------------
# Command listening
# ---------------------------------------------------------------------------

# Maximum seconds to wait for a command after the wake word
COMMAND_TIMEOUT_S: float = 4.0

# Minimum audio frames to capture before processing (avoids empty transcripts)
COMMAND_MIN_FRAMES: int = 5

# ---------------------------------------------------------------------------
# Grammar - Vosk vocabulary restriction
# ---------------------------------------------------------------------------

_COLUMNS = ["a", "b", "c", "d", "e", "f", "g", "h"]
_ROWS    = ["1", "2", "3", "4", "5", "6", "7", "8"]

# All 64 squares as spoken tokens
CHESS_SQUARES: list[str] = [f"{c}{r}" for c in _COLUMNS for r in _ROWS]

# Spoken digit words that map to row numbers
SPOKEN_DIGITS: dict[str, str] = {
    "one":   "1",
    "two":   "2",
    "three": "3",
    "four":  "4",
    "five":  "5",
    "six":   "6",
    "seven": "7",
    "eight": "8",
}

# Full grammar word list injected into KaldiRecognizer
VOSK_GRAMMAR_WORDS: list[str] = (
    ["move", "from", "to"]
    + CHESS_SQUARES
    + _COLUMNS
    + _ROWS
    + list(SPOKEN_DIGITS.keys())
    + ["magnus", "magnets"]
    + ["[unk]"]
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = "INFO"   # DEBUG | INFO | WARNING | ERROR
LOG_FORMAT: str = "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s"
LOG_DATE_FORMAT: str = "%H:%M:%S"
