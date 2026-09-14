"""
config.py
=========
Central configuration for the ChessAI 2.0 voice command system.

All tuneable constants live here. Edit this file to adapt the system
to your hardware (mic index, model paths, thresholds, timeouts, language).

Optimised for Raspberry Pi 3B+ running Raspberry Pi OS Bookworm (64-bit).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

# Active language: "pt-BR" (Português) or "en-US" (English)
LANGUAGE: str = "pt-BR"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root directory of this voice module (wherever config.py lives)
VOICE_DIR = Path(__file__).parent.resolve()

# Vosk models directory
MODELS_DIR = VOICE_DIR / "models"
VOSK_MODEL_EN_PATH = MODELS_DIR / "vosk-model-small-en-us-0.15"
VOSK_MODEL_PT_PATH = MODELS_DIR / "vosk-model-small-pt-0.3"

def get_model_path(lang: str | None = None) -> Path:
    """Return the Vosk model directory Path for the requested language."""
    selected_lang = (lang or LANGUAGE).lower()
    if selected_lang.startswith("pt"):
        return VOSK_MODEL_PT_PATH
    return VOSK_MODEL_EN_PATH

# Default model path pointing to the active language
VOSK_MODEL_PATH = get_model_path(LANGUAGE)

# openWakeWord ONNX model for "MAGNUS"
OWW_MODEL_PATH = MODELS_DIR / "magnus.onnx"

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
MIC_DEVICE_INDEX: int | None = None

# PyAudio internal ring-buffer size (in frames)
PYAUDIO_BUFFER_FRAMES: int = VAD_FRAME_SAMPLES

# ---------------------------------------------------------------------------
# Wake-word detection
# ---------------------------------------------------------------------------

# Canonical wake word string (used in log messages and output dicts)
WAKE_WORD: str = "MAGNUS"

# openWakeWord confidence threshold  [0.0 - 1.0]
OWW_THRESHOLD: float = 0.5
OWW_TRIGGER_LEVEL: int = 1

# Vosk-keyword fallback phrases that trigger wake word
WAKE_WORD_PHRASES_EN: list[str] = ["magnus"]
WAKE_WORD_PHRASES_PT: list[str] = ["magnus"]

def get_wake_word_phrases(lang: str | None = None) -> list[str]:
    selected_lang = (lang or LANGUAGE).lower()
    if selected_lang.startswith("pt"):
        return WAKE_WORD_PHRASES_PT
    return WAKE_WORD_PHRASES_EN

WAKE_WORD_PHRASES = get_wake_word_phrases(LANGUAGE)

# ---------------------------------------------------------------------------
# WebRTC VAD
# ---------------------------------------------------------------------------

# Aggressiveness mode  0 (lenient) - 3 (aggressive silence filtering)
VAD_AGGRESSIVENESS: int = 2

# How many consecutive silent frames before we consider speech ended
VAD_SILENCE_THRESHOLD: int = 20   # frames  (~600 ms at 30 ms/frame)

# ---------------------------------------------------------------------------
# Command & Confirmation Timeouts
# ---------------------------------------------------------------------------

# Maximum seconds to wait for a command after the wake word
COMMAND_TIMEOUT_S: float = 10.0

# Maximum seconds to wait for a YES/NO confirmation response
CONFIRM_TIMEOUT_S: float = 10.0

# Minimum audio frames to capture before processing (avoids empty transcripts)
COMMAND_MIN_FRAMES: int = 5

# ---------------------------------------------------------------------------
# Vocabulary, Grammar & Dictionaries
# ---------------------------------------------------------------------------

_COLUMNS = ["a", "b", "c", "d", "e", "f", "g", "h"]
_ROWS    = ["1", "2", "3", "4", "5", "6", "7", "8"]

# NATO phonetic alphabet for high-accuracy column recognition
NATO_PHONETICS: dict[str, str] = {
    "alpha":   "a",
    "bravo":   "b",
    "charlie": "c",
    "delta":   "d",
    "echo":    "e",
    "foxtrot": "f",
    "golf":    "g",
    "hotel":   "h",
}

# NATO phonetic alphabet for Portuguese
NATO_PHONETICS_PT: dict[str, str] = {
    "alfa":    "a",
    "bravo":   "b",
    "charlie": "c",
    "delta":   "d",
    "eco":     "e",
    "foxtrote":"f",
    "golf":    "g",
    "hotel":   "h",
}

# English Spoken Digits
SPOKEN_DIGITS_EN: dict[str, str] = {
    "one":   "1",
    "two":   "2",
    "three": "3",
    "four":  "4",
    "five":  "5",
    "six":   "6",
    "seven": "7",
    "eight": "8",
    "nine":  "9",
    "ten":   "10",
}

# Portuguese Spoken Digits
SPOKEN_DIGITS_PT: dict[str, str] = {
    "um":     "1",
    "dois":   "2",
    "três":   "3",
    "tres":   "3",
    "quatro": "4",
    "cinco":  "5",
    "seis":   "6",
    "sete":   "7",
    "oito":   "8",
    "nove":   "9",
    "dez":    "10",
}

# Spoken Column Phonetics
SPOKEN_COLUMNS_EN: dict[str, str] = {
    **NATO_PHONETICS,
    "ay": "a", 
    "bee": "b", 
    "see": "c", 
    "dee": "d",
    "ee": "e", 
    "eff": "f", 
    "gee": "g", 
    "age": "h", "hey": "h", "eight": "h"
}

SPOKEN_COLUMNS_PT: dict[str, str] = {
    **NATO_PHONETICS_PT,
    "á": "a",
    "be": "b",  "bê": "b",
    "ce": "c",  "cê": "c",
    "dê": "d",
    "é": "e", "ê": "e",
    "aga": "h", "agá": "h",
}

def get_spoken_digits(lang: str | None = None) -> dict[str, str]:
    selected_lang = (lang or LANGUAGE).lower()
    if selected_lang.startswith("pt"):
        return SPOKEN_DIGITS_PT
    return SPOKEN_DIGITS_EN

def get_spoken_columns(lang: str | None = None) -> dict[str, str]:
    selected_lang = (lang or LANGUAGE).lower()
    if selected_lang.startswith("pt"):
        return SPOKEN_COLUMNS_PT
    return SPOKEN_COLUMNS_EN

# Confirmation affirmative/negative word lists
CONFIRM_YES_EN: list[str] = ["yes", "yeah", "yep", "confirm"]
CONFIRM_NO_EN:  list[str] = ["no", "nope", "cancel"]

CONFIRM_YES_PT: list[str] = ["sim", "confirma", "confirmo", "positivo"]
CONFIRM_NO_PT:  list[str] = ["não", "cancela", "cancelar", "negativo"]

# Command Grammars for Vosk KaldiRecognizer
def get_grammar_words(lang: str | None = None) -> list[str]:
    """Return the list of allowed vocabulary words for the command recognizer."""
    selected_lang = (lang or LANGUAGE).lower()
    if selected_lang.startswith("pt"):
        words = set(
            ["jogar", "novo", "jogo", "continuar", "retomar", "desistir", "mover"]
            + list(SPOKEN_DIGITS_PT.keys())
            + _COLUMNS
            + list(SPOKEN_COLUMNS_PT.keys())
            + ["magnus", "[unk]"]
        )
    else:
        words = set(
            ["play", "new", "game", "resume", "resign", "move"]
            + list(SPOKEN_DIGITS_EN.keys())
            + _COLUMNS
            + list(SPOKEN_COLUMNS_EN.keys())
            + ["magnus", "magnets", "[unk]"]
        )
    return sorted(list(words))


def get_confirmation_grammar(lang: str | None = None) -> list[str]:
    """Return the restricted vocabulary words for the confirmation recognizer."""
    selected_lang = (lang or LANGUAGE).lower()
    if selected_lang.startswith("pt"):
        words = set(CONFIRM_YES_PT + CONFIRM_NO_PT + ["[unk]"])
    else:
        words = set(CONFIRM_YES_EN + CONFIRM_NO_EN + ["[unk]"])
    return sorted(list(words))


# Backward compatibility aliases
SPOKEN_DIGITS = SPOKEN_DIGITS_PT if LANGUAGE.lower().startswith("pt") else SPOKEN_DIGITS_EN
VOSK_GRAMMAR_WORDS = get_grammar_words(LANGUAGE)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = "INFO"   # DEBUG | INFO | WARNING | ERROR
LOG_FORMAT: str = "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s"
LOG_DATE_FORMAT: str = "%H:%M:%S"
