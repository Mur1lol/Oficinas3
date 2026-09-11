# ChessAI 2.0 — Voice Command System

Offline voice recognition pipeline for Raspberry Pi 3B+.

Detects the wake word **MAGNUS** and parses chess `MOVE <SRC> <DST>` commands
into validated structured output — no internet connection required.

---

## Architecture

```
voice/
+-- config.py           # All constants (model paths, timeouts, grammar)
+-- audio_capture.py    # PyAudio wrapper -> 16 kHz PCM frame generator
+-- wake_word.py        # openWakeWord ONNX / Vosk keyword fallback
+-- stt_engine.py       # Vosk STT + WebRTC VAD, grammar-restricted
+-- parser.py           # Raw text -> MoveCommand("E2", "E4")
+-- validator.py        # Validates A-H x 1-8 coordinates
+-- main.py             # State machine + public API + CLI
+-- requirements.txt    # Pinned Python dependencies
+-- models/
|   +-- vosk-model-small-en-us-0.15/   # Vosk model (download separately)
|   +-- magnus.onnx                    # OWW model (train separately)
+-- scripts/
|   +-- install_pi.sh   # One-shot setup for Raspberry Pi OS Bookworm
+-- tests/
    +-- test_parser.py      # 25+ parametrised parser tests
    +-- test_validator.py   # All 64 squares + edge cases
    +-- test_stt_mock.py    # STT integration + WAV file tests
    +-- audio/              # WAV test fixtures (generate with --generate)
```

---

## State Machine

```
IDLE
  |
  v
WAKE_WORD_LISTEN  <---------------------------+
  |                                           |
  | "MAGNUS" detected                         |
  v                                           |
COMMAND_LISTEN  (4 s timeout)                 |
  |                                           |
  | speech recognised                         | (timeout/error)
  v                                           |
PARSE                                         |
  |                                           |
  v                                           |
VALIDATE                                      |
  |                                           |
  v                                           |
EMIT result dict  ----------------------------+
```

---

## Quick Start (Raspberry Pi)

```bash
# 1. Clone / copy the voice/ directory onto the Pi
cd ~/chessai

# 2. Run the installer (downloads Vosk model, creates venv)
bash voice/scripts/install_pi.sh

# 3. Activate the virtual environment
source voice/.venv/bin/activate

# 4. Run
python voice/main.py
```

---

## Wake Word Model (MAGNUS)

The system ships with a **Vosk keyword fallback** that works immediately.
For the best accuracy, train the openWakeWord ONNX model:

1. Open the [openWakeWord training Colab](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1BZNV9LBOAsb5)
2. Enter wake word: `MAGNUS` (phonetic hint: `MAG-NUS`)
3. Download the generated `.onnx` file
4. Copy to `voice/models/magnus.onnx`

The pipeline auto-detects the ONNX file on startup.

---

## CLI Usage

```bash
# Continuous live mic loop (Ctrl+C to stop)
python main.py

# Listen for exactly one command, print JSON, exit
python main.py --once

# Transcribe a WAV file (no wake-word gate)
python main.py --file tests/audio/move_e2_e4.wav

# List audio input devices
python main.py --list-devices

# Enable debug logging
python main.py --debug
```

---

## Integration API

```python
from voice.main import VoiceCommandPipeline

pipeline = VoiceCommandPipeline()

# Wait for wake word + one command
result = pipeline.listen_once()
# {"wake_word": "MAGNUS", "command": "MOVE", "from": "E2", "to": "E4", "valid": True}

if result["valid"]:
    src = result["from"]   # "E2"
    dst = result["to"]     # "E4"
```

---

## Expected Output

**Success:**
```json
{
  "wake_word": "MAGNUS",
  "command": "MOVE",
  "from": "E2",
  "to": "E4",
  "valid": true
}
```

**Timeout:**
```json
{
  "valid": false,
  "reason": "No command heard within 4 seconds after wake word. Returning to listen mode."
}
```

**Invalid square:**
```json
{
  "valid": false,
  "reason": "Invalid destination square -- Invalid row '9' in 'E9'. Valid rows: 1-8."
}
```

---

## Supported Spoken Formats

| You say | Parsed as |
|---|---|
| "Move E2 E4" | MOVE E2 E4 |
| "move e two e four" | MOVE E2 E4 |
| "MOVE G1 F3" | MOVE G1 F3 |
| "move from a seven to a eight" | MOVE A7 A8 |
| "Move from E2 to E4" | MOVE E2 E4 |

---

## Running Tests

```bash
cd voice/
# Parser + validator tests (no model required)
python -m pytest tests/test_parser.py tests/test_validator.py -v

# Full suite (requires Vosk model)
python -m pytest tests/ -v

# Generate WAV test fixtures
python tests/test_stt_mock.py --generate
```

---

## Error Handling

| Situation | Behaviour |
|---|---|
| Wake word not detected | Keeps listening indefinitely |
| Timeout after wake word | Returns `{"valid": false, ...}`, resets to listening |
| Invalid chess square | Returns `{"valid": false, "reason": "..."}` |
| Null move (src = dst) | Returns `{"valid": false, ...}` |
| Empty transcript | Returns `{"valid": false, ...}` |
| No Vosk model | Raises `FileNotFoundError` at startup |
| No ONNX model | Falls back to Vosk keyword detection silently |
| Background noise | VAD filters silence; Vosk grammar limits false matches |

---

## Performance Profile (Pi 3B+)

| Component | CPU (idle) | CPU (active) | RAM |
|---|---|---|---|
| AudioStream (PyAudio) | < 1% | < 1% | ~2 MB |
| WebRTC VAD | < 1% | < 1% | < 1 MB |
| openWakeWord (OWW) | 10–15% | 10–15% | ~50 MB |
| Vosk STT | 0% (off) | 40–70% | ~100 MB |
| **Total (wake mode)** | **~15%** | — | **~155 MB** |
| **Total (STT active)** | — | **~70%** | **~155 MB** |

STT is only active for ~2–4 seconds per command cycle.

---

## Hardware Requirements

- Raspberry Pi 3B+
- Raspberry Pi OS Bookworm 64-bit
- Python 3.11
- USB microphone (16 kHz capable)
- 1 GB RAM (system + voice pipeline < 500 MB)
