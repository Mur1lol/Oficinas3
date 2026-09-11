#!/usr/bin/env bash
# =============================================================================
# scripts/install_pi.sh
# Installation script for ChessAI 2.0 Voice Command System
# Target: Raspberry Pi 3B+, Raspberry Pi OS Bookworm 64-bit, Python 3.11
# =============================================================================
set -euo pipefail

VOICE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${VOICE_DIR}/models"
VOSK_MODEL_NAME="vosk-model-small-en-us-0.15"
VOSK_MODEL_URL="https://alphacephei.com/vosk/models/${VOSK_MODEL_NAME}.zip"

echo "========================================================"
echo "  ChessAI 2.0 — Voice System Installer"
echo "  Target: Raspberry Pi 3B+ (Bookworm 64-bit)"
echo "========================================================"
echo ""

# --- 1. System packages ---
echo "[1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    portaudio19-dev \
    libopenblas-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    wget \
    unzip

echo "      System packages installed."
echo ""

# --- 2. Python virtual environment ---
echo "[2/5] Creating Python virtual environment at ${VOICE_DIR}/.venv ..."
python3 -m venv "${VOICE_DIR}/.venv"
source "${VOICE_DIR}/.venv/bin/activate"
pip install --upgrade pip wheel setuptools -q
echo "      Virtual environment ready."
echo ""

# --- 3. Python dependencies ---
echo "[3/5] Installing Python dependencies..."
pip install -r "${VOICE_DIR}/requirements.txt"

# Fallback: if onnxruntime failed (common on armv7/32-bit), skip OWW
if ! python3 -c "import onnxruntime" 2>/dev/null; then
    echo "      WARNING: onnxruntime not installed — openWakeWord unavailable."
    echo "      The system will use the Vosk keyword fallback for wake-word detection."
    echo "      To try again: pip install onnxruntime"
fi
echo "      Python dependencies installed."
echo ""

# --- 4. Download Vosk model ---
echo "[4/5] Downloading Vosk model (${VOSK_MODEL_NAME}, ~40 MB)..."
mkdir -p "${MODEL_DIR}"

if [ -d "${MODEL_DIR}/${VOSK_MODEL_NAME}" ]; then
    echo "      Model already exists — skipping download."
else
    wget -q --show-progress \
        -O "${MODEL_DIR}/${VOSK_MODEL_NAME}.zip" \
        "${VOSK_MODEL_URL}"
    unzip -q "${MODEL_DIR}/${VOSK_MODEL_NAME}.zip" -d "${MODEL_DIR}"
    rm "${MODEL_DIR}/${VOSK_MODEL_NAME}.zip"
    echo "      Vosk model extracted to ${MODEL_DIR}/${VOSK_MODEL_NAME}/"
fi
echo ""

# --- 5. MAGNUS wake-word model (manual step) ---
echo "[5/5] openWakeWord MAGNUS model:"
if [ -f "${MODEL_DIR}/magnus.onnx" ]; then
    echo "      magnus.onnx found — openWakeWord will be used."
else
    echo ""
    echo "  ┌──────────────────────────────────────────────────────────────┐"
    echo "  │  OPTIONAL: Train the MAGNUS wake-word model                 │"
    echo "  │                                                              │"
    echo "  │  1. Open the openWakeWord training Colab notebook:          │"
    echo "  │     https://colab.research.google.com/drive/                │"
    echo "  │     1q1oe2zOyZp7UsB3jJiQ1BZNV9LBOAsb5                      │"
    echo "  │                                                              │"
    echo "  │  2. Enter wake word: MAGNUS                                  │"
    echo "  │     Phonetic hint:   MAG-NUS                                 │"
    echo "  │                                                              │"
    echo "  │  3. Download the generated .onnx file                        │"
    echo "  │                                                              │"
    echo "  │  4. Copy it to:                                              │"
    echo "  │     ${MODEL_DIR}/magnus.onnx  │"
    echo "  │                                                              │"
    echo "  │  The system works NOW without it (Vosk keyword fallback).   │"
    echo "  └──────────────────────────────────────────────────────────────┘"
fi
echo ""

# --- 6. Microphone check ---
echo "Checking audio devices..."
python3 - <<'PYEOF'
import sys
sys.path.insert(0, "voice")
try:
    from audio_capture import list_audio_devices
    devices = list_audio_devices()
    if devices:
        print("  Available input devices:")
        for d in devices:
            print(f"    [{d['index']}] {d['name']} ({d['channels']}ch, {d['sample_rate']} Hz)")
    else:
        print("  WARNING: No input devices found. Connect your USB microphone.")
except Exception as e:
    print(f"  Could not list audio devices: {e}")
PYEOF
echo ""

# --- Done ---
echo "========================================================"
echo "  Installation complete!"
echo ""
echo "  To run the voice system:"
echo "    source ${VOICE_DIR}/.venv/bin/activate"
echo "    cd ${VOICE_DIR}"
echo "    python main.py"
echo ""
echo "  To run tests:"
echo "    python -m pytest tests/ -v"
echo ""
echo "  To list audio devices:"
echo "    python main.py --list-devices"
echo "========================================================"
