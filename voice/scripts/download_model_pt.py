"""
scripts/download_model_pt.py
============================
Downloads and extracts the official Vosk small Portuguese model (vosk-model-small-pt-0.3, ~31 MB).
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

VOICE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = VOICE_DIR / "models"
MODEL_NAME = "vosk-model-small-pt-0.3"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
TARGET_DIR = MODELS_DIR / MODEL_NAME
ZIP_PATH = MODELS_DIR / f"{MODEL_NAME}.zip"


def download_and_extract() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_DIR.exists():
        print(f"[OK] Model already exists at: {TARGET_DIR}")
        return

    print(f"Downloading {MODEL_NAME} (~31 MB) from {MODEL_URL} ...")

    def progress(count: int, block_size: int, total_size: int) -> None:
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading: {percent:3d}% complete")
        sys.stdout.flush()

    urllib.request.urlretrieve(MODEL_URL, ZIP_PATH, reporthook=progress)
    print("\nDownload complete. Extracting archive...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(MODELS_DIR)

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    print(f"[OK] Model successfully extracted to: {TARGET_DIR}")


if __name__ == "__main__":
    download_and_extract()
