"""Baixa a voz neural do Piper (TTS offline, bem menos robotica que o espeak).

    uv run python baixar_voz.py                      # en_US-lessac-medium (padrao, ~60 MB)
    uv run python baixar_voz.py en_US-amy-medium     # outra voz

Vozes disponiveis: https://rhasspy.github.io/piper-samples/
Depois ajuste game/config.py (TTS_PIPER_MODEL) se trocar de voz.
"""

import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
DEST = Path(__file__).resolve().parent / "voice" / "models"


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "en_US-lessac-medium"
    lang_region, voice, quality = name.split("-")          # en_US, lessac, medium
    lang = lang_region.split("_")[0]
    DEST.mkdir(parents=True, exist_ok=True)
    for suffix in (".onnx", ".onnx.json"):
        url = f"{BASE}/{lang}/{lang_region}/{voice}/{quality}/{name}{suffix}"
        out = DEST / f"{name}{suffix}"
        print(f"Baixando {url}")
        urllib.request.urlretrieve(url, out)
        print(f"  -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("\nPronto! Teste com: uv run python teste_integracao.py speaker")
    return 0


if __name__ == "__main__":
    sys.exit(main())
