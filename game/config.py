"""
game/config.py
==============
Constantes do jogo Jogador x Maquina.  AJUSTE os pinos e as medidas ao seu hardware.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Tabuleiro / Gantry ───────────────────────────────────────────────────────
# Curso util calibrado do ESP32 (Gantry.h: CURSO_X=350, CURSO_Y=650)
GANTRY_SIZE_MM = (350.0, 650.0)

# Tamanho de cada casa do tabuleiro em mm:
# Layout logico: 14 colunas x 8 linhas (layout fisico 27x15 contando as bordas internas)
# Eixo X (linhas 0..7): 7 passos de 40mm = 280 mm (cabe dentro do limite de 350 mm)
# Eixo Y (colunas 0..13): 13 passos de 40mm = 520 mm (cabe dentro do limite de 650 mm)
SQUARE_MM = 40.0

# O ponto (0, 0) do ESP32 fica na DIREITA EM CIMA (Top-Right) por conta dos sensores fisicos.
# Row 0 (Pretas / topo) -> X = 0.0 mm
# Col 13 (Cemiterio direito / direita) -> Y = 0.0 mm
HOME_CELL = (0, 13)       # celula logica (linha, coluna) correspondente a (0, 0)
HOME_PHYS = (0, 26)       # celula fisica (pr, pc) correspondente a (0, 0)

# Offset opcional se o tabuleiro tiver margem em relacao aos sensores de homing
GANTRY_OFFSET_MM = (0.0, 0.0)
GANTRY_TIMEOUT_S = 180.0  # tempo maximo esperando o ESP32 terminar uma jogada


def game_to_gantry(x: float, y: float) -> tuple[float, float]:
    """Coordenada em mm do planner -> coordenada absoluta do ESP32 (mm).

    X: 0..350 mm (linhas: 0 no topo/pretas -> 280 na base/brancas).
    Y: 0..650 mm (colunas: 0 na direita/cem_dir -> 520 na esquerda/cem_esq).
    Home (0, 0): Direita em cima (Top-Right, sensores fisicos).
    """
    gx = x + GANTRY_OFFSET_MM[0]
    gy = y + GANTRY_OFFSET_MM[1]
    gx = max(0.0, min(GANTRY_SIZE_MM[0], gx))
    gy = max(0.0, min(GANTRY_SIZE_MM[1], gy))
    return round(gx, 2), round(gy, 2)

# ── LEDs (um LED RGB de cada lado do tabuleiro) ──────────────────────────────
# Pinos BCM (R, G, B) da PCB C: brancas 17/27/22 (J5), pretas 5/6/13 (J6).
# O jogador joga de brancas, entao "player" = painel das brancas.
LED_PINS = {
    "player":   (17, 27, 22),
    "computer": (5, 6, 13),
}
LED_ACTIVE_HIGH = True    # catodo comum (PCB D: pino K no GND)
BLINK_ERROR_SECONDS = 3.0
BLINK_HZ = 2.0

# ── LCDs 20x4 I2C (um por painel, o mesmo texto nos dois) ─────────────────────
# 0x27 = painel das brancas (A0-A2 abertos), 0x26 = pretas (ponte em A0).
# Se o backpack for PCF8574A os enderecos sao 0x3F / 0x3E (veja com i2cdetect -y 1).
LCD_I2C_ADDRESSES = (0x27, 0x26)
LCD_I2C_PORT = 1
LCD_COLS = 20
LCD_ROWS = 4

# ── Speaker (TTS offline em ingles) ──────────────────────────────────────────
TTS_RATE = 150            # palavras por minuto (apenas fallback pyttsx3/espeak)
TTS_VOICE_HINT = "english"
# Voz neural (Piper). Baixe com: uv run python baixar_voz.py
# Se o modelo nao existir, cai para pyttsx3/espeak (voz robotica).
TTS_PIPER_MODEL = ROOT / "voice" / "models" / "en_US-amy-low.onnx"

# ── Voz ───────────────────────────────────────────────────────────────────────
VOICE_CONFIG = ROOT / "voice.toml"

# ── Stockfish ─────────────────────────────────────────────────────────────────
# nivel -> (Skill Level, UCI_Elo ou None)
SKILL_MAP = {
    1: (0, 800),
    2: (3, 1000),
    3: (6, 1200),
    4: (9, 1500),
    5: (12, 1700),
    6: (15, 2000),
    7: (18, 2300),
    8: (20, None),
}
DEFAULT_LEVEL = 4
ENGINE_THINK_SECONDS = 0.5


def find_stockfish() -> str | None:
    if sys.platform.startswith("win"):
        bundled = ROOT / "stockfish-windows-x86-64-universal" / "stockfish" / \
            "stockfish-windows-x86-64-universal.exe"
        if bundled.is_file():
            return str(bundled)

    # No Linux / Raspberry Pi: procura no PATH ou no diretorio padrao do Debian/Ubuntu
    for candidate in [
        shutil.which("stockfish"),
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
    ]:
        if candidate and Path(candidate).is_file():
            return str(candidate)

    return shutil.which("stockfish")
