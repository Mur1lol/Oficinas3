"""
game/config.py
==============
Constantes do jogo Jogador x Maquina.  AJUSTE os pinos e as medidas ao seu hardware.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Tabuleiro / Gantry ───────────────────────────────────────────────────────
SQUARE_MM = 50.0          # cada casa tem 5 cm; 16 colunas x 8 linhas = 800 x 400 mm
# Origem do Gantry (X=0, Y=0) = canto da FRENTE, a esquerda (lado das brancas / rank 1).
# True  -> a fileira 1 (brancas) fica perto de Y=0.
# False -> a fileira 8 fica perto de Y=0 (inverte o eixo Y).
INVERT_Y = True
HOME_CELL = (7, 0)        # celula logica (linha, coluna) mais proxima do home do Gantry
GANTRY_TIMEOUT_S = 180.0  # tempo maximo esperando o ESP32 terminar uma jogada

# ── LEDs (um LED RGB de cada lado do tabuleiro) ──────────────────────────────
# Pinos BCM (R, G, B).  >>> CONFIRA NA SUA PCB <<<
LED_PINS = {
    "player":   (17, 27, 22),
    "computer": (5, 6, 13),
}
LED_ACTIVE_HIGH = True    # False se for LED RGB de anodo comum
BLINK_ERROR_SECONDS = 3.0
BLINK_HZ = 2.0

# ── LCD 16x2 I2C ─────────────────────────────────────────────────────────────
LCD_I2C_ADDRESS = 0x27
LCD_I2C_PORT = 1
LCD_COLS = 16
LCD_ROWS = 2

# ── Speaker (TTS offline em ingles) ──────────────────────────────────────────
TTS_RATE = 150            # palavras por minuto
TTS_VOICE_HINT = "english"

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
    bundled = ROOT / "stockfish-windows-x86-64-universal" / "stockfish" / \
        "stockfish-windows-x86-64-universal.exe"
    if bundled.is_file():
        return str(bundled)
    return shutil.which("stockfish")
