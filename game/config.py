"""
game/config.py
==============
Constantes do jogo Jogador x Maquina.  AJUSTE os pinos e as medidas ao seu hardware.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Tabuleiro / Gantry ───────────────────────────────────────────────────────
SQUARE_MM = 50.0          # cada casa tem 5 cm (50 mm) por conta dos vaos
# Origem do Gantry (X=0, Y=0) = canto da FRENTE, a esquerda (lado das brancas / rank 1).
# True  -> a fileira 1 (brancas) fica perto de Y=0.
# False -> a fileira 8 fica perto de Y=0 (inverte o eixo Y).
INVERT_Y = True
HOME_CELL = (7, 0)        # celula logica (linha, coluna) mais proxima do home do Gantry

# O jogo pensa em x = 0..800 (16 colunas) e y = 0..400 (8 linhas). No gantry montado
# o lado de 800 mm e o eixo Y (perfis de 900, frente -> tras) e o de 400 mm e o X
# (viga, esquerda -> direita, olhando com os motores na frente). Home = (400, 0).
# game_to_gantry() converte; ajuste os flags com o teste "cantos" (teste_integracao.py).
GANTRY_SWAP_XY = True     # True: x do jogo -> Y do gantry, y do jogo -> X do gantry
GANTRY_FLIP_X = False     # depois da troca: X_gantry = 400 - X_gantry
GANTRY_FLIP_Y = False     # depois da troca: Y_gantry = 800 - Y_gantry
GANTRY_OFFSET_MM = (0.0, 0.0)   # (X, Y) somado no fim: onde fica o canto do tabuleiro
GANTRY_SIZE_MM = (400.0, 800.0) # curso util (X, Y), igual ao CURSO_X/CURSO_Y do ESP32


def game_to_gantry(x: float, y: float) -> tuple[float, float]:
    """Coordenada do planner (mm) -> coordenada do ESP32 (mm)."""
    if GANTRY_SWAP_XY:
        x, y = y, x
    if GANTRY_FLIP_X:
        x = GANTRY_SIZE_MM[0] - x
    if GANTRY_FLIP_Y:
        y = GANTRY_SIZE_MM[1] - y
    return round(x + GANTRY_OFFSET_MM[0], 2), round(y + GANTRY_OFFSET_MM[1], 2)


GANTRY_TIMEOUT_S = 180.0  # tempo maximo esperando o ESP32 terminar uma jogada

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
