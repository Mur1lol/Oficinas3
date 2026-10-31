"""Ponto de entrada principal do VoiceChess.

Exemplos de uso:
    python main.py                       # Modo real completo (microfone + ESP32 + perifericos)
    python main.py --no-return-home      # Mantem o gantry na ultima posicao entre lances
    python main.py --text --sim-gantry --sim-peripherals   # Modo 100% simulado no terminal
"""

import sys
from game.__main__ import main

if __name__ == "__main__":
    sys.exit(main())

# # Modo 100% simulado (comandos por teclado, sem ESP32 e sem periféricos reais)
# python main.py --text --sim-gantry --sim-peripherals
# # Modo real na Raspberry Pi (áudio, ESP32 e periféricos reais)
# python main.py --port /dev/ttyUSB0
# # Modo de teste com ESP32 real, mas digitando comandos no terminal
# python main.py --text --sim-peripherals --port COM3
