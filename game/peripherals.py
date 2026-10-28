"""
game/peripherals.py
===================
LEDs RGB (GPIO), LCD 16x2 (I2C) e Speaker (TTS offline) da Raspberry Pi.

Cada periferico tenta usar o hardware real; se a biblioteca ou o hardware nao
existir (ex.: rodando no PC), cai para uma versao que apenas imprime no terminal.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from . import config as cfg

log = logging.getLogger(__name__)

COLORS = {
    "off":   (0, 0, 0),
    "blue":  (0, 0, 1),
    "green": (0, 1, 0),
    "red":   (1, 0, 0),
}
COLOR_PT = {"off": "apagado", "blue": "AZUL", "green": "VERDE", "red": "VERMELHO"}


# ---------------------------------------------------------------------------
# LEDs
# ---------------------------------------------------------------------------

class StatusLeds:
    """Dois LEDs RGB: 'player' (lado do jogador) e 'computer' (lado da maquina)."""

    def __init__(self, simulate: bool = False) -> None:
        self._leds: dict[str, object] = {}
        self.simulated = True
        if not simulate:
            try:
                from gpiozero import RGBLED
                for side, (r, g, b) in cfg.LED_PINS.items():
                    self._leds[side] = RGBLED(r, g, b, active_high=cfg.LED_ACTIVE_HIGH)
                self.simulated = False
            except Exception as exc:                         # noqa: BLE001
                log.warning("LEDs reais indisponiveis (%s) - usando simulacao.", exc)
                self._leds = {}
        self._state = {side: "off" for side in cfg.LED_PINS}
        self._lock = threading.Lock()

    def state(self, side: str) -> str:
        """Cor atual do LED ('off', 'blue', 'green' ou 'red')."""
        return self._state.get(side, "off")

    def set(self, side: str, color: str) -> None:
        with self._lock:
            self._write(side, color)

    def off(self, side: str | None = None) -> None:
        for s in ([side] if side else list(cfg.LED_PINS)):
            self.set(s, "off")

    def blink(self, side: str, color: str, seconds: float = cfg.BLINK_ERROR_SECONDS,
              hz: float = cfg.BLINK_HZ) -> None:
        """Pisca (bloqueante) durante `seconds` e termina apagado."""
        half = 0.5 / hz
        end = time.monotonic() + seconds
        on = True
        while time.monotonic() < end:
            self.set(side, color if on else "off")
            on = not on
            time.sleep(half)
        self.set(side, "off")

    def _write(self, side: str, color: str) -> None:
        if self._state.get(side) == color:
            return
        self._state[side] = color
        if self.simulated:
            print(f"   [LED {side}] {COLOR_PT[color]}")
        else:
            self._leds[side].color = COLORS[color]

    def close(self) -> None:
        self.off()
        for led in self._leds.values():
            try:
                led.close()
            except Exception:                                # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# LCD
# ---------------------------------------------------------------------------

class Lcd:
    """LCD 16x2 via I2C (PCF8574). Fallback: imprime no terminal."""

    def __init__(self, simulate: bool = False) -> None:
        self._lcd = None
        if not simulate:
            try:
                from RPLCD.i2c import CharLCD
                self._lcd = CharLCD("PCF8574", cfg.LCD_I2C_ADDRESS, port=cfg.LCD_I2C_PORT,
                                    cols=cfg.LCD_COLS, rows=cfg.LCD_ROWS)
            except Exception as exc:                         # noqa: BLE001
                log.warning("LCD real indisponivel (%s) - usando simulacao.", exc)
        self.simulated = self._lcd is None
        self._last = None

    def show(self, line1: str = "", line2: str = "") -> None:
        lines = (line1[:cfg.LCD_COLS].ljust(cfg.LCD_COLS), line2[:cfg.LCD_COLS].ljust(cfg.LCD_COLS))
        if lines == self._last:
            return
        self._last = lines
        if self._lcd is None:
            print(f"   [LCD] |{lines[0]}|\n         |{lines[1]}|")
            return
        try:
            self._lcd.cursor_pos = (0, 0)
            self._lcd.write_string(lines[0])
            self._lcd.cursor_pos = (1, 0)
            self._lcd.write_string(lines[1])
        except Exception as exc:                             # noqa: BLE001
            log.warning("Falha ao escrever no LCD: %s", exc)

    def close(self) -> None:
        if self._lcd is not None:
            try:
                self._lcd.clear()
                self._lcd.close(clear=True)
            except Exception:                                # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Speaker
# ---------------------------------------------------------------------------

class Speaker:
    """TTS offline (pyttsx3/espeak) numa thread propria. Fallback: imprime no terminal."""

    _STOP = object()

    def __init__(self, simulate: bool = False) -> None:
        self.simulated = simulate
        self._q: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)

    def _worker(self) -> None:
        engine = None
        if not self.simulated:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", cfg.TTS_RATE)
                for v in engine.getProperty("voices"):
                    ident = f"{v.id} {v.name} {getattr(v, 'languages', '')}".lower()
                    if "en" in ident and ("english" in ident or "en-" in ident or "en_" in ident):
                        engine.setProperty("voice", v.id)
                        break
            except Exception as exc:                         # noqa: BLE001
                log.warning("TTS indisponivel (%s) - usando simulacao.", exc)
                engine = None
                self.simulated = True
        else:
            engine = None
        self._ready.set()

        while True:
            item = self._q.get()
            if item is self._STOP:
                self._q.task_done()
                return
            text, done = item
            try:
                if engine is None:
                    print(f"   [SPEAKER] \"{text}\"")
                else:
                    engine.say(text)
                    engine.runAndWait()
            except Exception as exc:                         # noqa: BLE001
                log.warning("Erro no TTS: %s", exc)
            finally:
                if done is not None:
                    done.set()
                self._q.task_done()

    def say(self, text: str, wait: bool = True) -> None:
        """Fala `text`. wait=False retorna na hora (a fala segue em segundo plano)."""
        done = threading.Event() if wait else None
        self._q.put((text, done))
        if done is not None:
            done.wait()

    def wait_idle(self) -> None:
        """Bloqueia ate acabar de falar (use antes de abrir o microfone)."""
        self._q.join()

    def close(self) -> None:
        self._q.put(self._STOP)
