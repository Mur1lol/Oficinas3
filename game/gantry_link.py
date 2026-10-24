"""
game/gantry_link.py
===================
Interface do jogo com o Gantry (ESP32).

* ``EspGantry``       : usa comunication.ESP32Communication (serial/USB).
* ``SimulatedGantry`` : sem ESP32; imprime os passos (para testar no PC).

Ambas tem: connect(), home(), execute(x, y, eletroima), close().
"""

from __future__ import annotations

import time

from . import config as cfg


class GantryError(RuntimeError):
    """O ESP32 nao concluiu o movimento (timeout, fim de curso, erro...)."""


class EspGantry:
    def __init__(self, port: str | None = None) -> None:
        self._port = port
        self._esp = None

    def connect(self) -> None:
        from comunication import ESP32Communication
        self._esp = ESP32Communication(porta=self._port, verbose=False)
        try:
            self._esp.conectar()
        except Exception as exc:                              # noqa: BLE001
            raise GantryError(f"Falha ao conectar no ESP32: {exc}") from exc

    def home(self) -> None:
        try:
            self._esp.homing()
        except Exception as exc:                              # noqa: BLE001
            raise GantryError(f"Falha no homing: {exc}") from exc

    def execute(self, x, y, eletroima) -> None:
        """Bloqueia ate o ESP32 responder 'ok' (movimento concluido)."""
        try:
            self._esp.gantry(x, y, eletroima, timeout=cfg.GANTRY_TIMEOUT_S)
        except Exception as exc:                              # noqa: BLE001
            raise GantryError(str(exc)) from exc

    def close(self) -> None:
        if self._esp is not None:
            try:
                self._esp.acionar_eletroima(False, timeout=2)
            except Exception:                                 # noqa: BLE001
                pass
            self._esp.fechar()


class SimulatedGantry:
    SIM_SECONDS_PER_STEP = 0.15

    def connect(self) -> None:
        print("   [GANTRY] (simulado) conectado")

    def home(self) -> None:
        print("   [GANTRY] (simulado) homing ok")

    def execute(self, x, y, eletroima) -> None:
        print(f"   [GANTRY] (simulado) {len(x)} passos:")
        for xi, yi, mi in zip(x, y, eletroima):
            print(f"      -> x={xi:6.1f} y={yi:6.1f} ima={'ON ' if mi else 'off'}")
        time.sleep(self.SIM_SECONDS_PER_STEP * len(x))

    def close(self) -> None:
        pass
