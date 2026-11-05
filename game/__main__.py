"""
Uso:
    python -m game                       # microfone + ESP32 + camera + perifericos reais
    python -m game --mode pvp            # modo Jogador vs Jogador (PvP)
    python -m game --mode pve            # modo Jogador vs Maquina (PvE)
    python -m game --text                # comandos digitados no terminal (sem microfone)
    python -m game --sim-gantry          # sem ESP32 (imprime os passos do Gantry)
    python -m game --sim-peripherals     # LEDs/LCD/speaker apenas no terminal
    python -m game --sim-cam             # ignora as imagens da camera (visao simulada)
    python -m game --no-camera           # desativa a camera USB
    python -m game --port COM3           # porta serial do ESP32
"""

import argparse
import logging
import warnings

# Silencia avisos informativos (ex.: pkg_resources deprecation do webrtcvad)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

try:
    from voice.audio_capture import suppress_alsa_handler
    suppress_alsa_handler()
except Exception:
    pass

from vision import CameraSystem

from . import config as cfg
from .gantry_link import EspGantry, SimulatedGantry
from .peripherals import Lcd, Speaker, StatusLeds
from .session import VoiceChessGame


def main() -> None:
    ap = argparse.ArgumentParser(description="VoiceChess - PvE & PvP com Visao e Gantry")
    ap.add_argument("--text", action="store_true", help="comandos por teclado em vez de voz")
    ap.add_argument("--sim-gantry", action="store_true", help="nao usa o ESP32")
    ap.add_argument("--sim-peripherals", action="store_true", help="LEDs/LCD/speaker simulados")
    ap.add_argument("--sim-cam", action="store_true", help="ignora as imagens da camera (visao simulada)")
    ap.add_argument("--no-camera", action="store_true", help="desativa o sistema de visao (camera)")
    ap.add_argument("--camera-device", type=int, default=None, help="indice da camera USB (padrao: config.py)")
    ap.add_argument("--mode", choices=["pve", "pvp"], default=None, help="modo de jogo inicial (pve ou pvp)")
    ap.add_argument("--white-control", choices=["voice", "manual"], default=None,
                    help="controle das Brancas (voice=Gantry, manual=MAGNUS VALIDATE)")
    ap.add_argument("--black-control", choices=["voice", "manual", "computer"], default=None,
                    help="controle das Pretas em PvP (voice=Gantry, manual=MAGNUS VALIDATE)")
    ap.add_argument("--port", default=None, help="porta serial do ESP32 (ex.: COM3, /dev/ttyUSB0)")
    ap.add_argument("--voice-config", default=str(cfg.VOICE_CONFIG))
    ap.add_argument("--no-return-home", action="store_true",
                    help="mantem o gantry na ultima posicao entre lances")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    leds = StatusLeds(simulate=args.sim_peripherals)
    lcd = Lcd(simulate=args.sim_peripherals)
    speaker = Speaker(simulate=args.sim_peripherals)
    gantry = SimulatedGantry() if args.sim_gantry else EspGantry(args.port)

    camera = None
    if not args.no_camera:
        if args.sim_cam:
            from vision import SimulatedCamera
            camera = SimulatedCamera()
            camera.initialize()
        else:
            try:
                cam_kwargs = {}
                if args.camera_device is not None:
                    cam_kwargs["device_index"] = args.camera_device
                cam = CameraSystem(**cam_kwargs)
                if cam.initialize():
                    camera = cam
                else:
                    logging.warning("Camera nao pode ser inicializada; prosseguindo sem visao.")
            except Exception as exc:
                logging.warning("Falha ao iniciar CameraSystem (%s); prosseguindo sem visao.", exc)

    if args.text:
        from .listener import TextListener
        factory = lambda on_l, on_c, before: TextListener(on_l, on_c, before)       # noqa: E731
    else:
        from .listener import VoiceListener
        factory = lambda on_l, on_c, before: VoiceListener(                          # noqa: E731
            args.voice_config, on_l, on_c, before)

    VoiceChessGame(
        leds,
        lcd,
        speaker,
        gantry,
        factory,
        return_home=not args.no_return_home,
        camera=camera,
        game_mode=args.mode,
        white_control=args.white_control,
        black_control=args.black_control,
    ).run()


if __name__ == "__main__":
    main()
