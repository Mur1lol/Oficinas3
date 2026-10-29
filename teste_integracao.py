"""Teste dos perifericos da Pi e da comunicacao Pi <-> ESP32, usando as classes do jogo.

Rode na raiz do projeto (onde ficam game/ e comunication/):

    python3 teste_integracao.py perifericos          # LEDs dos dois paineis + LCDs
    python3 teste_integracao.py status               # so pergunta o status ao ESP32
    python3 teste_integracao.py homing               # homing (canto da frente a direita)
    python3 teste_integracao.py cantos               # passeia pelos 4 cantos + centro
    python3 teste_integracao.py ima                  # liga o eletroima 2 s e desliga
    python3 teste_integracao.py lance e2e4           # planeja um lance e executa SEM ima
    python3 teste_integracao.py lance e2e4 --ima     # idem, com o eletroima

Opcoes: --porta /dev/ttyUSB0 (padrao: procura sozinho), -v (mostra o JSON trocado).
Ctrl+C para no meio. O ESP32 reinicia quando a porta abre: espere ~2 s.
"""

import argparse
import sys
import time

from game import config as cfg


def perifericos(_args):
    from game.peripherals import Lcd, StatusLeds

    leds = StatusLeds()
    lcd = Lcd()
    print(f"LEDs reais: {not leds.simulated}   LCDs reais: {len(lcd._lcds)} de "
          f"{len(cfg.LCD_I2C_ADDRESSES)}")
    try:
        for side in ("player", "computer"):
            for color in ("red", "green", "blue"):
                lcd.show(f"LED {side}", color.upper())
                print(f"LED {side}: {color}")
                leds.set(side, color)
                time.sleep(1.0)
            leds.off(side)
        lcd.show("VOICE CHESS", "TESTE OK")
        print("piscando vermelho nos dois lados (como no lance invalido)")
        leds.blink("player", "red", 2)
        leds.blink("computer", "red", 2)
    finally:
        time.sleep(1)
        leds.close()
        lcd.close()


def conectar(args):
    from comunication import ESP32Communication

    esp = ESP32Communication(porta=args.porta, verbose=args.v)
    esp.conectar()
    print(f"conectado em {esp.porta}")
    return esp


def status(args):
    with conectar(args) as esp:
        print(esp.obter_status())


def homing(args):
    with conectar(args) as esp:
        print("homing... (ate ~90 s)")
        t = time.monotonic()
        esp.homing()
        print(f"ok em {time.monotonic() - t:.1f} s:", esp.obter_status())


def cantos(args):
    m = 20.0  # fica 20 mm longe das bordas
    sx, sy = cfg.GANTRY_SIZE_MM
    pts = [(sx - m, m), (m, m), (m, sy - m), (sx - m, sy - m), (sx / 2, sy / 2)]
    nomes = ["frente-direita", "frente-esquerda", "tras-esquerda", "tras-direita", "centro"]
    with conectar(args) as esp:
        if not esp.obter_status().get("homed"):
            print("sem home: rodando o homing antes")
            esp.homing()
        for (x, y), nome in zip(pts, nomes):
            print(f"-> {nome}: X={x:.0f} Y={y:.0f}")
            esp.gantry([x], [y], [0])
        print("fim:", esp.obter_status())


def ima(args):
    with conectar(args) as esp:
        esp.acionar_eletroima(True)
        print("eletroima LIGADO (2 s)")
        time.sleep(2)
        esp.acionar_eletroima(False)
        print("eletroima desligado")


def lance(args):
    import chess

    from game.planner import MovePlanner

    planner = MovePlanner()
    move = chess.Move.from_uci(args.uci)
    plan = planner.plan(move)
    ima_lista = plan.eletroima if args.ima else [0] * len(plan.steps)
    print(f"{args.uci}: {len(plan.steps)} passos (jogo -> ESP32)")
    pontos = []
    for (x, y, m), m_env in zip(plan.steps, ima_lista):
        gx, gy = cfg.game_to_gantry(x, y)
        pontos.append((gx, gy))
        print(f"  jogo ({x:5.1f}, {y:5.1f})  ->  ESP32 X={gx:5.1f} Y={gy:5.1f}  ima={m_env}")
    if args.so_mostrar:
        return
    with conectar(args) as esp:
        if not esp.obter_status().get("homed"):
            print("sem home: rodando o homing antes")
            esp.homing()
        esp.gantry([p[0] for p in pontos], [p[1] for p in pontos], ima_lista,
                   timeout=cfg.GANTRY_TIMEOUT_S)
        print("ok:", esp.obter_status())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("teste", choices=["perifericos", "status", "homing", "cantos", "ima", "lance"])
    ap.add_argument("uci", nargs="?", default="e2e4", help="lance para o teste 'lance'")
    ap.add_argument("--porta", default=None)
    ap.add_argument("--ima", action="store_true", help="no teste 'lance', usa o eletroima")
    ap.add_argument("--so-mostrar", action="store_true", help="no 'lance', nao envia ao ESP32")
    ap.add_argument("-v", action="store_true", help="mostra o JSON enviado e recebido")
    args = ap.parse_args()
    try:
        globals()[args.teste](args)
    except KeyboardInterrupt:
        print("\ninterrompido")
    except Exception as exc:  # noqa: BLE001
        print(f"ERRO: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
