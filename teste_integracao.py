"""Teste dos perifericos da Pi e da comunicacao Pi <-> ESP32, usando as classes do jogo.

Rode na raiz do projeto (onde ficam game/ e comunication/):

    python3 teste_integracao.py perifericos          # LEDs + LCDs + Speaker juntos
    python3 teste_integracao.py leds                 # teste dedicado dos LEDs RGB
    python3 teste_integracao.py lcd                  # teste dedicado dos dois LCDs 20x4
    python3 teste_integracao.py speaker              # teste de voz/audio (TTS)
    python3 teste_integracao.py microfone            # teste de captura e medidor de volume (VU)
    python3 teste_integracao.py voz                  # teste completo de reconhecimento de voz
    python3 teste_integracao.py status               # status do ESP32
    python3 teste_integracao.py homing               # homing do gantry (canto da frente a direita)
    python3 teste_integracao.py cantos               # passeia pelos 4 cantos + centro
    python3 teste_integracao.py ima                  # liga o eletroima 2 s e desliga
    python3 teste_integracao.py lance e2e4           # planeja um lance e executa SEM ima
    python3 teste_integracao.py lance e2e4 --ima     # idem, com o eletroima

Opcoes: --porta /dev/ttyUSB0 (padrao: procura sozinho), -v (mostra o JSON trocado).
Ctrl+C para no meio. O ESP32 reinicia quando a porta abre: espere ~2 s.
"""

import argparse
import math
import sys
import time

from game import config as cfg


def leds(_args):
    """Teste dedicado dos LEDs RGB dos paineis do jogador e do computador."""
    from game.peripherals import StatusLeds

    leds = StatusLeds()
    print(f"\n[*] Iniciando teste dos LEDs RGB (Hardware real: {not leds.simulated})")
    print(f"    Pinos Player (BCM):   {cfg.LED_PINS['player']}")
    print(f"    Pinos Computer (BCM): {cfg.LED_PINS['computer']}\n")

    cores = [
        ("red", "Vermelho"),
        ("green", "Verde"),
        ("blue", "Azul"),
        ("yellow", "Amarelo"),
        ("magenta", "Magenta"),
        ("cyan", "Ciano"),
        ("white", "Branco"),
    ]

    try:
        for side, side_name in (("player", "Jogador (Brancas)"), ("computer", "Computador (Pretas)")):
            print(f"--- Testando painel: {side_name} ---")
            for cor_id, cor_nome in cores:
                print(f"  -> Acendendo: {cor_nome}")
                leds.set(side, cor_id)
                time.sleep(0.8)
            leds.off(side)
            time.sleep(0.4)

        print("\n--- Teste de Piscar (Alerta de lance invalido) ---")
        print("  Piscando vermelho em ambos os paineis...")
        leds.blink("player", "red", 3)
        leds.blink("computer", "red", 3)
        time.sleep(2.5)
        print("[OK] Teste de LEDs concluido com sucesso!")
    finally:
        leds.close()


def lcd(_args):
    """Teste dedicado dos displays LCD 20x4 via I2C."""
    from game.peripherals import Lcd

    lcd = Lcd()
    print(f"\n[*] Iniciando teste dos Displays LCD 20x4 I2C")
    print(f"    Displays detectados: {len(lcd._lcds)} de {len(cfg.LCD_I2C_ADDRESSES)}")
    print(f"    Enderecos I2C: {cfg.LCD_I2C_ADDRESSES}\n")

    try:
        print("-> Escrevendo mensagem de inicializacao nas 4 linhas...")
        lcd.show(
            "=== VOICE CHESS ===",
            "Painel 20x4 I2C",
            "Linha 3: Testando...",
            "Status: 100% OK!"
        )
        time.sleep(3.0)

        print("-> Testando contagem regressiva nos displays...")
        for i in range(3, 0, -1):
            lcd.show(
                "=== VOICE CHESS ===",
                f"Contagem: {i}",
                "Aguardando lance...",
                ""
            )
            time.sleep(1.0)

        lcd.show(
            "=== VOICE CHESS ===",
            "Sua vez: Brancas",
            "Diga: MAGNUS",
            "Ex: Pawn to E4"
        )
        print("[OK] Teste de LCD concluido com sucesso!")
        time.sleep(2.0)
    finally:
        lcd.close()


def speaker(_args):
    """Teste do sintetizador de voz (TTS offline pyttsx3 / espeak)."""
    from game.peripherals import Speaker

    print("\n[*] Iniciando teste do Speaker / TTS...")
    spk = Speaker()
    print(f"    Sintetizador real: {not spk.simulated} (Taxa: {cfg.TTS_RATE} wpm)\n")

    frases = [
        "Welcome to Voice Chess.",
        "Testing speech synthesis system.",
        "Pawn to e4.",
        "Checkmate. White wins the game."
    ]

    try:
        for frase in frases:
            print(f" -> Falando: \"{frase}\"")
            spk.say(frase, wait=True)
            time.sleep(0.5)
        print("[OK] Teste de Speaker concluido com sucesso!")
    finally:
        time.sleep(0.5)


def microfone(_args):
    """Teste de captura de audio do microfone com visualizador de volume (VU meter)."""
    print("\n[*] Iniciando teste do Microfone...")

    try:
        import pyaudio
    except ImportError:
        print("[ERRO] Biblioteca 'pyaudio' nao instalada.")
        print("Instale com: sudo apt install portaudio19-dev && pip install pyaudio")
        return

    pa = pyaudio.PyAudio()
    dev_count = pa.get_device_count()
    print(f"\nDispositivos de audio detectados ({dev_count}):")
    input_indices = []
    for i in range(dev_count):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            print(f"  [{i}] {info.get('name')} (Canais: {info.get('maxInputChannels')}, Rate: {int(info.get('defaultSampleRate'))} Hz)")
            input_indices.append(i)

    if not input_indices:
        print("[ERRO] Nenhum microfone de entrada foi encontrado pelo PyAudio!")
        print("Dica: execute 'arecord -l' no terminal do Linux para verificar os dispositivos USB.")
        pa.terminate()
        return

    # Tenta abrir o primeiro microfone disponível ou o padrão
    idx = input_indices[0]
    rate = 16000
    chunk = 1024

    print(f"\n[*] Abrindo microfone índice [{idx}] a {rate} Hz...")
    try:
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True,
                         input_device_index=idx, frames_per_buffer=chunk)
    except Exception as exc:
        print(f"[FALHA] Nao foi possivel abrir dispositivo [{idx}]: {exc}")
        pa.terminate()
        return

    print("[*] Gravando e medindo volume em tempo real por 8 segundos (fale no microfone!)...\n")
    start = time.monotonic()
    max_volume = 0
    try:
        while time.monotonic() - start < 8.0:
            data = stream.read(chunk, exception_on_overflow=False)
            # Calcula RMS (Root Mean Square) do sinal de 16-bit
            count = len(data) // 2
            format_str = f"<{count}h"
            import struct
            shorts = struct.unpack(format_str, data)
            sum_squares = sum(s * s for s in shorts)
            rms = math.sqrt(sum_squares / count) if count > 0 else 0
            # Normaliza para porcentagem de 0 a 100
            pct = min(100, int((rms / 3000.0) * 100))
            if pct > max_volume:
                max_volume = pct
            bar_len = pct // 3
            bar = "#" * bar_len + "-" * (33 - bar_len)
            sys.stdout.write(f"\r  Volume: [{bar}] {pct:3d}% | Pico: {max_volume:3d}%")
            sys.stdout.flush()
            time.sleep(0.04)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    print("\n\n" + "-" * 50)
    print(f"Pico maximo de volume detectado: {max_volume}%")
    if max_volume > 10:
        print("[SUCESSO] O microfone esta captando som perfeitamente!")
    else:
        print("[AVISO] Volume muito baixo captado. Fale mais proximo ou ajuste o ganho com 'alsamixer'.")
    print("-" * 50)


def voz(_args):
    """Teste do reconhecimento de voz completo (Magnus -> Lance -> Confirmacao)."""
    print("\n[*] Iniciando pipeline de reconhecimento de voz VoiceChess...")
    try:
        from voice.cli import cli as voice_cli
        print("Diga: 'Magnus' para chamar, depois o lance (ex: 'Pawn to e4'), e 'Yes' para confirmar.")
        sys.argv = ["voice", "run"]
        voice_cli()
    except Exception as exc:
        print(f"[ERRO] Falha ao executar o modulo de voz: {exc}")


def perifericos(_args):
    """Teste combinado: LEDs + LCDs + Speaker simultaneamente."""
    from game.peripherals import Lcd, Speaker, StatusLeds

    leds = StatusLeds()
    lcd = Lcd()
    spk = Speaker()

    print(f"\n[*] Teste Combinado de Perifericos:")
    print(f"    LEDs reais:    {not leds.simulated}")
    print(f"    LCDs reais:    {len(lcd._lcds)} de {len(cfg.LCD_I2C_ADDRESSES)}")
    print(f"    Speaker real:  {not spk.simulated}\n")

    try:
        lcd.show("VOICE CHESS", "Iniciando Perifericos", "Aguarde...", "")
        spk.say("Voice Chess peripherals test started.", wait=True)

        for side, side_name in (("player", "Brancas"), ("computer", "Pretas")):
            lcd.show(f"LED: {side_name}", "Cor: VERDE", "Status: OK", "")
            leds.set(side, "green")
            spk.say(f"{side_name} ready.", wait=True)
            time.sleep(0.5)
            leds.off(side)

        lcd.show("VOICE CHESS", "TESTE CONCLUIDO", "Todos perifericos OK", "")
        spk.say("All peripherals tested successfully.", wait=True)
        print("\n[OK] Perifericos integrados e testados com sucesso!")
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
    ap.add_argument("teste", choices=[
        "perifericos", "leds", "lcd", "speaker", "microfone", "voz",
        "status", "homing", "cantos", "ima", "lance"
    ])
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
