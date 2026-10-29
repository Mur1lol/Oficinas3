#!/usr/bin/env python3
"""
test_communication.py
=====================
Script interativo para teste dos motores e eletroímã (ESP32 <-> Raspberry Pi).

Opções do menu:
    1 - Mover no eixo X (distância em mm)
    2 - Mover no eixo Y (distância em mm)
    3 - Ligar Eletroímã
    4 - Desligar Eletroímã
    5 - Homing (Zero de máquina)
    0 - Sair
"""

import argparse
import sys
import time

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("[ERRO] Biblioteca 'pyserial' nao encontrada!")
    print("Instale executando: pip install pyserial (ou uv pip install pyserial)")
    sys.exit(1)

from comunication import ESP32Communication

# Limites do gantry
MAX_X_MM = 350.0
MAX_Y_MM = 650.0


def listar_portas():
    """Lista todas as portas seriais disponíveis no sistema."""
    ports = serial.tools.list_ports.comports()
    print("\n--- Portas Seriais Detectadas ---")
    if not ports:
        print("  Nenhuma porta serial encontrada.")
        return []

    for idx, p in enumerate(ports, start=1):
        print(f"  [{idx}] {p.device} ({p.description})")
    return ports


def mover_x(esp: ESP32Communication):
    """Move uma distância relativa no eixo X."""
    print(f"\n[*] Mover no Eixo X (Limite: 0 a {MAX_X_MM:.0f} mm)")
    dist_str = input("Distancia em mm para mover em X (ex: 50 ou -50) [padrao: 50]: ").strip()
    try:
        dist_mm = float(dist_str) if dist_str else 50.0
    except ValueError:
        print("Valor invalido.")
        return

    print(f"[*] Movendo X em {dist_mm:+.1f} mm...")
    try:
        esp.mover_eixo("X", dist_mm, timeout=30.0)
        time.sleep(0.2)
        st = esp.obter_status(timeout=2.0)
        ima_st = "LIGADO" if st.get("eletroima") else "desligado"
        print(f" -> [OK] Posicao: X={st.get('x', '?')} mm | Y={st.get('y', '?')} mm | Eletroima: {ima_st}")
    except Exception as exc:
        print(f" -> [ERRO] {exc}")


def mover_y(esp: ESP32Communication):
    """Move uma distância relativa no eixo Y."""
    print(f"\n[*] Mover no Eixo Y (Limite: 0 a {MAX_Y_MM:.0f} mm)")
    dist_str = input("Distancia em mm para mover em Y (ex: 50 ou -50) [padrao: 50]: ").strip()
    try:
        dist_mm = float(dist_str) if dist_str else 50.0
    except ValueError:
        print("Valor invalido.")
        return

    print(f"[*] Movendo Y em {dist_mm:+.1f} mm...")
    try:
        esp.mover_eixo("Y", dist_mm, timeout=30.0)
        time.sleep(0.2)
        st = esp.obter_status(timeout=2.0)
        ima_st = "LIGADO" if st.get("eletroima") else "desligado"
        print(f" -> [OK] Posicao: X={st.get('x', '?')} mm | Y={st.get('y', '?')} mm | Eletroima: {ima_st}")
    except Exception as exc:
        print(f" -> [ERRO] {exc}")


def ligar_eletroima(esp: ESP32Communication):
    """Liga o eletroímã imediatamente."""
    print("\n[*] Acionando Eletroima...")
    try:
        esp.acionar_eletroima(True, timeout=5.0)
        print(" -> [OK] Eletroima LIGADO!")
    except Exception as exc:
        print(f" -> [ERRO] {exc}")


def desligar_eletroima(esp: ESP32Communication):
    """Desliga o eletroímã imediatamente."""
    print("\n[*] Desligando Eletroima...")
    try:
        esp.acionar_eletroima(False, timeout=5.0)
        print(" -> [OK] Eletroima DESLIGADO!")
    except Exception as exc:
        print(f" -> [ERRO] {exc}")


def executar_homing(esp: ESP32Communication):
    """Executa o ciclo de Homing e define as coordenadas de repouso em (3, 8)."""
    print("\n[*] Executando ciclo de Homing (zero de maquina)...")
    try:
        t0 = time.monotonic()
        esp.homing(timeout=120.0)
        t_total = time.monotonic() - t0
        time.sleep(0.2)
        st = esp.obter_status(timeout=2.0)
        print(f" -> [OK] Homing concluido em {t_total:.1f}s!")
        print(f"    Posicao atual: X={st.get('x', 3.0)} mm | Y={st.get('y', 8.0)} mm | Homed: {st.get('homed')}")
    except Exception as exc:
        print(f" -> [ERRO] Falha no Homing: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Teste de Motores e Eletroímã (ESP32 <-> Raspberry Pi)"
    )
    parser.add_argument("--port", "-p", help="Porta serial manual (ex: /dev/ttyUSB0 ou COM3)")
    parser.add_argument("--baud", "-b", type=int, default=115200, help="Baudrate (padrao: 115200)")
    args = parser.parse_args()

    print("==================================================")
    print("  Teste de Motores e Eletroima (ESP32 <-> RPi)    ")
    print("==================================================")

    listar_portas()

    print(f"\nTentando conectar ao ESP32 (baudrate: {args.baud})...")
    esp = ESP32Communication(porta=args.port, baudrate=args.baud, timeout=1.0, verbose=False)

    try:
        esp.conectar()
    except Exception as exc:
        print(f"\n[ERRO CRITICO] Nao foi possivel conectar ao ESP32: {exc}")
        sys.exit(1)

    print("[OK] Conexao serial estabelecida com sucesso!")
    try:
        st_inicial = esp.obter_status(timeout=3.0)
        print(f"Status inicial: X={st_inicial.get('x', '?')} mm, Y={st_inicial.get('y', '?')} mm, Homed={st_inicial.get('homed')}")
    except Exception:
        pass

    try:
        while True:
            print("\n------------------------------------")
            print("         MENU DE TESTES             ")
            print("------------------------------------")
            print("1 - Mover no eixo X")
            print("2 - Mover no eixo Y")
            print("3 - Ligar Eletroima")
            print("4 - Desligar Eletroima")
            print("5 - Homing")
            print("0 - Sair")
            print("------------------------------------")

            opcao = input("Escolha uma opcao [0-5]: ").strip()
            if opcao == "1":
                mover_x(esp)
            elif opcao == "2":
                mover_y(esp)
            elif opcao == "3":
                ligar_eletroima(esp)
            elif opcao == "4":
                desligar_eletroima(esp)
            elif opcao == "5":
                executar_homing(esp)
            elif opcao in ("0", "sair", "exit", "q"):
                print("Encerrando...")
                break
            else:
                print("Opcao invalida. Digite de 0 a 5.")
    finally:
        esp.fechar()


if __name__ == "__main__":
    main()
