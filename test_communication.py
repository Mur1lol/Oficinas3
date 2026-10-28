#!/usr/bin/env python3
"""
test_communication.py
=====================
Script de testes para validação da comunicação Serial/USB entre Raspberry Pi e ESP32.

Pode ser executado diretamente na Raspberry Pi ou no PC de desenvolvimento:
    python test_communication.py
    python test_communication.py --port /dev/ttyUSB0
    python test_communication.py --auto
"""

import argparse
import json
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


def listar_portas():
    """Lista todas as portas seriais disponíveis no sistema."""
    ports = serial.tools.list_ports.comports()
    print("\n--- Portas Seriais Detectadas ---")
    if not ports:
        print("  Nenhuma porta serial encontrada.")
        print("  Dica na Raspberry Pi:")
        print("    1. Verifique se o cabo USB está conectado.")
        print("    2. Verifique se o cabo possui linhas de dados (nao apenas carga).")
        print("    3. Execute 'lsusb' e 'dmesg | grep tty' no terminal para debugar.")
        return []

    for idx, p in enumerate(ports, start=1):
        print(f"  [{idx}] {p.device}")
        print(f"      Descricao: {p.description}")
        print(f"      HWID:      {p.hwid}")
    return ports


def teste_status(esp: ESP32Communication):
    """Envia comando de status e valida os campos retornados."""
    print("\n[*] Solicitando status do ESP32...")
    try:
        status = esp.obter_status(timeout=5.0)
        print(" -> Resposta recebida:")
        print(json.dumps(status, indent=2, ensure_ascii=False))

        # Validacoes basicas
        if status.get("comando") == "status":
            print(" -> [OK] ESP32 respondeu corretamente ao comando de status!")
            print(f"    Posicao X: {status.get('x')} mm | Y: {status.get('y')} mm")
            print(f"    Homed: {status.get('homed')} | Drivers: {status.get('drivers')}")
            print(f"    Eletroima: {status.get('eletroima')}")
            print(f"    Fim de curso X: {status.get('fim_x')} | Y: {status.get('fim_y')}")
            return True
        else:
            print(" -> [AVISO] Resposta inesperada:", status)
            return False
    except Exception as exc:
        print(f" -> [FALHA] Erro ao obter status: {exc}")
        return False


def teste_eletroima(esp: ESP32Communication):
    """Testa ligar e desligar o eletroímã com intervalo de 2 segundos."""
    print("\n[*] Testando acionamento do eletroima...")
    try:
        print(" -> Ligando eletroima (esperando 'ok')...")
        esp.acionar_eletroima(ligar=True, timeout=5.0)
        print(" -> [OK] Eletroima LIGADO com sucesso!")
        print("    Aguardando 2 segundos com ima acionado...")
        time.sleep(2)

        print(" -> Desligando eletroima...")
        esp.acionar_eletroima(ligar=False, timeout=5.0)
        print(" -> [OK] Eletroima DESLIGADO com sucesso!")
        return True
    except Exception as exc:
        print(f" -> [FALHA] Erro no teste do eletroima: {exc}")
        return False


def teste_fins_de_curso(esp: ESP32Communication):
    """Monitora os fins de curso em tempo real por leitura contínua."""
    print("\n[*] Monitoramento em tempo real dos fins de curso (Ctrl+C para sair):")
    print("    Pressione manualmente os switches de fim de curso X e Y para testar.")
    try:
        while True:
            st = esp.obter_status(timeout=2.0)
            fim_x = "ACIONADO [X]" if st.get("fim_x") else "aberto"
            fim_y = "ACIONADO [Y]" if st.get("fim_y") else "aberto"
            sys.stdout.write(f"\r    Fim de Curso X: {fim_x:<15} | Fim de Curso Y: {fim_y:<15}")
            sys.stdout.flush()
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\n\n -> Monitoramento finalizado.")


def teste_homing(esp: ESP32Communication):
    """Executa o ciclo de homing do robô."""
    print("\n[!] ATENCAO: Certifique-se de que a mesa esta desobstruida e os fins de curso instalados.")
    conf = input("Deseja executar o Homing agora? (s/N): ").strip().lower()
    if conf not in ("s", "sim", "y", "yes"):
        print(" -> Homing cancelado pelo usuario.")
        return False

    print("[*] Iniciando ciclo de Homing (tempo maximo: 30s)...")
    try:
        resp = esp.homing(timeout=30.0)
        print(" -> [OK] Homing concluido com sucesso!")
        print(f"    Resposta: {resp}")
        return True
    except Exception as exc:
        print(f" -> [FALHA] Falha no Homing: {exc}")
        return False


def console_interativo(esp: ESP32Communication):
    """Console interativo para enviar comandos manuais ao ESP32."""
    print("\n=== Console Interativo Serial ===")
    print("Digite comandos para enviar ao ESP32. Exemplos:")
    print("  status         -> requisita status em JSON")
    print("  on             -> habilita drivers dos motores")
    print("  off            -> desabilita drivers dos motores")
    print("  m on           -> liga eletroima")
    print("  m off          -> desliga eletroima")
    print("  homing (ou h)  -> executa homing")
    print("  ou envie JSON  -> {\"comando\":\"status\"}")
    print("  sair           -> encerra o console interativo\n")

    while True:
        try:
            cmd = input("ESP32 >> ").strip()
            if not cmd:
                continue
            if cmd.lower() in ("sair", "exit", "quit", "q"):
                break

            # Se for formato JSON
            if cmd.startswith("{"):
                try:
                    payload = json.loads(cmd)
                    esp.enviar_comando(payload.get("comando"), payload.get("resposta"))
                except json.JSONDecodeError:
                    print("JSON invalido.")
                    continue
            elif cmd.lower() == "status":
                esp.enviar_comando("status")
            elif cmd.lower() in ("on", "off", "m on", "m off", "h", "homing"):
                esp.ser.write((cmd + "\n").encode())
            else:
                esp.ser.write((cmd + "\n").encode())

            # Aguarda e lê respostas que chegarem
            time.sleep(0.15)
            while esp.ser.in_waiting:
                linha = esp.ser.readline().decode("utf-8", errors="ignore").strip()
                if linha:
                    print(f"  << {linha}")
        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as exc:
            print(f"Erro: {exc}")


def diagnostico_completo(esp: ESP32Communication):
    """Executa sequência automatizada de diagnóstico."""
    print("\n" + "=" * 50)
    print("       DIAGNOSTICO COMPLETO ESP32 <-> RASPBERRY")
    print("=" * 50)

    sucessos = 0
    total = 2

    # 1. Status
    if teste_status(esp):
        sucessos += 1

    # 2. Eletroímã
    if teste_eletroima(esp):
        sucessos += 1

    print("\n" + "-" * 50)
    print(f"Resultado: {sucessos}/{total} testes concluidos com sucesso.")
    if sucessos == total:
        print("[SUCESSO] Comunicacao entre Raspberry e ESP32 100% OPERACIONAL!")
    else:
        print("[AVISO] Alguns testes apresentaram falha. Verifique os logs acima.")
    print("-" * 50)


def teste_drivers_torque(esp: ESP32Communication):
    """Testa a habilitação dos drivers de passo (EN_MOT) e torque de retenção."""
    print("\n[*] Teste de Habilitacao dos Motores (Torque de Retencao)...")
    print(" -> Enviando 'on' para travar motores...")
    esp.ser.write(b"on\n")
    time.sleep(0.3)
    resp = esp.ser.readline().decode("utf-8", errors="ignore").strip()
    print(f" -> Resposta do ESP32: {resp}")
    print(" -> Teste com a mao: os motores devem estar travados e resistindo ao movimento.")
    input(" -> Pressione [ENTER] quando estiver pronto para destravar os motores...")
    esp.ser.write(b"off\n")
    time.sleep(0.3)
    resp = esp.ser.readline().decode("utf-8", errors="ignore").strip()
    print(f" -> Resposta do ESP32: {resp}")
    print(" -> [OK] Drivers desligados. Os motores estao livres novamente!")
    return True


def teste_pequeno_movimento(esp: ESP32Communication):
    """Testa um deslocamento seguro de 20mm para verificar a cinemática CoreXY."""
    print("\n[*] ATENCAO: Posicione o carro aproximadamente no CENTRO da mesa!")
    conf = input("O gantry possui curso livre para se mover 20mm? (s/N): ").strip().lower()
    if conf not in ("s", "sim", "y", "yes"):
        print(" -> Movimento cancelado.")
        return False

    print("[*] Enviando comando gantry: deslocamento de 20mm...")
    try:
        esp.gantry(x=[20.0], y=[20.0], eletroima=[0], timeout=15.0)
        print(" -> [OK] Movimento executado e confirmado pelo ESP32!")
        return True
    except Exception as exc:
        print(f" -> [FALHA] Falha no movimento: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Utilitário de Teste de Comunicação Raspberry Pi <-> ESP32"
    )
    parser.add_argument("--port", "-p", help="Porta serial manual (ex: /dev/ttyUSB0 ou COM3)")
    parser.add_argument("--baud", "-b", type=int, default=115200, help="Baudrate (padrao: 115200)")
    parser.add_argument("--auto", "-a", action="store_true", help="Executa diagnostico automatico e sai")
    args = parser.parse_args()

    print("==================================================")
    print("  Teste de Comunicacao ESP32 <-> Raspberry Pi     ")
    print("==================================================")

    # 1. Listar portas
    listar_portas()

    # 2. Conectar ao ESP32
    print(f"\nTentando conectar ao ESP32 (baudrate: {args.baud})...")
    esp = ESP32Communication(porta=args.port, baudrate=args.baud, timeout=1.0, verbose=True)

    try:
        esp.conectar()
    except Exception as exc:
        print(f"\n[ERRO CRITICO] Nao foi possivel conectar: {exc}")
        print("\nVerificacoes recomendadas na Raspberry Pi:")
        print("1. O usuario tem permissao na porta serial? Execute:")
        print("   sudo usermod -a -G dialout $USER")
        print("   (depois reinicie a sessao ou faca logout/login)")
        print("2. A porta esta ocupada por outro processo? Verifique com:")
        print("   lsof | grep ttyUSB")
        print("3. O firmware no ESP32 foi gravado corretamente a 115200 baud?")
        sys.exit(1)

    print("[OK] Conexao serial estabelecida!")

    if args.auto:
        diagnostico_completo(esp)
        esp.fechar()
        return

    # Menu interativo
    try:
        while True:
            print("\n--- MENU DE TESTES ---")
            print("1 - Teste de Status (Ping / JSON)")
            print("2 - Teste do Eletroíma (Liga 2s e Desliga)")
            print("3 - Teste dos Fins de Curso (Monitoramento continuo)")
            print("4 - Teste dos Drivers de Passo (Trava / Destrava motores)")
            print("5 - Teste de Movimento Curto Seguro (20mm)")
            print("6 - Teste de Homing (Zero de maquina)")
            print("7 - Console Serial Interativo (Comandos manuais)")
            print("8 - Executar Diagnostico Completo")
            print("0 - Sair")

            opcao = input("\nEscolha uma opcao [0-8]: ").strip()
            if opcao == "1":
                teste_status(esp)
            elif opcao == "2":
                teste_eletroima(esp)
            elif opcao == "3":
                teste_fins_de_curso(esp)
            elif opcao == "4":
                teste_drivers_torque(esp)
            elif opcao == "5":
                teste_pequeno_movimento(esp)
            elif opcao == "6":
                teste_homing(esp)
            elif opcao == "7":
                console_interativo(esp)
            elif opcao == "8":
                diagnostico_completo(esp)
            elif opcao in ("0", "sair", "exit", "q"):
                print("Encerrando...")
                break
            else:
                print("Opcao invalida, tente novamente.")
    finally:
        esp.fechar()


if __name__ == "__main__":
    main()
