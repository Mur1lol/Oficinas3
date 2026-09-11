"""
scripts/test_mic.py
===================
Script para testar a captura de áudio/voz do microfone no computador.

Recursos:
1. Lista todos os dispositivos de entrada de áudio disponíveis.
2. Permite escolher um microfone específico ou usar o padrão / auto-detectado.
3. Exibe um medidor de volume (VU meter) em tempo real no terminal enquanto grava.
4. Salva a gravação em 'test_mic_output.wav' para você ouvir e verificar a qualidade.

Como usar:
    python scripts/test_mic.py
    python scripts/test_mic.py --device 1 --seconds 5
"""

import argparse
import math
import os
import struct
import sys
import time
import wave
from pathlib import Path

# Adiciona a pasta raiz (voice/) ao sys.path para importar config e audio_capture
VOICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VOICE_DIR))

import config
from audio_capture import AudioStream, list_audio_devices


def calculate_rms(pcm_bytes: bytes) -> float:
    """Calcula o Root-Mean-Square (RMS) das amostras de 16-bit PCM."""
    if not pcm_bytes:
        return 0.0
    count = len(pcm_bytes) // 2
    if count == 0:
        return 0.0
    shorts = struct.unpack(f"{count}h", pcm_bytes)
    sum_squares = sum(s * s for s in shorts)
    return math.sqrt(sum_squares / count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Teste de captura de áudio do microfone")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Índice do dispositivo de áudio (use --list para ver todos)",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=5,
        help="Tempo de gravação de teste em segundos (padrão: 5)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lista todos os dispositivos de entrada de áudio e sai",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(VOICE_DIR / "test_mic_output.wav"),
        help="Caminho do arquivo WAV de saída (padrão: test_mic_output.wav)",
    )

    args = parser.parse_args()

    devices = list_audio_devices()
    if args.list:
        print("\n=== Dispositivos de entrada de áudio disponíveis ===")
        for d in devices:
            print(f"  [{d['index']}] {d['name']} ({d['channels']} ch, {d['sample_rate']} Hz)")
        print("\nPara usar um dispositivo específico: python scripts/test_mic.py --device <NUMERO>")
        return

    print("=" * 60)
    print("  ChessAI 2.0 - Teste de Captura de Voz / Microfone")
    print("=" * 60)

    print("\nDispositivos detectados no sistema:")
    for d in devices:
        marker = "-> " if args.device == d["index"] else "   "
        print(f"{marker}[{d['index']}] {d['name']}")

    # Garante UTF-8 no terminal Windows para evitar erro de charmap
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    selected_device = args.device if args.device is not None else config.MIC_DEVICE_INDEX

    print(f"\nDispositivo selecionado: {selected_device if selected_device is not None else 'Auto-detectar (config.py)'}")
    print(f"Taxa de amostragem: {config.SAMPLE_RATE} Hz | Canais: {config.CHANNELS} (Mono)")
    print(f"Gravando por {args.seconds} segundos...")
    print("Fale algo no microfone ('MAGNUS', 'Move E2 E4', etc.)...\n")

    recorded_frames: list[bytes] = []
    max_rms = 0.0
    sum_rms = 0.0
    frame_count = 0

    try:
        with AudioStream(device_index=selected_device) as stream:
            start_time = time.time()
            total_frames_target = int(args.seconds * 1000 / config.VAD_FRAME_MS)

            while frame_count < total_frames_target:
                frame = stream.read_frame(timeout=1.0)
                if frame is None:
                    continue

                recorded_frames.append(frame)
                frame_count += 1

                rms = calculate_rms(frame)
                sum_rms += rms
                if rms > max_rms:
                    max_rms = rms

                # Barra de volume visual no terminal (escala ate ~3000 RMS)
                meter_length = 25
                normalized = min(1.0, rms / 3000.0)
                filled = int(normalized * meter_length)
                bar = "#" * filled + "-" * (meter_length - filled)
                elapsed = time.time() - start_time

                status = "Voz/Som detectado!" if rms > 300 else "Silêncio..."
                sys.stdout.write(f"\r[{elapsed:4.1f}s / {args.seconds}s] Volume: [{bar}] RMS: {rms:5.0f} ({status})  ")
                sys.stdout.flush()

        print("\n\nGravação concluída!")
    except Exception as e:
        print(f"\n[ERRO] Falha ao abrir o fluxo de áudio: {e}")
        print("Dica: tente passar um índice de dispositivo explícito com --device <INDEX>.")
        return

    # Salva em arquivo WAV
    out_path = Path(args.output)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(config.CHANNELS)
        wf.setsampwidth(config.SAMPLE_WIDTH)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(b"".join(recorded_frames))

    avg_rms = sum_rms / max(1, frame_count)
    print("=" * 60)
    print(f"Arquivo salvo em: {out_path.resolve()}")
    print(f"Estatísticas:")
    print(f"  - Total de frames: {frame_count}")
    print(f"  - RMS Médio: {avg_rms:.1f}")
    print(f"  - RMS Máximo: {max_rms:.1f}")

    if max_rms < 100:
        print("\n[ALERTA] O sinal gravado foi muito baixo ou totalmente silencioso.")
        print("Possíveis causas:")
        print("  1. O microfone selecionado está mudo no Windows ou chave física desligada.")
        print("  2. O índice de dispositivo padrão não é o seu microfone ativo.")
        print("     -> Execute: python scripts/test_mic.py --list")
        print("     -> E teste especificando seu microfone: python scripts/test_mic.py --device <NUMERO>")
    else:
        print("\n[SUCESSO] Sinal de voz detectado com sucesso!")
        print(f"Abra o arquivo '{out_path.name}' para escutar e verificar a qualidade.")
    print("=" * 60)


if __name__ == "__main__":
    main()
