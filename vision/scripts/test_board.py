#!/usr/bin/env python3
"""
vision/scripts/test_board.py
============================
Quick terminal test to inspect board detection on a headless Raspberry Pi.

Captures a frame, applies calibration from camera_calibration.json,
runs occupancy & color detection on all 64 squares, and prints an
ASCII chessboard directly in the PuTTY terminal.

Usage:
------
    python vision/scripts/test_board.py
    python vision/scripts/test_board.py --device 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

import vision.config as cfg
from vision.calibration import CalibrationData, PerspectiveCalibrator
from vision.camera import CameraController
from vision.piece_detector import PieceDetector
from vision.segmentation import BoardSegmenter


def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceChess Terminal Board Inspector")
    parser.add_argument("--device", "-d", type=int, default=cfg.CAMERA_DEVICE_INDEX, help="Indice da camera")
    parser.add_argument("--calib", "-c", default=cfg.CALIBRATION_FILE, help="Caminho do camera_calibration.json")
    args = parser.parse_args()

    cal_path = Path(args.calib)
    if not cal_path.is_file():
        print(f"\n[ERRO] Arquivo de calibração não encontrado em: {cal_path}")
        print("Execute primeiro o calibrador web para definir os 4 cantos:")
        print("    python vision/scripts/web_calibrator.py\n")
        sys.exit(1)

    print(f"\nCarregando calibração de: {cal_path}")
    cal_data = CalibrationData.load(cal_path)
    calibrator = PerspectiveCalibrator(cal_data)

    ref_path = Path(cfg.VISION_DIR) / "reference_empty.png"
    ref_img = cv2.imread(str(ref_path)) if ref_path.is_file() else None
    if ref_img is None:
        print("[AVISO] Imagem de referência (tabuleiro vazio) não encontrada em reference_empty.png.")
        print("        A detecção usará apenas variância de textura (menos precisa).")
    else:
        print("Referência de tabuleiro vazio carregada com sucesso.")

    print(f"Conectando à câmera {args.device}...")
    try:
        with CameraController(device_index=args.device) as cam:
            frame = cam.capture_with_retry()
    except Exception as e:
        print(f"[ERRO] Falha ao capturar imagem da câmera: {e}")
        sys.exit(1)

    if frame is None:
        print("[ERRO] Frame capturado é nulo.")
        sys.exit(1)

    print("Corrigindo perspectiva e segmentando casas...")
    warped = calibrator.correct_perspective(frame)
    segmenter = BoardSegmenter(board_orientation=cal_data.board_orientation)
    squares = segmenter.segment_board(warped)
    detector = PieceDetector(reference_image=ref_img)

    results = detector.analyze_all(squares)
    occupied_count = 0
    white_count = 0
    black_count = 0
    hand_squares = []

    for name, analysis in results.items():
        if getattr(analysis, "is_hand", False):
            hand_squares.append(name)
        if analysis.occupied and getattr(analysis, "is_piece", True):
            occupied_count += 1
            if analysis.color == "white":
                white_count += 1
            elif analysis.color == "black":
                black_count += 1

    # Renderiza Tabuleiro ASCII
    cols = "ABCDEFGH"
    rows = "87654321"
    type_letters = {"pawn": "P", "rook": "R", "knight": "N", "bishop": "B", "queen": "Q", "king": "K"}

    print("\n" + "=" * 53)
    print("           TABULEIRO RECONHECIDO (VISÃO)")
    print("=" * 53)
    print("      A    B    C    D    E    F    G    H")
    print("   +----+----+----+----+----+----+----+----+")
    for r in rows:
        line = f" {r} |"
        for c in cols:
            sq = f"{c}{r}"
            st = results.get(sq)
            if not st or not st.occupied or not getattr(st, "is_piece", True):
                char = "  "
            else:
                col_c = (st.color or "?")[0].upper()
                pt_c = type_letters.get(st.piece, "?")
                char = f"{col_c}{pt_c}"
            line += f" {char} |"
        print(line + f" {r}")
        print("   +----+----+----+----+----+----+----+----+")
    print("      A    B    C    D    E    F    G    H\n")

    identified = sum(1 for st in results.values() if st.occupied and getattr(st, "is_piece", True) and st.piece)
    print(f"Total de peças detectadas: {occupied_count}")
    print(f"  - Peças Brancas (W): {white_count}")
    print(f"  - Peças Pretas  (B): {black_count}")
    print(f"  - Identificadas com tipo : {identified}/{occupied_count}")
    print(f"  - Casas Vazias     : {64 - occupied_count}\n")

    if hand_squares:
        print(f"⚠️  [AVISO] Mão ou obstrução detectada nas casas: {', '.join(sorted(hand_squares))}")
        print("           Essas casas foram desconsideradas para evitar classificações falsas.\n")


if __name__ == "__main__":
    main()
