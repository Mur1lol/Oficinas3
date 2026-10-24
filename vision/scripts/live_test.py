"""
Live camera test: draws, on every board square, whether it is
EMPTY, occupied by a PIECE (colour + type) or occupied by an unrecognised
OBJECT.

    python vision/scripts/live_test.py [--device 0] [--scale 1.5]

Keys:
    R        capture the EMPTY board as reference (saved to
             vision/reference_empty.png and reloaded on next start)
    Q / ESC  quit

Run calibrate_camera.py first.  Do not move the camera after pressing R.
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vision import config as cfg
from vision.calibration import PerspectiveCalibrator
from vision.camera import CameraController
from vision.piece_detector import PieceDetector
from vision.segmentation import BoardSegmenter, CEM_W, SQUARE_SZ, COLUMNS, ROWS

REF_PATH = Path(cfg.VISION_DIR) / "reference_empty.png"

NAMES = {"pawn": "Peao", "rook": "Torre", "knight": "Cavalo",
         "bishop": "Bispo", "queen": "Rainha", "king": "Rei"}
LETTER = {"pawn": "P", "rook": "R", "knight": "N", "bishop": "B", "queen": "Q", "king": "K"}

GREEN, ORANGE, WHITE, BLACK, RED = (0, 200, 0), (0, 140, 255), (255, 255, 255), (0, 0, 0), (0, 0, 255)


def put_text(img, text, org, scale, color, thick=1, outline=True):
    if outline:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_overlay(board, results):
    """Annotate each square: green dot = empty; box + label = occupied."""
    view = board.copy()
    counts = {"empty": 0, "piece": 0, "object": 0}
    for ri, rl in enumerate(ROWS):
        for ci, cl in enumerate(COLUMNS):
            r = results[f"{cl}{rl}"]
            x0, y0 = CEM_W + ci * SQUARE_SZ, ri * SQUARE_SZ
            cx, cy = x0 + SQUARE_SZ // 2, y0 + SQUARE_SZ // 2
            if not r.occupied:
                counts["empty"] += 1
                cv2.circle(view, (cx, cy), 4, GREEN, -1)
                continue
            if r.piece:
                counts["piece"] += 1
                col = WHITE if r.color == "white" else (60, 60, 60)
                cv2.rectangle(view, (x0 + 2, y0 + 2), (x0 + SQUARE_SZ - 2, y0 + SQUARE_SZ - 2), RED if r.color == "black" else (255, 200, 0), 2)
                put_text(view, LETTER[r.piece], (cx - 10, cy + 8), 0.9, col if r.color == "white" else (200, 200, 200), 2)
                put_text(view, NAMES[r.piece], (x0 + 4, y0 + SQUARE_SZ - 6), 0.35, WHITE)
            else:
                counts["object"] += 1
                cv2.rectangle(view, (x0 + 2, y0 + 2), (x0 + SQUARE_SZ - 2, y0 + SQUARE_SZ - 2), ORANGE, 2)
                put_text(view, "?", (cx - 8, cy + 8), 0.9, ORANGE, 2)
                put_text(view, "Objeto", (x0 + 4, y0 + SQUARE_SZ - 6), 0.35, ORANGE)
    return view, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=cfg.CAMERA_DEVICE_INDEX)
    ap.add_argument("--scale", type=float, default=1.5)
    args = ap.parse_args()

    if Path(cfg.CALIBRATION_FILE).exists():
        cal = PerspectiveCalibrator.load(cfg.CALIBRATION_FILE)
    else:
        print("Sem calibracao - usando modo nao calibrado.")
        cal = PerspectiveCalibrator.default_uncalibrated()

    seg, det = BoardSegmenter(), PieceDetector()
    ref_squares = None

    def set_ref(ref_board):
        nonlocal ref_squares
        det.set_reference(ref_board)
        ref_squares = seg.segment_board(ref_board)

    if REF_PATH.exists():
        set_ref(cv2.imread(str(REF_PATH)))
        print(f"Referencia carregada de {REF_PATH}")

    with CameraController(device_index=args.device) as cam:
        print("R = definir referencia (tabuleiro VAZIO) | Q = sair")
        last = None
        while True:
            board = cal.correct_perspective(cam.capture())
            key = cv2.waitKey(150) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                cv2.imwrite(str(REF_PATH), board)
                set_ref(board)
                print("Referencia definida (tabuleiro vazio) e salva.")

            if ref_squares is None:
                # Without a reference, occupancy is unreliable: do not guess.
                view = seg.draw_grid(board)
                put_text(view, "SEM REFERENCIA: deixe o tabuleiro VAZIO e pressione R",
                         (10, 30), 0.6, RED, 2)
            else:
                results = det.analyze_all(seg.segment_board(board), ref_squares)
                view, c = draw_overlay(board, results)
                put_text(view, f"Vazias: {c['empty']}  Pecas: {c['piece']}  Objetos: {c['object']}",
                         (10, 20), 0.6, WHITE, 2)
                summary = {k: (r.color, r.piece) for k, r in results.items() if r.occupied}
                if summary != last:
                    last = summary
                    print("\n--- Casas ocupadas ---")
                    for sq, (color, piece) in sorted(summary.items()):
                        print(f"{sq}: {'PECA ' + color + ' ' + NAMES[piece] if piece else 'OBJETO (nao reconhecido)'}")

            if args.scale != 1.0:
                view = cv2.resize(view, None, fx=args.scale, fy=args.scale)
            cv2.imshow("VoiceChess - visao (R=referencia, Q=sair)", view)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
