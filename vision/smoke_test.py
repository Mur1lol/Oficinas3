"""
vision/smoke_test.py
====================
Quick sanity test for the vision module. No camera required.
Run from the Oficinas/ directory:

    python -m vision.smoke_test
    # or
    python vision/smoke_test.py
"""
import sys
from pathlib import Path

# Add Oficinas/ to path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing imports...")
from vision.board_state import BoardState, SquareState
from vision.piece_detector import PieceDetector
from vision.segmentation import BoardSegmenter
from vision.calibration import CalibrationData, PerspectiveCalibrator
from vision.state_comparator import StateComparator
from vision.diag_manager import DiagManager
from vision import CameraSystem
print("All imports OK")

import numpy as np
import vision.config as cfg

# Test 1: BoardState
state = BoardState.empty_board()
d = state.to_dict()
n = len(d["board"])
print(f"BoardState: {n} squares, confidence={state.overall_confidence}")

# Test 2: Segmenter
seg = BoardSegmenter()
W = cfg.BOARD_OUTPUT_SIZE_PX + 2 * cfg.CEMETERY_REGION_WIDTH_PX
H = cfg.BOARD_OUTPUT_SIZE_PX
blank = np.zeros((H, W, 3), dtype="uint8")
squares = seg.segment_board(blank)
print(f"Segmented {len(squares)} squares")

cems = seg.segment_cemeteries(blank)
print(f"Cemetery slots: white={len(cems['white'])}, black={len(cems['black'])}")

# Test 3: PieceDetector
det = PieceDetector()
roi = np.full((80, 80, 3), 200, dtype="uint8")  # bright = white piece
analysis = det.analyze_square("E2", roi)
print(f"Analysis E2: occupied={analysis.occupied}, color={analysis.color}, conf={analysis.confidence:.2f}")

# Test 4: StateComparator
before = BoardState.empty_board()
after  = BoardState.empty_board()
comp = StateComparator()
result = comp.compare(before, after, {"from": "E2", "to": "E4"})
print(f"Compare: success={result.success}, error={result.error}")

# Test 5: CalibrationData round-trip
cal = CalibrationData(
    board_corners=[[10,10],[630,10],[630,630],[10,630]],
    board_orientation="normal"
)
cal2 = CalibrationData.from_dict(cal.to_dict())
print(f"CalibrationData round-trip: valid={cal2.is_valid()}")

# Test 6: PerspectiveCalibrator (with fake corners)
p = PerspectiveCalibrator(cal2)
corrected = p.correct_perspective(blank)
print(f"Corrected shape: {corrected.shape}")

print("\nAll smoke tests PASSED.")
