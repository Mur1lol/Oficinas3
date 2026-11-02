import sys
sys.path.insert(0, '.')
import cv2
import numpy as np

img_screenshot = cv2.imread(r'C:\Users\Usuario\.gemini\antigravity-ide\brain\40a1b5cc-e3ac-43bd-8213-e233cf37c001\.user_uploaded\media_1793568860544.png')
crop = img_screenshot[37:368, 16:621]
warped_screenshot = cv2.resize(crop, (1174, 640))
ref_full = cv2.imread('vision/reference_empty.png')

from vision.segmentation import BoardSegmenter
from vision.piece_classifier import PieceClassifier, _extract_icon
import vision.config as cfg

segmenter = BoardSegmenter(board_orientation='normal')
ref_sqs = segmenter.segment_board(ref_full)
curr_sqs = segmenter.segment_board(warped_screenshot)
clf = PieceClassifier()

print("Testing balanced piece vs hand detection...")

# Let's inspect the values for every square
cols = "ABCDEFGH"
rows = "87654321"

def get_neighbors(square: str):
    c = cols.index(square[0])
    r = rows.index(square[1])
    nbrs = []
    for dc in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if dc == 0 and dr == 0:
                continue
            nc, nr = c + dc, r + dr
            if 0 <= nc < 8 and 0 <= nr < 8:
                nbrs.append(f"{cols[nc]}{rows[nr]}")
    return nbrs

# We test with threshold = 0.10 (same as calibrator slider)
thresh = 0.10

raw_squares = {}
for sq in sorted(curr_sqs.keys()):
    roi = curr_sqs[sq]
    ref_roi = ref_sqs.get(sq)
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    gray = cv2.resize(gray, (80, 80))
    
    ref_gray = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY) if ref_roi.ndim == 3 else ref_roi
    ref_gray = cv2.resize(ref_gray, (80, 80))
    diff = cv2.absdiff(gray, ref_gray)
    
    mad = float(np.mean(diff)) / 255.0
    
    Y, X = np.ogrid[:80, :80]
    dist_c = np.sqrt((X - 40) ** 2 + (Y - 40) ** 2)
    border_mask = np.ones((80, 80), dtype=bool)
    border_mask[8:72, 8:72] = False
    center_mad = float(np.mean(diff[dist_c <= 20])) / 255.0
    border_mad = float(np.mean(diff[border_mask])) / 255.0
    ratio = center_mad / max(0.001, border_mad)
    
    # Ring & Color
    ring_val = float(gray[(dist_c >= 12) & (dist_c <= 16)].mean())
    split_thresh = getattr(cfg, "PIECE_PUCK_SPLIT_THRESHOLD", 112.0)
    det_color = "white" if ring_val < split_thresh else "black"
    
    # Icon contrast
    if det_color == "black":
        norm = ring_val - gray.astype(np.float32)
    else:
        norm = gray.astype(np.float32) - ring_val
    norm = np.maximum(0.0, norm) * (dist_c <= 12.0)
    max_contrast = float(norm.max())
    
    # Classification
    piece, conf = clf.classify(roi, det_color)
    
    raw_squares[sq] = {
        'mad': mad,
        'center_mad': center_mad,
        'border_mad': border_mad,
        'ratio': ratio,
        'color': det_color,
        'max_contrast': max_contrast,
        'piece': piece,
        'conf': conf,
    }

# Now: Detection logic
# 1. Occupancy by threshold (slider controls it directly!)
#    A square is candidate occupied if mad >= thresh.
# 2. Hand identification:
#    A candidate square is classified as a HAND (not a piece) if:
#    - It clearly spills across borders: border_mad >= 0.18 and ratio < 0.95 and piece is None.
#    - OR it has large difference (mad >= 0.15) but virtually no icon (max_contrast < 25.0) and ratio < 1.05.
# 3. Cluster propagation:
#    If a square has border_mad >= 0.18 and is adjacent to >= 2 hand squares and piece is None (or conf < 0.60):
#    it is part of the hand cluster.
# 4. All other candidate squares with mad >= thresh are PIECES (occupied=True, is_piece=True)!

results = {}
for sq, d in raw_squares.items():
    if d['mad'] < thresh:
        results[sq] = {'occupied': False, 'is_piece': False, 'is_hand': False, 'color': None, 'piece': None}
        continue
        
    # Check if hand
    is_hand_spill = (d['border_mad'] >= 0.18 and d['ratio'] < 0.95 and d['piece'] is None and d['max_contrast'] < 60.0)
    is_hand_no_icon = (d['max_contrast'] < 25.0 and d['mad'] >= 0.15 and d['ratio'] < 1.05)
    
    if is_hand_spill or is_hand_no_icon:
        results[sq] = {'occupied': False, 'is_piece': False, 'is_hand': True, 'color': None, 'piece': None}
    else:
        results[sq] = {'occupied': True, 'is_piece': True, 'is_hand': False, 'color': d['color'], 'piece': d['piece']}

# Cluster check
initial_hands = {sq for sq, r in results.items() if r['is_hand']}
for sq, r in results.items():
    if r['occupied']:
        nbrs = get_neighbors(sq)
        adj_hands = [n for n in nbrs if n in initial_hands]
        d = raw_squares[sq]
        # If surrounded by hand squares, border is covered, and no strong piece classification
        if len(adj_hands) >= 2 and d['border_mad'] >= 0.18 and (d['piece'] is None or d['conf'] < 0.60):
            print(f"Cluster absorbed hand square {sq} (hands={adj_hands}, conf={d['conf']:.2f})")
            r['occupied'] = False
            r['is_piece'] = False
            r['is_hand'] = True
            r['color'] = None
            r['piece'] = None

hand_expected = ['D3', 'E3', 'F3', 'G3', 'H3', 'E4', 'F4', 'G4', 'H4', 'F5', 'G2', 'H2']
pieces_expected = [
    'A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1',
    'A2', 'B2', 'C2', 'D2', 'E2', 'F2',
    'A7', 'B7', 'C7', 'D7', 'E7', 'F7', 'G7', 'H7',
    'A8', 'B8', 'C8', 'D8', 'E8', 'F8', 'G8', 'H8',
]

hand_fp = [sq for sq in hand_expected if results[sq]['occupied']]
piece_fn = [sq for sq in pieces_expected if not results[sq]['occupied']]

print("\n--- RESULTS ---")
print(f"Hand False Positives (should be 0): {hand_fp}")
print(f"Piece False Negatives (should be 0): {piece_fn}")
print(f"Total pieces detected: {sum(1 for r in results.values() if r['occupied'])}")
print(f"Total hands detected: {sum(1 for r in results.values() if r['is_hand'])}")
