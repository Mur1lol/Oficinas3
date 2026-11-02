import sys
sys.path.insert(0, '.')
import cv2
import numpy as np
import glob
from pathlib import Path

# Load img1 (clean 32 pieces) and img2 (hand)
img1 = cv2.imread(r'C:\Users\Usuario\.gemini\antigravity-ide\brain\40a1b5cc-e3ac-43bd-8213-e233cf37c001\.user_uploaded\media_1793628615707.png')
crop1 = img1[0:574, 8:588]
board1 = cv2.resize(crop1, (640, 640))

img2 = cv2.imread(r'C:\Users\Usuario\.gemini\antigravity-ide\brain\40a1b5cc-e3ac-43bd-8213-e233cf37c001\.user_uploaded\media_1793629149061.png')
# img2 size is 393x409
# Let's align img2 to board:
board2 = cv2.resize(img2, (640, 640))

ref = cv2.imread('vision/reference_empty.png')
ref_board = ref[:, 267:907]

cols = 'ABCDEFGH'
rows = '87654321'

pieces_map = {
    'A8': ('rook', 'black'), 'B8': ('knight', 'black'), 'C8': ('bishop', 'black'), 'D8': ('queen', 'black'),
    'E8': ('king', 'black'), 'F8': ('bishop', 'black'), 'G8': ('knight', 'black'), 'H8': ('rook', 'black'),
    'A7': ('pawn', 'black'), 'B7': ('pawn', 'black'), 'C7': ('pawn', 'black'), 'D7': ('pawn', 'black'),
    'E7': ('pawn', 'black'), 'F7': ('pawn', 'black'), 'G7': ('pawn', 'black'), 'H7': ('pawn', 'black'),
    'A2': ('pawn', 'white'), 'B2': ('pawn', 'white'), 'C2': ('pawn', 'white'), 'D2': ('pawn', 'white'),
    'E2': ('pawn', 'white'), 'F2': ('pawn', 'white'), 'G2': ('pawn', 'white'), 'H2': ('pawn', 'white'),
    'A1': ('rook', 'white'), 'B1': ('knight', 'white'), 'C1': ('bishop', 'white'), 'D1': ('queen', 'white'),
    'E1': ('king', 'white'), 'F1': ('bishop', 'white'), 'G1': ('knight', 'white'), 'H1': ('rook', 'white'),
}

print(f"Total pieces in map: {len(pieces_map)}")
