"""
vision/piece_classifier.py
==========================
Classifies the *type* of a piece (pawn, rook, knight, bishop, queen, king)
by multi-scale template matching against the reference images stored in
``pieces/`` (``<Type>_<Color>.png``).

No training is needed: each reference image is converted to a blurred edge
map (robust to lighting and to the piece's fill colour) and matched against
the edge map of the square ROI at several scales and rotations (0° and 180°
by default, because black pieces face the opposite side of the board).

Usage
-----
    clf = PieceClassifier()
    piece, conf = clf.classify(roi, color="white")   # ("knight", 0.81)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config as cfg

log = logging.getLogger(__name__)

_PIECE_TYPES = ("pawn", "rook", "knight", "bishop", "queen", "king")
_TEMPLATE_SIZE = 64  # px, reference icon resized to this before scaling


def _edge_map(gray: np.ndarray) -> np.ndarray:
    """Blurred Canny edge map, tolerant to small misalignments."""
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.GaussianBlur(edges, (5, 5), 0)


def _load_icon(path: Path) -> Optional[np.ndarray]:
    """Load a PNG (with optional alpha) as grayscale over a white background."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        img = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


class PieceClassifier:
    """
    Parameters
    ----------
    pieces_dir : str | Path | None
        Folder with ``<Type>_<Color>.png`` references (default: ``cfg.PIECES_DIR``).
    scales : tuple[float, ...]
        Template sizes as a fraction of the ROI's shorter side.
    rotations : tuple[int, ...]
        Rotations (degrees, multiples of 90) tried for each template.
    """

    def __init__(
        self,
        pieces_dir: str | Path | None = None,
        scales: tuple[float, ...] = cfg.PIECE_TEMPLATE_SCALES,
        rotations: tuple[int, ...] = cfg.PIECE_TEMPLATE_ROTATIONS,
    ) -> None:
        self._dir = Path(pieces_dir or cfg.PIECES_DIR)
        self._scales = scales
        self._rotations = rotations
        # {color: {piece_type: [edge map per rotation]}}
        self._templates: dict[str, dict[str, list[np.ndarray]]] = {
            "white": {}, "black": {},
        }
        self._load()

    @property
    def is_ready(self) -> bool:
        return all(len(v) == len(_PIECE_TYPES) for v in self._templates.values())

    # ------------------------------------------------------------------

    def _load(self) -> None:
        for color in self._templates:
            for ptype in _PIECE_TYPES:
                path = self._dir / f"{ptype.capitalize()}_{color.capitalize()}.png"
                icon = _load_icon(path)
                if icon is None:
                    log.warning("[Classifier] Missing reference image: %s", path)
                    continue
                icon = cv2.resize(icon, (_TEMPLATE_SIZE, _TEMPLATE_SIZE),
                                  interpolation=cv2.INTER_AREA)
                variants = []
                for rot in self._rotations:
                    k = (rot // 90) % 4
                    variants.append(_edge_map(np.rot90(icon, k).copy()))
                self._templates[color][ptype] = variants
        log.info("[Classifier] Loaded templates: white=%d black=%d",
                 len(self._templates["white"]), len(self._templates["black"]))

    def _score(self, roi_edges: np.ndarray, variants: list[np.ndarray]) -> float:
        h, w = roi_edges.shape
        best = -1.0
        for tpl in variants:
            for s in self._scales:
                size = int(min(h, w) * s)
                if size < 16 or size > min(h, w):
                    continue
                t = cv2.resize(tpl, (size, size), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(roi_edges, t, cv2.TM_CCOEFF_NORMED)
                best = max(best, float(res.max()))
        return best

    def classify(
        self, roi: np.ndarray, color: Optional[str] = None
    ) -> tuple[Optional[str], float]:
        """
        Returns ``(piece_type, confidence)``; ``(None, 0.0)`` when unsure.

        ``color`` restricts matching to that colour's references (recommended);
        if None, all references are searched.
        """
        if not self.is_ready:
            return None, 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        roi_edges = _edge_map(gray)

        colors = [color] if color in self._templates else list(self._templates)
        scores: dict[str, float] = {}
        for c in colors:
            for ptype, variants in self._templates[c].items():
                scores[ptype] = max(scores.get(ptype, -1.0),
                                    self._score(roi_edges, variants))

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        (best_type, best), (_, second) = ranked[0], ranked[1]
        if best < cfg.PIECE_TYPE_MIN_SCORE:
            return None, max(0.0, best)

        margin = best - second
        confidence = float(np.clip(0.5 * best + 2.5 * margin, 0.0, 1.0))
        if margin < cfg.PIECE_TYPE_MIN_MARGIN:
            return None, confidence
        return best_type, confidence
