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

# Amostras reais: a casa e normalizada para 80x80; busca no centro (56x56)
# com um molde central de 44x44 -> tolera ~6 px de deslocamento da peca.
_SAMPLE_NORM = 80
_SEARCH_MARGIN = 12   # (80 - 56) / 2
_TPL_MARGIN = 18      # (80 - 44) / 2


def _prep_square(img: np.ndarray) -> np.ndarray:
    """Casa BGR/cinza -> cinza 80x80 com contraste equalizado (CLAHE)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (_SAMPLE_NORM, _SAMPLE_NORM), interpolation=cv2.INTER_AREA)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(img)


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
        samples_dir: str | Path | None = None,
    ) -> None:
        self._dir = Path(pieces_dir or cfg.PIECES_DIR)
        self._samples_dir = Path(samples_dir or cfg.PIECE_SAMPLES_DIR)
        self._scales = scales
        self._rotations = rotations
        # {color: {piece_type: [edge map per rotation]}}
        self._templates: dict[str, dict[str, list[np.ndarray]]] = {
            "white": {}, "black": {},
        }
        # {color: {piece_type: [molde cinza 44x44, ja com as 4 rotacoes]}}
        self._samples: dict[str, dict[str, list[np.ndarray]]] = {
            "white": {}, "black": {},
        }
        self._load()
        self._load_samples()

    @property
    def has_samples(self) -> bool:
        return any(self._samples[c] for c in self._samples)

    @property
    def is_ready(self) -> bool:
        if self.has_samples:
            return True
        return all(len(v) == len(_PIECE_TYPES) for v in self._templates.values())

    def sample_counts(self) -> dict[str, int]:
        """Numero de amostras (sem contar rotacoes) por '<tipo>_<cor>'."""
        return {f"{p}_{c}": len(v) // 4
                for c, d in self._samples.items() for p, v in d.items()}

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

    def _load_samples(self) -> None:
        """Le vision/pieces/samples/<tipo>_<cor>/*.png (casas recortadas pela camera)."""
        if not self._samples_dir.is_dir():
            return
        for sub in sorted(self._samples_dir.iterdir()):
            if not sub.is_dir() or "_" not in sub.name:
                continue
            ptype, color = sub.name.lower().split("_", 1)
            if ptype not in _PIECE_TYPES or color not in self._samples:
                continue
            for f in sorted(sub.glob("*.png")) + sorted(sub.glob("*.jpg")):
                img = cv2.imread(str(f))
                if img is None:
                    continue
                sq = _prep_square(img)
                m = _TPL_MARGIN
                tpl = sq[m:_SAMPLE_NORM - m, m:_SAMPLE_NORM - m]
                # Peca pode estar girada em qualquer direcao vista de cima
                for k in range(4):
                    self._samples[color].setdefault(ptype, []).append(
                        np.ascontiguousarray(np.rot90(tpl, k)))
        if self.has_samples:
            log.info("[Classifier] Real samples loaded: %s", self.sample_counts())

    def _classify_samples(
        self, roi: np.ndarray, color: Optional[str]
    ) -> tuple[Optional[str], float]:
        sq = _prep_square(roi)
        m = _SEARCH_MARGIN
        search = sq[m:_SAMPLE_NORM - m, m:_SAMPLE_NORM - m]

        # Usa so a cor detectada se houver amostras dela; senao procura em todas
        colors = [color] if color in self._samples and self._samples[color] else \
            [c for c in self._samples if self._samples[c]]
        scores: dict[str, float] = {}
        for c in colors:
            for ptype, tpls in self._samples[c].items():
                best = max(float(cv2.matchTemplate(search, t, cv2.TM_CCOEFF_NORMED).max())
                           for t in tpls)
                scores[ptype] = max(scores.get(ptype, -1.0), best)
        if not scores:
            return None, 0.0

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_type, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best - second
        confidence = float(np.clip(0.5 * best + 2.5 * margin, 0.0, 1.0))
        if best < cfg.PIECE_SAMPLE_MIN_SCORE or margin < cfg.PIECE_SAMPLE_MIN_MARGIN:
            return None, confidence
        return best_type, confidence

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
        if self.has_samples:
            return self._classify_samples(roi, color)
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
