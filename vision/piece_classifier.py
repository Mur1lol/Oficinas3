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

# Amostras reais: normalizacao 80x80; busca 32x32 ao redor do centro (40, 40)
# com molde central de 22x22 -> tolera +/- 5 px de deslocamento da peca.
_SAMPLE_NORM = 80
_SEARCH_BOX = (22, 58)  # 36x36 search window
_TPL_BOX = (28, 52)     # 24x24 template window


def _prep_square(img: np.ndarray) -> np.ndarray:
    """Casa BGR/cinza -> cinza 80x80 com contraste equalizado (CLAHE)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (_SAMPLE_NORM, _SAMPLE_NORM), interpolation=cv2.INTER_AREA)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(img)


def find_puck_center(gray: np.ndarray) -> tuple[int, int]:
    """Localiza o centro do disco da peca tolerando inclinacao e perspectiva de camera."""
    sub = gray[18:62, 22:58]
    blurred = cv2.GaussianBlur(sub, (3, 3), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    thresh = np.percentile(mag, 80)
    ys, xs = np.where(mag >= thresh)
    if len(xs) > 10:
        cx = int(np.mean(xs) + 22)
        cy = int(np.mean(ys) + 18)
    else:
        cx, cy = 40, 40
    cx = max(30, min(50, cx))
    cy = max(26, min(54, cy))
    return cx, cy


def _extract_icon(
    img: np.ndarray, color: Optional[str] = None, radius_mask: float = 14.0
) -> tuple[np.ndarray, str]:
    """
    Normaliza a imagem da casa recortada (80x80) para isolar o icone gravado no disco:
    1. Localiza o centro real do disco (cx, cy) para compensar perspectiva.
    2. Mede o anel do corpo do disco ao redor de (cx, cy).
    3. Subtrai o corpo para que o icone fique sempre brilhante sobre fundo preto.
    4. Translada o icone para que (cx, cy) fique exatamente em (40, 40).
    5. Aplica mascara circular (raio 14px) centrada em (40, 40).
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = cv2.resize(gray, (_SAMPLE_NORM, _SAMPLE_NORM), interpolation=cv2.INTER_AREA)

    cx, cy = find_puck_center(gray)

    h, w = gray.shape
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    ring_val = float(gray[(dist >= 11) & (dist <= 15)].mean())
    split_thresh = getattr(cfg, "PIECE_PUCK_SPLIT_THRESHOLD", 112.0)
    det_color = color if color in ("white", "black") else ("white" if ring_val < split_thresh else "black")

    if det_color == "black":
        norm = ring_val - gray.astype(np.float32)
    else:
        norm = gray.astype(np.float32) - ring_val
    norm = np.maximum(0.0, norm)

    # Centraliza o icone em (40, 40)
    dx = 40 - cx
    dy = 40 - cy
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(norm, M, (_SAMPLE_NORM, _SAMPLE_NORM), borderValue=0)

    dist_c = np.sqrt((X - 40) ** 2 + (Y - 40) ** 2)
    masked = shifted * (dist_c <= radius_mask)

    max_val = float(masked.max())
    min_contrast = getattr(cfg, "PIECE_MIN_ICON_CONTRAST", 15.0)
    if max_val < min_contrast:
        return np.zeros_like(gray), det_color

    norm_img = (masked / max_val * 255.0).astype(np.uint8)
    return norm_img, det_color


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
        # {color: {piece_type: [moldes normalizados]}}
        self._samples: dict[str, dict[str, list[np.ndarray]]] = {
            "white": {}, "black": {},
        }
        # Moldes agrupados por tipo (compartilhados entre brancas e pretas com todas as rotacoes)
        self._pooled_samples: dict[str, list[np.ndarray]] = {}
        self._sample_counts: dict[str, int] = {}

        self._load()
        self._load_samples()

    @property
    def has_samples(self) -> bool:
        return any(self._pooled_samples.values())

    @property
    def is_ready(self) -> bool:
        if self.has_samples:
            return True
        return all(len(v) == len(_PIECE_TYPES) for v in self._templates.values())

    def sample_counts(self) -> dict[str, int]:
        """Numero de fotos de amostra por '<tipo>_<cor>'."""
        return dict(self._sample_counts)

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
        angles = list(range(0, 360, 15))  # 24 angulos de rotacao para cobrir qualquer orientacao
        for sub in sorted(self._samples_dir.iterdir()):
            if not sub.is_dir() or "_" not in sub.name:
                continue
            parts = sub.name.lower().split("_", 1)
            if len(parts) != 2:
                continue
            ptype, color = parts
            if ptype not in _PIECE_TYPES or color not in ("white", "black"):
                continue
            files = sorted(sub.glob("*.png")) + sorted(sub.glob("*.jpg"))
            self._sample_counts[f"{ptype}_{color}"] = len(files)
            for f in files:
                img = cv2.imread(str(f))
                if img is None:
                    continue
                icon, _ = _extract_icon(img, color=color)
                # Molde central 22x22 isolando o simbolo
                tpl = icon[_TPL_BOX[0]:_TPL_BOX[1], _TPL_BOX[0]:_TPL_BOX[1]]
                h_t, w_t = tpl.shape
                for ang in angles:
                    if ang == 0:
                        r_tpl = tpl
                    else:
                        M = cv2.getRotationMatrix2D((w_t / 2.0, h_t / 2.0), ang, 1.0)
                        r_tpl = cv2.warpAffine(tpl, M, (w_t, h_t))
                    self._samples[color].setdefault(ptype, []).append(r_tpl)
                    self._pooled_samples.setdefault(ptype, []).append(r_tpl)
        if self.has_samples:
            log.info("[Classifier] Real samples loaded: %s", self.sample_counts())

    def _classify_samples(
        self, roi: np.ndarray, color: Optional[str]
    ) -> tuple[Optional[str], float]:
        icon, det_color = _extract_icon(roi, color=color)
        if icon.max() == 0:
            return None, 0.0
        # Janela de busca 32x32 ao redor do centro (40, 40) -> tolera +/- 5 px de desalinhamento
        search = icon[_SEARCH_BOX[0]:_SEARCH_BOX[1], _SEARCH_BOX[0]:_SEARCH_BOX[1]]

        scores: dict[str, float] = {}
        for ptype in _PIECE_TYPES:
            tpls = self._pooled_samples.get(ptype)
            if not tpls:
                continue
            best = max(float(cv2.matchTemplate(search, t, cv2.TM_CCOEFF_NORMED).max()) for t in tpls)
            scores[ptype] = best

        if not scores:
            return None, 0.0

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_type, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best - second
        confidence = float(np.clip(0.6 * best + 2.0 * margin, 0.0, 1.0))
        min_conf = getattr(cfg, "PIECE_SAMPLE_MIN_CONFIDENCE", 0.65)
        if best < cfg.PIECE_SAMPLE_MIN_SCORE or margin < cfg.PIECE_SAMPLE_MIN_MARGIN or confidence < min_conf:
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
