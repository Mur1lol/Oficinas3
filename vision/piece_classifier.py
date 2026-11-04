"""
vision/piece_classifier.py
==========================
Classifies the *type* of a piece (pawn, rook, knight, bishop, queen, king)
by multi-scale template matching against the reference images stored in
``pieces/`` (``<Type>_<Color>.png``) or real camera samples stored in
``pieces/samples/<type>_<color>/``.

Also provides:
* ``locate_puck_center(gray)`` — dynamic localization of the physical circular
  puck within a square ROI (tolerant to off-center pieces, cast shadows, and
  printed square target dots).
* ``detect_puck_color(gray, center)`` — illumination-invariant relative contrast
  classifier separating white and black pieces around the located center.

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

# Amostras reais: normalizacao 80x80; com auto-centralizacao do disco,
# busca 28x28 ao redor do centro (40, 40) com molde central de 22x22.
_SAMPLE_NORM = 80
_SEARCH_BOX = (26, 54)  # 28x28 search window around centered icon
_TPL_BOX = (29, 51)     # 22x22 template window

# ---------------------------------------------------------------------------
# Precomputed concentric kernels for dynamic puck localization (80x80 scale)
# ---------------------------------------------------------------------------

_KY, _KX = np.ogrid[-25:26, -25:26]
_KR = np.sqrt(_KX ** 2 + _KY ** 2)


def _norm_kernel(mask: np.ndarray) -> np.ndarray:
    k = mask.astype(np.float32)
    s = float(k.sum())
    return k / s if s > 0 else k


_K_ICON  = _norm_kernel(_KR <= 8.5)
_K_MOAT  = _norm_kernel((_KR >= 11.0) & (_KR <= 14.2))
_RIM_BAND = (_KR >= 14.8) & (_KR <= 19.0)
_K_RIM_N = _norm_kernel(_RIM_BAND & (_KY < -4))
_K_RIM_S = _norm_kernel(_RIM_BAND & (_KY >  4))
_K_RIM_W = _norm_kernel(_RIM_BAND & (_KX < -4))
_K_RIM_E = _norm_kernel(_RIM_BAND & (_KX >  4))

_YG80, _XG80 = np.ogrid[:_SAMPLE_NORM, :_SAMPLE_NORM]


def _to_gray80(img: np.ndarray) -> np.ndarray:
    """Convert BGR/gray image to 80x80 float32 grayscale."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if gray.shape[0] != _SAMPLE_NORM or gray.shape[1] != _SAMPLE_NORM:
        gray = cv2.resize(gray, (_SAMPLE_NORM, _SAMPLE_NORM), interpolation=cv2.INTER_AREA)
    return np.asarray(gray, dtype=np.float32)


def locate_puck_center(gray: np.ndarray) -> tuple[int, int, float, float]:
    """
    Locate the physical puck center ``(cx, cy)`` in an 80x80 grayscale ROI.

    Combines 4 concentric geometric invariants of the physical pieces:
    1. High gradient & local standard deviation inside the central icon (r <= 8.5 px).
    2. Low gradient & uniform intensity in the solid puck body moat (11.0 <= r <= 14.2 px).
    3. 360-degree circular outer rim edges across all 4 quadrants (14.8 <= r <= 19.0 px).

    Returns
    -------
    (cx, cy, structural_score, icon_std)
        ``cx, cy`` in [14..66] on the 80x80 grid, ``structural_score`` measuring
        3D puck + icon presence (empty squares < 2500, occupied > 10000), and
        ``icon_std`` (local std inside r <= 8.5 around ``(cx, cy)``).
    """
    # Percentile-normalize the square ROI before structural scoring so that
    # dim rooms (0.5x brightness) and bright spotlights (1.35x brightness)
    # produce identical gradient and moat-std scales.
    g_raw = _to_gray80(gray)
    p5, p95 = np.percentile(g_raw, (5, 95))
    span = max(25.0, float(p95 - p5))
    g_norm = np.clip((g_raw - p5) * (220.0 / span) + 18.0, 0.0, 255.0).astype(np.float32)
    g_eq = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4)).apply(
        g_norm.astype(np.uint8)
    ).astype(np.float32)
    g = 0.65 * g_norm + 0.35 * g_eq
    g2 = g * g
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)

    m_icon = cv2.filter2D(g, cv2.CV_32F, _K_ICON)
    std_icon = np.sqrt(np.maximum(0.0, cv2.filter2D(g2, cv2.CV_32F, _K_ICON) - m_icon * m_icon))

    g_norm2 = g_norm * g_norm
    m_moat = cv2.filter2D(g_norm, cv2.CV_32F, _K_MOAT)
    std_moat = np.sqrt(np.maximum(0.0, cv2.filter2D(g_norm2, cv2.CV_32F, _K_MOAT) - m_moat * m_moat))

    r_icon = cv2.filter2D(grad, cv2.CV_32F, _K_ICON)
    r_moat = cv2.filter2D(grad, cv2.CV_32F, _K_MOAT)

    r_N = cv2.filter2D(grad, cv2.CV_32F, _K_RIM_N)
    r_S = cv2.filter2D(grad, cv2.CV_32F, _K_RIM_S)
    r_W = cv2.filter2D(grad, cv2.CV_32F, _K_RIM_W)
    r_E = cv2.filter2D(grad, cv2.CV_32F, _K_RIM_E)
    r_rim_sym = (
        np.maximum(1.0, r_N)
        * np.maximum(1.0, r_S)
        * np.maximum(1.0, r_W)
        * np.maximum(1.0, r_E)
    ) ** 0.25

    s_inner = np.maximum(0.0, r_icon - 0.7 * r_moat) * (
        std_icon * 20.0 / ((std_moat + 5.0) ** 2)
    )
    resp = s_inner * r_rim_sym

    sub = resp[14:67, 14:67]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    cx, cy = int(ix + 14), int(iy + 14)
    return cx, cy, float(resp[cy, cx]), float(std_icon[cy, cx])


def detect_puck_color(
    img: np.ndarray,
    center: Optional[tuple[int, int]] = None,
    split_thresh: Optional[float] = None,
) -> tuple[str, float]:
    """
    Illumination-invariant piece colour detection around the located puck center.

    Combines three concentric relative signals around ``(cx, cy)``:
    1. ``icon_asym``: asymmetry between bright icon pixels (p85) and dark icon
       pixels (p15) relative to the puck body ring (r = 10.5..14.5 px).
       White pieces have a bright icon on a dark body (icon_asym >> 0);
       black pieces have a dark icon on a light body (icon_asym << 0).
    2. ``rim_diff``: difference between outer rim (r = 15.5..18.0 px) and body ring.
    3. ``body_val`` vs ``split_thresh`` (fallback absolute reference).

    Because ``icon_asym`` and ``rim_diff`` are differences of concentric rings on
    the same puck, additive lighting changes cancel out completely and shadows
    preserve polarity.
    """
    g = _to_gray80(img)
    if center is None:
        cx, cy, _, _ = locate_puck_center(g)
    else:
        cx, cy = int(center[0]), int(center[1])

    d = np.sqrt((_XG80 - cx) ** 2 + (_YG80 - cy) ** 2)
    body_mask = (d >= 10.5) & (d <= 14.5)
    rim_mask  = (d >= 15.5) & (d <= 18.0)
    icon_mask = d <= 8.5

    body_val = float(g[body_mask].mean()) if body_mask.any() else 112.0
    rim_val  = float(g[rim_mask].mean()) if rim_mask.any() else body_val
    icon_px  = g[icon_mask]
    p85 = float(np.percentile(icon_px, 85)) if icon_px.size else body_val
    p15 = float(np.percentile(icon_px, 15)) if icon_px.size else body_val

    if split_thresh is None:
        split_thresh = float(getattr(cfg, "PIECE_PUCK_SPLIT_THRESHOLD", 112.0))

    icon_asym = (p85 - body_val) - (body_val - p15)
    rim_diff  = rim_val - body_val
    score = icon_asym + 1.5 * rim_diff + (split_thresh - body_val) * 0.5

    color = "white" if score >= 0.0 else "black"
    conf = min(1.0, 0.65 + abs(score) / 250.0)
    return color, float(conf)


def _prep_square(img: np.ndarray) -> np.ndarray:
    """Casa BGR/cinza -> cinza 80x80 com contraste equalizado (CLAHE)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (_SAMPLE_NORM, _SAMPLE_NORM), interpolation=cv2.INTER_AREA)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(img)


def _extract_icon(
    img: np.ndarray,
    color: Optional[str] = None,
    radius_mask: float = 11.5,
    center: Optional[tuple[int, int]] = None,
) -> tuple[np.ndarray, str]:
    """
    Normaliza a imagem da casa recortada (80x80) para isolar o icone gravado no disco:
    1. Localiza dinamicamente o centro real ``(cx, cy)`` do disco e translada para ``(40, 40)``.
    2. Mede o anel do corpo do disco (raio 10.5-14.5px ao redor do centro real).
    3. Determina a polaridade (branca tem disco escuro com icone claro; preta tem
       disco claro com icone escuro).
    4. Inverte e subtrai o corpo para que o icone fique sempre brilhante sobre fundo preto.
    5. Aplica mascara circular e refina o centro pelo centro de massa do icone.
    """
    gray = _to_gray80(img)
    if center is None:
        cx, cy, _, _ = locate_puck_center(gray)
    else:
        cx, cy = int(center[0]), int(center[1])

    # Translada a imagem para centralizar o disco em (40, 40)
    if cx != 40 or cy != 40:
        M = np.float32([[1.0, 0.0, float(40 - cx)], [0.0, 1.0, float(40 - cy)]])
        centered = cv2.warpAffine(
            gray, M, (_SAMPLE_NORM, _SAMPLE_NORM),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
    else:
        centered = gray

    dist = np.sqrt((_XG80 - 40) ** 2 + (_YG80 - 40) ** 2)
    body_val = float(centered[(dist >= 10.5) & (dist <= 14.5)].mean())

    if color in ("white", "black"):
        det_color = color
    else:
        det_color, _ = detect_puck_color(centered, center=(40, 40))

    if det_color == "black":
        norm = body_val - centered
    else:
        norm = centered - body_val
    norm = np.maximum(0.0, norm)

    mask = (dist <= radius_mask).astype(np.float32)
    norm = norm * mask

    # Refinamento sub-pixel pelo centro de massa do icone isolado
    tot = float(norm.sum())
    if tot > 100.0:
        icx = float((norm * _XG80).sum() / tot)
        icy = float((norm * _YG80).sum() / tot)
        if abs(icx - 40.0) <= 3.5 and abs(icy - 40.0) <= 3.5:
            M2 = np.float32([[1.0, 0.0, 40.0 - icx], [0.0, 1.0, 40.0 - icy]])
            norm = cv2.warpAffine(
                norm, M2, (_SAMPLE_NORM, _SAMPLE_NORM),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
            )

    max_val = float(norm.max())
    if max_val > 5.0:
        norm = (norm / max_val * 255.0).astype(np.uint8)
    else:
        norm = norm.astype(np.uint8)

    return norm, det_color


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
                # Molde central 22x22 isolando o simbolo ja centralizado em (40, 40)
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
        self,
        roi: np.ndarray,
        color: Optional[str],
        center: Optional[tuple[int, int]] = None,
    ) -> tuple[Optional[str], float]:
        icon, _ = _extract_icon(roi, color=color, center=center)
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
        self,
        roi: np.ndarray,
        color: Optional[str] = None,
        center: Optional[tuple[int, int]] = None,
    ) -> tuple[Optional[str], float]:
        """
        Returns ``(piece_type, confidence)``; ``(None, 0.0)`` when unsure.

        ``color`` restricts matching to that colour's references (recommended);
        if None, all references are searched.
        """
        if self.has_samples:
            return self._classify_samples(roi, color, center=center)
        if not self.is_ready:
            return None, 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        if center is None:
            cx, cy, _, _ = locate_puck_center(gray)
        else:
            cx, cy = int(center[0]), int(center[1])

        if gray.shape[0] == _SAMPLE_NORM and gray.shape[1] == _SAMPLE_NORM and (cx != 40 or cy != 40):
            M = np.float32([[1.0, 0.0, float(40 - cx)], [0.0, 1.0, float(40 - cy)]])
            gray = cv2.warpAffine(gray, M, (_SAMPLE_NORM, _SAMPLE_NORM), borderMode=cv2.BORDER_REPLICATE)

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
