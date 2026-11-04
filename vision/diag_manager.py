"""
vision/diag_manager.py
======================
Manages diagnostic image storage for the VoiceChess vision subsystem.

During development, images are saved with timestamped filenames to allow
offline review and debugging of the detection pipeline.

In production (SAVE_DIAGNOSTIC_IMAGES = False in vision_config.py),
all save calls become no-ops with zero overhead.

A rotating limit (MAX_DIAGNOSTIC_FILES) prevents the SD card from filling up.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import config as cfg

log = logging.getLogger(__name__)


class DiagManager:
    """
    Saves diagnostic images to *diag_dir*, keeping only:
      - ``last_capture.jpg``: the most recent photo captured/analyzed.
      - ``last_ok.jpg``     : the most recent photo where validation/position was OK.
    Any other ``.jpg`` files in *diag_dir* are automatically cleaned up.
    """

    LAST_CAPTURE_NAME = "last_capture.jpg"
    LAST_OK_NAME      = "last_ok.jpg"

    def __init__(
        self,
        diag_dir: str | Path = cfg.DIAGNOSTIC_DIR,
        enabled: bool = cfg.SAVE_DIAGNOSTIC_IMAGES,
        max_files: int = cfg.MAX_DIAGNOSTIC_FILES,
    ) -> None:
        self._dir     = Path(diag_dir)
        self._enabled = enabled
        self._max     = max_files

        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._cleanup_extra_files()
            log.info("[Diag] Diagnostic images → %s (keeping only last_capture.jpg & last_ok.jpg)", self._dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, image: np.ndarray, label: str = "capture", *, is_ok: bool = False) -> Path | None:
        """
        Save *image* as ``last_capture.jpg`` (and also as ``last_ok.jpg`` when
        *is_ok* is True), removing any old diagnostic ``.jpg`` files.
        """
        if not self._enabled:
            return None

        self._dir.mkdir(parents=True, exist_ok=True)
        capture_path = self._dir / self.LAST_CAPTURE_NAME
        ok = cv2.imwrite(str(capture_path), image)
        if not ok:
            log.warning("[Diag] Failed to save: %s", capture_path.name)
            return None

        if is_ok:
            ok_path = self._dir / self.LAST_OK_NAME
            cv2.imwrite(str(ok_path), image)
            log.debug("[Diag] Updated %s and %s (%s)", capture_path.name, ok_path.name, label)
        else:
            log.debug("[Diag] Updated %s (%s)", capture_path.name, label)

        self._cleanup_extra_files()
        return capture_path

    def save_ok(self, image: np.ndarray, label: str = "ok") -> Path | None:
        """Save *image* as both ``last_capture.jpg`` and ``last_ok.jpg``."""
        return self.save(image, label=label, is_ok=True)

    def save_squares(
        self,
        squares: dict[str, np.ndarray],
        label: str = "squares",
    ) -> Path | None:
        """
        Create a contact sheet of all 64 square ROIs and save it.
        """
        if not self._enabled:
            return None

        col_labels = "ABCDEFGH"
        row_labels = "87654321"

        rows = []
        for row_label in row_labels:
            row_imgs = []
            for col_label in col_labels:
                sq = f"{col_label}{row_label}"
                roi = squares.get(sq)
                if roi is None:
                    roi = np.zeros((80, 80, 3), dtype=np.uint8)
                annotated = roi.copy()
                cv2.putText(annotated, sq, (2, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                row_imgs.append(annotated)
            rows.append(np.hstack(row_imgs))

        sheet = np.vstack(rows)
        return self.save(sheet, label)

    def list_files(self) -> list[Path]:
        """Return all saved diagnostic files sorted by modification time."""
        if not self._dir.exists():
            return []
        return sorted(self._dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cleanup_extra_files(self) -> None:
        """Remove any .jpg file in diag_dir except last_capture.jpg and last_ok.jpg."""
        if not self._dir.exists():
            return
        keep = {self.LAST_CAPTURE_NAME, self.LAST_OK_NAME}
        for p in self._dir.glob("*.jpg"):
            if p.name not in keep:
                try:
                    p.unlink()
                    log.debug("[Diag] Removed old file: %s", p.name)
                except OSError as exc:
                    log.warning("[Diag] Could not delete %s: %s", p.name, exc)
