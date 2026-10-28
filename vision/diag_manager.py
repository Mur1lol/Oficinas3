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
    Saves annotated images to *diag_dir* for offline inspection.

    Parameters
    ----------
    diag_dir : str | Path
        Directory where diagnostic images are written.
    enabled : bool
        If False, all save calls are no-ops.
    max_files : int
        Maximum number of .jpg files to keep; oldest are deleted when exceeded.
    """

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
            log.info("[Diag] Diagnostic images → %s", self._dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, image: np.ndarray, label: str) -> Path | None:
        """
        Save *image* as a JPEG with a timestamped filename.

        Parameters
        ----------
        image : np.ndarray
            BGR image to save.
        label : str
            Short tag appended to the filename, e.g. "raw", "corrected",
            "after_move".

        Returns
        -------
        Path | None
            The saved file path, or None if saving is disabled.
        """
        if not self._enabled:
            return None

        ts   = time.strftime("%Y%m%d_%H%M%S")
        name = f"{ts}_{label}.jpg"
        path = self._dir / name

        ok = cv2.imwrite(str(path), image)
        if ok:
            log.debug("[Diag] Saved: %s", path.name)
            self._rotate()
        else:
            log.warning("[Diag] Failed to save: %s", path.name)
            return None

        return path

    def save_squares(
        self,
        squares: dict[str, np.ndarray],
        label: str = "squares",
    ) -> Path | None:
        """
        Create a contact sheet of all 64 square ROIs and save it.

        Parameters
        ----------
        squares : dict[str, np.ndarray]
            From BoardSegmenter.segment_board().
        label : str
            Filename tag.
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
                # Add label overlay
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

    def _rotate(self) -> None:
        """Delete oldest files if the file count exceeds the limit."""
        files = self.list_files()
        while len(files) > self._max:
            oldest = files.pop(0)
            try:
                oldest.unlink()
                log.debug("[Diag] Rotated out: %s", oldest.name)
            except OSError as exc:
                log.warning("[Diag] Could not delete %s: %s", oldest.name, exc)
