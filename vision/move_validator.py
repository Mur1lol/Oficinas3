"""
vision/move_validator.py
========================
Orchestrates the full visual validation of a chess move.

State machine
-------------
    IDLE
      │  validate_move() called
      ▼
    CAMERA_CAPTURING  (capture before-state)
      │
      ▼
    IMAGE_PROCESSING  (process before-state)
      │  gantry moves (or immediate in test mode)
      ▼
    CAMERA_CAPTURING  (capture after-state)
      │
      ▼
    IMAGE_PROCESSING  (process after-state)
      │
      ▼
    BOARD_VALIDATING  (compare states)
      │
      ├── success ──► CAPTURE_CONFIRMED
      └── fail    ──► retry? ──► CAPTURE_FAILED (if max retries reached)

Gantry integration
------------------
In the current test phase, the validator does NOT wait for a gantry signal.
The ``gantry_done_event`` parameter is optional; if None, the validator
proceeds immediately after POST_MOVE_SETTLE_SECONDS.

When the real gantry is integrated, pass a threading.Event that is set()
when the gantry signals move completion.
"""

from __future__ import annotations

import logging
import sys
import time
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np

from . import config as cfg
from .board_state import BoardState, SquareState
from .camera import CameraController
from .calibration import PerspectiveCalibrator
from .segmentation import BoardSegmenter
from .piece_detector import PieceDetector
from .state_comparator import StateComparator, CompareResult
from .diag_manager import DiagManager

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera state enum
# ---------------------------------------------------------------------------

class CameraState(Enum):
    IDLE              = auto()
    CAMERA_CAPTURING  = auto()
    IMAGE_PROCESSING  = auto()
    BOARD_VALIDATING  = auto()
    CAPTURE_CONFIRMED = auto()
    CAPTURE_FAILED    = auto()


# ---------------------------------------------------------------------------
# MoveValidator
# ---------------------------------------------------------------------------

class MoveValidator:
    """
    High-level move validation controller.

    Parameters
    ----------
    camera : CameraController
    calibrator : PerspectiveCalibrator
    segmenter : BoardSegmenter
    detector : PieceDetector
    comparator : StateComparator
    diag : DiagManager
    max_attempts : int
        Maximum capture+process attempts before giving up.
    settle_seconds : float
        Seconds to wait after gantry signals done (vibration settling).
    min_confidence : float
        Minimum confidence for a valid board state.
    """

    def __init__(
        self,
        camera:      CameraController,
        calibrator:  PerspectiveCalibrator,
        segmenter:   BoardSegmenter,
        detector:    PieceDetector,
        comparator:  StateComparator,
        diag:        DiagManager,
        max_attempts:    int   = cfg.MAX_CAPTURE_ATTEMPTS,
        settle_seconds:  float = cfg.POST_MOVE_SETTLE_SECONDS,
        min_confidence:  float = cfg.MIN_CONFIDENCE,
    ) -> None:
        self._camera     = camera
        self._calibrator = calibrator
        self._segmenter  = segmenter
        self._detector   = detector
        self._comparator = comparator
        self._diag       = diag
        self._max        = max_attempts
        self._settle     = settle_seconds
        self._min_conf   = min_confidence
        self._state      = CameraState.IDLE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture_state(self, label: str = "state") -> BoardState | None:
        """
        Capture and process the current board state.

        Returns None if all attempts fail or confidence is too low.
        """
        self._state = CameraState.CAMERA_CAPTURING

        for attempt in range(1, self._max + 1):
            frame = self._camera.capture_with_retry(max_attempts=1)
            if frame is None:
                log.warning("[Validator] Capture attempt %d/%d failed.", attempt, self._max)
                time.sleep(cfg.RETRY_DELAY_SECONDS)
                continue

            self._diag.save(frame, f"{label}_raw_attempt{attempt}")

            self._state = CameraState.IMAGE_PROCESSING
            corrected   = self._calibrator.correct_perspective(frame)
            self._diag.save(corrected, f"{label}_corrected_attempt{attempt}")

            board_state = self._process_board(corrected)

            if board_state.is_confident(self._min_conf):
                self._state = CameraState.IDLE
                return board_state

            log.warning(
                "[Validator] Attempt %d/%d: confidence %.2f < %.2f — retrying.",
                attempt, self._max,
                board_state.overall_confidence, self._min_conf,
            )
            time.sleep(cfg.RETRY_DELAY_SECONDS)

        self._state = CameraState.CAPTURE_FAILED
        log.error("[Validator] All %d capture attempts failed.", self._max)
        return None

    def validate_move(
        self,
        move: dict,
        gantry_done_event=None,
        before_state: BoardState | None = None,
    ) -> CompareResult:
        """
        Full move validation sequence.

        Parameters
        ----------
        move : dict
            {"from": "E2", "to": "E4"} — must contain uppercase square names.
        gantry_done_event : threading.Event | None
            Set by the gantry when movement is complete.
            If None, waits POST_MOVE_SETTLE_SECONDS unconditionally.
        before_state : BoardState | None
            If provided, skips the pre-move capture.

        Returns
        -------
        CompareResult
        """
        # ── 1. Pre-move state ─────────────────────────────────────────
        if before_state is None:
            log.info("[Validator] Capturing pre-move state...")
            before_state = self.capture_state("before")
            if before_state is None:
                return CompareResult.uncertain(attempts=self._max)

        # ── 2. Wait for gantry ────────────────────────────────────────
        if gantry_done_event is not None:
            log.info("[Validator] Waiting for gantry signal...")
            gantry_done_event.wait(timeout=60.0)
        else:
            log.info("[Validator] Settling for %.1f s...", self._settle)
            time.sleep(self._settle)

        # ── 3. Post-move state ────────────────────────────────────────
        log.info("[Validator] Capturing post-move state...")
        for attempt in range(1, self._max + 1):
            after_state = self.capture_state(f"after_attempt{attempt}")
            if after_state is None:
                continue

            # ── 4. Validate ───────────────────────────────────────────
            self._state = CameraState.BOARD_VALIDATING
            result = self._comparator.compare(
                before_state, after_state, move, attempts=attempt
            )

            if result.success:
                self._state = CameraState.CAPTURE_CONFIRMED
                log.info("[Validator] Move confirmed on attempt %d.", attempt)
                return result

            log.warning(
                "[Validator] Validation failed (attempt %d/%d): %s",
                attempt, self._max, result.message,
            )
            if attempt < self._max:
                time.sleep(cfg.RETRY_DELAY_SECONDS)

        self._state = CameraState.CAPTURE_FAILED
        return CompareResult.uncertain(attempts=self._max)

    @property
    def state(self) -> CameraState:
        return self._state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_board(self, corrected: np.ndarray) -> BoardState:
        """
        Segment and analyse a corrected image → BoardState.
        """
        squares    = self._segmenter.segment_board(corrected)
        cem_slots  = self._segmenter.segment_cemeteries(corrected)

        analyses   = self._detector.analyze_all(squares)
        confidences = [a.confidence for a in analyses.values()]

        board = {
            sq: SquareState(
                occupied   = a.occupied,
                color      = a.color,
                piece      = a.piece,
                confidence = a.confidence,
            )
            for sq, a in analyses.items()
        }

        white_cem = [
            SquareState(
                occupied   = a.occupied,
                color      = a.color,
                piece      = a.piece,
                confidence = a.confidence,
            )
            for a in [
                self._detector.analyze_cemetery_slot(i, roi, "white")
                for i, roi in enumerate(cem_slots.get("white", []))
            ]
        ]
        black_cem = [
            SquareState(
                occupied   = a.occupied,
                color      = a.color,
                piece      = a.piece,
                confidence = a.confidence,
            )
            for a in [
                self._detector.analyze_cemetery_slot(i, roi, "black")
                for i, roi in enumerate(cem_slots.get("black", []))
            ]
        ]

        overall = float(sum(confidences) / len(confidences)) if confidences else 0.0

        state = BoardState(
            board      = board,
            cemeteries = {"white": white_cem, "black": black_cem},
            overall_confidence = overall,
        )
        log.debug("[Validator] Board state: %s", state)
        return state
