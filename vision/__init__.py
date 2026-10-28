"""
vision/__init__.py
==================
Public façade for the VoiceChess vision subsystem.

Usage
-----
    from vision import CameraSystem

    cam = CameraSystem()
    cam.initialize()                    # warmup + load calibration

    state = cam.capture_board_state()   # -> BoardState | None
    result = cam.validate_move(         # -> CompareResult
        {"from": "E2", "to": "E4"}
    )

    cam.shutdown()
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Optional

from . import config as cfg
from .camera         import CameraController
from .calibration    import PerspectiveCalibrator, CalibrationData
from .segmentation   import BoardSegmenter
from .piece_detector import PieceDetector
from .board_state    import BoardState
from .state_comparator import StateComparator, CompareResult
from .move_validator import MoveValidator, CameraState
from .diag_manager   import DiagManager

log = logging.getLogger(__name__)

__all__ = ["CameraSystem"]


class CameraSystem:
    """
    High-level façade that wires all vision components together.

    Parameters
    ----------
    calibration_path : str | None
        Path to camera_calibration.json.  If None or the file doesn't
        exist, the system starts in uncalibrated mode (passthrough resize).
    device_index : int
        OpenCV camera device index.
    """

    def __init__(
        self,
        calibration_path: str | None = None,
        device_index: int = cfg.CAMERA_DEVICE_INDEX,
    ) -> None:
        cal_path = calibration_path or cfg.CALIBRATION_FILE

        # ── Components ─────────────────────────────────────────────────
        self._camera     = CameraController(device_index=device_index)
        self._diag       = DiagManager()
        self._segmenter  = BoardSegmenter()
        self._comparator = StateComparator(min_confidence=cfg.MIN_CONFIDENCE)

        # Load calibration if available
        if Path(cal_path).exists():
            try:
                self._calibrator = PerspectiveCalibrator.load(cal_path)
                log.info("[CameraSystem] Calibration loaded from %s", cal_path)
            except Exception as exc:
                log.warning(
                    "[CameraSystem] Could not load calibration (%s) — uncalibrated mode.",
                    exc,
                )
                self._calibrator = PerspectiveCalibrator.default_uncalibrated()
        else:
            log.warning(
                "[CameraSystem] Calibration file not found: %s — uncalibrated mode.",
                cal_path,
            )
            self._calibrator = PerspectiveCalibrator.default_uncalibrated()

        # Detector starts without a reference image (set after initialize())
        self._detector = PieceDetector()

        # Validator wires everything
        self._validator = MoveValidator(
            camera     = self._camera,
            calibrator = self._calibrator,
            segmenter  = self._segmenter,
            detector   = self._detector,
            comparator = self._comparator,
            diag       = self._diag,
        )

        self._reference_state: Optional[BoardState] = None
        self._initialised = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Open the camera, perform warmup, and capture the reference state.

        Returns
        -------
        bool
            True if the camera opened and a reference state was captured.
        """
        try:
            self._camera.open()
        except RuntimeError as exc:
            log.error("[CameraSystem] Initialization failed: %s", exc)
            return False

        # Capture reference (empty board or current state)
        log.info("[CameraSystem] Capturing reference state...")
        ref_state = self._validator.capture_state("reference")
        if ref_state is not None:
            self._reference_state = ref_state
            # Provide the reference image to the detector for MAD occupancy
            # (We re-capture one raw frame for the reference image)
            raw = self._camera.capture_with_retry(max_attempts=2)
            if raw is not None:
                corrected_ref = self._calibrator.correct_perspective(raw)
                self._detector.set_reference(corrected_ref)
                self._diag.save(corrected_ref, "reference_corrected")
                log.info("[CameraSystem] Reference image set for piece detector.")
        else:
            log.warning(
                "[CameraSystem] Could not capture reference state — "
                "occupancy detection will use variance-only fallback."
            )

        self._initialised = True
        log.info("[CameraSystem] Initialised. Calibrated=%s", self._calibrator.is_calibrated)
        return True

    def shutdown(self) -> None:
        """Release the camera and clean up."""
        self._camera.close()
        self._initialised = False
        log.info("[CameraSystem] Shut down.")

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------

    def capture_board_state(self) -> Optional[BoardState]:
        """
        Capture and return the current board state.

        Returns None if the capture fails or confidence is too low.
        """
        if not self._initialised:
            log.warning("[CameraSystem] Not initialised. Call initialize() first.")
            return None
        return self._validator.capture_state("board_state")

    def validate_move(
        self,
        move: dict,
        gantry_done_event: Optional[threading.Event] = None,
        before_state: Optional[BoardState] = None,
    ) -> CompareResult:
        """
        Validate that a move was physically executed.

        Parameters
        ----------
        move : dict
            {"from": "E2", "to": "E4"}
        gantry_done_event : threading.Event | None
            Gantry completion signal.  None → wait POST_MOVE_SETTLE_SECONDS.
        before_state : BoardState | None
            Pre-captured state; if None, captures automatically.

        Returns
        -------
        CompareResult
        """
        if not self._initialised:
            log.warning("[CameraSystem] Not initialised.")
            return CompareResult(
                success=False,
                error="NOT_INITIALISED",
                message="CameraSystem não foi inicializado.",
            )
        return self._validator.validate_move(
            move=move,
            gantry_done_event=gantry_done_event,
            before_state=before_state or self._reference_state,
        )

    def calibrate_interactive(self) -> bool:
        """
        Run the interactive calibration wizard.

        Returns True if calibration was completed and saved.
        """
        from .calibration import InteractiveCalibrator

        if not self._camera.is_ready:
            try:
                self._camera.open()
            except RuntimeError as exc:
                log.error("[CameraSystem] Cannot open camera for calibration: %s", exc)
                return False

        raw = self._camera.capture_with_retry()
        if raw is None:
            log.error("[CameraSystem] Failed to capture image for calibration.")
            return False

        ic  = InteractiveCalibrator()
        cal = ic.run(raw, device_index=cfg.CAMERA_DEVICE_INDEX,
                     resolution=cfg.CAMERA_RESOLUTION)
        cal.save(cfg.CALIBRATION_FILE)
        self._calibrator.update(cal)
        log.info("[CameraSystem] Calibration saved and applied.")
        return True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self._calibrator.is_calibrated

    @property
    def is_ready(self) -> bool:
        return self._initialised and self._camera.is_ready

    @property
    def camera_state(self) -> CameraState:
        return self._validator.state

    @property
    def reference_state(self) -> Optional[BoardState]:
        return self._reference_state
