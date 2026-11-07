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

import base64
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2

from . import config as cfg
from .camera         import CameraController
from .calibration    import PerspectiveCalibrator, CalibrationData
from .segmentation   import BoardSegmenter
from .piece_detector import PieceDetector
from .board_state    import BoardState
from .state_comparator import StateComparator, CompareResult
from .move_validator import MoveValidator, CameraState
from .diag_manager   import DiagManager
from .board_verifier import (
    GameSettings,
    ManualValidationResult,
    PositionVerificationResult,
    annotate_validation_image,
    detect_and_validate_manual_move,
    record_validation_event,
    verify_board_position,
)

log = logging.getLogger(__name__)

__all__ = [
    "CameraSystem",
    "SimulatedCamera",
    "GameSettings",
    "ManualValidationResult",
    "PositionVerificationResult",
]


class SimulatedCamera:
    """
    Simula o sistema de visão ignorando as capturas e imagens reais da câmera.
    Usado para desenvolvimento, quando a iluminação está desfavorável ou com a flag --sim-cam.
    """

    def __init__(self) -> None:
        self._initialised = False

    @property
    def is_ready(self) -> bool:
        return self._initialised

    def initialize(self) -> bool:
        self._initialised = True
        log.info("[CameraSystem] (simulado) Câmera simulada inicializada — imagens ignoradas.")
        return True

    def shutdown(self) -> None:
        self._initialised = False
        log.info("[CameraSystem] (simulado) Câmera simulada encerrada.")

    def verify_gantry_position(
        self,
        board,
        last_move=None,
        settle_seconds: float = 0.0,
    ) -> PositionVerificationResult:
        log.info("[CameraSystem] (simulado) verify_gantry_position: validação de imagem ignorada (OK).")
        res = PositionVerificationResult(
            success=True,
            move_uci=last_move.uci().upper() if last_move else None,
            origin_ok=True,
            dest_ok=True,
            color_ok=True,
            mismatches=[],
            matching_squares=64,
            confidence=1.0,
            message="Câmera simulada: validação de imagem ignorada com sucesso.",
            speech="Position verified.",
            lcd_lines=("POSITION VERIFIED", "CAMERA SIMULATED", "", ""),
        )
        try:
            record_validation_event(res.to_dict())
        except Exception:
            pass
        return res

    def validate_manual_turn(
        self,
        board,
        is_phys_valid_fn: Optional[Callable] = None,
    ) -> ManualValidationResult:
        res = ManualValidationResult(
            valid=False,
            error_code="SIM_CAM",
            message="Câmera simulada ativa (imagens ignoradas). Fale o lance por voz (ex: Magnus Move E2 E4).",
            speech="Camera is simulated. Please announce your move by voice.",
            lcd_lines=("CAMERA SIMULATED", "USE VOICE MOVE", "SAY: MAGNUS MOVE", "EX: E2 TO E4"),
        )
        try:
            record_validation_event(res.to_dict())
        except Exception:
            pass
        return res

    def capture_board_state(self) -> Optional[BoardState]:
        return None

    def validate_move(
        self,
        move: dict,
        gantry_done_event: Optional[threading.Event] = None,
        before_state: Optional[BoardState] = None,
    ) -> CompareResult:
        return CompareResult(
            success=True,
            message="Câmera simulada: movimento validado.",
        )


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
    simulate : bool
        If True, returns a SimulatedCamera instance that ignores real camera images.
    """

    def __new__(cls, *args, simulate: bool = False, **kwargs):
        if simulate:
            return SimulatedCamera()
        return super().__new__(cls)

    def __init__(
        self,
        calibration_path: str | None = None,
        device_index: int = cfg.CAMERA_DEVICE_INDEX,
    ) -> None:
        cal_path = calibration_path or cfg.CALIBRATION_FILE

        # ── Components ─────────────────────────────────────────────────
        self._camera     = CameraController(device_index=device_index)
        self._diag       = DiagManager()
        self._comparator = StateComparator(min_confidence=cfg.MIN_CONFIDENCE)

        # Load calibration if available
        orientation = "normal"
        if Path(cal_path).exists():
            try:
                self._calibrator = PerspectiveCalibrator.load(cal_path)
                orientation = self._calibrator.calibration.board_orientation
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

        self._segmenter = BoardSegmenter(board_orientation=orientation)

        # Load saved empty-board reference image if available on disk
        ref_img = None
        ref_path = Path(getattr(cfg, "REFERENCE_EMPTY_FILE", cfg.VISION_DIR / "reference_empty.png"))
        if ref_path.is_file():
            ref_img = cv2.imread(str(ref_path))
            if ref_img is not None:
                log.info("[CameraSystem] Loaded empty-board reference from %s", ref_path)

        settings = GameSettings.load()
        self._detector = PieceDetector(
            reference_image=ref_img,
            occupied_diff_threshold=settings.diff_threshold,
            board_orientation=orientation,
        )

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
        self._last_warped: Optional[ object ] = None
        self._initialised = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Open the camera, perform warmup, and capture the initial board state.

        Returns
        -------
        bool
            True if the camera opened and an initial state was captured.
        """
        try:
            self._camera.open()
        except RuntimeError as exc:
            log.error("[CameraSystem] Initialization failed: %s", exc)
            return False

        # Capture current board state
        log.info("[CameraSystem] Capturing initial board state...")
        ref_state = self._validator.capture_state("reference")
        if ref_state is not None:
            self._reference_state = ref_state
            # Only overwrite detector reference image if no saved empty reference exists
            if self._detector._ref is None:
                raw = self._camera.capture_with_retry(max_attempts=2)
                if raw is not None:
                    corrected_ref = self._calibrator.correct_perspective(raw)
                    self._detector.set_reference(corrected_ref)
                    self._diag.save(corrected_ref, "reference_corrected")
                    log.info("[CameraSystem] Live reference image set for piece detector.")
        else:
            log.warning(
                "[CameraSystem] Could not capture initial state — "
                "occupancy detection will use structural fallback if no reference exists."
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

    def _reload_runtime_calibration_and_settings(self) -> None:
        """Reload calibration & threshold if updated via the Web UI."""
        settings = GameSettings.load()
        self._detector._diff_thresh = settings.diff_threshold
        cal_path = Path(cfg.CALIBRATION_FILE)
        if cal_path.is_file():
            try:
                cal = CalibrationData.load(cal_path)
                if cal.is_valid():
                    self._calibrator.update(cal)
                    self._segmenter = BoardSegmenter(board_orientation=cal.board_orientation)
                    self._detector._orientation = cal.board_orientation
            except Exception:
                pass
        ref_path = Path(getattr(cfg, "REFERENCE_EMPTY_FILE", cfg.VISION_DIR / "reference_empty.png"))
        if ref_path.is_file() and self._detector._ref is None:
            ref_img = cv2.imread(str(ref_path))
            if ref_img is not None:
                self._detector.set_reference(ref_img, orientation=self._detector._orientation)

    def capture_board_state(self) -> Optional[BoardState]:
        """
        Capture and return the current board state.

        Returns None if the capture fails or confidence is too low.
        """
        if not self._initialised:
            log.warning("[CameraSystem] Not initialised. Call initialize() first.")
            return None
        self._reload_runtime_calibration_and_settings()
        return self._validator.capture_state("board_state")

    def _capture_warped_and_state(
        self,
        label: str = "capture",
        prior_board: Optional[object] = None,
        prior_mode: str = "auto",
        unbiased_squares: Optional[set[str] | list[str] | tuple[str, ...]] = None,
    ) -> tuple[Optional[object], Optional[BoardState]]:
        """Capture a single corrected frame and its processed BoardState."""
        self._reload_runtime_calibration_and_settings()
        frame = self._camera.capture_with_retry(max_attempts=cfg.MAX_CAPTURE_ATTEMPTS)
        if frame is None:
            return None, None
        warped = self._calibrator.correct_perspective(frame)
        self._diag.save(warped, f"{label}_corrected")
        state = self._validator._process_board(
            warped,
            prior_board=prior_board,
            prior_mode=prior_mode,
            unbiased_squares=unbiased_squares,
        )
        self._reference_state = state
        return warped, state

    def verify_gantry_position(
        self,
        board,
        last_move=None,
        settle_seconds: float = cfg.POST_MOVE_SETTLE_SECONDS,
    ) -> PositionVerificationResult:
        """
        Capture a photo after the Gantry finishes a move and verify that all
        pieces are in their expected positions according to ``board``.
        """
        if not self._initialised:
            return PositionVerificationResult(
                success=False,
                message="CameraSystem não inicializado.",
                speech="",
                lcd_lines=("CAMERA OFFLINE", "SKIP VERIFICATION", "", ""),
            )
        if settle_seconds > 0:
            time.sleep(settle_seconds)

        unbiased = None
        if last_move is not None:
            import chess
            unbiased = {
                chess.square_name(last_move.from_square).upper(),
                chess.square_name(last_move.to_square).upper(),
            }
        warped, state = self._capture_warped_and_state(
            "post_gantry",
            prior_board=board,
            prior_mode="post_move",
            unbiased_squares=unbiased,
        )
        if state is None or warped is None:
            res = PositionVerificationResult(
                success=False,
                move_uci=last_move.uci().upper() if last_move else None,
                message="Falha ao capturar imagem da câmera após movimento do Gantry.",
                speech="Camera capture failed.",
                lcd_lines=("CAMERA ERROR", "CAPTURE FAILED", "", ""),
            )
            record_validation_event(res.to_dict())
            return res

        res = verify_board_position(board, state, last_move=last_move)
        orientation = self._calibrator.calibration.board_orientation
        annotated = annotate_validation_image(warped, state, orientation=orientation, verify_result=res)
        self._diag.save(
            annotated,
            "post_gantry_verified",
            is_ok=bool(res.success and len(res.mismatches) == 0),
        )

        thumb_b64 = None
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            thumb_b64 = base64.b64encode(buf).decode("utf-8")

        event = res.to_dict()
        event["annotated_image"] = thumb_b64
        record_validation_event(event)
        return res

    def validate_manual_turn(
        self,
        board,
        is_phys_valid_fn: Optional[Callable] = None,
    ) -> ManualValidationResult:
        """
        Triggered when a manual player says 'MAGNUS VALIDATE'.
        Captures a photo, infers the move played on the physical board,
        checks chess legality against ``board``, and returns the validation result.
        """
        if not self._initialised:
            return ManualValidationResult(
                valid=False,
                error_code="CAMERA_NOT_INITIALISED",
                message="Câmera não inicializada.",
                speech="Camera is not available to validate the move.",
                lcd_lines=("CAMERA OFFLINE", "CANNOT VALIDATE", "", ""),
            )

        warped, state = self._capture_warped_and_state(
            "manual_validate",
            prior_board=board,
            prior_mode="pre_move",
        )
        if state is None or warped is None:
            res = ManualValidationResult(
                valid=False,
                error_code="CAPTURE_FAILED",
                message="Falha ao capturar imagem da câmera.",
                speech="Could not capture camera image. Please try again.",
                lcd_lines=("CAMERA ERROR", "CAPTURE FAILED", "SAY: MAGNUS", "VALIDATE"),
            )
            record_validation_event(res.to_dict())
            return res

        res = detect_and_validate_manual_move(board, state, is_phys_valid_fn=is_phys_valid_fn)
        orientation = self._calibrator.calibration.board_orientation
        annotated = annotate_validation_image(warped, state, orientation=orientation, manual_result=res)
        self._diag.save(annotated, "manual_validate_annotated", is_ok=bool(res.valid))

        thumb_b64 = None
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            thumb_b64 = base64.b64encode(buf).decode("utf-8")

        event = res.to_dict()
        event["annotated_image"] = thumb_b64
        record_validation_event(event)
        return res

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
