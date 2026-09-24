"""
vision/camera.py
================
USB camera control for the VoiceChess vision subsystem.

Wraps OpenCV's VideoCapture into a context manager with warmup,
retry logic, and quality validation.

Usage
-----
    from vision.camera import CameraController

    with CameraController() as cam:
        frame = cam.capture()           # -> np.ndarray (BGR)
        frame = cam.capture_with_retry()  # -> np.ndarray | None
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import config as cfg

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CameraState
# ---------------------------------------------------------------------------

class CameraState:
    IDLE       = "CAMERA_IDLE"
    CAPTURING  = "CAMERA_CAPTURING"
    FAILED     = "CAPTURE_FAILED"


# ---------------------------------------------------------------------------
# CameraController
# ---------------------------------------------------------------------------

class CameraController:
    """
    Thread-safe (single-consumer) USB camera wrapper.

    Parameters
    ----------
    device_index : int
        OpenCV camera device index. 0 = first webcam.
    resolution : tuple[int, int]
        Capture resolution (width, height).
    warmup_seconds : float
        Seconds to discard frames after open() to allow auto-exposure.
    """

    def __init__(
        self,
        device_index: int = cfg.CAMERA_DEVICE_INDEX,
        resolution: tuple[int, int] = cfg.CAMERA_RESOLUTION,
        warmup_seconds: float = cfg.CAMERA_WARMUP_SECONDS,
    ) -> None:
        self._device_index  = device_index
        self._resolution    = resolution
        self._warmup_seconds = warmup_seconds
        self._cap: cv2.VideoCapture | None = None
        self._state = CameraState.IDLE

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraController":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the camera and perform auto-exposure warmup."""
        if self._cap is not None and self._cap.isOpened():
            return

        log.info(
            "[Camera] Opening device %d at %dx%d ...",
            self._device_index, *self._resolution,
        )
        self._cap = cv2.VideoCapture(self._device_index, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            # Fallback: try without backend hint (Linux / Pi)
            self._cap = cv2.VideoCapture(self._device_index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"[Camera] Failed to open device {self._device_index}. "
                "Check that the USB camera is connected."
            )

        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("[Camera] Actual resolution: %dx%d", actual_w, actual_h)

        # Warmup: discard frames until auto-exposure stabilises
        self._warmup()
        log.info("[Camera] Ready.")

    def close(self) -> None:
        """Release the camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._state = CameraState.IDLE
            log.info("[Camera] Released.")

    @property
    def is_ready(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(self) -> np.ndarray:
        """
        Capture a single frame.

        Returns
        -------
        np.ndarray
            BGR image array.

        Raises
        ------
        RuntimeError
            If the camera is not open or the frame cannot be read.
        """
        if not self.is_ready:
            raise RuntimeError("[Camera] Camera is not open. Call open() first.")

        self._state = CameraState.CAPTURING
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._state = CameraState.FAILED
            raise RuntimeError("[Camera] Failed to read frame from camera.")

        self._state = CameraState.IDLE
        log.debug("[Camera] Frame captured: %s", frame.shape)
        return frame

    def capture_with_retry(
        self,
        max_attempts: int = cfg.MAX_CAPTURE_ATTEMPTS,
        delay: float = cfg.RETRY_DELAY_SECONDS,
    ) -> np.ndarray | None:
        """
        Attempt to capture a valid frame up to *max_attempts* times.

        A frame is considered invalid if it is completely black (camera not
        ready) or has extremely low variance (lens cap, obstruction).

        Returns None if all attempts fail.
        """
        for attempt in range(1, max_attempts + 1):
            try:
                frame = self.capture()
            except RuntimeError as exc:
                log.warning("[Camera] Attempt %d/%d failed: %s", attempt, max_attempts, exc)
                time.sleep(delay)
                continue

            if self._is_valid_frame(frame):
                return frame

            log.warning(
                "[Camera] Attempt %d/%d: frame quality too low (dark or obstructed).",
                attempt, max_attempts,
            )
            time.sleep(delay)

        log.error("[Camera] All %d capture attempts failed.", max_attempts)
        self._state = CameraState.FAILED
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _warmup(self) -> None:
        """Discard frames during the warmup period."""
        log.info("[Camera] Warming up for %.1f s ...", self._warmup_seconds)
        deadline = time.monotonic() + self._warmup_seconds
        while time.monotonic() < deadline:
            self._cap.read()   # discard
        log.debug("[Camera] Warmup complete.")

    @staticmethod
    def _is_valid_frame(frame: np.ndarray) -> bool:
        """Return True if the frame has enough brightness and variance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        variance = float(np.var(gray))

        if mean_brightness < 10.0:
            log.debug("[Camera] Frame rejected: too dark (mean=%.1f)", mean_brightness)
            return False
        if variance < 50.0:
            log.debug("[Camera] Frame rejected: too uniform (var=%.1f)", variance)
            return False
        return True

    @property
    def state(self) -> str:
        return self._state
