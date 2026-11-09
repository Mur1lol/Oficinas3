"""
vision/camera.py
================
USB camera control for the VoiceChess vision subsystem.

Wraps OpenCV's VideoCapture into a context manager with warmup,
retry logic, and quality validation. Also supports:
- Auto-discovery of USB cameras on Linux/Raspberry Pi via ``/dev/v4l/by-id/``
  and ``/sys/class/video4linux/``
- Clean diagnostics when the USB camera is unplugged or locked by another PID
- Transparent frame sharing when ``web_calibrator.py`` is actively streaming
  from the real camera on the same machine.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

# Suppress OpenCV C++ V4L2/obsensor stderr spam before importing cv2
os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")

import cv2
import numpy as np

from . import config as cfg

log = logging.getLogger(__name__)


@contextmanager
def _quiet_opencv():
    """Temporarily silence OpenCV C++ warnings/errors during device probing."""
    prev = None
    try:
        if hasattr(cv2, "setLogLevel"):
            prev = cv2.getLogLevel()
            cv2.setLogLevel(0)
        elif hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
            prev = cv2.utils.logging.getLogLevel()
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        prev = None
    try:
        yield
    finally:
        if prev is not None:
            try:
                if hasattr(cv2, "setLogLevel"):
                    cv2.setLogLevel(prev)
                elif hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
                    cv2.utils.logging.setLogLevel(prev)
            except Exception:
                pass


def diagnose_linux_cameras() -> tuple[list[int], str]:
    """
    Inspect ``/dev/v4l/by-id`` and ``/sys/class/video4linux`` on Linux/Pi.

    Returns
    -------
    tuple[list[int], str]
        (ordered list of candidate video device indices, human-readable diagnostic summary)
    """
    if sys.platform == "win32":
        return [0, 1, 2], "Windows DirectShow"

    usb_indices: list[int] = []
    usb_names: list[str] = []

    by_id = Path("/dev/v4l/by-id")
    if by_id.is_dir():
        for link in sorted(by_id.iterdir()):
            if "index0" in link.name or "video" in link.name:
                try:
                    target = link.resolve().name  # e.g. "video0"
                    if target.startswith("video") and target[5:].isdigit():
                        idx = int(target[5:])
                        if idx not in usb_indices:
                            usb_indices.append(idx)
                            usb_names.append(f"/dev/{target} ({link.name})")
                except Exception:
                    pass

    v4l_sys = Path("/sys/class/video4linux")
    all_nodes: list[str] = []
    if v4l_sys.is_dir():
        for node in sorted(v4l_sys.iterdir()):
            if not (node.name.startswith("video") and node.name[5:].isdigit()):
                continue
            idx = int(node.name[5:])
            try:
                name = (node / "name").read_text(encoding="utf-8").strip()
            except Exception:
                name = "unknown"
            all_nodes.append(f"/dev/{node.name}={name}")
            # Skip Raspberry Pi internal SoC codec/ISP nodes (video10..video31)
            if any(k in name.lower() for k in ("bcm2835", "rpi-", "pisp", "hevc", "isp")):
                continue
            if idx not in usb_indices:
                usb_indices.append(idx)
                usb_names.append(f"/dev/{node.name} ({name})")

    # Check if any process is currently holding /dev/video* open
    holders: list[str] = []
    proc_root = Path("/proc")
    if proc_root.is_dir():
        my_pid = str(os.getpid())
        for p in proc_root.iterdir():
            if not p.name.isdigit() or p.name == my_pid:
                continue
            fd_dir = p / "fd"
            try:
                for fd in fd_dir.iterdir():
                    target = os.readlink(fd)
                    if target.startswith("/dev/video"):
                        cmdline = (p / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
                        holders.append(f"PID {p.name} ({cmdline or 'process'}) usando {target}")
                        break
            except Exception:
                continue

    if not usb_indices:
        msg = "Nenhuma webcam USB encontrada em /dev/v4l/by-id ou /dev/video* (verifique o cabo USB com 'lsusb')."
        if all_nodes:
            msg += f" Nós internos encontrados: {', '.join(all_nodes[:5])}"
        return [0, 1, 2], msg

    msg = f"Câmeras USB detectadas: {', '.join(usb_names)}."
    if holders:
        msg += f" Em uso por: {'; '.join(holders)}."
    return usb_indices, msg


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
        allow_web_bridge: bool = True,
    ) -> None:
        self._device_index   = device_index
        self._resolution     = resolution
        self._warmup_seconds = warmup_seconds
        self._allow_web_bridge = allow_web_bridge
        self._cap: cv2.VideoCapture | None = None
        self._using_web_bridge: bool = False
        self._state = CameraState.IDLE
        self.last_error_detail: str = ""

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
        if self.is_ready:
            return

        self._using_web_bridge = False
        discovered, diag_msg = diagnose_linux_cameras()
        candidates = [self._device_index] + [i for i in discovered if i != self._device_index]

        log.info(
            "[Camera] Opening device %d (candidates=%s) ...",
            self._device_index, candidates,
        )

        with _quiet_opencv():
            for idx in candidates:
                cap = self._try_open_index(idx)
                if cap is not None and cap.isOpened():
                    self._cap = cap
                    self._device_index = idx
                    break

        if self._cap is None or not self._cap.isOpened():
            # Check if web_calibrator.py is running on localhost:5000 with a REAL camera open
            if self._allow_web_bridge and self._fetch_web_bridge_frame() is not None:
                self._using_web_bridge = True
                log.info(
                    "[Camera] USB camera is open in web_calibrator — "
                    "bridging live frames via http://127.0.0.1:5000/api/raw_camera"
                )
                return

            self.last_error_detail = diag_msg
            raise RuntimeError(
                f"[Camera] Failed to open USB camera (tried {candidates}). {diag_msg}"
            )

        # Set resolution and minimize internal driver frame buffer (avoids stale frames)
        with _quiet_opencv():
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])

            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("[Camera] Actual resolution (device %d): %dx%d", self._device_index, actual_w, actual_h)

        # Warmup: discard frames until auto-exposure stabilises
        self._warmup()
        log.info("[Camera] Ready.")

    @staticmethod
    def _try_open_index(idx: int) -> cv2.VideoCapture | None:
        if sys.platform == "win32":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                return cap
        else:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap
                cap.release()
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
            cap.release()
        return None

    @staticmethod
    def _fetch_web_bridge_frame(url: str = "http://127.0.0.1:5000/api/raw_camera") -> np.ndarray | None:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status != 200:
                    return None
                data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def close(self) -> None:
        """Release the camera."""
        self._using_web_bridge = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._state = CameraState.IDLE
            log.info("[Camera] Released.")

    @property
    def is_ready(self) -> bool:
        if self._using_web_bridge:
            return True
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(self, flush_buffer: bool = True) -> np.ndarray:
        """
        Capture a single live frame.

        Parameters
        ----------
        flush_buffer : bool
            When True (default), drains queued frames from the OS/V4L2/DirectShow
            ring buffer via ``grab()`` before decoding so the returned image
            always reflects the current physical board state right now rather
            than an old frame buffered before the move.

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
        if self._using_web_bridge:
            frame = self._fetch_web_bridge_frame()
            if frame is None:
                self._state = CameraState.FAILED
                raise RuntimeError("[Camera] Failed to read frame from web_calibrator bridge.")
            self._state = CameraState.IDLE
            return frame

        assert self._cap is not None
        if flush_buffer:
            for _ in range(4):
                self._cap.grab()
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
        if self._cap is None:
            return
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
