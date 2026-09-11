"""
audio_capture.py
================
Wraps PyAudio into a context-manager that streams fixed-size PCM frames
suitable for WebRTC VAD, Vosk KaldiRecognizer, and openWakeWord.

Frame size: 30 ms @ 16 kHz / 16-bit / mono = 960 bytes per frame.

Usage
-----
    from audio_capture import AudioStream

    with AudioStream() as stream:
        for frame in stream:
            process(frame)   # bytes object, exactly VAD_FRAME_BYTES long
"""

from __future__ import annotations

import logging
import queue
from typing import Iterator

import pyaudio

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------

def list_audio_devices() -> list[dict]:
    """Return a list of dicts describing every PyAudio input device."""
    pa = pyaudio.PyAudio()
    devices = []
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxInputChannels"],
                    "sample_rate": int(info["defaultSampleRate"]),
                })
    finally:
        pa.terminate()
    return devices


def find_usb_mic_index() -> int | None:
    """
    Auto-detect the first USB / physical microphone by scanning device names.
    Returns None if none found (PyAudio will use the system default).
    """
    keywords = ("usb", "microphone", "microfone", "mic", "headset")
    ignore = ("mapeador", "microsoft", "driver de captura", "mixagem", "steam")
    for dev in list_audio_devices():
        name_lower = dev["name"].lower()
        if any(ign in name_lower for ign in ignore):
            continue
        if any(kw in name_lower for kw in keywords):
            log.info(
                "Auto-detected USB mic: index=%d  name='%s'",
                dev["index"], dev["name"],
            )
            return dev["index"]
    log.warning("No USB microphone found — using system default input device.")
    return None


# ---------------------------------------------------------------------------
# AudioStream
# ---------------------------------------------------------------------------

class AudioStream:
    """
    Thread-safe audio capture using a background producer thread and
    an internal queue so the consumer never blocks the audio callback.

    Parameters
    ----------
    device_index : int | None
        PyAudio device index.  None = auto-detect USB mic.
    sample_rate : int
        Target sample rate (Hz).
    frame_bytes : int
        Exact byte length of each yielded frame.
    channels : int
        Number of audio channels (1 = mono).
    sample_width : int
        Bytes per sample (2 = 16-bit PCM).
    maxsize : int
        Maximum frames held in the internal queue before dropping.
    """

    def __init__(
        self,
        device_index: int | None = config.MIC_DEVICE_INDEX,
        sample_rate: int = config.SAMPLE_RATE,
        frame_bytes: int = config.VAD_FRAME_BYTES,
        channels: int = config.CHANNELS,
        sample_width: int = config.SAMPLE_WIDTH,
        maxsize: int = 50,
    ) -> None:
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._frame_bytes = frame_bytes
        self._channels = channels
        self._sample_width = sample_width
        self._format = pyaudio.paInt16

        self._pa: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=maxsize)
        self._active = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "AudioStream":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open PyAudio and start the capture stream."""
        if self._active:
            return

        self._pa = pyaudio.PyAudio()

        dev_index = self._device_index
        if dev_index is None:
            dev_index = find_usb_mic_index()

        frames_per_buffer = (
            self._frame_bytes // self._sample_width // self._channels
        )

        log.info(
            "Opening audio stream: device=%s  rate=%d Hz  frame=%d bytes",
            dev_index if dev_index is not None else "default",
            self._sample_rate,
            self._frame_bytes,
        )

        self._stream = self._pa.open(
            format=self._format,
            channels=self._channels,
            rate=self._sample_rate,
            input=True,
            input_device_index=dev_index,
            frames_per_buffer=frames_per_buffer,
            stream_callback=self._callback,
        )
        self._active = True
        self._stream.start_stream()
        log.info("Audio stream started.")

    def close(self) -> None:
        """Stop and release all audio resources."""
        if not self._active:
            return
        self._active = False
        self._queue.put(None)  # sentinel — unblocks __iter__

        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

        log.info("Audio stream closed.")

    # ------------------------------------------------------------------
    # Internal callback  (runs in PyAudio's internal thread)
    # ------------------------------------------------------------------

    def _callback(
        self,
        in_data: bytes,
        frame_count: int,
        time_info: dict,
        status_flags: int,
    ) -> tuple[None, int]:
        if not self._active:
            return None, pyaudio.paComplete

        if status_flags:
            log.debug("PyAudio status flag: %s", status_flags)

        if len(in_data) == self._frame_bytes:
            try:
                self._queue.put_nowait(in_data)
            except queue.Full:
                log.debug("Audio queue full — dropping frame.")
        elif len(in_data) > self._frame_bytes:
            try:
                self._queue.put_nowait(in_data[: self._frame_bytes])
            except queue.Full:
                pass
        # shorter frames are silently dropped (malformed hardware delivery)

        return None, pyaudio.paContinue

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[bytes]:
        """Yield PCM frames indefinitely until .close() is called."""
        while True:
            frame = self._queue.get()
            if frame is None:
                break
            yield frame

    def read_frame(self, timeout: float = 1.0) -> bytes | None:
        """
        Read a single frame with an optional timeout.
        Returns None if no frame arrives within *timeout* seconds.
        """
        try:
            frame = self._queue.get(timeout=timeout)
            return frame
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def frame_duration_ms(self) -> int:
        samples_per_frame = (
            self._frame_bytes // self._sample_width // self._channels
        )
        return int(samples_per_frame * 1000 / self._sample_rate)
