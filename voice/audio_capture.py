import queue
import time
from collections import deque

from .config import FRAME_BYTES, FRAME_MS, SAMPLE_RATE, Settings


class AudioError(RuntimeError):
    pass


def list_audio_devices() -> list[dict]:
    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        return [
            dict(pa.get_device_info_by_index(i))
            for i in range(pa.get_device_count())
            if pa.get_device_info_by_index(i)["maxInputChannels"] > 0
        ]
    finally:
        pa.terminate()


def select_device(devices: list[dict], name: str) -> int:
    if not devices:
        raise AudioError("No audio input devices found.")
    matches = [d for d in devices if name.casefold() in d["name"].casefold()]
    if matches:
        return int(matches[0]["index"])
    usb_devices = [d for d in devices if "usb" in d["name"].casefold()]
    if usb_devices:
        return int(usb_devices[0]["index"])
    return int(devices[0]["index"])


class FrameConverter:
    def __init__(self, rate: int):
        import soxr

        self.resampler = soxr.ResampleStream(rate, SAMPLE_RATE, 1, dtype="int16")
        self.pending = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        import numpy as np

        if len(data) % 2:
            raise AudioError("Malformed PCM data")
        self.pending.extend(
            self.resampler.resample_chunk(np.frombuffer(data, dtype="<i2")).tobytes()
        )
        frames = []
        while len(self.pending) >= FRAME_BYTES:
            frames.append(bytes(self.pending[:FRAME_BYTES]))
            del self.pending[:FRAME_BYTES]
        return frames

    def reset(self):
        self.pending.clear()
        self.resampler.clear()


class AudioStream:
    def __init__(self, settings: Settings, *, maxsize: int = 50):
        self.settings = settings
        self._queue = queue.Queue(maxsize=maxsize)
        self._frames = deque()
        self._pa = self._stream = self._converter = None
        self._active = False
        self._error = None
        self._cutoff = 0.0
        self.received_frames = 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    def open(self):
        import pyaudio

        if self._active:
            return
        self._error = None
        self._cutoff = 0.0
        try:
            self._converter = FrameConverter(self.settings.capture_rate)
            self._pa = pyaudio.PyAudio()
            devices = [
                dict(self._pa.get_device_info_by_index(i))
                for i in range(self._pa.get_device_count())
                if self._pa.get_device_info_by_index(i)["maxInputChannels"] > 0
            ]
            index = select_device(devices, self.settings.device_name)
            self._pa.is_format_supported(
                self.settings.capture_rate,
                input_device=index,
                input_channels=1,
                input_format=pyaudio.paInt16,
            )
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.settings.capture_rate,
                input=True,
                input_device_index=index,
                frames_per_buffer=self.settings.capture_rate * FRAME_MS // 1000,
                stream_callback=self._callback,
                start=False,
            )
            self._active = True
            self._stream.start_stream()
        except Exception as error:
            self.close()
            raise AudioError(f"Could not start microphone: {error}") from error

    def _callback(self, data, frame_count, time_info, status):
        import pyaudio

        if not self._active:
            return None, pyaudio.paComplete
        # Buffer overflow/underflow (status & 1 or 2) e comum no Linux/RPi ao inicializar ou
        # quando o speaker fala. Nao abortamos por overflow/underflow transitorio.
        if not data or len(data) != frame_count * 2:
            self._error = AudioError(f"Audio capture invalid data (len={len(data) if data else 0})")
            return None, pyaudio.paAbort
        # Map the device timestamp to monotonic time to discard pre-prompt audio.
        age = max(0.0, time_info.get("current_time", 0) - time_info.get("input_buffer_adc_time", 0))
        try:
            self._queue.put_nowait((time.monotonic() - age, data))
            self.received_frames += 1
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait((time.monotonic() - age, data))
            except Exception:
                pass
        return None, pyaudio.paContinue

    def check(self):
        if self._error:
            raise self._error
        if not self._active:
            raise AudioError("Microphone is closed")
        if self._stream is not None and not self._stream.is_active():
            raise AudioError("Microphone stopped providing audio")

    def read_frame(self, timeout: float) -> bytes | None:
        deadline = time.monotonic() + timeout
        while True:
            self.check()
            if self._frames:
                return self._frames.popleft()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                stamp, data = self._queue.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                continue
            if stamp >= self._cutoff:
                self._frames.extend(self._converter.feed(data))

    def discard_pending(self):
        self.check()
        self._cutoff = time.monotonic()
        self._frames.clear()
        self._converter.reset()
        self._drain()

    def _drain(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def close(self):
        self._active = False
        try:
            if self._stream is not None:
                try:
                    self._stream.stop_stream()
                finally:
                    self._stream.close()
        finally:
            self._stream = None
            if self._pa is not None:
                self._pa.terminate()
                self._pa = None
            self._frames.clear()
            self._drain()
