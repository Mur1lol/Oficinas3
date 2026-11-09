import ctypes
import os
import queue
import sys
import time
from collections import deque
from contextlib import contextmanager

from .config import FRAME_BYTES, FRAME_MS, SAMPLE_RATE, Settings

_c_alsa_handler = None


def suppress_alsa_handler():
    """Silencia mensagens de erro/aviso da biblioteca ALSA (libasound) em nivel C."""
    global _c_alsa_handler
    if _c_alsa_handler is not None:
        return
    try:
        ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
            None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
        )

        def _null_handler(filename, line, function, err, fmt):
            pass

        _c_alsa_handler = ERROR_HANDLER_FUNC(_null_handler)
        for lib in ("libasound.so", "libasound.so.2"):
            try:
                asound = ctypes.cdll.LoadLibrary(lib)
                asound.snd_lib_error_set_handler(_c_alsa_handler)
                break
            except OSError:
                continue
    except Exception:
        pass


@contextmanager
def suppress_c_stderr():
    """Redireciona temporariamente o stderr do C (FD 2) para devnull durante varredura ALSA/JACK."""
    try:
        sys.stderr.flush()
        old_stderr_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        try:
            yield
        finally:
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)
    except Exception:
        yield


class AudioError(RuntimeError):
    pass


def list_audio_devices() -> list[dict]:
    import pyaudio

    suppress_alsa_handler()
    with suppress_c_stderr():
        pa = pyaudio.PyAudio()
        try:
            return [
                dict(pa.get_device_info_by_index(i))
                for i in range(pa.get_device_count())
                if pa.get_device_info_by_index(i)["maxInputChannels"] > 0
            ]
        finally:
            pa.terminate()


def select_device(devices: list[dict], name: str, slot: int = 0) -> int:
    if not devices:
        raise AudioError("No audio input devices found.")
    matches = [d for d in devices if name.casefold() in d["name"].casefold()]
    if matches:
        idx = min(max(0, slot), len(matches) - 1)
        return int(matches[idx]["index"])
    usb_devices = [d for d in devices if "usb" in d["name"].casefold()]
    if usb_devices:
        idx = min(max(0, slot), len(usb_devices) - 1)
        return int(usb_devices[idx]["index"])
    idx = min(max(0, slot), len(devices) - 1)
    return int(devices[idx]["index"])


def select_side_devices(
    devices: list[dict],
    name: str,
    p1_slot: int = 1,
    p2_slot: int = 0,
) -> dict[str, int]:
    """
    Mapeia os dois microfones USB para os lados do tabuleiro:
    - 'player'   (Player 1 / Brancas) -> slot ``p1_slot`` (padrao: 1, segunda placa USB)
    - 'computer' (Player 2 / Pretas)  -> slot ``p2_slot`` (padrao: 0, primeira placa USB)
    Se houver apenas 1 microfone conectado, ambos os lados usam o mesmo dispositivo.
    """
    return {
        "player": select_device(devices, name, slot=p1_slot),
        "computer": select_device(devices, name, slot=p2_slot),
    }


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
    def __init__(self, settings: Settings, *, maxsize: int = 50, initial_side: str = "player"):
        self.settings = settings
        self._queue = queue.Queue(maxsize=maxsize)
        self._frames = deque()
        self._pa = self._stream = self._converter = None
        self._active = False
        self._error = None
        self._cutoff = 0.0
        self.received_frames = 0
        self._side = initial_side if initial_side in ("player", "computer") else "player"
        self._side_devices: dict[str, int] = {}
        self._device_names: dict[int, str] = {}
        self._current_device_index: int | None = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    def _open_input_stream(self, index: int) -> None:
        import pyaudio

        with suppress_c_stderr():
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
        self._current_device_index = index
        self._active = True
        self._stream.start_stream()

    def _close_input_stream(self) -> None:
        self._active = False
        if self._stream is not None:
            try:
                self._stream.stop_stream()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def open(self):
        import pyaudio
        from loguru import logger

        if self._active:
            return
        self._error = None
        self._cutoff = 0.0
        suppress_alsa_handler()
        try:
            self._converter = FrameConverter(self.settings.capture_rate)
            with suppress_c_stderr():
                self._pa = pyaudio.PyAudio()
                devices = [
                    dict(self._pa.get_device_info_by_index(i))
                    for i in range(self._pa.get_device_count())
                    if self._pa.get_device_info_by_index(i)["maxInputChannels"] > 0
                ]
            self._device_names = {int(d["index"]): d["name"] for d in devices}
            p1_slot = getattr(self.settings, "player1_usb_slot", 1)
            p2_slot = getattr(self.settings, "player2_usb_slot", 0)
            self._side_devices = select_side_devices(
                devices, self.settings.device_name, p1_slot=p1_slot, p2_slot=p2_slot
            )
            p1_idx = self._side_devices["player"]
            p2_idx = self._side_devices["computer"]
            logger.info(
                "[VOZ] Microfones mapeados: Player 1 (Brancas) -> idx {} ({!r}) | "
                "Player 2 (Pretas) -> idx {} ({!r})",
                p1_idx,
                self._device_names.get(p1_idx, "?"),
                p2_idx,
                self._device_names.get(p2_idx, "?"),
            )

            target_idx = self._side_devices.get(self._side, p1_idx)
            fallback_idx = p2_idx if target_idx == p1_idx else p1_idx
            try:
                self._open_input_stream(target_idx)
            except Exception:
                if fallback_idx != target_idx:
                    logger.warning(
                        "[VOZ] Falha ao abrir microfone idx {}, usando fallback idx {}",
                        target_idx,
                        fallback_idx,
                    )
                    self._open_input_stream(fallback_idx)
                else:
                    raise
            side_label = "Player 1 (Brancas)" if self._side == "player" else "Player 2 (Pretas)"
            logger.info(
                "[VOZ] Microfone ativo inicial: {} [idx {}]",
                side_label,
                self._current_device_index,
            )
        except Exception as error:
            self.close()
            raise AudioError(f"Could not start microphone: {error}") from error

    def set_side(self, side: str) -> None:
        """
        Alterna o microfone ativo conforme o turno/lado:
        - 'player'   -> Microfone do Player 1 (Brancas)
        - 'computer' -> Microfone do Player 2 (Pretas)
        """
        from loguru import logger

        if side not in ("player", "computer"):
            return
        self._side = side
        if not self._active or self._pa is None or not self._side_devices:
            return
        target_idx = self._side_devices.get(side)
        if target_idx is None or target_idx == self._current_device_index:
            return

        prev_idx = self._current_device_index
        side_label = "Player 1 (Brancas)" if side == "player" else "Player 2 (Pretas)"
        try:
            self._close_input_stream()
            self._error = None
            self._cutoff = time.monotonic()
            self._frames.clear()
            if self._converter is not None:
                self._converter.reset()
            self._drain()
            self._open_input_stream(target_idx)
            logger.info(
                "[VOZ] Microfone ativo trocado para: {} [idx {} - {!r}]",
                side_label,
                target_idx,
                self._device_names.get(target_idx, "?"),
            )
        except Exception as exc:
            logger.warning(
                "[VOZ] Nao foi possivel alternar para microfone idx {} ({}): {}. Mantendo idx {}.",
                target_idx,
                side_label,
                exc,
                prev_idx,
            )
            if prev_idx is not None:
                try:
                    self._open_input_stream(prev_idx)
                except Exception as fatal:
                    self._error = AudioError(f"Could not reopen microphone: {fatal}")

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
            self._current_device_index = None
            if self._pa is not None:
                self._pa.terminate()
                self._pa = None
            self._frames.clear()
            self._drain()
