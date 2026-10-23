import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from voice.audio_capture import AudioError, AudioStream, FrameConverter, select_device
from voice.config import FRAME_BYTES, Settings


@pytest.fixture
def pa(monkeypatch):
    stream = Mock()
    stream.is_active.return_value = True
    driver = Mock()
    driver.get_device_count.return_value = 1
    driver.get_device_info_by_index.return_value = {
        "index": 0,
        "name": "USB Audio Device",
        "maxInputChannels": 1,
    }
    driver.open.return_value = stream
    module = SimpleNamespace(
        paInt16=8, paComplete=1, paContinue=0, paAbort=2, PyAudio=Mock(return_value=driver)
    )
    monkeypatch.setitem(sys.modules, "pyaudio", module)
    return driver, stream, module


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_streaming_conversion_preserves_fragmented_samples(rate):
    samples = (np.sin(np.arange(rate) * 2 * np.pi * 440 / rate) * 12000).astype("<i2")
    full = FrameConverter(rate).feed(samples.tobytes())
    converter = FrameConverter(rate)
    fragmented = []
    for chunk in np.array_split(samples, 113):
        fragmented.extend(converter.feed(chunk.tobytes()))
    np.testing.assert_allclose(
        np.frombuffer(b"".join(full), dtype="<i2").astype(float),
        np.frombuffer(b"".join(fragmented), dtype="<i2"),
        atol=2,
        rtol=0,
    )
    assert all(len(frame) == FRAME_BYTES for frame in fragmented)
    assert 30 <= len(fragmented) <= 33


def test_conversion_reset_drops_old_audio():
    converter = FrameConverter(48000)
    converter.feed(np.full(1500, 20000, dtype="<i2").tobytes())
    converter.reset()
    assert not converter.pending
    silence = np.frombuffer(b"".join(converter.feed(bytes(48000 * 2))), dtype="<i2")
    assert np.max(np.abs(silence)) <= 1


def test_odd_pcm_rejected():
    with pytest.raises(AudioError):
        FrameConverter(48000).feed(b"x")


def test_device_selection():
    devices = [{"index": 3, "name": "USB Audio Device"}]
    assert select_device(devices, "usb audio") == 3
    for candidates in ([], devices * 2):
        with pytest.raises(AudioError):
            select_device(candidates, "USB")


def test_callback_is_active_before_capture_starts(pa):
    driver, device, _ = pa
    audio = AudioStream(Settings())

    def start():
        assert audio._active
        assert driver.open.call_args.kwargs["start"] is False
        audio._callback(bytes(2880), 1440, {}, 0)

    device.start_stream.side_effect = start
    with audio:
        assert audio.received_frames == 1
    device.close.assert_called_once()
    driver.terminate.assert_called_once()


def test_open_failure_releases_partial_resources(pa):
    driver, _, _ = pa
    driver.open.side_effect = OSError("invalid sample rate")
    audio = AudioStream(Settings())
    with pytest.raises(AudioError, match="invalid sample rate"):
        audio.open()
    driver.terminate.assert_called_once()
    audio.close()


def test_start_failure_closes_stream(pa):
    driver, device, _ = pa
    device.start_stream.side_effect = OSError("start failed")
    with pytest.raises(AudioError):
        AudioStream(Settings()).open()
    device.close.assert_called_once()
    driver.terminate.assert_called_once()


def test_full_queue_closes_without_blocking(pa):
    audio = AudioStream(Settings(), maxsize=1)
    audio.open()
    audio._callback(bytes(2880), 1440, {}, 0)
    audio._callback(bytes(2880), 1440, {}, 0)
    with pytest.raises(AudioError, match="overflow"):
        audio.read_frame(0.1)
    start = time.monotonic()
    audio.close()
    audio.close()
    assert time.monotonic() - start < 0.2


@pytest.mark.parametrize("status,data", [(2, bytes(2880)), (0, b""), (0, b"a")])
def test_status_and_malformed_frames_abort(pa, status, data):
    with AudioStream(Settings()) as audio:
        audio._callback(data, 1440, {}, status)
        with pytest.raises(AudioError):
            audio.check()


def test_dead_device_and_no_frame_timeout(pa):
    _, device, _ = pa
    with AudioStream(Settings()) as audio:
        start = time.monotonic()
        assert audio.read_frame(0.02) is None
        assert time.monotonic() - start < 0.2
        device.is_active.return_value = False
        with pytest.raises(AudioError, match="stopped"):
            audio.read_frame(1)


def test_discard_pending_rejects_late_old_callback(pa):
    with AudioStream(Settings()) as audio:
        audio._frames.append(b"old")
        audio._queue.put((0, b"old"))
        audio.discard_pending()
        assert not audio._frames
        assert audio._queue.empty()
        audio._queue.put((audio._cutoff - 1, bytes(2880)))
        audio._converter = Mock()
        assert audio.read_frame(0.02) is None
        audio._converter.feed.assert_not_called()


def test_capture_can_reopen_after_failure(pa):
    audio = AudioStream(Settings(), maxsize=1)
    audio.open()
    audio._error = AudioError("old failure")
    audio.close()
    audio.open()
    audio.check()
    audio.close()
