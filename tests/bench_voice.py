import math
import os
import resource
import signal
import sys
import threading
import time

import numpy as np
from loguru import logger

from voice.audio_capture import AudioStream
from voice.cli import report_state
from voice.config import load_settings
from voice.pipeline import VoiceCommandPipeline


def main():
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}")
    logger.add("/tmp/voicechess-bench.log", mode="w", format="{time:HH:mm:ss} | {level} | {message}")
    settings = load_settings()
    pipeline = VoiceCommandPipeline(settings, on_state=report_state)
    duration = float(os.environ.get("VOICE_BENCH_SECONDS", "600"))
    count = clipped = peak = frames = confirmed = rejected = 0
    energy = 0.0
    start = time.monotonic()
    cpu = time.process_time()
    stream = AudioStream(settings)
    read_frame = stream.read_frame

    def measure(timeout):
        nonlocal count, clipped, peak, frames, energy
        frame = read_frame(timeout)
        if frame:
            samples = np.frombuffer(frame, dtype="<i2").astype(np.int32)
            count += len(samples)
            frames += 1
            energy += float(np.dot(samples.astype(float), samples))
            peak = max(peak, int(np.max(np.abs(samples))))
            clipped += int(np.count_nonzero(np.abs(samples) >= 32767))
        return frame

    stream.read_frame = measure
    timer = threading.Timer(duration, os.kill, args=(os.getpid(), signal.SIGINT))
    timer.daemon = True
    try:
        with stream:
            timer.start()
            while True:
                result = pipeline.listen_once(stream)
                confirmed += result.confirmed
                rejected += not result.confirmed
                logger.info("Cycle: {}", result.to_dict())
    except KeyboardInterrupt:
        logger.info("SIGINT received; microphone released.")
    finally:
        timer.cancel()
        elapsed = time.monotonic() - start
        logger.info(
            "BENCH elapsed={:.2f}s cpu_one_core={:.2f}% peak_rss={:.2f}MiB "
            "frames={} samples={} rms={:.2f} peak={} clipping={:.6f}% confirmed={} rejected={}",
            elapsed, (time.process_time() - cpu) / elapsed * 100,
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            frames, count, math.sqrt(energy / count) if count else 0,
            peak, clipped / count * 100 if count else 0, confirmed, rejected,
        )


if __name__ == "__main__":
    main()
