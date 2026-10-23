import sys
import time

import numpy as np
from loguru import logger

from voice.audio_capture import AudioStream
from voice.config import load_settings


def main():
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {message}")
    with AudioStream(load_settings()) as stream:
        for _ in range(30):
            chunks = []
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                frame = stream.read_frame(0.1)
                if frame:
                    chunks.append(np.frombuffer(frame, dtype="<i2"))
            if not chunks:
                raise RuntimeError("No microphone frames")
            samples = np.concatenate(chunks).astype(float)
            logger.info(
                "MIC dc={:.1f} speech_rms={:.1f} peak={:.0f}",
                samples.mean(),
                samples.std(),
                np.abs(samples).max(),
            )


if __name__ == "__main__":
    main()
