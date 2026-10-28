import time

from .audio_capture import AudioError


class WakeWordDetector:
    """Optional ONNX adapter. Vosk is the default and shares the speech model."""

    def __init__(self, settings):
        from openwakeword.model import Model

        path = settings.model_dir / "magnus.onnx"
        if not path.is_file():
            raise FileNotFoundError(f"Wake word model missing: {path}")
        self.model = Model(wakeword_models=[str(path)], inference_framework="onnx")
        self.audio_timeout = settings.audio_timeout

    def wait(self, stream):
        import numpy as np

        self.model.reset()
        pending = bytearray()
        last_frame = time.monotonic()
        while True:
            frame = stream.read_frame(0.1)
            if frame is None:
                if time.monotonic() - last_frame >= self.audio_timeout:
                    raise AudioError("No audio frames received from microphone")
                continue
            last_frame = time.monotonic()
            pending.extend(frame)
            while len(pending) >= 2560:
                scores = self.model.predict(np.frombuffer(bytes(pending[:2560]), dtype="<i2"))
                del pending[:2560]
                if max(scores.values(), default=0) >= 0.5:
                    stream.check()
                    return
