import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2
WAKE_WORD = "MAGNUS"


@dataclass(frozen=True)
class Settings:
    language: str = "en-US"
    device_name: str = "USB Audio Device"
    model_dir: Path = Path(__file__).parent / "models"
    capture_rate: int = 48000
    command_timeout: float = 10.0
    confirmation_timeout: float = 10.0
    audio_timeout: float = 2.0
    silence_ms: int = 600
    wake_backend: str = "vosk"
    log_level: str = "INFO"

    def __post_init__(self):
        for name in ("language", "device_name", "wake_backend", "log_level"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        if self.language not in {"en-US", "pt-BR"}:
            raise ValueError("language must be en-US or pt-BR")
        if not self.device_name.strip():
            raise ValueError("device_name must identify a microphone")
        if type(self.capture_rate) is not int or self.capture_rate not in {16000, 44100, 48000}:
            raise ValueError("capture_rate must be 16000, 44100 or 48000")
        for name in ("command_timeout", "confirmation_timeout", "audio_timeout"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.silence_ms) is not int or not 90 <= self.silence_ms <= 3000:
            raise ValueError("silence_ms must be an integer between 90 and 3000")
        if self.wake_backend not in {"vosk", "openwakeword"}:
            raise ValueError("wake_backend must be vosk or openwakeword")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("log_level must be DEBUG, INFO, WARNING or ERROR")

    @property
    def model_path(self):
        model = (
            "vosk-model-small-en-us-0.15" if self.language == "en-US" else "vosk-model-small-pt-0.3"
        )
        return self.model_dir / model


def load_settings(path: Path | None = None) -> Settings:
    path = path or Path(__file__).resolve().parent.parent / "voice.toml"
    with path.open("rb") as source:
        data = tomllib.load(source)
    if set(data) != {"voice"} or not isinstance(data["voice"], dict):
        raise ValueError("Configuration must contain a [voice] table only")
    values = data["voice"]
    unknown = set(values) - {field.name for field in fields(Settings)}
    if unknown:
        raise ValueError(f"Unknown voice settings: {sorted(unknown)}")
    if "model_dir" in values:
        if not isinstance(values["model_dir"], str):
            raise ValueError("model_dir must be a path string")
        values["model_dir"] = (path.resolve().parent / values["model_dir"]).resolve()
    return Settings(**values)
