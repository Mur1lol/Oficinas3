import subprocess
import sys

import pytest
from click.testing import CliRunner

from voice.cli import cli
from voice.config import Settings, load_settings
from voice.parser import ACTIONS, FILLERS, MOVE_WORDS, grammar


@pytest.mark.parametrize("options", [{"language": "xx"}, {"command_timeout": 0},
    {"confirmation_timeout": float("inf")}, {"device_name": ""}, {"capture_rate": 22050},
    {"audio_timeout": -1}, {"silence_ms": 0}, {"wake_backend": "unknown"}])
def test_invalid_settings(options):
    with pytest.raises(ValueError):
        Settings(**options)


def test_config_paths_relative_to_file(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('[voice]\nmodel_dir="models"\nlanguage="en-US"\n')
    assert load_settings(path).model_dir == tmp_path / "models"
    path.write_text('[voice]\nunrecognised=true\n')
    with pytest.raises(ValueError, match="Unknown"):
        load_settings(path)


def test_command_grammar_contains_parser_words():
    words = set(grammar("command", "en-US"))
    assert MOVE_WORDS | FILLERS <= words
    for phrases in ACTIONS.values():
        for phrase in phrases:
            assert set(phrase.split()) <= words


def test_import_does_not_load_audio_or_models():
    code = 'import sys; import voice.parser; assert not {"pyaudio", "vosk", "soxr", "numpy"} & sys.modules.keys()'
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cli_has_only_three_commands():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert set(cli.commands) == {"run", "devices", "transcribe"}


def test_devices_use_loguru(monkeypatch):
    monkeypatch.setattr("voice.cli.list_audio_devices", lambda: [{"index": 1, "name": "USB", "defaultSampleRate": 48000}])
    result = CliRunner().invoke(cli, ["devices"])
    assert result.exit_code == 0
    assert "USB" in result.output


def test_interrupt_releases_microphone(monkeypatch):
    from unittest.mock import MagicMock

    stream = MagicMock()
    pipeline = MagicMock()
    pipeline.listen_once.side_effect = KeyboardInterrupt
    monkeypatch.setattr("voice.cli.AudioStream", lambda settings: stream)
    monkeypatch.setattr("voice.cli.VoiceCommandPipeline", lambda *args, **kwargs: pipeline)
    result = CliRunner().invoke(cli, ["run"])
    assert result.exit_code == 0
    stream.__exit__.assert_called_once()
