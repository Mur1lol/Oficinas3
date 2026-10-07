import sys
from pathlib import Path

import click
from loguru import logger

from .audio_capture import AudioStream, list_audio_devices
from .config import load_settings
from .parser import parse
from .pipeline import PipelineState, VoiceCommandPipeline
from .validator import validate


def report_state(state, command):
    if state == PipelineState.WAKE_WORD_LISTEN:
        logger.info("Say 'Magnus', then wait for the listening prompt.")
    elif state == PipelineState.COMMAND_LISTEN:
        logger.info("Listening for command...")
    elif state == PipelineState.CONFIRMATION_LISTEN:
        logger.info("Confirm {}? Say YES or NO.", command)


@click.group()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def cli(ctx, config_path):
    """VoiceChess microphone and recognition tools."""
    error = None
    try:
        ctx.obj = load_settings(config_path)
    except (OSError, ValueError, TypeError) as exc:
        error = exc
    logger.remove()
    level = "INFO" if error else ctx.obj.log_level
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level=level)
    if error:
        logger.error("Invalid configuration: {}", error)
        ctx.exit(1)


@cli.command()
@click.pass_context
def run(ctx):
    """Listen continuously for a call, command and confirmation."""
    try:
        pipeline = VoiceCommandPipeline(ctx.obj, on_state=report_state)
        with AudioStream(ctx.obj) as stream:
            while True:
                result = pipeline.listen_once(stream)
                if result.confirmed:
                    logger.success("Confirmed voice command: {}", result.to_dict())
                else:
                    logger.warning("{}", result.to_dict())
    except KeyboardInterrupt:
        logger.info("Voice stopped; microphone released.")
    except Exception as error:
        logger.opt(exception=ctx.obj.log_level == "DEBUG").error("Voice stopped: {}", error)
        ctx.exit(1)


@cli.command()
@click.pass_context
def devices(ctx):
    """List microphone names to use in the TOML configuration."""
    try:
        for device in list_audio_devices():
            logger.info("{} | {} | {} Hz", int(device["index"]), device["name"], device["defaultSampleRate"])
    except Exception as error:
        logger.error("Could not list microphones: {}", error)
        ctx.exit(1)


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def transcribe(ctx, audio_file):
    """Inspect a 16 kHz mono WAV; this does not confirm or execute a command."""
    from .stt_engine import VoskSTTEngine

    try:
        text = VoskSTTEngine(ctx.obj).transcribe_file(audio_file)
        command, reason = parse(text or "", ctx.obj.language)
        result = validate(command, reason)
        logger.info("Transcript: {!r}; syntax: {}; confirmed: False", text, result)
        if not result["valid"]:
            ctx.exit(1)
    except (OSError, ValueError, RuntimeError) as error:
        logger.error("Transcription failed: {}", error)
        ctx.exit(1)
