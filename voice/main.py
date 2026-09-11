"""
main.py
=======
VoiceCommandPipeline — top-level state machine for ChessAI 2.0.

State diagram
-------------
    IDLE
      |
      v
    WAKE_WORD_LISTEN   <-----------+
      |                            |
      | MAGNUS detected            |
      v                            |
    COMMAND_LISTEN                 |
      |                            |
      | speech recognised          | (timeout / invalid / error)
      v                            |
    PARSE                          |
      |                            |
      v                            |
    VALIDATE                       |
      |                            |
      v                            |
    EMIT -------------------------+

Public API
----------
    # Blocking loop (runs forever — for standalone use)
    pipeline = VoiceCommandPipeline()
    pipeline.run()

    # Single-shot (for integration into ChessAI game loop)
    pipeline = VoiceCommandPipeline()
    result = pipeline.listen_once()
    # {"wake_word": "MAGNUS", "command": "MOVE", "from": "E2", "to": "E4", "valid": True}

CLI usage
---------
    python main.py                 # live microphone loop
    python main.py --once          # wait for one command, print, exit
    python main.py --file audio.wav  # transcribe WAV file (no wake-word gate)
    python main.py --demo          # demo mode: mock audio, print result

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from enum import Enum, auto

import config
from audio_capture import AudioStream, list_audio_devices
from parser import parse
from stt_engine import VoskSTTEngine
from validator import validate
from wake_word import WakeWordDetector

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State machine states
# ---------------------------------------------------------------------------

class PipelineState(Enum):
    IDLE               = auto()
    WAKE_WORD_LISTEN   = auto()
    COMMAND_LISTEN     = auto()
    PARSE              = auto()
    VALIDATE           = auto()
    EMIT               = auto()
    ERROR              = auto()


# ---------------------------------------------------------------------------
# VoiceCommandPipeline
# ---------------------------------------------------------------------------

class VoiceCommandPipeline:
    """
    Orchestrates the full voice command pipeline.

    Parameters
    ----------
    command_timeout : float
        Seconds to wait for a command after wake word detection.
    """

    def __init__(
        self,
        command_timeout: float = config.COMMAND_TIMEOUT_S,
    ) -> None:
        self._timeout = command_timeout
        self._state = PipelineState.IDLE

        log.info("Initialising VoiceCommandPipeline...")
        log.info("  Command timeout : %.1f s", self._timeout)
        log.info("  Vosk model      : %s", config.VOSK_MODEL_PATH)
        log.info("  OWW model       : %s", config.OWW_MODEL_PATH)

        # Load engines once — they are expensive to initialise
        self._stt = VoskSTTEngine()
        self._wake = WakeWordDetector()

        log.info("Pipeline ready. Wake word engine: %s", self._wake.engine_name)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def listen_once(self) -> dict:
        """
        Block until the wake word is heard, then listen for one command.

        Returns
        -------
        dict
            Validated result dict. Always contains {"valid": bool}.

        Example
        -------
            {
                "wake_word": "MAGNUS",
                "command": "MOVE",
                "from": "E2",
                "to": "E4",
                "valid": True,
            }
        """
        with AudioStream() as stream:
            return self._run_cycle(stream)

    def run(self) -> None:
        """
        Blocking loop: listen forever, printing each recognised command.

        Designed for standalone operation. Press Ctrl+C to stop.
        """
        log.info("Starting continuous voice command loop. Press Ctrl+C to stop.")
        try:
            with AudioStream() as stream:
                while True:
                    result = self._run_cycle(stream)
                    print("\n" + json.dumps(result, indent=2) + "\n")
        except KeyboardInterrupt:
            log.info("Interrupted by user — shutting down.")

    # ------------------------------------------------------------------
    # Internal state machine
    # ------------------------------------------------------------------

    def _run_cycle(self, stream: AudioStream) -> dict:
        """Execute one full wake-word -> command -> result cycle."""

        # --- IDLE -> WAKE_WORD_LISTEN ---
        self._set_state(PipelineState.WAKE_WORD_LISTEN)
        self._wake.wait_for_wake_word(stream)

        # --- WAKE_WORD_LISTEN -> COMMAND_LISTEN ---
        self._set_state(PipelineState.COMMAND_LISTEN)
        print(f"[{config.WAKE_WORD}] Listening for command...")

        text = self._stt.transcribe(stream, timeout=self._timeout)

        # Timeout handling
        if text is None:
            self._set_state(PipelineState.ERROR)
            result = {
                "valid": False,
                "reason": (
                    f"No command heard within {self._timeout:.0f} seconds "
                    "after wake word. Returning to listen mode."
                ),
            }
            log.warning("Timeout after wake word.")
            self._set_state(PipelineState.WAKE_WORD_LISTEN)
            return result

        # --- COMMAND_LISTEN -> PARSE ---
        self._set_state(PipelineState.PARSE)
        command, parse_error = parse(text)

        # --- PARSE -> VALIDATE ---
        self._set_state(PipelineState.VALIDATE)
        result = validate(command, reason=parse_error)

        # --- VALIDATE -> EMIT ---
        self._set_state(PipelineState.EMIT)
        return result

    def _set_state(self, state: PipelineState) -> None:
        log.debug("State: %s -> %s", self._state.name, state.name)
        self._state = state

    @property
    def state(self) -> PipelineState:
        return self._state


# ---------------------------------------------------------------------------
# File-based transcription (no wake word, no mic)
# ---------------------------------------------------------------------------

def transcribe_file(wav_path: str) -> dict:
    """
    Transcribe a WAV file and return a validated result dict.

    The WAV must be 16 kHz / 16-bit / mono.  No wake-word detection.
    """
    stt = VoskSTTEngine()
    text = stt.transcribe_file(wav_path)
    if text is None:
        return {"valid": False, "reason": "No speech recognised in file."}
    command, err = parse(text)
    return validate(command, reason=err)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ChessAI 2.0 Voice Command Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     # continuous live mic loop
  python main.py --once              # one command then exit
  python main.py --file move.wav     # transcribe a WAV file
  python main.py --list-devices      # list audio input devices
        """,
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Listen for one command, print result as JSON, then exit.",
    )
    p.add_argument(
        "--file",
        metavar="WAV",
        help="Transcribe a 16 kHz/16-bit/mono WAV file (no wake-word gate).",
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available audio input devices and exit.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=config.COMMAND_TIMEOUT_S,
        help=f"Command timeout in seconds (default: {config.COMMAND_TIMEOUT_S}).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # List devices and exit
    if args.list_devices:
        print("\nAvailable audio input devices:")
        for dev in list_audio_devices():
            print(
                f"  [{dev['index']}] {dev['name']}"
                f"  ({dev['channels']}ch, {dev['sample_rate']} Hz)"
            )
        return

    # File-based transcription
    if args.file:
        print(f"Transcribing: {args.file}")
        result = transcribe_file(args.file)
        print(json.dumps(result, indent=2))
        return

    # Live pipeline
    pipeline = VoiceCommandPipeline(command_timeout=args.timeout)

    if args.once:
        result = pipeline.listen_once()
        print(json.dumps(result, indent=2))
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
