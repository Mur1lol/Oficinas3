"""
main.py
=======
VoiceCommandPipeline — top-level state machine for ChessAI 2.0.

State diagram
-------------
    IDLE
      |
      v
    WAKE_WORD_LISTEN   <---------------------------------------+
      |                                                        |
      | MAGNUS detected                                        |
      v                                                        |
    COMMAND_LISTEN                                             |
      |                                                        |
      | speech recognised                                      | (timeout / invalid / error)
      v                                                        |
    PARSE                                                      |
      |                                                        |
      v                                                        |
    VALIDATE                                                   |
      |                                                        |
      | valid command                                          |
      v                                                        |
    CONFIRMATION_LISTEN (YES/NO, SIM/NÃO)                      |
      |                                                        |
      | user confirms                                          |
      v                                                        |
    EMIT ------------------------------------------------------+
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
from parser import parse, parse_confirmation
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
    IDLE                = auto()
    WAKE_WORD_LISTEN    = auto()
    COMMAND_LISTEN      = auto()
    PARSE               = auto()
    VALIDATE            = auto()
    CONFIRMATION_LISTEN = auto()
    EMIT                = auto()
    ERROR               = auto()


# ---------------------------------------------------------------------------
# VoiceCommandPipeline
# ---------------------------------------------------------------------------

class VoiceCommandPipeline:
    """
    Orchestrates the full voice command pipeline with bilingual support
    and YES/NO confirmation.

    Parameters
    ----------
    command_timeout : float
        Seconds to wait for a command after wake word detection.
    confirm_timeout : float
        Seconds to wait for YES/NO confirmation.
    language : str
        "pt-BR" or "en-US"
    """

    def __init__(
        self,
        command_timeout: float = config.COMMAND_TIMEOUT_S,
        confirm_timeout: float = config.CONFIRM_TIMEOUT_S,
        language: str | None = None,
    ) -> None:
        self.language = language or config.LANGUAGE
        self._timeout = command_timeout
        self._confirm_timeout = confirm_timeout
        self._state = PipelineState.IDLE

        model_path = config.get_model_path(self.language)
        log.info("Initialising VoiceCommandPipeline (%s)...", self.language)
        log.info("  Command timeout : %.1f s", self._timeout)
        log.info("  Confirm timeout : %.1f s", self._confirm_timeout)
        log.info("  Vosk model      : %s", model_path)
        log.info("  OWW model       : %s", config.OWW_MODEL_PATH)

        # Load engines once
        self._stt = VoskSTTEngine(language=self.language)
        self._wake = WakeWordDetector(language=self.language)

        log.info("Pipeline ready. Wake word engine: %s", self._wake.engine_name)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def listen_once(self) -> dict:
        """
        Block until wake word is heard, listen for one command, request confirmation, and return result.
        """
        with AudioStream() as stream:
            return self._run_cycle(stream)

    def run(self) -> None:
        """
        Continuous loop: listen for wake word, command, confirmation, and print output.
        """
        log.info("Starting continuous voice command loop. Press Ctrl+C to stop.")
        try:
            with AudioStream() as stream:
                while True:
                    result = self._run_cycle(stream)
                    print("\n" + json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        except KeyboardInterrupt:
            log.info("Interrupted by user — shutting down.")

    # ------------------------------------------------------------------
    # Internal state machine
    # ------------------------------------------------------------------

    def _run_cycle(self, stream: AudioStream) -> dict:
        """Execute one full wake-word -> command -> confirmation -> result cycle."""
        is_pt = self.language.lower().startswith("pt")

        # 1. WAKE_WORD_LISTEN
        self._set_state(PipelineState.WAKE_WORD_LISTEN)
        self._wake.wait_for_wake_word(stream)

        # 2. COMMAND_LISTEN
        self._set_state(PipelineState.COMMAND_LISTEN)
        listen_prompt = "Ouvindo comando..." if is_pt else "Listening for command..."
        print(f"[{config.WAKE_WORD}] {listen_prompt}")

        text = self._stt.transcribe(stream, timeout=self._timeout)

        # Timeout handling for command
        if text is None:
            self._set_state(PipelineState.ERROR)
            reason = (
                f"Nenhum comando ouvido em {self._timeout:.0f} segundos. Retornando ao modo de escuta."
                if is_pt else
                f"No command heard within {self._timeout:.0f} seconds after wake word. Returning to listen mode."
            )
            result = {"valid": False, "reason": reason}
            log.warning("Timeout after wake word.")
            self._set_state(PipelineState.WAKE_WORD_LISTEN)
            return result

        # 3. PARSE
        self._set_state(PipelineState.PARSE)
        command, parse_error = parse(text, language=self.language)

        # 4. VALIDATE
        self._set_state(PipelineState.VALIDATE)
        result = validate(command, reason=parse_error)

        if not result.get("valid", False):
            self._set_state(PipelineState.ERROR)
            return result

        # 5. CONFIRMATION_LISTEN (YES/NO, SIM/NÃO)
        self._set_state(PipelineState.CONFIRMATION_LISTEN)

        if result["command"] == "MOVE":
            cmd_str = f"MOVE {result['from']} {result['to']}"
        else:
            cmd_str = result["command"]

        if is_pt:
            confirm_prompt = f"[{config.WAKE_WORD}] Confirmar {cmd_str}? Diga SIM ou NÃO."
        else:
            confirm_prompt = f"[{config.WAKE_WORD}] Confirm {cmd_str}? Say YES or NO."

        print(confirm_prompt)
        log.info(confirm_prompt)

        confirm_text = self._stt.transcribe_confirmation(stream, timeout=self._confirm_timeout)
        confirmed = parse_confirmation(confirm_text, language=self.language)

        if confirmed is True:
            if is_pt:
                print(f"[{config.WAKE_WORD}] Comando confirmado!")
            else:
                print(f"[{config.WAKE_WORD}] Command confirmed!")
            result["confirmed"] = True
            self._set_state(PipelineState.EMIT)
            return result

        elif confirmed is False:
            if is_pt:
                print(f"[{config.WAKE_WORD}] Comando cancelado.")
            else:
                print(f"[{config.WAKE_WORD}] Command cancelled.")
            result["confirmed"] = False
            result["valid"] = False
            result["reason"] = "Comando cancelado pelo usuário (disse NÃO)." if is_pt else "Command cancelled by user (said NO)."
            self._set_state(PipelineState.ERROR)
            return result

        else:
            # Inconclusive or timeout
            if is_pt:
                print(f"[{config.WAKE_WORD}] Confirmação não ouvida ou expirada.")
            else:
                print(f"[{config.WAKE_WORD}] Confirmation timed out or not understood.")
            result["confirmed"] = False
            result["valid"] = False
            result["reason"] = (
                f"Confirmação expirou após {self._confirm_timeout:.0f}s sem resposta SIM/NÃO."
                if is_pt else
                f"Confirmation timed out after {self._confirm_timeout:.0f}s without YES/NO response."
            )
            self._set_state(PipelineState.ERROR)
            return result

    def _set_state(self, state: PipelineState) -> None:
        log.debug("State: %s -> %s", self._state.name, state.name)
        self._state = state

    @property
    def state(self) -> PipelineState:
        return self._state


# ---------------------------------------------------------------------------
# File-based transcription
# ---------------------------------------------------------------------------

def transcribe_file(wav_path: str, language: str | None = None) -> dict:
    """Transcribe a WAV file and return a validated result dict."""
    lang = language or config.LANGUAGE
    stt = VoskSTTEngine(language=lang)
    text = stt.transcribe_file(wav_path)
    if text is None:
        return {"valid": False, "reason": "No speech recognised in file."}
    command, err = parse(text, language=lang)
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
  python main.py                     # continuous live mic loop (uses config.LANGUAGE)
  python main.py --lang pt-BR        # run in Portuguese
  python main.py --lang en-US        # run in English
  python main.py --once              # one command + confirmation then exit
  python main.py --list-devices      # list audio input devices
        """,
    )
    p.add_argument(
        "--lang", "--language",
        dest="language",
        choices=["pt-BR", "en-US", "pt", "en"],
        default=config.LANGUAGE,
        help=f"Active language (default: {config.LANGUAGE}).",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Listen for one command, request confirmation, print result as JSON, then exit.",
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
        "--confirm-timeout",
        type=float,
        default=config.CONFIRM_TIMEOUT_S,
        help=f"Confirmation timeout in seconds (default: {config.CONFIRM_TIMEOUT_S}).",
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

    # Normalize language flag
    lang = args.language
    if lang.lower().startswith("pt"):
        lang = "pt-BR"
    elif lang.lower().startswith("en"):
        lang = "en-US"
    config.LANGUAGE = lang

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
        print(f"Transcribing: {args.file} (lang={lang})")
        result = transcribe_file(args.file, language=lang)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # Live pipeline
    pipeline = VoiceCommandPipeline(
        command_timeout=args.timeout,
        confirm_timeout=args.confirm_timeout,
        language=lang,
    )

    if args.once:
        result = pipeline.listen_once()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
