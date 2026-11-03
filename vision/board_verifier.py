"""
vision/board_verifier.py
========================
Bridges the physical vision ``BoardState`` with the logical ``chess.Board``
for the two game-mode camera workflows:

1. **Post-Gantry Position Verification (PvE & Gantry moves in PvP)**:
   After the Gantry finishes executing a move, captures the board state and
   verifies that the physical pieces match the logical ``chess.Board``
   (checking the move's origin/destination squares as well as full-board
   consistency).

2. **Manual Move Detection & Validation ("MAGNUS VALIDATE" in PvP)**:
   When a player in manual mode physically moves a piece and says
   ``MAGNUS VALIDATE``, compares the new physical ``BoardState`` against the
   logical ``chess.Board`` (before the move), infers which move was played
   (quiet move, capture, en passant, castling, promotion), checks chess
   legality, and returns the spoken announcement + LCD lines.

Also manages shared ``GameSettings`` (``vision/game_settings.json``) and
``ValidationHistory`` (``vision/validation_history.json``) so the Web UI and
the main game loop stay synchronised during development and play.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import chess
import cv2
import numpy as np

from . import config as cfg
from .board_state import BoardState

log = logging.getLogger(__name__)

GAME_SETTINGS_FILE = Path(cfg.VISION_DIR) / "game_settings.json"
VALIDATION_HISTORY_FILE = Path(cfg.VISION_DIR) / "validation_history.json"
LOGICAL_BOARD_FILE = Path(cfg.VISION_DIR) / "logical_board_state.json"

PIECE_NAME_EN = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

PIECE_NAME_PT = {
    chess.PAWN: "Peão",
    chess.KNIGHT: "Cavalo",
    chess.BISHOP: "Bispo",
    chess.ROOK: "Torre",
    chess.QUEEN: "Rainha",
    chess.KING: "Rei",
}

STR_TO_PROMOTION = {
    "queen": chess.QUEEN,
    "rook": chess.ROOK,
    "bishop": chess.BISHOP,
    "knight": chess.KNIGHT,
}


def _spoken_sq(name: str) -> str:
    return " ".join(name.upper())


# ---------------------------------------------------------------------------
# GameSettings (shared between Web UI and main game)
# ---------------------------------------------------------------------------

@dataclass
class GameSettings:
    """
    Persistent configuration for game mode and camera validation.

    Attributes
    ----------
    game_mode : str
        "pve" (Player vs Computer) or "pvp" (Player vs Player).
    level : int
        Stockfish difficulty level (1–8) when in PvE mode.
    white_control : str
        "voice" (Voice + Gantry) or "manual" (Manual on board + MAGNUS VALIDATE).
    black_control : str
        In PvE: "computer".
        In PvP: "voice" (Voice + Gantry) or "manual" (Manual + MAGNUS VALIDATE).
    verify_gantry_moves : bool
        If True, captures a photo after every Gantry move to verify piece positions.
    diff_threshold : float
        MAD threshold for occupancy detection.
    """
    game_mode: str = "pve"
    level: int = 4
    white_control: str = "voice"
    black_control: str = "computer"
    verify_gantry_moves: bool = True
    diff_threshold: float = cfg.OCCUPIED_DIFF_THRESHOLD

    def normalize(self) -> "GameSettings":
        self.game_mode = "pvp" if str(self.game_mode).lower() == "pvp" else "pve"
        self.level = max(1, min(8, int(self.level or 4)))
        if self.game_mode == "pve":
            # In PvE, White is the human player (default voice/gantry, or manual) and Black is computer
            self.white_control = "manual" if str(self.white_control).lower() == "manual" else "voice"
            self.black_control = "computer"
        else:
            self.white_control = "manual" if str(self.white_control).lower() == "manual" else "voice"
            self.black_control = "manual" if str(self.black_control).lower() == "manual" else "voice"
        self.verify_gantry_moves = bool(self.verify_gantry_moves)
        self.diff_threshold = float(self.diff_threshold or cfg.OCCUPIED_DIFF_THRESHOLD)
        return self

    def control_for(self, color: bool) -> str:
        """Return control mode ('voice', 'manual', or 'computer') for chess.WHITE / chess.BLACK."""
        self.normalize()
        if color == chess.WHITE:
            return self.white_control
        return "computer" if self.game_mode == "pve" else self.black_control

    def to_dict(self) -> dict:
        self.normalize()
        return asdict(self)

    def save(self, path: str | Path = GAME_SETTINGS_FILE) -> None:
        self.normalize()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path = GAME_SETTINGS_FILE) -> "GameSettings":
        p = Path(path)
        if not p.is_file():
            return cls().normalize()
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            allowed = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**allowed).normalize()
        except Exception as exc:
            log.warning("[GameSettings] Failed to load %s (%s) — using defaults.", p, exc)
            return cls().normalize()


# ---------------------------------------------------------------------------
# Validation Results
# ---------------------------------------------------------------------------

@dataclass
class PositionVerificationResult:
    """Result of verifying the physical board after a Gantry move (PvE / Gantry turn)."""
    success: bool
    move_uci: Optional[str] = None
    origin_ok: bool = True
    dest_ok: bool = True
    color_ok: bool = True
    mismatches: list[dict] = field(default_factory=list)
    matching_squares: int = 64
    confidence: float = 1.0
    message: str = ""
    speech: str = ""
    lcd_lines: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "kind": "post_gantry",
            "success": self.success,
            "move_uci": self.move_uci,
            "origin_ok": self.origin_ok,
            "dest_ok": self.dest_ok,
            "color_ok": self.color_ok,
            "mismatches": self.mismatches,
            "matching_squares": self.matching_squares,
            "confidence": round(self.confidence, 3),
            "message": self.message,
            "speech": self.speech,
            "lcd_lines": list(self.lcd_lines),
        }


@dataclass
class ManualValidationResult:
    """Result of validating a manual human move when the player says 'MAGNUS VALIDATE'."""
    valid: bool
    move: Optional[chess.Move] = None
    move_uci: Optional[str] = None
    san: Optional[str] = None
    from_square: Optional[str] = None
    to_square: Optional[str] = None
    piece_name: Optional[str] = None
    captured_piece: Optional[str] = None
    error_code: Optional[str] = None
    message: str = ""
    speech: str = ""
    lcd_lines: tuple[str, ...] = ()
    changed_squares: list[dict] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "kind": "manual_validate",
            "valid": self.valid,
            "success": self.valid,
            "move_uci": self.move_uci,
            "san": self.san,
            "from_square": self.from_square,
            "to_square": self.to_square,
            "piece_name": self.piece_name,
            "captured_piece": self.captured_piece,
            "error_code": self.error_code,
            "message": self.message,
            "speech": self.speech,
            "lcd_lines": list(self.lcd_lines),
            "changed_squares": self.changed_squares,
            "confidence": round(self.confidence, 3),
        }


# ---------------------------------------------------------------------------
# Core Verification & Manual Move Inference Logic
# ---------------------------------------------------------------------------

def verify_board_position(
    board: chess.Board,
    state: BoardState,
    last_move: Optional[chess.Move] = None,
    max_non_move_mismatches: int = 2,
) -> PositionVerificationResult:
    """
    Compare the physical ``state`` captured by the camera against the logical
    ``board`` (after ``last_move`` has been pushed to ``board``).

    Checks:
    1. If ``last_move`` is provided:
       * ``from_square`` must be physically empty.
       * ``to_square`` must be physically occupied by the colour that lies on
         ``to_square`` in ``board``.
    2. Across all 64 squares: counts any mismatches between ``board`` and ``state``.
    """
    mismatches: list[dict] = []
    origin_ok = True
    dest_ok = True
    color_ok = True

    from_name = chess.square_name(last_move.from_square).upper() if last_move else None
    to_name = chess.square_name(last_move.to_square).upper() if last_move else None

    for sq in chess.SQUARES:
        sq_name = chess.square_name(sq).upper()
        logical_piece = board.piece_at(sq)
        exp_occ = logical_piece is not None
        exp_color = (
            ("white" if logical_piece.color == chess.WHITE else "black")
            if logical_piece else None
        )

        phys = state.board.get(sq_name)
        obs_occ = phys.occupied if phys else False
        obs_color = phys.color if (phys and obs_occ) else None

        if exp_occ != obs_occ or (exp_occ and exp_color != obs_color):
            mismatches.append({
                "square": sq_name,
                "expected": exp_color if exp_occ else "empty",
                "observed": obs_color if obs_occ else "empty",
                "is_move_square": sq_name in (from_name, to_name),
            })
            if sq_name == from_name and obs_occ:
                origin_ok = False
            elif sq_name == to_name:
                if not obs_occ:
                    dest_ok = False
                elif obs_color != exp_color:
                    color_ok = False

    non_move_mismatches = [m for m in mismatches if not m["is_move_square"]]
    move_squares_ok = origin_ok and dest_ok and color_ok
    success = move_squares_ok and (len(non_move_mismatches) <= max_non_move_mismatches)
    matching_squares = 64 - len(mismatches)
    move_uci = last_move.uci().upper() if last_move else None

    if success:
        if last_move:
            msg = f"Posição confirmada após {from_name}→{to_name} ({matching_squares}/64 casas conferem)."
            lcd = ("CAMERA VERIFIED", f"MOVE {from_name}->{to_name} OK", f"MATCH: {matching_squares}/64", "")
        else:
            msg = f"Tabuleiro verificado com sucesso ({matching_squares}/64 casas conferem)."
            lcd = ("CAMERA VERIFIED", "BOARD POSITION OK", f"MATCH: {matching_squares}/64", "")
        speech = ""
    else:
        issues = []
        if not origin_ok and from_name:
            issues.append(f"origem {from_name} continua ocupada")
        if not dest_ok and to_name:
            issues.append(f"destino {to_name} está vazio")
        elif not color_ok and to_name:
            issues.append(f"cor incorreta em {to_name}")
        if non_move_mismatches:
            sq_list = ", ".join(m["square"] for m in non_move_mismatches[:5])
            issues.append(f"divergência em {sq_list}")
        msg = "Divergência na verificação: " + "; ".join(issues) + "."
        speech = "Warning. Board position mismatch detected. Please check the pieces."
        bad_sq = mismatches[0]["square"] if mismatches else "BOARD"
        lcd = ("CAMERA WARNING!", f"CHECK SQUARE {bad_sq}", f"MATCH: {matching_squares}/64", "CHECK PIECES")

    return PositionVerificationResult(
        success=success,
        move_uci=move_uci,
        origin_ok=origin_ok,
        dest_ok=dest_ok,
        color_ok=color_ok,
        mismatches=mismatches,
        matching_squares=matching_squares,
        confidence=state.overall_confidence,
        message=msg,
        speech=speech,
        lcd_lines=lcd,
    )


def detect_and_validate_manual_move(
    board: chess.Board,
    state: BoardState,
    is_phys_valid_fn: Optional[Callable[[chess.Move], bool]] = None,
) -> ManualValidationResult:
    """
    Infer and validate a manual chess move by comparing the logical ``board``
    (before the move) with the camera's physical ``state`` (after the move).

    Handles:
    * Standard quiet moves (1 own piece vacated -> 1 empty square now occupied by own colour)
    * Captures (1 own piece vacated -> 1 opponent square now occupied by own colour)
    * En passant (1 own pawn vacated + 1 empty square occupied + 1 opponent pawn vacated)
    * Castling (King + Rook vacated -> King + Rook destination squares occupied)
    * Pawn promotion (uses classified piece type on destination if available, else Queen)
    """
    turn_color = "white" if board.turn == chess.WHITE else "black"
    opp_color = "black" if turn_color == "white" else "white"
    turn_label_en = "White" if turn_color == "white" else "Black"
    turn_label_pt = "Brancas" if turn_color == "white" else "Pretas"

    vacated_own: list[str] = []
    vacated_opp: list[str] = []
    entered_empty_own: list[str] = []
    replaced_opp_by_own: list[str] = []
    unexpected_changes: list[dict] = []
    changed_squares: list[dict] = []

    for sq in chess.SQUARES:
        sq_name = chess.square_name(sq).upper()
        p = board.piece_at(sq)
        exp_occ = p is not None
        exp_color = ("white" if p.color == chess.WHITE else "black") if p else None

        phys = state.board.get(sq_name)
        obs_occ = phys.occupied if phys else False
        obs_color = phys.color if (phys and obs_occ) else None
        obs_piece = phys.piece if (phys and obs_occ) else None

        if exp_occ == obs_occ and (not exp_occ or exp_color == obs_color):
            continue

        change_entry = {
            "square": sq_name,
            "before": exp_color if exp_occ else "empty",
            "after": obs_color if obs_occ else "empty",
            "detected_piece": obs_piece,
        }
        changed_squares.append(change_entry)

        if exp_occ and exp_color == turn_color and not obs_occ:
            vacated_own.append(sq_name)
        elif exp_occ and exp_color == opp_color and not obs_occ:
            vacated_opp.append(sq_name)
        elif not exp_occ and obs_occ and obs_color == turn_color:
            entered_empty_own.append(sq_name)
        elif exp_occ and exp_color == opp_color and obs_occ and obs_color == turn_color:
            replaced_opp_by_own.append(sq_name)
        else:
            unexpected_changes.append(change_entry)

    # ── 1. No move detected at all ────────────────────────────────────
    if not changed_squares:
        return ManualValidationResult(
            valid=False,
            error_code="NO_MOVE_DETECTED",
            message=f"Nenhum movimento detectado no tabuleiro (vez das {turn_label_pt}).",
            speech=f"Invalid move. No move detected on the board. It is {turn_label_en}'s turn.",
            lcd_lines=("INVALID MOVE!", "NO MOVE DETECTED", f"TURN: {turn_label_en.upper()}", "MOVE & VALIDATE"),
            changed_squares=[],
            confidence=state.overall_confidence,
        )

    # ── 2. Check if opponent's piece was moved instead ────────────────
    if not vacated_own and vacated_opp:
        return ManualValidationResult(
            valid=False,
            error_code="WRONG_SIDE_MOVED",
            message=f"Lance inválido: é a vez das {turn_label_pt}, mas uma peça adversária foi movida ({', '.join(vacated_opp)}).",
            speech=f"Invalid move. It is {turn_label_en}'s turn, but an opponent piece was moved.",
            lcd_lines=("INVALID MOVE!", f"TURN: {turn_label_en.upper()}", "WRONG COLOR MOVED", "FIX & VALIDATE"),
            changed_squares=changed_squares,
            confidence=state.overall_confidence,
        )

    # ── 3. Score all legal moves against the observed board state ─────
    # In addition to exact transition matching, scoring legal moves makes
    # castling, en passant, and single-square transitions rock-solid.
    from_sq_cand: Optional[str] = None
    to_sq_cand: Optional[str] = None

    # Case A: Castling (2 own pieces vacated, 2 empty squares entered)
    if len(vacated_own) == 2 and len(entered_empty_own) == 2 and not replaced_opp_by_own and not vacated_opp and not unexpected_changes:
        king_home = "E1" if turn_color == "white" else "E8"
        if king_home in vacated_own:
            for dst in ("G1", "C1", "G8", "C8"):
                if dst in entered_empty_own:
                    from_sq_cand = king_home
                    to_sq_cand = dst
                    break

    # Case B: Quiet move (1 own vacated -> 1 empty entered)
    elif len(vacated_own) == 1 and len(entered_empty_own) == 1 and not replaced_opp_by_own and not vacated_opp and not unexpected_changes:
        from_sq_cand = vacated_own[0]
        to_sq_cand = entered_empty_own[0]

    # Case C: Standard capture (1 own vacated -> 1 opp replaced by own)
    elif len(vacated_own) == 1 and len(replaced_opp_by_own) == 1 and not entered_empty_own and not vacated_opp and not unexpected_changes:
        from_sq_cand = vacated_own[0]
        to_sq_cand = replaced_opp_by_own[0]

    # Case D: En passant capture (1 own vacated, 1 empty entered, 1 opp pawn vacated)
    elif len(vacated_own) == 1 and len(entered_empty_own) == 1 and len(vacated_opp) == 1 and not replaced_opp_by_own and not unexpected_changes:
        from_sq_cand = vacated_own[0]
        to_sq_cand = entered_empty_own[0]

    # Case E: Fallback — if 1 own vacated and 1 destination (even with 1 minor noise square),
    # check if there is a unique legal move matching (vacated_own, entered_empty_own + replaced_opp_by_own)
    if from_sq_cand is None and len(vacated_own) == 1:
        dests = entered_empty_own + replaced_opp_by_own
        if len(dests) == 1 and len(unexpected_changes) == 0:
            from_sq_cand = vacated_own[0]
            to_sq_cand = dests[0]

    if from_sq_cand is None or to_sq_cand is None:
        sq_summary = ", ".join(f"{c['square']}({c['before']}->{c['after']})" for c in changed_squares[:4])
        return ManualValidationResult(
            valid=False,
            error_code="AMBIGUOUS_BOARD_CHANGE",
            message=f"Lance inválido: alterações no tabuleiro não formam um único lance ({sq_summary}).",
            speech="Invalid move. Multiple or unrecognized piece changes on the board.",
            lcd_lines=("INVALID MOVE!", "CHECK BOARD PIECES", sq_summary[:20], "FIX & VALIDATE"),
            changed_squares=changed_squares,
            confidence=state.overall_confidence,
        )

    # ── 4. Check legality of (from_sq_cand -> to_sq_cand) in chess.Board ──
    from_idx = chess.parse_square(from_sq_cand.lower())
    to_idx = chess.parse_square(to_sq_cand.lower())
    moving_piece = board.piece_at(from_idx)
    piece_en = PIECE_NAME_EN.get(moving_piece.piece_type, "piece") if moving_piece else "piece"
    piece_pt = PIECE_NAME_PT.get(moving_piece.piece_type, "Peça") if moving_piece else "Peça"

    # Determine promotion if pawn reaches last rank
    candidates: list[chess.Move] = []
    if moving_piece and moving_piece.piece_type == chess.PAWN and chess.square_rank(to_idx) in (0, 7):
        dest_phys = state.board.get(to_sq_cand)
        det_promo = STR_TO_PROMOTION.get(dest_phys.piece if dest_phys else "", None)
        promo_order = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        if det_promo in promo_order:
            promo_order.remove(det_promo)
            promo_order.insert(0, det_promo)
        candidates = [chess.Move(from_idx, to_idx, promotion=p) for p in promo_order]
    else:
        candidates = [chess.Move(from_idx, to_idx)]

    matched_move: Optional[chess.Move] = None
    for cand in candidates:
        if cand in board.legal_moves:
            if is_phys_valid_fn is None or is_phys_valid_fn(cand):
                matched_move = cand
                break

    if matched_move is None:
        # Also check if the move is pseudo-legal or illegal because of check!
        base_move = chess.Move(from_idx, to_idx)
        if board.is_into_check(base_move):
            reason_pt = f"Rei das {turn_label_pt} ficaria em xeque ({piece_pt} {from_sq_cand}→{to_sq_cand})."
            reason_en = f"Invalid move. {piece_en.capitalize()} {_spoken_sq(from_sq_cand)} to {_spoken_sq(to_sq_cand)} leaves your king in check."
            lcd_err = "KING IN CHECK!"
        else:
            reason_pt = f"Movimento ilegal para {piece_pt}: {from_sq_cand} → {to_sq_cand}."
            reason_en = f"Invalid move. {piece_en.capitalize()} {_spoken_sq(from_sq_cand)} to {_spoken_sq(to_sq_cand)} is not legal."
            lcd_err = "NOT LEGAL ON BOARD"

        return ManualValidationResult(
            valid=False,
            from_square=from_sq_cand,
            to_square=to_sq_cand,
            piece_name=piece_en,
            error_code="ILLEGAL_MOVE",
            message=f"Lance inválido: {reason_pt}",
            speech=reason_en,
            lcd_lines=(
                "INVALID MOVE!",
                f"{piece_en.upper()} {from_sq_cand} -> {to_sq_cand}",
                lcd_err,
                "FIX & VALIDATE",
            ),
            changed_squares=changed_squares,
            confidence=state.overall_confidence,
        )

    # ── 5. Valid legal move! Build announcement & SAN ─────────────────
    san_str = board.san(matched_move)
    captured_en: Optional[str] = None
    captured_pt: Optional[str] = None
    cap_sq_name = to_sq_cand

    if board.is_en_passant(matched_move):
        captured_en = "pawn"
        captured_pt = "Peão"
        ep_sq = chess.square(chess.square_file(to_idx), chess.square_rank(from_idx))
        cap_sq_name = chess.square_name(ep_sq).upper()
    elif board.is_capture(matched_move):
        cap_p = board.piece_at(to_idx)
        if cap_p:
            captured_en = PIECE_NAME_EN.get(cap_p.piece_type, "piece")
            captured_pt = PIECE_NAME_PT.get(cap_p.piece_type, "Peça")

    if board.is_castling(matched_move):
        side_str = "kingside" if chess.square_file(to_idx) == 6 else "queenside"
        speech = f"Valid move. {turn_label_en} castles {side_str}, {_spoken_sq(from_sq_cand)} to {_spoken_sq(to_sq_cand)}."
        msg = f"Lance válido! Roque ({san_str}): {piece_pt} {from_sq_cand} → {to_sq_cand}."
        lcd = (
            "VALID MOVE!",
            f"CASTLING ({san_str})",
            f"KING {from_sq_cand} -> {to_sq_cand}",
            f"{turn_label_en.upper()} OK",
        )
    elif captured_en:
        speech = (
            f"Valid move. Capturing {captured_en} on {_spoken_sq(cap_sq_name)}, "
            f"moving {piece_en} from {_spoken_sq(from_sq_cand)} to {_spoken_sq(to_sq_cand)}."
        )
        msg = f"Lance válido! {piece_pt} {from_sq_cand} × {to_sq_cand} (capturou {captured_pt}, SAN: {san_str})."
        lcd = (
            "VALID MOVE!",
            f"CAPTURE {captured_en.upper()}",
            f"{piece_en.upper()} {from_sq_cand} -> {to_sq_cand}",
            f"SAN: {san_str}",
        )
    else:
        speech = f"Valid move. Moving {piece_en} from {_spoken_sq(from_sq_cand)} to {_spoken_sq(to_sq_cand)}."
        msg = f"Lance válido! {piece_pt} de {from_sq_cand} para {to_sq_cand} (SAN: {san_str})."
        lcd = (
            "VALID MOVE!",
            f"{piece_en.upper()} {from_sq_cand} -> {to_sq_cand}",
            f"SAN: {san_str}",
            f"{turn_label_en.upper()} OK",
        )

    if matched_move.promotion:
        promo_en = PIECE_NAME_EN.get(matched_move.promotion, "queen")
        speech += f" Promoting to {promo_en}."

    return ManualValidationResult(
        valid=True,
        move=matched_move,
        move_uci=matched_move.uci().upper(),
        san=san_str,
        from_square=from_sq_cand,
        to_square=to_sq_cand,
        piece_name=piece_en,
        captured_piece=captured_en,
        error_code=None,
        message=msg,
        speech=speech,
        lcd_lines=lcd,
        changed_squares=changed_squares,
        confidence=state.overall_confidence,
    )


# ---------------------------------------------------------------------------
# Annotated visual overlay for Web UI & Diagnostics
# ---------------------------------------------------------------------------

def annotate_validation_image(
    warped: np.ndarray,
    state: BoardState,
    orientation: str = "normal",
    manual_result: Optional[ManualValidationResult] = None,
    verify_result: Optional[PositionVerificationResult] = None,
) -> np.ndarray:
    """
    Draw a clear visual overlay on a perspective-corrected board image
    showing detected pieces, centers, and validation status per square.
    """
    from .segmentation import BoardSegmenter, CEM_W, BOARD_SZ

    sq_sz = BOARD_SZ // 8
    cols = "ABCDEFGH"
    rows = "87654321"
    type_letter = {
        "pawn": "P", "rook": "R", "knight": "N",
        "bishop": "B", "queen": "Q", "king": "K",
    }

    def sq_xy(name: str) -> tuple[int, int]:
        c, r = cols.index(name[0]), rows.index(name[1])
        if orientation == "flipped":
            c, r = 7 - c, 7 - r
        return CEM_W + c * sq_sz, r * sq_sz

    annotated = BoardSegmenter(board_orientation=orientation).draw_grid(warped)

    # 1. Draw subtle indicators for all occupied squares
    for name, sq_st in state.board.items():
        if not sq_st.occupied:
            continue
        x0, y0 = sq_xy(name)
        tag = (sq_st.color or "?")[0].upper() + type_letter.get(sq_st.piece, "?")
        cv2.rectangle(annotated, (x0 + 4, y0 + 4), (x0 + 36, y0 + 24), (15, 23, 42), -1)
        cv2.putText(
            annotated, tag, (x0 + 6, y0 + 19),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 2,
        )

    # 2. Highlight ManualValidationResult (from_square -> to_square)
    if manual_result is not None:
        if manual_result.from_square and manual_result.to_square:
            fx, fy = sq_xy(manual_result.from_square)
            tx, ty = sq_xy(manual_result.to_square)
            main_color = (0, 230, 118) if manual_result.valid else (0, 60, 255)
            # Origin box (blue/cyan)
            cv2.rectangle(annotated, (fx + 3, fy + 3), (fx + sq_sz - 3, fy + sq_sz - 3), (255, 190, 0), 3)
            # Destination box (green if valid, red if invalid)
            cv2.rectangle(annotated, (tx + 3, ty + 3), (tx + sq_sz - 3, ty + sq_sz - 3), main_color, 4)
            # Arrow from origin center to destination center
            p1 = (fx + sq_sz // 2, fy + sq_sz // 2)
            p2 = (tx + sq_sz // 2, ty + sq_sz // 2)
            cv2.arrowedLine(annotated, p1, p2, main_color, 3, tipLength=0.22)
        else:
            for ch in manual_result.changed_squares:
                cx0, cy0 = sq_xy(ch["square"])
                cv2.rectangle(annotated, (cx0 + 3, cy0 + 3), (cx0 + sq_sz - 3, cy0 + sq_sz - 3), (0, 60, 255), 3)

    # 3. Highlight PositionVerificationResult mismatches
    if verify_result is not None:
        for m in verify_result.mismatches:
            mx, my = sq_xy(m["square"])
            cv2.rectangle(annotated, (mx + 3, my + 3), (mx + sq_sz - 3, my + sq_sz - 3), (0, 40, 255), 4)
            cv2.putText(
                annotated, f"EXP:{m['expected'][:1].upper()}",
                (mx + 5, my + sq_sz - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 40, 255), 2,
            )

    return annotated


# ---------------------------------------------------------------------------
# Validation History & Logical Board Persistence (for Web UI live monitoring)
# ---------------------------------------------------------------------------

def record_validation_event(event: dict, max_entries: int = 30) -> None:
    """Append a validation event to ``vision/validation_history.json``."""
    entry = {
        "timestamp": time.strftime("%H:%M:%S"),
        "epoch_ms": int(time.time() * 1000),
        **event,
    }
    history: list[dict] = []
    if VALIDATION_HISTORY_FILE.is_file():
        try:
            with open(VALIDATION_HISTORY_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    history = loaded
        except Exception:
            history = []
    history.insert(0, entry)
    history = history[:max_entries]
    try:
        with open(VALIDATION_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as exc:
        log.debug("[BoardVerifier] Could not write validation history: %s", exc)


def load_validation_history() -> list[dict]:
    if not VALIDATION_HISTORY_FILE.is_file():
        return []
    try:
        with open(VALIDATION_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def clear_validation_history() -> None:
    if VALIDATION_HISTORY_FILE.is_file():
        try:
            VALIDATION_HISTORY_FILE.unlink()
        except Exception:
            pass


def save_logical_board_snapshot(
    board: chess.Board,
    last_move_uci: Optional[str] = None,
    status_text: str = "",
) -> None:
    """Save the current logical board FEN so the Web UI can mirror an active game."""
    payload = {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "fullmove_number": board.fullmove_number,
        "is_check": board.is_check(),
        "is_game_over": board.is_game_over(),
        "last_move_uci": last_move_uci,
        "status_text": status_text,
        "updated_at": time.strftime("%H:%M:%S"),
    }
    try:
        with open(LOGICAL_BOARD_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        log.debug("[BoardVerifier] Could not save logical board snapshot: %s", exc)


def load_logical_board_snapshot() -> Optional[dict]:
    if not LOGICAL_BOARD_FILE.is_file():
        return None
    try:
        with open(LOGICAL_BOARD_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

