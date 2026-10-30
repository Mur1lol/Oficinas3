"""
game/phrases.py
===============
Textos falados no speaker e mostrados no LCD (16x2), em ingles.
"""

from __future__ import annotations

import chess

PIECE_NAME = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}


def spoken_square(name: str) -> str:
    """'E2' -> 'E 2' (o TTS le melhor separado)."""
    return " ".join(name.upper())


def _piece_at(board: chess.Board, square: int) -> str | None:
    p = board.piece_at(square)
    return PIECE_NAME[p.piece_type] if p else None


def move_info(board: chess.Board, move: chess.Move) -> dict:
    """Dados do lance (antes de ser jogado) usados nas frases."""
    frm, to = chess.square_name(move.from_square).upper(), chess.square_name(move.to_square).upper()
    captured = None
    cap_square = to
    if board.is_en_passant(move):
        captured = "pawn"
        cap_square = chess.square_name(
            chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))).upper()
    elif board.is_capture(move):
        captured = _piece_at(board, move.to_square)
    return {
        "piece": _piece_at(board, move.from_square) or "piece",
        "from": frm, "to": to,
        "captured": captured, "cap_square": cap_square,
        "promotion": PIECE_NAME.get(move.promotion) if move.promotion else None,
    }


def announce(board: chess.Board, move: chess.Move) -> tuple[str, tuple[str, ...]]:
    """Frase falada + 4 linhas de LCD ao EXECUTAR o lance."""
    i = move_info(board, move)
    moving = f"moving {i['piece']} {spoken_square(i['from'])} to {spoken_square(i['to'])}"
    if i["captured"]:
        speech = f"capturing {i['captured']} {spoken_square(i['cap_square'])} and {moving}"
        lcd = (
            "EXECUTING MOVE",
            f"CAPTURE {i['captured'].upper()}",
            f"{i['piece'].upper()} {i['from']} -> {i['to']}",
            "GANTRY MOVING...",
        )
    else:
        speech = moving
        lcd = (
            "EXECUTING MOVE",
            f"{i['piece'].upper()} {i['from']} -> {i['to']}",
            "GANTRY MOVING...",
            "PLEASE WAIT",
        )
    if i["promotion"]:
        speech += f" and promoting to {i['promotion']}"
        lcd = (
            "PROMOTING PAWN",
            f"{i['from']} -> {i['to']}={i['promotion'][:1].upper()}",
            "GANTRY MOVING...",
            "PLEASE WAIT",
        )
    return speech, lcd


def confirm_move(board: chess.Board, from_sq: str, to_sq: str) -> tuple[str, tuple[str, ...]]:
    """Frase falada + 4 linhas de LCD para o jogador confirmar o lance (SIM/NAO)."""
    piece = None
    try:
        piece = _piece_at(board, chess.parse_square(from_sq.lower()))
    except ValueError:
        pass
    what = f"{piece} " if piece else ""
    piece_str = piece.upper() if piece else "PIECE"
    speech = f"move {what}{spoken_square(from_sq)} to {spoken_square(to_sq)}. Say yes or no."
    lcd = (
        "CONFIRM MOVE?",
        f"{piece_str} {from_sq.upper()} -> {to_sq.upper()}",
        "SAY: YES OR NO",
        "",
    )
    return speech, lcd


def game_result(board: chess.Board) -> tuple[str, tuple[str, str]]:
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"Checkmate. {winner} wins.", ("CHECKMATE", f"{winner.upper()} WINS")
    if board.is_stalemate():
        return "Stalemate. The game is a draw.", ("STALEMATE", "DRAW")
    return "The game is a draw.", ("GAME OVER", "DRAW")
