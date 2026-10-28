"""
game/planner.py
===============
Converte um lance de xadrez nos movimentos do Gantry.

Reaproveita o grid logico (navigation/board.py) e o A* (navigation/pathfinder.py):
o carro so anda pelos corredores entre as casas e nunca atravessa uma peca.

Saida (`Plan`): tres listas do mesmo tamanho, no formato do comando "gantry":
    x[i], y[i]  : posicao absoluta (mm) do passo i
    eletroima[i]: 1 = ligado durante o deslocamento ate o passo i, 0 = desligado

Convencao: o ESP32 liga/desliga o eletroima ANTES de se mover ate o ponto i.
  - Pegar a peca  = ponto repetido (parado) com eletroima = 1.
  - Soltar a peca = ponto repetido (parado) com eletroima = 0.

Uso:
    planner = MovePlanner()
    plan = planner.plan(move)       # atualiza o estado interno
    ... envia plan.x, plan.y, plan.eletroima ao ESP32 ...
    planner.rollback()              # se o ESP32 falhar, volta ao estado anterior
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field

import chess

from . import config as cfg

sys.path.insert(0, str(cfg.ROOT / "navigation"))
from board import (BoardLogic, PIECE, EMPTY, LOGIC_ROWS, PHYS_ROWS,          # noqa: E402
                   COL_BOARD_START, COL_CEM_ESQ_START, COL_CEM_ESQ_END,
                   COL_CEM_DIR_START, COL_CEM_DIR_END)
from pathfinder import find_path_logical                                     # noqa: E402

# ── Layout fixo do cemiterio (mesmo de main.py) ──────────────────────────────
W_PAWN_COL = COL_CEM_ESQ_START
W_MAJ_COL  = COL_CEM_ESQ_START + 1
B_MAJ_COL  = COL_CEM_DIR_START
B_PAWN_COL = COL_CEM_DIR_START + 1
_MAJOR_ORDER = [chess.ROOK, chess.ROOK, chess.KNIGHT, chess.KNIGHT,
                chess.BISHOP, chess.BISHOP, chess.QUEEN, chess.KING]
W_MAJOR_SLOTS = [(pt, (row, W_MAJ_COL)) for row, pt in enumerate(_MAJOR_ORDER)]
B_MAJOR_SLOTS = [(pt, (row, B_MAJ_COL)) for row, pt in enumerate(_MAJOR_ORDER)]


class PlanningError(RuntimeError):
    """Nao foi possivel planejar o movimento (sem caminho, cemiterio cheio...)."""


@dataclass
class Plan:
    steps: list[tuple[float, float, int]] = field(default_factory=list)

    @property
    def x(self) -> list[float]:
        return [s[0] for s in self.steps]

    @property
    def y(self) -> list[float]:
        return [s[1] for s in self.steps]

    @property
    def eletroima(self) -> list[int]:
        return [s[2] for s in self.steps]


def square_to_cell(sq: int) -> tuple[int, int]:
    """Casa do xadrez -> celula logica global (linha, coluna)."""
    return 7 - chess.square_rank(sq), COL_BOARD_START + chess.square_file(sq)


class MovePlanner:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.board = chess.Board()
        self.cem_occ: dict[tuple[int, int], int] = {}
        self.robot_pos = cfg.HOME_CELL
        self.carrying = False
        self.logic = BoardLogic()
        self._steps: list[tuple[float, float, int]] = []
        self._snapshot = None
        self._rebuild()

    # ── estado ───────────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        self.logic = BoardLogic()
        for sq in chess.SQUARES:
            if self.board.piece_at(sq):
                r, c = square_to_cell(sq)
                self.logic.grid[r][c] = PIECE
        for (r, c) in self.cem_occ:
            self.logic.grid[r][c] = PIECE

    def rollback(self) -> None:
        """Desfaz o ultimo plan() (use quando o ESP32 nao concluir o movimento)."""
        if self._snapshot is None:
            return
        self.board, self.cem_occ, self.robot_pos, self.carrying = self._snapshot
        self._snapshot = None
        self._rebuild()

    # ── coordenadas ──────────────────────────────────────────────────────────

    @staticmethod
    def phys_to_mm(pr: int, pc: int) -> tuple[float, float]:
        half = cfg.SQUARE_MM / 2.0
        y_idx = (PHYS_ROWS - 1 - pr) if cfg.INVERT_Y else pr
        return pc * half, y_idx * half

    def cell_mm(self, cell: tuple[int, int]) -> tuple[float, float]:
        return self.phys_to_mm(cell[0] * 2 + 1, cell[1] * 2 + 1)

    # ── cemiterio ────────────────────────────────────────────────────────────

    def cem_slot_for(self, piece_type: int, captured_color: bool):
        white = captured_color == chess.WHITE
        pawn_col = W_PAWN_COL if white else B_PAWN_COL
        majors = W_MAJOR_SLOTS if white else B_MAJOR_SLOTS
        grid = self.logic.grid
        if piece_type == chess.PAWN:
            for row in range(LOGIC_ROWS):
                if grid[row][pawn_col] == EMPTY:
                    return (row, pawn_col)
        else:
            for pt, slot in majors:
                if pt == piece_type and grid[slot[0]][slot[1]] == EMPTY:
                    return slot
        # reserva: terceira coluna do cemiterio
        spare = COL_CEM_ESQ_END if white else COL_CEM_DIR_END
        for row in range(LOGIC_ROWS):
            if grid[row][spare] == EMPTY:
                return (row, spare)
        raise PlanningError("Cemiterio cheio.")

    def promotable_slot(self, piece_type: int, color: bool):
        majors = W_MAJOR_SLOTS if color == chess.WHITE else B_MAJOR_SLOTS
        for pt, slot in majors:
            if pt == piece_type and self.cem_occ.get(slot) == piece_type:
                return slot
        return None

    def is_phys_valid(self, move: chess.Move) -> bool:
        """Promocao so e possivel com uma peca ja capturada do tipo desejado."""
        if move.promotion is not None:
            return self.promotable_slot(move.promotion, self.board.turn) is not None
        return True

    def legal_physical_moves(self) -> list[chess.Move]:
        return [m for m in self.board.legal_moves if self.is_phys_valid(m)]

    # ── primitivas do robo ───────────────────────────────────────────────────

    def _emit(self, x: float, y: float, ima: int) -> None:
        step = (x, y, ima)
        if not self._steps or self._steps[-1] != step:
            self._steps.append(step)

    def _navigate(self, dest: tuple[int, int]) -> None:
        if self.robot_pos == dest:
            return
        result = find_path_logical(self.logic, self.robot_pos, dest)
        if not result:
            raise PlanningError(f"Sem caminho: {self.robot_pos} -> {dest}")
        pts = [self.phys_to_mm(*c) for c in result["path"]]
        ima = 1 if self.carrying else 0
        for i in range(1, len(pts)):
            if i < len(pts) - 1:       # descarta pontos no meio de uma reta
                a, b, c = pts[i - 1], pts[i], pts[i + 1]
                if (b[0] - a[0], b[1] - a[1]) == (c[0] - b[0], c[1] - b[1]):
                    continue
            self._emit(pts[i][0], pts[i][1], ima)
        self.robot_pos = dest

    def _pickup(self) -> None:
        r, c = self.robot_pos
        self.logic.grid[r][c] = EMPTY
        self.carrying = True
        x, y = self.cell_mm(self.robot_pos)
        self._emit(x, y, 1)

    def _release(self) -> None:
        r, c = self.robot_pos
        self.logic.grid[r][c] = PIECE
        self.carrying = False
        x, y = self.cell_mm(self.robot_pos)
        self._emit(x, y, 0)

    def _carry(self, src: tuple[int, int], dst: tuple[int, int]) -> None:
        self._navigate(src)
        self._pickup()
        self._navigate(dst)
        self._release()

    # ── planejamento ─────────────────────────────────────────────────────────

    def plan(self, move: chess.Move) -> Plan:
        """Gera os passos do Gantry para `move` e aplica o lance ao estado interno."""
        if move not in self.board.legal_moves:
            raise PlanningError(f"Lance ilegal: {move}")
        if not self.is_phys_valid(move):
            raise PlanningError("Promocao sem peca capturada disponivel no cemiterio.")

        self._snapshot = (self.board.copy(), dict(self.cem_occ), self.robot_pos, self.carrying)
        self._steps = []
        try:
            self._plan(move)
        except Exception:
            self.rollback()
            raise
        return Plan(list(self._steps))

    def _plan(self, move: chess.Move) -> None:
        cb = self.board
        color = cb.turn
        opp = chess.BLACK if color == chess.WHITE else chess.WHITE
        fr, to = square_to_cell(move.from_square), square_to_cell(move.to_square)
        to_piece = cb.piece_at(move.to_square)

        x, y = self.cell_mm(self.robot_pos)      # garante o ponto de partida (home)
        self._emit(x, y, 0)

        if cb.is_castling(move):
            kingside = chess.square_file(move.to_square) == 6
            rank = 0 if color == chess.WHITE else 7
            rook_from = chess.square(7 if kingside else 0, rank)
            rook_to = chess.square(5 if kingside else 3, rank)
            self._carry(fr, to)
            self._carry(square_to_cell(rook_from), square_to_cell(rook_to))

        elif move.promotion is not None:
            res_slot = self.promotable_slot(move.promotion, color)
            if cb.is_capture(move):
                cap_type = to_piece.piece_type if to_piece else chess.PAWN
                slot = self.cem_slot_for(cap_type, opp)
                self._carry(to, slot)
                self.cem_occ[slot] = cap_type
            pawn_slot = self.cem_slot_for(chess.PAWN, color)
            self._carry(fr, pawn_slot)
            self.cem_occ[pawn_slot] = chess.PAWN
            self._carry(res_slot, to)
            del self.cem_occ[res_slot]

        elif cb.is_en_passant(move):
            self._carry(fr, to)
            ep_sq = chess.square(chess.square_file(move.to_square),
                                 chess.square_rank(move.from_square))
            slot = self.cem_slot_for(chess.PAWN, opp)
            self._carry(square_to_cell(ep_sq), slot)
            self.cem_occ[slot] = chess.PAWN

        elif cb.is_capture(move):
            cap_type = to_piece.piece_type if to_piece else chess.PAWN
            slot = self.cem_slot_for(cap_type, opp)
            self._carry(to, slot)                 # 1) peca capturada vai ao cemiterio
            self.cem_occ[slot] = cap_type
            self._carry(fr, to)                   # 2) peca movente ocupa a casa

        else:
            self._carry(fr, to)

        self.board.push(move)
        self.carrying = False
        self._rebuild()
