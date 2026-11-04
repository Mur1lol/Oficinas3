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
from board import (BoardLogic, PIECE, EMPTY, LOGIC_ROWS, PHYS_ROWS, PHYS_COLS,  # noqa: E402
                   COL_BOARD_START, COL_BOARD_END, COL_CEM_DIR_PEAO, COL_CEM_DIR_NOBRE,
                   COL_CEM_ESQ_NOBRE, COL_CEM_ESQ_PEAO)
from pathfinder import find_path_logical, find_path_physical                     # noqa: E402

# ── Layout das pecas nos cemiterios conforme mapeamento fisico ───────────────
# Coluna 0: peoes cemiterio direito
# Coluna 1: torre, cavalo, bispo, rei, rainha, bispo, cavalo, torre
# Coluna 12: torre, cavalo, bispo, rei, rainha, bispo, cavalo, torre
# Coluna 13: peoes cemiterio esquerdo
_MAJOR_ORDER = [
    chess.ROOK,
    chess.KNIGHT,
    chess.BISHOP,
    chess.KING,
    chess.QUEEN,
    chess.BISHOP,
    chess.KNIGHT,
    chess.ROOK,
]
R_MAJOR_SLOTS = [(pt, (row, COL_CEM_DIR_NOBRE)) for row, pt in enumerate(_MAJOR_ORDER)]
L_MAJOR_SLOTS = [(pt, (row, COL_CEM_ESQ_NOBRE)) for row, pt in enumerate(_MAJOR_ORDER)]


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
    """
    Casa do xadrez -> celula logica global (linha, coluna).

    Linha 0 = Rank 8 (H8..A8)
    Linha 7 = Rank 1 (H1..A1)
    Coluna 3 = File H (H8..H1)
    Coluna 10 = File A (A8..A1)
    """
    return 7 - chess.square_rank(sq), 10 - chess.square_file(sq)


class MovePlanner:
    def __init__(
        self,
        return_home: bool = True,
        overshoot_mm: float = cfg.RELEASE_OVERSHOOT_MM,
    ) -> None:
        self.return_home = return_home
        self.overshoot_mm = overshoot_mm
        self.reset()

    @staticmethod
    def home_phys() -> tuple[int, int]:
        """Coordenada fisica (pr, pc) correspondente a (0.0, 0.0) mm no Top-Right (Direita em cima)."""
        return 0, 0

    def reset(self) -> None:
        self.board = chess.Board()
        self.cem_occ: dict[tuple[int, int], int] = {}
        self.robot_phys = self.home_phys()
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
        self.board, self.cem_occ, self.robot_phys, self.carrying = self._snapshot
        self._snapshot = None
        self._rebuild()

    def apply_manual_move(self, move: chess.Move) -> None:
        """
        Aplica um lance feito manualmente pelo jogador (validado pela camera)
        ao estado logico do tabuleiro e do cemiterio, sem mover o Gantry.
        """
        if move not in self.board.legal_moves:
            raise PlanningError(f"Lance manual ilegal: {move}")
        self._snapshot = (self.board.copy(), dict(self.cem_occ), self.robot_phys, self.carrying)
        cb = self.board
        color = cb.turn
        opp = chess.BLACK if color == chess.WHITE else chess.WHITE
        to_piece = cb.piece_at(move.to_square)

        if move.promotion is not None:
            if cb.is_capture(move):
                cap_type = to_piece.piece_type if to_piece else chess.PAWN
                try:
                    slot = self.cem_slot_for(cap_type, opp)
                    self.cem_occ[slot] = cap_type
                except PlanningError:
                    pass
            try:
                pawn_slot = self.cem_slot_for(chess.PAWN, color)
                self.cem_occ[pawn_slot] = chess.PAWN
            except PlanningError:
                pass
            res_slot = self.promotable_slot(move.promotion, color)
            if res_slot is not None and res_slot in self.cem_occ:
                del self.cem_occ[res_slot]
        elif cb.is_en_passant(move):
            try:
                slot = self.cem_slot_for(chess.PAWN, opp)
                self.cem_occ[slot] = chess.PAWN
            except PlanningError:
                pass
        elif cb.is_capture(move):
            cap_type = to_piece.piece_type if to_piece else chess.PAWN
            try:
                slot = self.cem_slot_for(cap_type, opp)
                self.cem_occ[slot] = cap_type
            except PlanningError:
                pass

        self.board.push(move)
        self.carrying = False
        self._rebuild()

    # ── coordenadas ──────────────────────────────────────────────────────────

    @staticmethod
    def phys_to_mm(pr: int, pc: int) -> tuple[float, float]:
        """
        Converte celula fisica (pr, pc) no grid 15x27 para mm (X, Y) do ESP32.

        pr: 0..14 (0 = topo/em cima/Pretas -> 14 = base/em baixo/Brancas). Eixo X (0..350 mm).
        pc: 0..26 (0 = direita/cem_dir -> 26 = esquerda/cem_esq). Eixo Y (0..650 mm).
        Cada celula mede 25 mm x 25 mm (2.5 cm x 2.5 cm).
        Home (0.0, 0.0) fica exatamente em (pr=0, pc=0).
        """
        x_mm = pr * cfg.CELL_MM
        y_mm = pc * cfg.CELL_MM
        return x_mm, y_mm

    def cell_mm(self, cell: tuple[int, int]) -> tuple[float, float]:
        """Coordenada logica (row, col) no grid 14x8 -> (X, Y) em mm."""
        return self.phys_to_mm(cell[0] * 2, cell[1] * 2)

    # ── cemiterio ────────────────────────────────────────────────────────────

    def cem_slot_for(self, piece_type: int, captured_color: bool) -> tuple[int, int]:
        # Colunas 0 e 1 (cemiterio direito): pecas pretas
        # Colunas 12 e 13 (cemiterio esquerdo): pecas brancas
        black = (captured_color == chess.BLACK)
        pawn_col = COL_CEM_DIR_PEAO if black else COL_CEM_ESQ_PEAO
        majors = R_MAJOR_SLOTS if black else L_MAJOR_SLOTS
        grid = self.logic.grid
        if piece_type == chess.PAWN:
            for row in range(LOGIC_ROWS):
                if grid[row][pawn_col] == EMPTY:
                    return (row, pawn_col)
        else:
            for pt, slot in majors:
                if pt == piece_type and grid[slot[0]][slot[1]] == EMPTY:
                    return slot
            for pt, slot in majors:
                if grid[slot[0]][slot[1]] == EMPTY:
                    return slot
        for row in range(LOGIC_ROWS):
            if grid[row][pawn_col] == EMPTY:
                return (row, pawn_col)
        raise PlanningError("Cemiterio cheio.")

    def promotable_slot(self, piece_type: int, color: bool):
        # Brancas no cemiterio esquerdo (L_MAJOR_SLOTS), pretas no direito (R_MAJOR_SLOTS)
        majors = L_MAJOR_SLOTS if color == chess.WHITE else R_MAJOR_SLOTS
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

    def _navigate_phys(self, dest_phys: tuple[int, int]) -> None:
        if self.robot_phys == dest_phys:
            return
        result = find_path_physical(self.logic, self.robot_phys, dest_phys)
        if not result:
            raise PlanningError(f"Sem caminho fisico: {self.robot_phys} -> {dest_phys}")
        pts = [self.phys_to_mm(*c) for c in result["path"]]
        ima = 1 if self.carrying else 0
        for i in range(1, len(pts)):
            if i < len(pts) - 1:       # descarta pontos no meio de uma reta
                a, b, c = pts[i - 1], pts[i], pts[i + 1]
                if (b[0] - a[0], b[1] - a[1]) == (c[0] - b[0], c[1] - b[1]):
                    continue
            x, y = pts[i][0], pts[i][1]
            if i == len(pts) - 1 and self.carrying and self.overshoot_mm > 0:
                p_prev = pts[i - 1]
                dx = x - p_prev[0]
                dy = y - p_prev[1]
                # Nao aplica overshoot quando o movimento aproxima a peca de X=0 ou Y=0,
                # evitando acionar os sensores de fim de curso (homing) em 0.0 mm.
                if abs(dx) > 0 and x > 0.0:
                    dir_x = 1.0 if dx > 0 else -1.0
                    x = max(0.0, min(cfg.GANTRY_SIZE_MM[0], x + dir_x * self.overshoot_mm))
                elif abs(dy) > 0 and y > 0.0:
                    dir_y = 1.0 if dy > 0 else -1.0
                    y = max(0.0, min(cfg.GANTRY_SIZE_MM[1], y + dir_y * self.overshoot_mm))
                x = round(x, 2)
                y = round(y, 2)
            self._emit(x, y, ima)
        self.robot_phys = dest_phys

    def _navigate(self, dest: tuple[int, int]) -> None:
        dest_phys = (dest[0] * 2, dest[1] * 2)
        self._navigate_phys(dest_phys)

    def _pickup(self, cell: tuple[int, int]) -> None:
        r, c = cell
        x, y = self.cell_mm(cell)
        # Garante que o Gantry esteja posicionado na casa com o eletroima DESLIGADO (0)
        # antes de emitir o comando parado com eletroima LIGADO (1).
        if not self._steps or self._steps[-1] != (x, y, 0):
            self._emit(x, y, 0)
        self.logic.grid[r][c] = EMPTY
        self.carrying = True
        self._emit(x, y, 1)

    def _release(self, cell: tuple[int, int]) -> None:
        r, c = cell
        self.logic.grid[r][c] = PIECE
        self.carrying = False
        x, y = self.cell_mm(cell)
        self._emit(x, y, 0)

    def _carry(self, src: tuple[int, int], dst: tuple[int, int]) -> None:
        self._navigate(src)
        self._pickup(src)
        self._navigate(dst)
        self._release(dst)

    # ── planejamento ─────────────────────────────────────────────────────────

    def plan(self, move: chess.Move) -> Plan:
        """Gera os passos do Gantry para `move` e aplica o lance ao estado interno."""
        if move not in self.board.legal_moves:
            raise PlanningError(f"Lance ilegal: {move}")
        if not self.is_phys_valid(move):
            raise PlanningError("Promocao sem peca capturada disponivel no cemiterio.")

        self._snapshot = (self.board.copy(), dict(self.cem_occ), self.robot_phys, self.carrying)
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

        if self.return_home:
            self.robot_phys = self.home_phys()
            self._emit(0.0, 0.0, 0)

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
        if self.return_home:
            self._navigate_phys(self.home_phys())

    def plan_reorganize(self, color: bool | None = None) -> Plan:
        """
        Organiza as pecas de volta as casas iniciais.
        color=None: organiza ambos os lados (todas as 32 pecas).
        color=chess.BLACK ou chess.WHITE: organiza apenas a cor especificada.
        """
        self._snapshot = (self.board.copy(), dict(self.cem_occ), self.robot_phys, self.carrying)
        self._steps = []

        # Na organizacao, o Gantry sempre parte de (0, 0) apos o Homing com o ima desligado
        self.carrying = False
        self.robot_phys = self.home_phys()
        self._emit(0.0, 0.0, 0)

        # 1. Alvos de cada cor:
        # Pretas: Linha 0 (Rank 8) Nobres, Linha 1 (Rank 7) Peoes
        black_nobles = {
            (0, 10): chess.ROOK, (0, 9): chess.KNIGHT, (0, 8): chess.BISHOP, (0, 7): chess.QUEEN,
            (0, 6): chess.KING, (0, 5): chess.BISHOP, (0, 4): chess.KNIGHT, (0, 3): chess.ROOK,
        }
        black_pawns = {(1, c): chess.PAWN for c in range(3, 11)}

        # Brancas: Linha 6 (Rank 2) Peoes, Linha 7 (Rank 1) Nobres
        white_pawns = {(6, c): chess.PAWN for c in range(3, 11)}
        white_nobles = {
            (7, 10): chess.ROOK, (7, 9): chess.KNIGHT, (7, 8): chess.BISHOP, (7, 7): chess.QUEEN,
            (7, 6): chess.KING, (7, 5): chess.BISHOP, (7, 4): chess.KNIGHT, (7, 3): chess.ROOK,
        }

        all_targets: dict[tuple[int, int], tuple[bool, int]] = {}
        if color in (None, chess.BLACK):
            for cell, pt in black_nobles.items():
                all_targets[cell] = (chess.BLACK, pt)
            for cell, pt in black_pawns.items():
                all_targets[cell] = (chess.BLACK, pt)
        if color in (None, chess.WHITE):
            for cell, pt in white_pawns.items():
                all_targets[cell] = (chess.WHITE, pt)
            for cell, pt in white_nobles.items():
                all_targets[cell] = (chess.WHITE, pt)

        def find_empty_temp_square(near_cell: tuple[int, int] | None = None) -> tuple[int, int]:
            candidates: list[tuple[int, int]] = []
            for r in range(2, 6):
                for c in range(COL_BOARD_START, COL_BOARD_END + 1):
                    if self.logic.grid[r][c] == EMPTY:
                        candidates.append((r, c))
            if not candidates:
                raise PlanningError("Nenhuma casa livre para manobra encontrada.")
            if near_cell is None:
                return candidates[0]
            return min(
                candidates,
                key=lambda cell: abs(cell[0] - near_cell[0]) + abs(cell[1] - near_cell[1]),
            )

        # Se organizando apenas uma cor, limpa pecas da outra cor que estejam nas fileiras alvo
        if color is not None:
            target_rows = (0, 1) if color == chess.BLACK else (6, 7)
            for sq in chess.SQUARES:
                p = self.board.piece_at(sq)
                if p and p.color != color:
                    cell = square_to_cell(sq)
                    if cell[0] in target_rows:
                        temp = find_empty_temp_square(near_cell=cell)
                        self._carry(cell, temp)
                        self.board.remove_piece_at(sq)
                        temp_sq = chess.square(10 - temp[1], 7 - temp[0])
                        self.board.set_piece_at(temp_sq, p)

        # 2. Localizar todas as pecas no tabuleiro e no cemiterio
        pieces = []
        pid = 0
        for sq in chess.SQUARES:
            p = self.board.piece_at(sq)
            if p and (color is None or p.color == color):
                pieces.append({
                    "id": pid,
                    "color": p.color,
                    "type": p.piece_type,
                    "curr": square_to_cell(sq),
                    "is_cem": False,
                    "target": None,
                })
                pid += 1

        for slot, p_type in list(self.cem_occ.items()):
            if slot[1] in (COL_CEM_DIR_PEAO, COL_CEM_DIR_NOBRE):
                slot_color = chess.BLACK
            elif slot[1] in (COL_CEM_ESQ_NOBRE, COL_CEM_ESQ_PEAO):
                slot_color = chess.WHITE
            else:
                continue
            if color is None or slot_color == color:
                pieces.append({
                    "id": pid,
                    "color": slot_color,
                    "type": p_type,
                    "curr": slot,
                    "is_cem": True,
                    "target": None,
                })
                pid += 1

        # 3. Atribuicao otimizada dos alvos (menor distancia Manhattan entre posicao atual e casa inicial):
        unassigned = dict(all_targets)
        for p in pieces:
            if not p["is_cem"] and p["curr"] in unassigned:
                tc, tt = unassigned[p["curr"]]
                if tc == p["color"] and tt == p["type"]:
                    p["target"] = p["curr"]
                    del unassigned[p["curr"]]

        # Agrupa pecas sem alvo por (color, type) e associa cada uma ao alvo livre mais proximo
        unassigned_pieces = [p for p in pieces if p["target"] is None]
        while unassigned_pieces:
            best_pair = None
            best_dist = float("inf")
            for p in unassigned_pieces:
                for tgt, (tc, tt) in unassigned.items():
                    if tc == p["color"] and tt == p["type"]:
                        dist = abs(p["curr"][0] - tgt[0]) + abs(p["curr"][1] - tgt[1])
                        if dist < best_dist:
                            best_dist = dist
                            best_pair = (p, tgt)
            if best_pair is None:
                p0 = unassigned_pieces[0]
                raise PlanningError(f"Nao foi possivel alocar alvo para peca {p0['type']}.")
            p_sel, tgt_sel = best_pair
            p_sel["target"] = tgt_sel
            del unassigned[tgt_sel]
            unassigned_pieces.remove(p_sel)

        # 4. Mover pecas para os alvos com:
        #    a) Prioridade para pecas que NAO sao peoes (Nobres nas linhas 0 e 7 primeiro,
        #       evitando que peoes nas linhas 1 e 6 fechem a passagem e forcem o Gantry a andar pelos vaos).
        #    b) Rota gulosa (nearest-neighbor) a partir da posicao atual do Gantry (iniciando em 0,0).
        def _robot_dist_to_cell(cell: tuple[int, int]) -> int:
            pr, pc = cell[0] * 2, cell[1] * 2
            return abs(self.robot_phys[0] - pr) + abs(self.robot_phys[1] - pc)

        def _candidate_score(p: dict) -> tuple[int, int, int]:
            tgt = p["target"]
            # Se for nobre (linha 0 ou 7) e a casa de peao logo a frente ja estiver ocupada,
            # prioriza encaixar esse nobre antes que as casas laterais da fileira fechem o caminho aberto.
            front_row = 1 if tgt[0] == 0 else (6 if tgt[0] == 7 else None)
            front_occupied_priority = (
                0 if (front_row is not None and self.logic.grid[front_row][tgt[1]] != EMPTY) else 1
            )
            dist_to_pickup = _robot_dist_to_cell(p["curr"])
            carry_dist = (abs(p["curr"][0] - tgt[0]) + abs(p["curr"][1] - tgt[1])) * 2
            return (front_occupied_priority, dist_to_pickup, carry_dist)

        def _move_piece_to(p: dict, dest: tuple[int, int]) -> None:
            self._carry(p["curr"], dest)
            if p["is_cem"]:
                del self.cem_occ[p["curr"]]
                p["is_cem"] = False
            p["curr"] = dest

        home_rows = (0, 1, 6, 7) if color is None else ((0, 1) if color == chess.BLACK else (6, 7))
        while True:
            pieces_to_move = [p for p in pieces if p["curr"] != p["target"]]
            if not pieces_to_move:
                break

            nobles_to_move = [p for p in pieces_to_move if p["type"] != chess.PAWN]
            pawns_to_move = [p for p in pieces_to_move if p["type"] == chess.PAWN]

            # Fase A: Enquanto houver pecas nobres (nao-peoes) fora do lugar, prioriza elas!
            if nobles_to_move:
                ready_nobles = [
                    p for p in nobles_to_move
                    if self.logic.grid[p["target"][0]][p["target"][1]] == EMPTY
                ]
                if ready_nobles:
                    best_noble = min(ready_nobles, key=_candidate_score)
                    _move_piece_to(best_noble, best_noble["target"])
                    continue

                # Todos os alvos dos nobres pendentes estao ocupados por alguma peca bloqueadora
                noble_target_cells = {p["target"] for p in nobles_to_move}
                blockers = [p for p in pieces_to_move if p["curr"] in noble_target_cells]
                if not blockers:
                    blockers = nobles_to_move
                blocker = min(blockers, key=lambda p: _robot_dist_to_cell(p["curr"]))
                # Se o bloqueador for um peao (ou seu alvo estiver ocupado), move para casa temporaria livre
                # no centro do tabuleiro (linhas 2..5) para nao fechar a linha de peoes antes da hora
                if (
                    blocker["type"] != chess.PAWN
                    and self.logic.grid[blocker["target"][0]][blocker["target"][1]] == EMPTY
                ):
                    _move_piece_to(blocker, blocker["target"])
                else:
                    temp = find_empty_temp_square(near_cell=blocker["target"])
                    _move_piece_to(blocker, temp)
                continue

            # Fase B: Todos os nobres ja estao nas casas iniciais -> organiza os peoes pela melhor rota
            ready_pawns = [
                p for p in pawns_to_move
                if self.logic.grid[p["target"][0]][p["target"][1]] == EMPTY
            ]
            if ready_pawns:
                best_pawn = min(ready_pawns, key=_candidate_score)
                _move_piece_to(best_pawn, best_pawn["target"])
                continue

            # Se nenhum peao tem alvo livre (ex: peoes trocados), libera o mais proximo do robo
            in_home_blockers = [p for p in pawns_to_move if p["curr"][0] in home_rows]
            candidates_to_clear = in_home_blockers if in_home_blockers else pawns_to_move
            blocker = min(candidates_to_clear, key=lambda p: _robot_dist_to_cell(p["curr"]))
            temp = find_empty_temp_square(near_cell=blocker["target"])
            _move_piece_to(blocker, temp)

        # 5. Atualiza o estado do tabuleiro de xadrez
        if color is None:
            self.board = chess.Board()
        else:
            for sq in chess.SQUARES:
                p = self.board.piece_at(sq)
                if p and p.color == color:
                    self.board.remove_piece_at(sq)
            for tgt, (tc, tt) in all_targets.items():
                sq = chess.square(10 - tgt[1], 7 - tgt[0])
                self.board.set_piece_at(sq, chess.Piece(tt, tc))

        self.carrying = False
        self._rebuild()
        self._navigate_phys(self.home_phys())

        return Plan(list(self._steps))

    def plan_reorganize_batches(
        self,
        color: bool | None = None,
        max_carries_per_batch: int = 2,
    ) -> list[Plan]:
        """
        Divide a organizacao completa do tabuleiro em partes menores (lotes de ate
        ``max_carries_per_batch`` pecas transportadas por comando).
        Assim, cada parte leva ~20-40s e mantem comunicacao constante entre a
        Raspberry Pi e o ESP32, evitando estourar o timeout de 3 minutos.
        """
        full_plan = self.plan_reorganize(color=color)
        steps = full_plan.steps
        if not steps:
            return []

        batches: list[Plan] = []
        current_batch: list[tuple[float, float, int]] = []
        carries_in_batch = 0

        for i, step in enumerate(steps):
            current_batch.append(step)
            # Detecta o momento exato em que uma peca acabou de ser solta (ima 1 -> 0)
            if i > 0 and steps[i - 1][2] == 1 and step[2] == 0:
                carries_in_batch += 1
                if carries_in_batch >= max_carries_per_batch:
                    batches.append(Plan(list(current_batch)))
                    current_batch = []
                    carries_in_batch = 0

        if current_batch:
            # Se sobrou apenas o deslocamento final de volta ao (0, 0) sem carregar peca,
            # anexa ao ultimo lote para evitar um comando extra de 1 passo
            if batches and all(s[2] == 0 for s in current_batch):
                batches[-1].steps.extend(current_batch)
            else:
                batches.append(Plan(list(current_batch)))

        return batches
