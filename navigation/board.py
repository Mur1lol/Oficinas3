"""
board.py
========
Representacao do tabuleiro de xadrez/damas para navegacao robotica.

Layout logico (colunas):
    [cemiterio_dir]  [corredor]  [tabuleiro 8x8 (H8:H1 ate A8:A1)]  [corredor]  [cemiterio_esq]
    Col 0: Peoes      Col 2       Col 3 : H8:H1                      Col 11      Col 12: Nobres
    Col 1: Nobres                 Col 4 : G8:G1                                  Col 13: Peoes
                                  Col 5 : F8:F1
                                  Col 6 : E8:E1
                                  Col 7 : D8:D1
                                  Col 8 : C8:C1
                                  Col 9 : B8:B1
                                  Col 10: A8:A1

Total logico : 14 colunas x 8 linhas (cada casa logica mede 5 cm x 5 cm)

Layout fisico com bordas internas (27 colunas x 15 linhas):
    - Cada celula fisica mede 2,5 cm x 2,5 cm (sem bordas externas).
    - O ponto (0, 0) fica exatamente no canto superior direito (Direita em cima).
    - Casa logica (row, col) fica na posicao fisica (row * 2, col * 2).
    - As bordas/corredores internos entre casas ficam nas celulas com indices
      impares (row % 2 != 0 ou col % 2 != 0).
    - As colunas 2 e 11 sao corredores livres de navegacao (pc = 4 e pc = 22).
    - Total de colunas fisicas: 14 casas + 13 bordas internas = 27 colunas (0..26).
    - Total de linhas fisicas : 8 casas + 7 bordas internas = 15 linhas (0..14).
    - Curso fisico maximo: 14 passos * 25 mm = 350 mm em X (linhas).
                           26 passos * 25 mm = 650 mm em Y (colunas).
"""

# ─── Dimensoes ────────────────────────────────────────────────────────────────

BOARD_ROWS   = 8    # linhas do tabuleiro de jogo (Row 0 = Rank 8, Row 7 = Rank 1)
BOARD_COLS   = 8    # colunas do tabuleiro de jogo (Col 3 = H, Col 10 = A)

LOGIC_COLS   = 14   # 14 colunas no total (0..13)
LOGIC_ROWS   = 8    # 8 linhas no total (0..7)

# Layout fisico sem borda externa (cada celula 2,5 cm x 2,5 cm):
PHYS_COLS    = LOGIC_COLS * 2 - 1  # 27 colunas (indices 0..26)
PHYS_ROWS    = LOGIC_ROWS * 2 - 1  # 15 linhas (indices 0..14)

# Definicao das colunas no espaco logico:
COL_CEM_DIR_PEAO   = 0   # peoes cemiterio direito
COL_CEM_DIR_NOBRE  = 1   # nobres cemiterio direito
COL_CORREDOR_DIR   = 2   # corredor livre direito
COL_BOARD_START    = 3   # H8..H1 (file H = index 7)
COL_BOARD_END      = 10  # A8..A1 (file A = index 0)
COL_CORREDOR_ESQ   = 11  # corredor livre esquerdo
COL_CEM_ESQ_NOBRE  = 12  # nobres cemiterio esquerdo
COL_CEM_ESQ_PEAO   = 13  # peoes cemiterio esquerdo

# Valores de celula
EMPTY  = 0
PIECE  = 1

# ─── Tipos de zona ────────────────────────────────────────────────────────────

ZONE_BOARD   = "board"
ZONE_CEM_DIR = "cemetery_right"
ZONE_CEM_ESQ = "cemetery_left"
ZONE_VAO     = "gap"


def zone_of(col: int) -> str:
    """Retorna a zona logica de uma coluna."""
    if col in (COL_CEM_DIR_PEAO, COL_CEM_DIR_NOBRE):
        return ZONE_CEM_DIR
    if col in (COL_CORREDOR_DIR, COL_CORREDOR_ESQ):
        return ZONE_VAO
    if COL_BOARD_START <= col <= COL_BOARD_END:
        return ZONE_BOARD
    if col in (COL_CEM_ESQ_NOBRE, COL_CEM_ESQ_PEAO):
        return ZONE_CEM_ESQ
    raise ValueError(f"Coluna logica {col} fora do range [0, {LOGIC_COLS - 1}].")


# ─── Tabuleiro logico ─────────────────────────────────────────────────────────

class BoardLogic:
    """
    Tabuleiro logico completo (14 colunas x 8 linhas).
    """

    def __init__(self):
        self.grid = [[EMPTY] * LOGIC_COLS for _ in range(LOGIC_ROWS)]

    # ── Acesso basico ──

    def get(self, row: int, col: int) -> int:
        self._check(row, col)
        return self.grid[row][col]

    def set(self, row: int, col: int, value: int):
        self._check(row, col)
        if self.is_gap(col):
            raise ValueError(f"Coluna {col} e um corredor e nao pode conter pecas fixas.")
        self.grid[row][col] = value

    def place_piece(self, row: int, col: int):
        self.set(row, col, PIECE)

    def remove_piece(self, row: int, col: int):
        self.set(row, col, EMPTY)

    def is_empty(self, row: int, col: int) -> bool:
        return self.get(row, col) == EMPTY

    # ── Consultas de zona ──

    def is_gap(self, col: int) -> bool:
        """True se a coluna for um corredor de navegacao (cols 2 e 11)."""
        return col == COL_CORREDOR_DIR or col == COL_CORREDOR_ESQ

    def is_board(self, row: int, col: int) -> bool:
        return COL_BOARD_START <= col <= COL_BOARD_END

    def is_cemetery(self, row: int, col: int) -> bool:
        return col in (COL_CEM_DIR_PEAO, COL_CEM_DIR_NOBRE,
                       COL_CEM_ESQ_NOBRE, COL_CEM_ESQ_PEAO)

    def zone(self, col: int) -> str:
        return zone_of(col)

    # ── Coordenadas do tabuleiro de jogo (File A..H <-> Colunas 10..3) ──

    def board_to_logical(self, board_row: int, board_col: int) -> tuple[int, int]:
        """
        Converte coord do sub-tabuleiro 8x8 -> coord logica global.
        board_col: 0 (file A) a 7 (file H).
        board_row: 0 (rank 8) a 7 (rank 1).
        """
        if not (0 <= board_row < BOARD_ROWS and 0 <= board_col < BOARD_COLS):
            raise IndexError(f"Coord do tabuleiro ({board_row}, {board_col}) fora de 8x8.")
        return board_row, 10 - board_col

    def logical_to_board(self, row: int, col: int) -> tuple[int, int]:
        """Converte coord logica global -> coord do sub-tabuleiro 8x8 (row, file)."""
        if not self.is_board(row, col):
            raise ValueError(f"Coord ({row}, {col}) nao pertence ao tabuleiro de jogo.")
        return row, 10 - col

    # ── Coordenadas do cemiterio ──

    def next_cemetery_slot(self, side: str) -> tuple[int, int] | None:
        """
        Retorna a proxima posicao livre no cemiterio do lado indicado.
        side: 'right' (cols 0, 1) ou 'left' (cols 12, 13).
        """
        if side == "right":
            cols = (COL_CEM_DIR_PEAO, COL_CEM_DIR_NOBRE)
        elif side == "left":
            cols = (COL_CEM_ESQ_PEAO, COL_CEM_ESQ_NOBRE)
        else:
            raise ValueError(f"Side deve ser 'left' ou 'right', nao '{side}'.")

        for col in cols:
            for row in range(LOGIC_ROWS):
                if self.grid[row][col] == EMPTY:
                    return (row, col)
        return None

    # ── Mapeamento para o espaco fisico ──

    def to_physical(self, row: int, col: int) -> tuple[int, int]:
        """Converte coordenada logica -> coordenada fisica no grid 15x27."""
        self._check(row, col)
        return row * 2, col * 2

    def _check(self, row: int, col: int):
        if not (0 <= row < LOGIC_ROWS and 0 <= col < LOGIC_COLS):
            raise IndexError(f"Coordenada ({row}, {col}) fora do grid logico "
                             f"{LOGIC_ROWS}x{LOGIC_COLS}.")

    def display(self):
        """Exibe o tabuleiro logico completo no terminal."""
        col_hdr = "     " + "  ".join(f"{c:2d}" for c in range(LOGIC_COLS))
        print(col_hdr)
        print("     " + "---" * LOGIC_COLS)
        zone_row = "zona "
        for c in range(LOGIC_COLS):
            z = zone_of(c)
            if z == ZONE_CEM_DIR:
                zone_row += " CD"
            elif z == ZONE_CEM_ESQ:
                zone_row += " CE"
            elif z == ZONE_VAO:
                zone_row += " ||"
            else:
                zone_row += " T "
        print(zone_row)
        print("     " + "---" * LOGIC_COLS)
        for r in range(LOGIC_ROWS):
            row_str = f"{r:2d}  |"
            for c in range(LOGIC_COLS):
                if self.is_gap(c):
                    row_str += " | "
                elif self.grid[r][c] == PIECE:
                    row_str += " P "
                else:
                    row_str += " . "
            print(row_str + "|")
        print("     " + "---" * LOGIC_COLS)


# ─── Tabuleiro fisico ─────────────────────────────────────────────────────────

class BoardPhysical:
    """
    Tabuleiro fisico com bordas internas (PHYS_ROWS x PHYS_COLS = 15x27).

    Mapeamento:
        posicoes com row % 2 == 0 e col % 2 == 0  ->  casas reais (logicas)
        demais posicoes                           ->  bordas/corredores internos
        colunas fisicas 4 e 22 (cols logicas 2 e 11) -> corredores livres
    """

    def __init__(self, board_logic=None):
        self.grid = [[EMPTY] * PHYS_COLS for _ in range(PHYS_ROWS)]
        if board_logic is not None:
            self.sync_from(board_logic)

    def sync_from(self, board_logic):
        """Copia o estado logico para as celulas fisicas correspondentes."""
        src = board_logic.grid
        for r in range(LOGIC_ROWS):
            for c in range(LOGIC_COLS):
                pr, pc = r * 2, c * 2
                is_gap = (c == COL_CORREDOR_DIR or c == COL_CORREDOR_ESQ)
                self.grid[pr][pc] = EMPTY if is_gap else src[r][c]

    def get(self, row: int, col: int) -> int:
        self._check(row, col)
        return self.grid[row][col]

    def is_real_cell(self, row: int, col: int) -> bool:
        """True se a celula corresponde ao centro de uma casa real (indices pares)."""
        return (row % 2 == 0) and (col % 2 == 0)

    def is_corridor(self, row: int, col: int) -> bool:
        """True se a celula e uma borda interna ou corredor transitavel."""
        if not self.is_real_cell(row, col):
            return True
        # As colunas logicas 2 e 11 (fisicas 4 e 22) sao corredores livres
        return col == (COL_CORREDOR_DIR * 2) or col == (COL_CORREDOR_ESQ * 2)

    def is_passable(self, row: int, col: int) -> bool:
        """True se o robo pode passar por essa celula."""
        if not (0 <= row < PHYS_ROWS and 0 <= col < PHYS_COLS):
            return False
        if self.is_corridor(row, col):
            return True   # corredores sao sempre livres
        return self.grid[row][col] == EMPTY  # casa real: apenas se vazia

    def _check(self, row: int, col: int):
        if not (0 <= row < PHYS_ROWS and 0 <= col < PHYS_COLS):
            raise IndexError(f"Coordenada fisica ({row}, {col}) fora do grid "
                             f"{PHYS_ROWS}x{PHYS_COLS}.")

    def display(self, path=None):
        """Imprime o tabuleiro fisico 15x27."""
        path_set = set(path) if path else set()
        print(f"Tabuleiro fisico {PHYS_ROWS}x{PHYS_COLS}")
        print("  H=casa vazia  P=peca  .=borda/corredor  *=caminho")
        header = "    " + " ".join(f"{c:2d}" for c in range(PHYS_COLS))
        print(header)
        for r in range(PHYS_ROWS):
            parts = []
            for c in range(PHYS_COLS):
                if (r, c) in path_set:
                    parts.append(" *")
                elif self.is_real_cell(r, c) and not self.is_corridor(r, c):
                    parts.append(" P" if self.grid[r][c] == PIECE else " H")
                else:
                    parts.append(" .")
            print(f"{r:2d} " + " ".join(parts))


# ─── Compatibilidade retroativa ───────────────────────────────────────────────

class Board8x8(BoardLogic):
    """Alias de compatibilidade: opera sobre o grid logico 14x8."""

    def __init__(self):
        super().__init__()

    def _b2l(self, row: int, col: int) -> tuple[int, int]:
        if not (0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS):
            raise IndexError(f"Coordenada ({row}, {col}) fora do tabuleiro 8x8.")
        return row, 10 - col

    def get(self, row: int, col: int) -> int:
        lr, lc = self._b2l(row, col)
        return self.grid[lr][lc]

    def set(self, row: int, col: int, value: int):
        lr, lc = self._b2l(row, col)
        self.grid[lr][lc] = value

    def place_piece(self, row: int, col: int):
        self.set(row, col, PIECE)

    def remove_piece(self, row: int, col: int):
        self.set(row, col, EMPTY)

    def is_empty(self, row: int, col: int) -> bool:
        return self.get(row, col) == EMPTY

    def to_physical(self, row: int, col: int) -> tuple[int, int]:
        lr, lc = self._b2l(row, col)
        return lr * 2, lc * 2

    def _check(self, row: int, col: int):
        if not (0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS):
            raise IndexError(f"Coordenada ({row}, {col}) fora do tabuleiro 8x8.")


class Board17x17(BoardPhysical):
    """Alias de compatibilidade retroativa para BoardPhysical."""

    def __init__(self, board8=None):
        super().__init__(None)
        if board8 is not None and isinstance(board8, BoardLogic):
            self.sync_from(board8)

    def is_house(self, row: int, col: int) -> bool:
        return self.is_real_cell(row, col) and not self.is_corridor(row, col)


BOARD_SIZE = BOARD_ROWS   # = 8
PHYS_SIZE  = PHYS_ROWS    # = 15
