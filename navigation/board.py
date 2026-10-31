"""
board.py
========
Representacao do tabuleiro de xadrez/damas para navegacao robotica.

Layout logico (colunas):
    [cemiterio_esq 3x8] [tabuleiro 8x8] [cemiterio_dir 3x8]
    cols logicas: 0..2        3..10             11..13

Total logico : 14 colunas x 8 linhas

Layout fisico com bordas internas (27 colunas x 15 linhas):
    - Cada casa logica (row, col) fica na posicao fisica (row * 2, col * 2).
    - As bordas/corredores internos entre casas adjacentes ficam nas celulas
      com indices impares (row % 2 != 0 ou col % 2 != 0).
    - Total de colunas fisicas: 14 casas + 13 bordas internas = 27 colunas (0..26).
    - Total de linhas fisicas : 8 casas + 7 bordas internas = 15 linhas (0..14).

Zonas no grid logico (por col):
    COL_CEM_ESQ  = 0..2   -> cemiterio das pecas do jogador esquerdo
    COL_BOARD    = 3..10  -> tabuleiro de jogo 8x8 (a..h)
    COL_CEM_DIR  = 11..13 -> cemiterio das pecas do jogador direito

Mapeamento de coordenadas
-------------------------
  Casa logica  (row, col)   ->  celula fisica  (row * 2, col * 2)
  Borda interna adjacente   ->  celula fisica com row ou col impar.

Convencao de eixos
------------------
  row  ->  eixo Y logico / X do ESP32 (0 = topo/computer/Pretas, 7 = base/player/Brancas)
  col  ->  eixo X logico / Y do ESP32 (0 = esquerda/cem_esq, 13 = direita/cem_dir)
  Home fisico (0, 0) do ESP32 = canto superior direito (row 0, col 13 -> phys 0, 26).
"""

# ─── Dimensoes ────────────────────────────────────────────────────────────────

BOARD_ROWS   = 8    # linhas do tabuleiro de jogo
BOARD_COLS   = 8    # colunas do tabuleiro de jogo

CEM_COLS     = 3    # colunas de cemiterio em cada lado (esquerdo e direito)

# Layout logico total: 3 + 8 + 3 = 14 colunas x 8 linhas
LOGIC_COLS   = CEM_COLS + BOARD_COLS + CEM_COLS  # = 14
LOGIC_ROWS   = BOARD_ROWS                        # = 8

# Layout fisico considerando as bordas internas entre casas:
# 14 casas + 13 bordas internas = 27 colunas (indices 0..26)
# 8 casas + 7 bordas internas = 15 linhas (indices 0..14)
PHYS_COLS    = LOGIC_COLS * 2 - 1                # = 27
PHYS_ROWS    = LOGIC_ROWS * 2 - 1                # = 15

# Offsets de coluna no espaco logico
COL_CEM_ESQ_START  = 0
COL_CEM_ESQ_END    = CEM_COLS - 1                           # 2 (cols 0, 1, 2)
COL_BOARD_START    = CEM_COLS                               # 3
COL_BOARD_END      = CEM_COLS + BOARD_COLS - 1             # 10 (cols 3..10)
COL_CEM_DIR_START  = CEM_COLS + BOARD_COLS                  # 11
COL_CEM_DIR_END    = LOGIC_COLS - 1                         # 13 (cols 11..13)

# Valores de celula
EMPTY  = 0
PIECE  = 1

# ─── Tipos de zona ────────────────────────────────────────────────────────────

ZONE_BOARD   = "board"
ZONE_CEM_ESQ = "cemetery_left"
ZONE_CEM_DIR = "cemetery_right"
ZONE_VAO     = "gap"  # mantido para compatibilidade retroativa


def zone_of(col: int) -> str:
    """Retorna a zona logica de uma coluna."""
    if COL_CEM_ESQ_START <= col <= COL_CEM_ESQ_END:
        return ZONE_CEM_ESQ
    if COL_BOARD_START <= col <= COL_BOARD_END:
        return ZONE_BOARD
    if COL_CEM_DIR_START <= col <= COL_CEM_DIR_END:
        return ZONE_CEM_DIR
    raise ValueError(f"Coluna logica {col} fora do range [0, {LOGIC_COLS - 1}].")


# ─── Tabuleiro logico ─────────────────────────────────────────────────────────

class BoardLogic:
    """
    Tabuleiro logico completo (14 colunas x 8 linhas).

    Inclui:
        - Cemiterio esquerdo 3x8 (cols 0..2)
        - Tabuleiro de jogo  8x8 (cols 3..10)
        - Cemiterio direito  3x8 (cols 11..13)
    """

    def __init__(self):
        self.grid = [[EMPTY] * LOGIC_COLS for _ in range(LOGIC_ROWS)]

    # ── Acesso basico ──

    def get(self, row: int, col: int) -> int:
        self._check(row, col)
        return self.grid[row][col]

    def set(self, row: int, col: int, value: int):
        self._check(row, col)
        self.grid[row][col] = value

    def place_piece(self, row: int, col: int):
        self.set(row, col, PIECE)

    def remove_piece(self, row: int, col: int):
        self.set(row, col, EMPTY)

    def is_empty(self, row: int, col: int) -> bool:
        return self.get(row, col) == EMPTY

    # ── Consultas de zona ──

    def is_gap(self, col: int) -> bool:
        """Mantido para compatibilidade. No grid 14x8 nao ha colunas extras de vao."""
        return False

    def is_board(self, row: int, col: int) -> bool:
        return COL_BOARD_START <= col <= COL_BOARD_END

    def is_cemetery(self, row: int, col: int) -> bool:
        return (COL_CEM_ESQ_START <= col <= COL_CEM_ESQ_END or
                COL_CEM_DIR_START <= col <= COL_CEM_DIR_END)

    def zone(self, col: int) -> str:
        return zone_of(col)

    # ── Coordenadas do tabuleiro de jogo ──

    def board_to_logical(self, board_row: int, board_col: int) -> tuple[int, int]:
        """Converte coord do sub-tabuleiro 8x8 (0..7, 0..7) -> coord logica global."""
        if not (0 <= board_row < BOARD_ROWS and 0 <= board_col < BOARD_COLS):
            raise IndexError(f"Coord do tabuleiro ({board_row}, {board_col}) fora de 8x8.")
        return board_row, COL_BOARD_START + board_col

    def logical_to_board(self, row: int, col: int) -> tuple[int, int]:
        """Converte coord logica global -> coord do sub-tabuleiro 8x8."""
        if not self.is_board(row, col):
            raise ValueError(f"Coord ({row}, {col}) nao pertence ao tabuleiro de jogo.")
        return row, col - COL_BOARD_START

    # ── Coordenadas do cemiterio ──

    def next_cemetery_slot(self, side: str) -> tuple[int, int] | None:
        """
        Retorna a proxima posicao livre no cemiterio do lado indicado.

        Parameters
        ----------
        side : 'left' ou 'right'

        Returns
        -------
        (row, col) logico da proxima vaga livre, ou None se cemiterio cheio.
        """
        if side == "left":
            col_range = range(COL_CEM_ESQ_START, COL_CEM_ESQ_END + 1)
        elif side == "right":
            col_range = range(COL_CEM_DIR_START, COL_CEM_DIR_END + 1)
        else:
            raise ValueError(f"Side deve ser 'left' ou 'right', nao '{side}'.")

        for col in col_range:
            for row in range(LOGIC_ROWS):
                if self.grid[row][col] == EMPTY:
                    return (row, col)
        return None

    # ── Mapeamento para o espaco fisico ──

    def to_physical(self, row: int, col: int) -> tuple[int, int]:
        """Converte coordenada logica -> coordenada fisica no grid PHYS_ROWS x PHYS_COLS (15x27)."""
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
            if z == ZONE_CEM_ESQ:
                zone_row += " CE"
            elif z == ZONE_CEM_DIR:
                zone_row += " CD"
            else:
                zone_row += " T "
        print(zone_row)
        print("     " + "---" * LOGIC_COLS)
        for r in range(LOGIC_ROWS):
            row_str = f"{r:2d}  |"
            for c in range(LOGIC_COLS):
                if self.grid[r][c] == PIECE:
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
                self.grid[pr][pc] = src[r][c]

    def get(self, row: int, col: int) -> int:
        self._check(row, col)
        return self.grid[row][col]

    def is_real_cell(self, row: int, col: int) -> bool:
        """True se a celula corresponde ao centro de uma casa real (indices pares)."""
        return (row % 2 == 0) and (col % 2 == 0)

    def is_corridor(self, row: int, col: int) -> bool:
        """True se a celula e uma borda interna transitavel."""
        return not self.is_real_cell(row, col)

    def is_passable(self, row: int, col: int) -> bool:
        """True se o robo pode passar por essa celula."""
        if not (0 <= row < PHYS_ROWS and 0 <= col < PHYS_COLS):
            return False
        if self.is_corridor(row, col):
            return True   # bordas internas sao sempre livres
        return self.grid[row][col] == EMPTY  # casa real: apenas se vazia

    def _check(self, row: int, col: int):
        if not (0 <= row < PHYS_ROWS and 0 <= col < PHYS_COLS):
            raise IndexError(f"Coordenada fisica ({row}, {col}) fora do grid "
                             f"{PHYS_ROWS}x{PHYS_COLS}.")

    def display(self, path=None):
        """
        Imprime o tabuleiro fisico 15x27.

        Parameters
        ----------
        path : list of (row, col) | None
            Se fornecido, marca as celulas do caminho com '*'.
        """
        path_set = set(path) if path else set()
        print(f"Tabuleiro fisico {PHYS_ROWS}x{PHYS_COLS}")
        print("  H=casa vazia  P=peca  .=borda interna  *=caminho")
        header = "    " + " ".join(f"{c:2d}" for c in range(PHYS_COLS))
        print(header)
        for r in range(PHYS_ROWS):
            parts = []
            for c in range(PHYS_COLS):
                if (r, c) in path_set:
                    parts.append(" *")
                elif self.is_real_cell(r, c):
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
        return row, COL_BOARD_START + col

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
        return self.is_real_cell(row, col)


# Constantes legadas
BOARD_SIZE = BOARD_ROWS   # = 8
PHYS_SIZE  = PHYS_ROWS    # = 15
