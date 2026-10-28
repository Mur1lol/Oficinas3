"""
board.py
========
Representacao do tabuleiro de xadrez/damas para navegacao robotica.

Layout logico (colunas):
    [cemiterio_esq 3x8] [vao_esq 1x8] [tabuleiro 8x8] [vao_dir 1x8] [cemiterio_dir 3x8]
    cols logicas: 0..2        3              4..11            12             13..15

Total logico : 16 colunas x 8 linhas

Layout fisico expandido (cada casa logica -> celula fisica (row*2+1, col*2+1)):
    Total fisico: 33 colunas x 17 linhas
    (16 casas * 2 + 1 = 33 cols; 8 casas * 2 + 1 = 17 rows)

Zonas no grid logico (por col):
    COL_CEM_ESQ  = 0..2   -> cemiterio das pecas do jogador esquerdo
    COL_VAO_ESQ  = 3      -> corredor de transicao (sempre livre)
    COL_BOARD    = 4..11  -> tabuleiro de jogo 8x8
    COL_VAO_DIR  = 12     -> corredor de transicao (sempre livre)
    COL_CEM_DIR  = 13..15 -> cemiterio das pecas do jogador direito

Mapeamento de coordenadas
-------------------------
  Casa logica  (row, col)  ->  celula fisica  (row*2 + 1, col*2 + 1)
  Corredor entre casas adjacentes: celulas com ao menos um indice par.

Convencao de eixos
------------------
  row  ->  eixo Y  (0 = topo, 7 = base no logico)
  col  ->  eixo X  (0 = esquerda, 15 = direita no logico)
"""

# ─── Dimensoes ────────────────────────────────────────────────────────────────

BOARD_ROWS   = 8    # linhas do tabuleiro de jogo
BOARD_COLS   = 8    # colunas do tabuleiro de jogo

CEM_COLS     = 3    # colunas de cemiterio em cada lado
VAO_COLS     = 1    # coluna de vao em cada lado

# Layout logico total
LOGIC_COLS   = CEM_COLS + VAO_COLS + BOARD_COLS + VAO_COLS + CEM_COLS  # = 16
LOGIC_ROWS   = BOARD_ROWS                                                # = 8

# Layout fisico (expansao 2x+1)
PHYS_COLS    = LOGIC_COLS * 2 + 1   # = 33
PHYS_ROWS    = LOGIC_ROWS * 2 + 1   # = 17

# Offsets de coluna no espaco logico
COL_CEM_ESQ_START  = 0
COL_CEM_ESQ_END    = CEM_COLS - 1                           # 2
COL_VAO_ESQ        = CEM_COLS                               # 3
COL_BOARD_START    = CEM_COLS + VAO_COLS                    # 4
COL_BOARD_END      = CEM_COLS + VAO_COLS + BOARD_COLS - 1  # 11
COL_VAO_DIR        = CEM_COLS + VAO_COLS + BOARD_COLS       # 12
COL_CEM_DIR_START  = CEM_COLS + VAO_COLS + BOARD_COLS + VAO_COLS  # 13
COL_CEM_DIR_END    = LOGIC_COLS - 1                         # 15

# Valores de celula
EMPTY  = 0
PIECE  = 1

# ─── Tipos de zona ────────────────────────────────────────────────────────────

ZONE_BOARD   = "board"
ZONE_CEM_ESQ = "cemetery_left"
ZONE_CEM_DIR = "cemetery_right"
ZONE_VAO     = "gap"


def zone_of(col: int) -> str:
    """Retorna a zona logica de uma coluna."""
    if COL_CEM_ESQ_START <= col <= COL_CEM_ESQ_END:
        return ZONE_CEM_ESQ
    if col == COL_VAO_ESQ or col == COL_VAO_DIR:
        return ZONE_VAO
    if COL_BOARD_START <= col <= COL_BOARD_END:
        return ZONE_BOARD
    if COL_CEM_DIR_START <= col <= COL_CEM_DIR_END:
        return ZONE_CEM_DIR
    raise ValueError(f"Coluna logica {col} fora do range [0, {LOGIC_COLS - 1}].")


# ─── Tabuleiro logico ─────────────────────────────────────────────────────────

class BoardLogic:
    """
    Tabuleiro logico completo (16 colunas x 8 linhas).

    Inclui:
        - Tabuleiro de jogo 8x8 (cols 4..11)
        - Cemiterio esquerdo 3x8 (cols 0..2)
        - Cemiterio direito  3x8 (cols 13..15)
        - Vaos de transicao  1x8 (cols 3 e 12) -- sempre EMPTY/livres
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
            raise ValueError(f"Coluna {col} e um vao e nao pode conter pecas.")
        self.grid[row][col] = value

    def place_piece(self, row: int, col: int):
        self.set(row, col, PIECE)

    def remove_piece(self, row: int, col: int):
        self.set(row, col, EMPTY)

    def is_empty(self, row: int, col: int) -> bool:
        return self.get(row, col) == EMPTY

    # ── Consultas de zona ──

    def is_gap(self, col: int) -> bool:
        return col == COL_VAO_ESQ or col == COL_VAO_DIR

    def is_board(self, row: int, col: int) -> bool:
        return COL_BOARD_START <= col <= COL_BOARD_END

    def is_cemetery(self, row: int, col: int) -> bool:
        return (COL_CEM_ESQ_START <= col <= COL_CEM_ESQ_END or
                COL_CEM_DIR_START <= col <= COL_CEM_DIR_END)

    def zone(self, col: int) -> str:
        return zone_of(col)

    # ── Coordenadas do tabuleiro de jogo ──

    def board_to_logical(self, board_row: int, board_col: int) -> tuple:
        """Converte coord do sub-tabuleiro 8x8 (0..7, 0..7) -> coord logica global."""
        if not (0 <= board_row < BOARD_ROWS and 0 <= board_col < BOARD_COLS):
            raise IndexError(f"Coord do tabuleiro ({board_row}, {board_col}) fora de 8x8.")
        return board_row, COL_BOARD_START + board_col

    def logical_to_board(self, row: int, col: int) -> tuple:
        """Converte coord logica global -> coord do sub-tabuleiro 8x8."""
        if not self.is_board(row, col):
            raise ValueError(f"Coord ({row}, {col}) nao pertence ao tabuleiro de jogo.")
        return row, col - COL_BOARD_START

    # ── Coordenadas do cemiterio ──

    def next_cemetery_slot(self, side: str):
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
        return None  # cemiterio cheio

    # ── Mapeamento para o espaco fisico ──

    def to_physical(self, row: int, col: int) -> tuple:
        """Converte coordenada logica -> coordenada fisica no grid PHYS_ROWS x PHYS_COLS."""
        self._check(row, col)
        return row * 2 + 1, col * 2 + 1

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
            elif z == ZONE_VAO:
                zone_row += " |"
            else:
                zone_row += " T "
        print(zone_row)
        print("     " + "---" * LOGIC_COLS)
        for r in range(LOGIC_ROWS):
            row_str = f"{r:2d}  |"
            for c in range(LOGIC_COLS):
                if self.is_gap(c):
                    row_str += " |"
                elif self.grid[r][c] == PIECE:
                    row_str += " P "
                else:
                    row_str += " . "
            print(row_str + "|")
        print("     " + "---" * LOGIC_COLS)


# ─── Tabuleiro fisico ─────────────────────────────────────────────────────────

class BoardPhysical:
    """
    Tabuleiro fisico expandido (PHYS_ROWS x PHYS_COLS = 17x33).

    Mapeamento:
        posicoes com row%2==1 e col%2==1  ->  casas reais (logicas)
        demais posicoes                    ->  corredores (vaos fisicos)

    Os vaos logicos (cols 3 e 12 no espaco logico) expandem para cols fisicas
    7 e 25 (casas) + seus corredores adjacentes, garantindo que o robo possa
    transitar entre cemiterio e tabuleiro.
    """

    def __init__(self, board_logic=None):
        self.grid = [[EMPTY] * PHYS_COLS for _ in range(PHYS_ROWS)]
        if board_logic is not None:
            self.sync_from(board_logic)

    def sync_from(self, board_logic):
        """Copia o estado logico para as celulas fisicas correspondentes."""
        # Acessa .grid diretamente para evitar overrides de get/_check em subclasses compat
        src = board_logic.grid
        for r in range(LOGIC_ROWS):
            for c in range(LOGIC_COLS):
                pr, pc = r * 2 + 1, c * 2 + 1
                is_gap = (c == COL_VAO_ESQ or c == COL_VAO_DIR)
                self.grid[pr][pc] = EMPTY if is_gap else src[r][c]

    def get(self, row: int, col: int) -> int:
        self._check(row, col)
        return self.grid[row][col]

    def is_real_cell(self, row: int, col: int) -> bool:
        """True se a celula corresponde a uma casa logica (indices fisicos impares)."""
        return (row % 2 == 1) and (col % 2 == 1)

    def is_corridor(self, row: int, col: int) -> bool:
        return not self.is_real_cell(row, col)

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
        """
        Imprime o tabuleiro fisico.

        Parameters
        ----------
        path : list of (row, col) | None
            Se fornecido, marca as celulas do caminho com '*'.
        """
        path_set = set(path) if path else set()
        print(f"Tabuleiro fisico {PHYS_ROWS}x{PHYS_COLS}")
        print("  H=casa vazia  P=peca  .=corredor  *=caminho")
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
# Aliases para nao quebrar codigo existente que importava Board8x8 / Board17x17

class Board8x8(BoardLogic):
    """
    Alias de compatibilidade: wrapper que expoe a interface anterior do Board8x8,
    mas agora opera sobre o grid logico completo 16x8.

    Metodos legados mapeiam coordenadas 8x8 (0..7, 0..7) para o sub-grid
    do tabuleiro de jogo (cols 4..11 no espaco logico).
    """

    def __init__(self):
        super().__init__()

    def _b2l(self, row, col):
        """Converte coord do sub-tabuleiro 8x8 -> coord logica global."""
        if not (0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS):
            raise IndexError(f"Coordenada ({row}, {col}) fora do tabuleiro 8x8.")
        return row, COL_BOARD_START + col

    def get(self, row, col):
        lr, lc = self._b2l(row, col)
        return self.grid[lr][lc]

    def set(self, row, col, value):
        lr, lc = self._b2l(row, col)
        self.grid[lr][lc] = value

    def place_piece(self, row, col):
        self.set(row, col, PIECE)

    def remove_piece(self, row, col):
        self.set(row, col, EMPTY)

    def is_empty(self, row, col):
        return self.get(row, col) == EMPTY

    def to_physical(self, row, col):
        lr, lc = self._b2l(row, col)
        return lr * 2 + 1, lc * 2 + 1

    def _check(self, row, col):
        if not (0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS):
            raise IndexError(f"Coordenada ({row}, {col}) fora do tabuleiro 8x8.")


class Board17x17(BoardPhysical):
    """
    Alias de compatibilidade: subclasse de BoardPhysical.
    O grid agora e 17x33, mas a interface e identica.
    """

    def __init__(self, board8=None):
        # Inicializa o grid vazio primeiro
        super().__init__(None)
        if board8 is not None and isinstance(board8, BoardLogic):
            # sync_from agora calcula to_physical diretamente sem chamar override
            self.sync_from(board8)

    def is_house(self, row, col):
        """Alias legado para is_real_cell."""
        return self.is_real_cell(row, col)


# ─── Constantes legadas ───────────────────────────────────────────────────────
BOARD_SIZE = BOARD_ROWS   # = 8  (compatibilidade)
PHYS_SIZE  = PHYS_ROWS    # = 17 (compatibilidade -- linhas nao mudaram)
