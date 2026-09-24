"""
board.py
========
Representacao do tabuleiro de xadrez/damas para navegacao robotica.

Tabuleiro logico  : 8x8  -- casas reais onde as pecas ficam.
Tabuleiro fisico  : 17x17 -- expansao do 8x8 com corredores entre cada casa.

Mapeamento de coordenadas
-------------------------
  Casa logica  (row, col)  ->  celula fisica  (row*2 + 1, col*2 + 1)
  Corredor entre casas adjacentes: celulas com ao menos um indice par.

Convencao de eixos
------------------
  row  ->  eixo Y  (0 = topo, 7 = base no logico; 1 = topo, 15 = base no fisico)
  col  ->  eixo X  (0 = esquerda, 7 = direita no logico)
"""

BOARD_SIZE  = 8   # tabuleiro logico
PHYS_SIZE   = 17  # tabuleiro fisico  (8*2 + 1 = 17, com bordas de corredor)

# Com PHYS_OFFSET = 1:
#   casa logica (r, c)  ->  posicao fisica (r*2 + 1, c*2 + 1)
#   corredor entre (r,c) e (r,c+1) -> posicao fisica (r*2+1, c*2+2)
PHYS_OFFSET = 1

# Valores de celula
EMPTY = 0
PIECE = 1


class Board8x8:
    """Tabuleiro logico 8x8."""

    def __init__(self):
        self.grid = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]

    def get(self, row, col):
        self._check(row, col)
        return self.grid[row][col]

    def set(self, row, col, value):
        self._check(row, col)
        self.grid[row][col] = value

    def place_piece(self, row, col):
        self.set(row, col, PIECE)

    def remove_piece(self, row, col):
        self.set(row, col, EMPTY)

    def is_empty(self, row, col):
        return self.get(row, col) == EMPTY

    def to_physical(self, row, col):
        """Converte coordenada logica -> coordenada fisica no grid 17x17."""
        self._check(row, col)
        return row * 2 + PHYS_OFFSET, col * 2 + PHYS_OFFSET

    def _check(self, row, col):
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            raise IndexError(f"Coordenada ({row}, {col}) fora do tabuleiro 8x8.")

    def display(self):
        header = "   " + "  ".join(str(c) for c in range(BOARD_SIZE))
        print(header)
        print("  +" + "---" * BOARD_SIZE + "+")
        for r in range(BOARD_SIZE):
            row_str = " | ".join("P" if self.grid[r][c] == PIECE else "." for c in range(BOARD_SIZE))
            print(f"{r} | {row_str} |")
        print("  +" + "---" * BOARD_SIZE + "+")


class Board17x17:
    """Tabuleiro fisico 17x17, expansao do 8x8 com corredores.

    Layout (PHYS_OFFSET=1):
        posicoes com row%2==1 e col%2==1  ->  casas reais
        demais posicoes                    ->  corredores (vaos)
    """

    def __init__(self, board8=None):
        self.grid = [[EMPTY] * PHYS_SIZE for _ in range(PHYS_SIZE)]
        if board8 is not None:
            self.sync_from(board8)

    def sync_from(self, board8):
        """Copia o estado do tabuleiro logico para as celulas fisicas."""
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                pr, pc = board8.to_physical(r, c)
                self.grid[pr][pc] = board8.get(r, c)

    def get(self, row, col):
        self._check(row, col)
        return self.grid[row][col]

    def is_house(self, row, col):
        """True se a celula e uma casa real (indices fisicos impares)."""
        return (row % 2 == 1) and (col % 2 == 1)

    def is_corridor(self, row, col):
        return not self.is_house(row, col)

    def is_passable(self, row, col):
        """True se o robo pode passar por essa celula."""
        if not (0 <= row < PHYS_SIZE and 0 <= col < PHYS_SIZE):
            return False
        if self.is_corridor(row, col):
            return True   # corredores sao sempre livres
        return self.grid[row][col] == EMPTY  # casa livre

    def _check(self, row, col):
        if not (0 <= row < PHYS_SIZE and 0 <= col < PHYS_SIZE):
            raise IndexError(f"Coordenada fisica ({row}, {col}) fora do grid 17x17.")

    def display(self, path=None):
        """Imprime o tabuleiro fisico.

        Parameters
        ----------
        path : list of (row, col) | None
            Se fornecido, marca as celulas do caminho com '*'.
        """
        path_set = set(path) if path else set()
        print("Tabuleiro fisico 17x17")
        print("  H=casa vazia  P=peca  .=corredor  *=caminho")
        header = "    " + " ".join(f"{c:2d}" for c in range(PHYS_SIZE))
        print(header)
        for r in range(PHYS_SIZE):
            parts = []
            for c in range(PHYS_SIZE):
                if (r, c) in path_set:
                    parts.append(" *")
                elif self.is_house(r, c):
                    parts.append(" P" if self.grid[r][c] == PIECE else " H")
                else:
                    parts.append(" .")
            print(f"{r:2d} " + " ".join(parts))
