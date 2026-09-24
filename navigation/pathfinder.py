"""
pathfinder.py
=============
Algoritmo de pathfinding no tabuleiro fisico 17x17.

O robo parte de uma casa do tabuleiro logico 8x8 (origem) e precisa
chegar a outra casa (destino), navegando pelos corredores do grid 17x17.

Restricoes
----------
- O robo so se move horizontalmente ou verticalmente (4 direcoes).
- Corredores sao sempre transitaveis.
- Casas (house) so sao transitaveis se estiverem VAZIAS, exceto a casa
  de destino, que nao pode ser alcancada se estiver ocupada.
- Se o destino estiver ocupado, a funcao retorna None.

Retorno
-------
A funcao ``find_path`` retorna um dicionario com:
    {
        "moves_x": list[int],   # deslocamentos no eixo X (colunas)
        "moves_y": list[int],   # deslocamentos no eixo Y (linhas)
        "path"   : list[tuple], # sequencia de celulas fisicas percorridas
    }
ou None se o caminho nao existir / destino estiver ocupado.

Algoritmo
---------
Usa A* com heuristica de distancia de Manhattan no grid 17x17.

A* expande apenas as celulas mais promissoras (f = g + h), evitando
explorar na direcao oposta ao destino -- ao contrario do BFS que
explora em todas as direcoes igualmente.
"""

import heapq
from board import Board8x8, Board17x17, PIECE, BOARD_SIZE, PHYS_SIZE


# Direcoes de movimento: (delta_row, delta_col)
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def find_path(
    board8: Board8x8,
    origin: tuple,
    dest: tuple,
) -> dict | None:
    """Calcula o caminho no tabuleiro fisico 17x17 entre duas casas logicas.

    Parameters
    ----------
    board8  : Board8x8
        Estado atual do tabuleiro logico.
    origin  : (row, col)
        Casa de origem no tabuleiro 8x8 (posicao atual do robo).
    dest    : (row, col)
        Casa de destino no tabuleiro 8x8.

    Returns
    -------
    dict com chaves ``moves_x``, ``moves_y`` e ``path``, ou ``None`` se:
        - Destino estiver ocupado.
        - Nao existir caminho possivel.
    """
    or_r, or_c = origin
    ds_r, ds_c = dest

    # Valida limites
    for name, (r, c) in [("origem", origin), ("destino", dest)]:
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            raise ValueError(f"Coordenada de {name} ({r}, {c}) invalida para o tabuleiro 8x8.")

    # Se origem == destino, caminho trivial (antes de checar ocupacao)
    if origin == dest:
        pr, pc = board8.to_physical(or_r, or_c)
        return {"moves_x": [], "moves_y": [], "path": [(pr, pc)]}

    # Se destino ocupado, nao faz nada
    if board8.get(ds_r, ds_c) == PIECE:
        return None

    # Constroi o grid fisico
    board17 = Board17x17(board8)

    # Converte coordenadas logicas -> fisicas
    start = board8.to_physical(or_r, or_c)
    goal  = board8.to_physical(ds_r, ds_c)

    # A* no grid 17x17
    # f(n) = g(n) + h(n)
    #   g = custo real acumulado (numero de passos)
    #   h = heuristica de Manhattan ate o objetivo (admissivel e consistente)

    def h(a: tuple) -> int:
        return abs(a[0] - goal[0]) + abs(a[1] - goal[1])

    # heap: (f, g, celula)
    heap    = [(h(start), 0, start)]
    came_from = {start: None}   # celula -> celula anterior
    g_cost    = {start: 0}      # custo real ate cada celula

    while heap:
        f, g, cur = heapq.heappop(heap)

        if cur == goal:
            break

        # Celula ja processada com custo menor (entrada obsoleta no heap)
        if g > g_cost.get(cur, float('inf')):
            continue

        cur_r, cur_c = cur
        for dr, dc in _DIRS:
            nr, nc = cur_r + dr, cur_c + dc
            nxt = (nr, nc)

            # Celula destino: entrada permitida (sabemos que esta vazia)
            if nxt == goal:
                new_g = g + 1
                if new_g < g_cost.get(nxt, float('inf')):
                    g_cost[nxt]    = new_g
                    came_from[nxt] = cur
                    heapq.heappush(heap, (new_g + 0, new_g, nxt))  # h(goal)=0
                continue

            if not board17.is_passable(nr, nc):
                continue

            new_g = g + 1
            if new_g < g_cost.get(nxt, float('inf')):
                g_cost[nxt]    = new_g
                came_from[nxt] = cur
                heapq.heappush(heap, (new_g + h(nxt), new_g, nxt))

    if goal not in came_from:
        return None  # Caminho inexistente

    # Reconstroi o caminho
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()

    # Calcula vetores de movimento (delta entre celulas consecutivas)
    moves_x = []
    moves_y = []
    for i in range(1, len(path)):
        prev_r, prev_c = path[i - 1]
        cur_r,  cur_c  = path[i]
        moves_x.append(cur_c - prev_c)   # eixo X = colunas
        moves_y.append(cur_r - prev_r)   # eixo Y = linhas

    return {
        "moves_x": moves_x,
        "moves_y": moves_y,
        "path"   : path,
    }


def print_result(result: dict | None, board17: Board17x17 | None = None) -> None:
    """Exibe o resultado do pathfinding de forma legivel."""
    if result is None:
        print("[Pathfinder] Destino ocupado ou caminho inexistente. Nenhum movimento gerado.")
        return

    print(f"[Pathfinder] Caminho encontrado com {len(result['path'])} celulas fisicas.")
    print(f"  moves_x (delta col): {result['moves_x']}")
    print(f"  moves_y (delta row): {result['moves_y']}")

    if board17 is not None:
        board17.display(path=result["path"])
