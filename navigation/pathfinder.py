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
from board import (Board8x8, Board17x17, BoardLogic, BoardPhysical,
                   PIECE, BOARD_ROWS, BOARD_COLS, PHYS_ROWS, PHYS_COLS,
                   # aliases de compat
                   BOARD_SIZE, PHYS_SIZE)


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

    # A* no grid 17x17 (com estado de direcao para minimizar trocas de eixo)
    def h(a: tuple) -> int:
        return abs(a[0] - goal[0]) + abs(a[1] - goal[1])

    TURN_PENALTY = 6.0
    start_state = (start[0], start[1], -1)
    heap      = [(float(h(start)), 0.0, 0, start_state)]
    came_from = {start_state: None}
    g_cost    = {start_state: 0.0}

    goal_state = None
    while heap:
        f, g, turns, cur_state = heapq.heappop(heap)
        cur_r, cur_c, pdir = cur_state

        if (cur_r, cur_c) == goal:
            goal_state = cur_state
            break

        if g > g_cost.get(cur_state, float('inf')):
            continue

        for d_idx, (dr, dc) in enumerate(_DIRS):
            nr, nc = cur_r + dr, cur_c + dc
            nxt = (nr, nc)
            is_turn = (pdir != -1 and d_idx != pdir)
            turn_cost = TURN_PENALTY if is_turn else 0.0
            new_turns = turns + (1 if is_turn else 0)
            nxt_state = (nr, nc, d_idx)

            # Celula destino: entrada permitida (sabemos que esta vazia)
            if nxt == goal:
                new_g = g + 1.0 + turn_cost
                if new_g < g_cost.get(nxt_state, float('inf')):
                    g_cost[nxt_state]    = new_g
                    came_from[nxt_state] = cur_state
                    heapq.heappush(heap, (new_g, new_g, new_turns, nxt_state))
                continue

            if not board17.is_passable(nr, nc):
                continue

            new_g = g + 1.0 + turn_cost
            if new_g < g_cost.get(nxt_state, float('inf')):
                g_cost[nxt_state]    = new_g
                came_from[nxt_state] = cur_state
                heapq.heappush(heap, (new_g + h(nxt), new_g, new_turns, nxt_state))

    if goal_state is None:
        return None  # Caminho inexistente

    # Reconstroi o caminho
    path = []
    cur = goal_state
    while cur is not None:
        path.append((cur[0], cur[1]))
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


def find_path_physical(
    board_logic,
    start_phys: tuple[int, int],
    goal_phys: tuple[int, int],
) -> dict | None:
    """Calcula o caminho no grid fisico 15x27 entre quaisquer duas coordenadas fisicas (pr, pc)."""
    from board import BoardPhysical, PHYS_ROWS, PHYS_COLS

    sr, sc = start_phys
    gr, gc = goal_phys

    if not (0 <= sr < PHYS_ROWS and 0 <= sc < PHYS_COLS):
        raise ValueError(f"Origem fisica ({sr},{sc}) fora do grid.")
    if not (0 <= gr < PHYS_ROWS and 0 <= gc < PHYS_COLS):
        raise ValueError(f"Destino fisico ({gr},{gc}) fora do grid.")

    if start_phys == goal_phys:
        return {"moves_x": [], "moves_y": [], "path": [start_phys]}

    phys = BoardPhysical(board_logic)
    start = start_phys
    goal = goal_phys

    def h(a):
        return abs(a[0] - goal[0]) + abs(a[1] - goal[1])

    # Estado no grafo A*: (row, col, dir_idx) onde dir_idx in {-1, 0, 1, 2, 3}.
    # Incluir a direcao de chegada no estado permite ao A* encontrar o caminho com
    # o MENOR numero de mudancas de eixo (ex.: X -> Y -> X em 3 retas no Roque,
    # em vez de Y -> X -> Y -> X em 4 retas).
    start_state = (start[0], start[1], -1)
    heap = [(float(h(start)), 0.0, 0, start_state)]
    came_from = {start_state: None}
    g_cost = {start_state: 0.0}

    ADJACENT_PENALTY = 3.0
    TURN_PENALTY = 6.0

    goal_state = None
    while heap:
        f, g, turns, cur_state = heapq.heappop(heap)
        cr, cc, pdir = cur_state
        if (cr, cc) == goal:
            goal_state = cur_state
            break
        if g > g_cost.get(cur_state, float("inf")):
            continue

        for d_idx, (dr, dc) in enumerate(_DIRS):
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < PHYS_ROWS and 0 <= nc < PHYS_COLS):
                continue
            is_turn = (pdir != -1 and d_idx != pdir)
            turn_cost = TURN_PENALTY if is_turn else 0.0
            new_turns = turns + (1 if is_turn else 0)
            nxt_state = (nr, nc, d_idx)

            if (nr, nc) == goal:
                new_g = g + 1.0 + turn_cost
                if new_g < g_cost.get(nxt_state, float("inf")):
                    g_cost[nxt_state] = new_g
                    came_from[nxt_state] = cur_state
                    heapq.heappush(heap, (new_g, new_g, new_turns, nxt_state))
                continue
            if not phys.is_passable(nr, nc):
                continue

            step_cost = 1.0 + turn_cost
            if getattr(phys, "piece_adjacent", None) and phys.piece_adjacent[nr][nc]:
                dist_start = abs(nr - sr) + abs(nc - sc)
                dist_goal = abs(nr - gr) + abs(nc - gc)
                if dist_start > 1 and dist_goal > 1:
                    step_cost += ADJACENT_PENALTY

            new_g = g + step_cost
            if new_g < g_cost.get(nxt_state, float("inf")):
                g_cost[nxt_state] = new_g
                came_from[nxt_state] = cur_state
                heapq.heappush(heap, (new_g + h((nr, nc)), new_g, new_turns, nxt_state))

    if goal_state is None:
        return None

    path = []
    cur = goal_state
    while cur is not None:
        path.append((cur[0], cur[1]))
        cur = came_from[cur]
    path.reverse()

    moves_x, moves_y = [], []
    for i in range(1, len(path)):
        pr, pc = path[i - 1]
        cr, cc = path[i]
        moves_x.append(cc - pc)
        moves_y.append(cr - pr)

    return {"moves_x": moves_x, "moves_y": moves_y, "path": path}


def find_path_logical(
    board_logic,
    origin_logical: tuple,
    dest_logical: tuple,
) -> dict | None:
    """Calcula o caminho no grid fisico 15x27 entre duas coordenadas logicas globais."""
    from board import LOGIC_ROWS, LOGIC_COLS

    or_r, or_c = origin_logical
    ds_r, ds_c = dest_logical

    if not (0 <= or_r < LOGIC_ROWS and 0 <= or_c < LOGIC_COLS):
        raise ValueError(f"Origem logica ({or_r},{or_c}) fora do grid.")
    if not (0 <= ds_r < LOGIC_ROWS and 0 <= ds_c < LOGIC_COLS):
        raise ValueError(f"Destino logico ({ds_r},{ds_c}) fora do grid.")

    start = board_logic.to_physical(or_r, or_c)
    goal = board_logic.to_physical(ds_r, ds_c)
    return find_path_physical(board_logic, start, goal)


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
