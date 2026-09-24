"""
demo.py
=======
Demonstracao interativa do pathfinding com notacao de xadrez e animacao no terminal.

Uso:
    python demo.py

Durante a execucao o usuario digita comandos no formato:
    <ORIGEM> <DESTINO>   ex: A1 H8
    place <CASA>         ex: place C4   (coloca peca em uma casa)
    remove <CASA>        ex: remove C4  (remove peca de uma casa)
    reset                (limpa todo o tabuleiro)
    show                 (exibe o estado atual sem mover nada)
    quit / exit          (encerra)

Convencao de coordenadas (xadrez):
    Coluna: A..H  ->  col 0..7
    Linha:  1..8  ->  row 7..0  (1 = base, 8 = topo)
"""

import os
import sys
import time

from board import Board8x8, Board17x17, PIECE, EMPTY, PHYS_SIZE, BOARD_SIZE
from pathfinder import find_path

# ─── Constantes de animacao ───────────────────────────────────────────────────
FRAME_DELAY   = 0.08   # segundos entre frames
ROBOT_SYMBOL  = "R"    # simbolo do robo no grid fisico

# ─── ANSI helpers ─────────────────────────────────────────────────────────────
def _ansi(code): return f"\033[{code}m"

RESET  = _ansi(0)
BOLD   = _ansi(1)
DIM    = _ansi(2)
RED    = _ansi(31)
GREEN  = _ansi(32)
YELLOW = _ansi(33)
CYAN   = _ansi(36)
WHITE  = _ansi(37)
BG_DARK   = _ansi("48;5;234")   # fundo escuro
BG_HOUSE  = _ansi("48;5;239")   # casa vazia
BG_PIECE  = _ansi("48;5;52")    # casa com peca (vermelho escuro)
BG_ROBOT  = _ansi("48;5;22")    # posicao do robo (verde escuro)
BG_TRAIL  = _ansi("48;5;17")    # rastro do caminho (azul escuro)
BG_CORR   = _ansi("48;5;235")   # corredor

COL_LABELS = "ABCDEFGH"

# Detecta se o terminal suporta ANSI (Windows precisa do modo virtual)
def _enable_ansi():
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.SetConsoleMode(kernel.GetStdHandle(-11), 7)

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def hide_cursor():
    print("\033[?25l", end="", flush=True)

def show_cursor():
    print("\033[?25h", end="", flush=True)

def move_cursor_home():
    print("\033[H", end="", flush=True)


# ─── Conversao de notacao ─────────────────────────────────────────────────────

def notation_to_logical(s: str) -> tuple:
    """Converte 'A1'..'H8' para (row, col) do tabuleiro 8x8.

    Coluna: A=0, B=1, ..., H=7
    Linha:  1=row7, 2=row6, ..., 8=row0
    """
    s = s.strip().upper()
    if len(s) != 2:
        raise ValueError(f"Notacao invalida: '{s}'. Use ex: A1, H8.")
    col_char, row_char = s[0], s[1]
    if col_char not in COL_LABELS:
        raise ValueError(f"Coluna invalida: '{col_char}'. Use A..H.")
    if row_char not in "12345678":
        raise ValueError(f"Linha invalida: '{row_char}'. Use 1..8.")
    col = COL_LABELS.index(col_char)
    row = 8 - int(row_char)   # 1->row7, 8->row0
    return (row, col)

def logical_to_notation(row: int, col: int) -> str:
    return f"{COL_LABELS[col]}{8 - row}"


# ─── Renderizacao ─────────────────────────────────────────────────────────────

def _cell_str(board17: Board17x17, pr: int, pc: int,
              robot_pos=None, trail_set=None) -> str:
    """Retorna a string colorida para uma celula do grid fisico."""
    is_house = board17.is_house(pr, pc)
    val      = board17.get(pr, pc)

    if (pr, pc) == robot_pos:
        return f"{BG_ROBOT}{BOLD}{GREEN} {ROBOT_SYMBOL} {RESET}"
    if trail_set and (pr, pc) in trail_set:
        if is_house:
            return f"{BG_TRAIL}{BOLD}{CYAN} \xb7\xb7 {RESET}"
        return f"{BG_TRAIL}{DIM}{CYAN} .. {RESET}"
    if is_house:
        if val == PIECE:
            return f"{BG_PIECE}{BOLD}{RED} \u2659\u2659 {RESET}"
        return f"{BG_HOUSE}{DIM}{WHITE} \u25a1\u25a1 {RESET}"
    # corredor
    return f"{BG_CORR}    {RESET}"


def render_boards(board8: Board8x8, board17: Board17x17,
                  robot_pos=None, trail_set=None,
                  message: str = "") -> str:
    """Monta a string completa da tela (tabuleiros lado a lado + mensagem)."""
    lines = []

    # ── Cabecalho ──
    lines.append(f"{BOLD}{CYAN}  Tabuleiro 8x8 (logico){RESET}           "
                 f"{BOLD}{CYAN}  Tabuleiro 17x17 (fisico){RESET}")

    # ── Cabecalho de colunas 8x8 ──
    hdr8  = "    " + "  ".join(f"{BOLD}{YELLOW}{c}{RESET}" for c in COL_LABELS)
    hdr17 = "  " + " ".join(f"{BOLD}{YELLOW}{c:2d}{RESET}" for c in range(PHYS_SIZE))
    lines.append(hdr8 + "      " + hdr17)
    lines.append("")

    for r in range(BOARD_SIZE):
        # coluna esquerda: tabuleiro 8x8
        row_label = f"{BOLD}{YELLOW}{8 - r}{RESET}"
        cells8 = []
        for c in range(BOARD_SIZE):
            if board8.get(r, c) == PIECE:
                cells8.append(f"{RED}\u2659{RESET}")
            else:
                cells8.append(f"{DIM}.{RESET}")
        row8_str = f"  {row_label}  " + "  ".join(cells8)

        lines.append(row8_str)

    lines.append("")

    # ── Grid fisico 17x17 (exibido abaixo) ──
    lines.append(f"{BOLD}{CYAN}  Grid fisico 17x17:{RESET}")
    lines.append("    " + " ".join(f"{BOLD}{YELLOW}{c:2d}{RESET}" for c in range(PHYS_SIZE)))

    for pr in range(PHYS_SIZE):
        row_str = f"{BOLD}{YELLOW}{pr:2d}{RESET}  "
        row_str += "".join(_cell_str(board17, pr, pc, robot_pos, trail_set)
                           for pc in range(PHYS_SIZE))
        lines.append(row_str)

    lines.append("")
    if message:
        lines.append(f"  {message}")
    lines.append(f"  {DIM}Comandos: <A1 H8> | place <C4> | remove <C4> | reset | show | quit{RESET}")
    lines.append("")

    return "\n".join(lines)


def draw(board8, board17, robot_pos=None, trail_set=None, message=""):
    move_cursor_home()
    print(render_boards(board8, board17, robot_pos, trail_set, message),
          end="", flush=True)


# ─── Animacao ─────────────────────────────────────────────────────────────────

def animate(board8: Board8x8, origin_logical: tuple, path: list):
    """Anima o robo percorrendo o caminho fisico."""
    board17 = Board17x17(board8)

    # Conjunto de todas as celulas do caminho (para mostrar o rastro planejado)
    full_trail = set(path)
    visited_trail = set()

    orig_note = logical_to_notation(*origin_logical)

    for i, pos in enumerate(path):
        visited_trail.add(pos)
        remaining_trail = full_trail - visited_trail

        # Constroi a mensagem de progresso
        pct = int(100 * i / max(len(path) - 1, 1))
        bar_len = 20
        filled = int(bar_len * i / max(len(path) - 1, 1))
        bar = f"{GREEN}{'=' * filled}{DIM}{'-' * (bar_len - filled)}{RESET}"
        msg = (f"{BOLD}{CYAN}Movendo {orig_note}{RESET}  "
               f"[{bar}] {pct:3d}%  "
               f"Celula fisica {pos}")

        draw(board8, board17,
             robot_pos=pos,
             trail_set=remaining_trail,
             message=msg)
        time.sleep(FRAME_DELAY)

    # Frame final: robo na posicao de chegada, sem rastro
    dest_note = logical_to_notation(*_phys_to_logical(path[-1]))
    draw(board8, board17,
         robot_pos=path[-1],
         trail_set=set(),
         message=f"{BOLD}{GREEN}Chegou em {dest_note}!{RESET}  (pressione Enter para continuar)")
    input()


def _phys_to_logical(phys: tuple) -> tuple:
    """Converte celula fisica de volta para logica (apenas para casas)."""
    pr, pc = phys
    return (pr - 1) // 2, (pc - 1) // 2


# ─── Loop principal ───────────────────────────────────────────────────────────

def main():
    _enable_ansi()
    hide_cursor()
    clear_screen()

    board8  = Board8x8()
    board17 = Board17x17(board8)

    # Posicao atual do robo (comeca na A1 = (7,0))
    robot_logical = (7, 0)
    board8.place_piece(*robot_logical)
    board17.sync_from(board8)

    message = (f"{BOLD}{GREEN}Robo posicionado em {logical_to_notation(*robot_logical)}.{RESET}  "
               f"Digite um comando:")

    try:
        while True:
            draw(board8, board17,
                 robot_pos=board8.to_physical(*robot_logical),
                 message=message)

            try:
                raw = input(f"  {BOLD}>{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not raw:
                message = ""
                continue

            parts = raw.upper().split()
            cmd   = parts[0]

            # ── quit ──
            if cmd in ("QUIT", "EXIT", "Q"):
                break

            # ── reset ──
            elif cmd == "RESET":
                board8  = Board8x8()
                robot_logical = (7, 0)
                board8.place_piece(*robot_logical)
                board17 = Board17x17(board8)
                message = f"{GREEN}Tabuleiro resetado. Robo em {logical_to_notation(*robot_logical)}.{RESET}"

            # ── show ──
            elif cmd == "SHOW":
                message = f"{CYAN}Estado atual do tabuleiro.{RESET}"

            # ── place ──
            elif cmd == "PLACE" and len(parts) == 2:
                try:
                    r, c = notation_to_logical(parts[1])
                    board8.place_piece(r, c)
                    board17.sync_from(board8)
                    message = f"{YELLOW}Peca colocada em {logical_to_notation(r, c)}.{RESET}"
                except (ValueError, IndexError) as e:
                    message = f"{RED}Erro: {e}{RESET}"

            # ── remove ──
            elif cmd == "REMOVE" and len(parts) == 2:
                try:
                    r, c = notation_to_logical(parts[1])
                    board8.remove_piece(r, c)
                    board17.sync_from(board8)
                    message = f"{YELLOW}Peca removida de {logical_to_notation(r, c)}.{RESET}"
                except (ValueError, IndexError) as e:
                    message = f"{RED}Erro: {e}{RESET}"

            # ── movimento: ORIGEM DESTINO ──
            elif len(parts) == 2:
                try:
                    origin = notation_to_logical(parts[0])
                    dest   = notation_to_logical(parts[1])

                    # Verifica se ha peca na origem
                    if board8.get(*origin) != PIECE:
                        message = (f"{RED}Nao ha peca em {logical_to_notation(*origin)}. "
                                   f"Use 'place {parts[0]}' primeiro.{RESET}")
                        continue

                    result = find_path(board8, origin=origin, dest=dest)

                    if result is None:
                        dest_note = logical_to_notation(*dest)
                        if board8.get(*dest) == PIECE:
                            message = f"{RED}Destino {dest_note} esta ocupado. Movimento cancelado.{RESET}"
                        else:
                            message = f"{RED}Nao foi encontrado caminho ate {dest_note}.{RESET}"
                        continue

                    # Executa animacao
                    clear_screen()
                    animate(board8, origin, result["path"])

                    # Atualiza estado logico apos o movimento
                    board8.remove_piece(*origin)
                    board8.place_piece(*dest)
                    board17.sync_from(board8)
                    robot_logical = dest

                    dest_note = logical_to_notation(*dest)
                    orig_note = logical_to_notation(*origin)
                    message = (f"{GREEN}Movimento concluido: {orig_note} -> {dest_note}.{RESET}  "
                               f"moves_x={result['moves_x']}  moves_y={result['moves_y']}")

                    clear_screen()

                except (ValueError, IndexError) as e:
                    message = f"{RED}Erro: {e}{RESET}"

            else:
                message = f"{RED}Comando nao reconhecido: '{raw}'. Use ex: A1 H8, place C4, remove C4, reset, quit.{RESET}"

    finally:
        show_cursor()
        clear_screen()
        print("Encerrando. Ate logo!")


if __name__ == "__main__":
    main()
