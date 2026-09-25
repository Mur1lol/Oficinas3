"""
main.py
=======
Integracao do Stockfish com o sistema de navegacao robotica.

Fluxo de cada lance:
    1. Stockfish (ou humano) escolhe um lance UCI (ex: e2e4, e7e5, d1h5).
    2. Se for captura:
        a. O robo vai ate a casa da peca capturada.
        b. Leva a peca capturada ate a proxima vaga no cemiterio.
        c. Volta para a casa de origem da peca que vai mover.
    3. O robo move a peca da origem ate o destino.
    4. Estado logico e visual sao atualizados.

Uso:
    python main.py                   # Stockfish x Stockfish (auto)
    python main.py --modo humano     # Humano (brancas) x Stockfish
    python main.py --lances 20       # limita numero de lances (padrao: 40)
    python main.py --delay 0.05      # velocidade de animacao em segundos
"""

import os, sys, time, argparse, heapq
import chess, chess.engine

# Forca UTF-8 no terminal Windows (evita UnicodeEncodeError com simbolos de xadrez)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


from board import (
    BoardLogic, BoardPhysical, Board8x8, Board17x17,
    PIECE, EMPTY,
    BOARD_ROWS, BOARD_COLS, PHYS_ROWS, PHYS_COLS,
    LOGIC_ROWS, LOGIC_COLS,
    COL_BOARD_START,
    COL_CEM_ESQ_START, COL_CEM_ESQ_END,
    COL_CEM_DIR_START, COL_CEM_DIR_END,
    COL_VAO_ESQ, COL_VAO_DIR,
    ZONE_BOARD, ZONE_CEM_ESQ, ZONE_CEM_DIR, ZONE_VAO,
    zone_of,
)
from pathfinder import find_path_logical

# ─── Caminho do Stockfish ─────────────────────────────────────────────────────

STOCKFISH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "stockfish-windows-x86-64-universal", "stockfish",
    "stockfish-windows-x86-64-universal.exe"
)

FRAME_DELAY  = 0.07
ROBOT_SYMBOL = "R"

# ─── ANSI ─────────────────────────────────────────────────────────────────────

def _ansi(c): return f"\033[{c}m"

RESET   = _ansi(0);  BOLD  = _ansi(1);  DIM    = _ansi(2)
RED     = _ansi(31); GREEN = _ansi(32); YELLOW = _ansi(33)
BLUE    = _ansi(34); MAGENTA = _ansi(35); CYAN  = _ansi(36); WHITE = _ansi(37)

BG_HOUSE   = _ansi("48;5;239"); BG_PIECE  = _ansi("48;5;52")
BG_ROBOT   = _ansi("48;5;22");  BG_TRAIL  = _ansi("48;5;17")
BG_CORR    = _ansi("48;5;235"); BG_CEM    = _ansi("48;5;236")
BG_CEM_OCC = _ansi("48;5;88");  BG_VAO    = _ansi("48;5;234")

def _enable_ansi():
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        k.SetConsoleMode(k.GetStdHandle(-11), 7)

clear_screen = lambda: os.system("cls" if os.name == "nt" else "clear")
hide_cursor  = lambda: print("\033[?25l", end="", flush=True)
show_cursor  = lambda: print("\033[?25h", end="", flush=True)
cursor_home  = lambda: print("\033[H",    end="", flush=True)

# ─── Mapeamento chess <-> logico ──────────────────────────────────────────────

COL_LABELS = "ABCDEFGH"

_PIECE_SYM = {
    chess.PAWN:   ("♙","♟"), chess.KNIGHT: ("♘","♞"),
    chess.BISHOP: ("♗","♝"), chess.ROOK:   ("♖","♜"),
    chess.QUEEN:  ("♕","♛"), chess.KING:   ("♔","♚"),
}

def chess_sq_to_sub(sq):
    """chess.Square -> (row, col) do sub-tabuleiro 8x8 (row0=rank8)."""
    return 7 - chess.square_rank(sq), chess.square_file(sq)

def sub_to_global(row, col):
    return row, COL_BOARD_START + col

def sq_to_global(sq):
    r, c = chess_sq_to_sub(sq)
    return sub_to_global(r, c)

def notation(row, col):
    return f"{COL_LABELS[col]}{8-row}"

def piece_sym(p: chess.Piece):
    w, b = _PIECE_SYM.get(p.piece_type, ("?","?"))
    return w if p.color == chess.WHITE else b

# ─── Estado do jogo ───────────────────────────────────────────────────────────

class GameState:
    def __init__(self):
        self.cb        = chess.Board()
        self.logic     = BoardLogic()
        self.move_log  = []
        self._sync()

    def _sync(self):
        self.logic = BoardLogic()
        for sq in chess.SQUARES:
            p = self.cb.piece_at(sq)
            if p:
                gr, gc = sq_to_global(sq)
                self.logic.grid[gr][gc] = PIECE

    def apply(self, move: chess.Move):
        san   = self.cb.san(move)
        side  = "white" if self.cb.turn == chess.WHITE else "black"
        is_cap= self.cb.is_capture(move)
        self.cb.push(move)
        self._sync()
        # Repoe pecas no cemiterio (sync limpou tudo)
        for r, c in self._cem_white:
            self.logic.grid[r][c] = PIECE
        for r, c in self._cem_black:
            self.logic.grid[r][c] = PIECE
        self.move_log.append({"san": san, "side": side, "is_capture": is_cap})

    _cem_white: list = []   # vagas usadas no cem_esq (brancas capturadas)
    _cem_black: list = []   # vagas usadas no cem_dir (negras capturadas)

    def __init__(self):
        self.cb        = chess.Board()
        self.logic     = BoardLogic()
        self.move_log  = []
        self._cem_white = []
        self._cem_black = []
        self._sync()

# ─── Renderizacao ─────────────────────────────────────────────────────────────

def _cell(logic, pr, pc, robot_pos, trail_set, cb):
    is_real = (pr % 2 == 1) and (pc % 2 == 1)
    lr = (pr - 1) // 2
    lc = (pc - 1) // 2

    if (pr, pc) == robot_pos:
        return f"{BG_ROBOT}{BOLD}{GREEN} {ROBOT_SYMBOL}  {RESET}"
    if trail_set and (pr, pc) in trail_set:
        return f"{BG_TRAIL}{DIM}{CYAN} ·· {RESET}"

    if not is_real:
        zc = min(max(pc // 2, 0), LOGIC_COLS - 1)
        z = zone_of(zc)
        if z == ZONE_VAO:             return f"{BG_VAO}    {RESET}"
        if z in (ZONE_CEM_ESQ, ZONE_CEM_DIR): return f"{BG_CEM}    {RESET}"
        return f"{BG_CORR}    {RESET}"

    z = zone_of(lc)
    val = logic.grid[lr][lc]

    if z == ZONE_VAO:
        return f"{BG_VAO}    {RESET}"
    if z in (ZONE_CEM_ESQ, ZONE_CEM_DIR):
        if val == PIECE: return f"{BG_CEM_OCC}{RED} ✕  {RESET}"
        return f"{BG_CEM}{DIM}    {RESET}"

    # Tabuleiro de jogo
    bc = lc - COL_BOARD_START
    br = lr
    sq = chess.square(bc, 7 - br)
    p  = cb.piece_at(sq) if cb else None
    if p:
        col = RED if p.color == chess.BLACK else WHITE
        return f"{BG_PIECE}{BOLD}{col} {piece_sym(p)}  {RESET}"
    return f"{BG_HOUSE}{DIM}{WHITE} □  {RESET}"


def render(logic, cb, robot_pos=None, trail_set=None,
           message="", move_log=None, eval_score=""):
    lines = []
    lines.append(f"\n {BOLD}{CYAN}╔══════════════════════════════════════╗{RESET}")
    lines.append(f" {BOLD}{CYAN}║  Stockfish ×  Navegacao Robotica    ║{RESET}")
    lines.append(f" {BOLD}{CYAN}╚══════════════════════════════════════╝{RESET}\n")

    turn_s = f"{WHITE}Brancas{RESET}" if cb.turn==chess.WHITE else f"{DIM}Negras{RESET}"
    lines.append(f"  {BOLD}Turno:{RESET} {turn_s}   "
                 f"{BOLD}Aval:{RESET} {CYAN}{eval_score or '---'}{RESET}   "
                 f"{BOLD}Lance #{RESET}{len(move_log or [])+1}")
    lines.append("")

    # Tabuleiro 8x8 em notacao
    lines.append("  " + "  ".join(f"{BOLD}{YELLOW}{c}{RESET}" for c in "ABCDEFGH"))
    lines.append("  " + "─"*24)
    for rank in range(7, -1, -1):
        parts = []
        for file in range(8):
            sq = chess.square(file, rank)
            p  = cb.piece_at(sq)
            if p:
                col = RED if p.color==chess.BLACK else WHITE
                parts.append(f"{BOLD}{col}{piece_sym(p)}{RESET}")
            else:
                parts.append(f"{DIM}{'·' if (rank+file)%2==0 else '░'}{RESET}")
        lines.append(f"  {'  '.join(parts)} {BOLD}{YELLOW}{rank+1}{RESET}")
    lines.append("")

    # Grid fisico
    lines.append(f"  {BOLD}{CYAN}Grid fisico {PHYS_ROWS}×{PHYS_COLS}  "
                 f"[{MAGENTA}CEM{CYAN}|{YELLOW}VAO{CYAN}|{GREEN}TABULEIRO"
                 f"{CYAN}|{YELLOW}VAO{CYAN}|{MAGENTA}CEM{CYAN}]{RESET}")

    hdr = "     "
    for pc in range(PHYS_COLS):
        if pc % 2 == 1:
            lc = (pc-1)//2
            z = zone_of(lc)
            colors = {ZONE_CEM_ESQ: MAGENTA, ZONE_CEM_DIR: MAGENTA,
                      ZONE_VAO: YELLOW, ZONE_BOARD: GREEN}
            hdr += f"{colors.get(z, WHITE)}{lc:2d}{RESET} "
        else:
            hdr += "   "
    lines.append(hdr)

    for pr in range(PHYS_ROWS):
        row = f"  {BOLD}{YELLOW}{pr:2d}{RESET} "
        row += "".join(_cell(logic, pr, pc, robot_pos, trail_set, cb)
                       for pc in range(PHYS_COLS))
        lines.append(row)
    lines.append("")

    if move_log:
        recent = move_log[-6:]
        lines.append(f"  {BOLD}Historico:{RESET}")
        for i, m in enumerate(recent):
            idx = len(move_log)-len(recent)+i+1
            sym = f"{WHITE}♙{RESET}" if m["side"]=="white" else f"{DIM}♟{RESET}"
            cap = f" {RED}✕{RESET}" if m.get("is_capture") else ""
            lines.append(f"    {DIM}{idx:2d}.{RESET} {sym} {BOLD}{m['san']}{RESET}{cap}")
    lines.append("")
    lines.append(f"  {MAGENTA}CEM_ESQ{RESET}=pecas brancas capturadas(cols 0-2)  "
                 f"{MAGENTA}CEM_DIR{RESET}=negras capturadas(cols 13-15)")
    lines.append("")
    if message: lines.append(f"  {message}")
    lines.append("")
    return "\n".join(lines)


def draw(logic, cb, robot_pos=None, trail_set=None,
         message="", move_log=None, eval_score=""):
    cursor_home()
    print(render(logic, cb, robot_pos, trail_set, message, move_log, eval_score),
          end="", flush=True)

# ─── Animacao ─────────────────────────────────────────────────────────────────

def animate(logic, cb, path, label, move_log, eval_score, delay):
    full = set(path); visited = set()
    for i, pos in enumerate(path):
        visited.add(pos)
        pct = int(100*i/max(len(path)-1, 1))
        filled = int(20*i/max(len(path)-1,1))
        bar = f"{GREEN}{'█'*filled}{DIM}{'─'*(20-filled)}{RESET}"
        draw(logic, cb, robot_pos=pos, trail_set=full-visited,
             message=f"{BOLD}{CYAN}{label}{RESET}  [{bar}] {pct:3d}%",
             move_log=move_log, eval_score=eval_score)
        time.sleep(delay)
    draw(logic, cb, robot_pos=path[-1], trail_set=set(),
         message=f"{BOLD}{GREEN}{label} — chegou!{RESET}",
         move_log=move_log, eval_score=eval_score)
    time.sleep(delay*3)

# ─── Pathfinding auxiliar ─────────────────────────────────────────────────────

def path_between(logic: BoardLogic, orig, dest, allow_occupied_dest=False):
    """Wrapper que libera dest do grid antes de calcular o caminho."""
    snap = logic.grid[dest[0]][dest[1]]
    if allow_occupied_dest:
        logic.grid[dest[0]][dest[1]] = EMPTY
    result = find_path_logical(logic, orig, dest)
    logic.grid[dest[0]][dest[1]] = snap
    return result

# ─── Execucao de um lance ─────────────────────────────────────────────────────

def execute_move(state: GameState, move: chess.Move,
                 delay: float, eval_score: str):
    cb  = state.cb          # estado ANTES do lance
    log = state.move_log

    # Copia do grid para animacao (sem alterar o estado oficial ainda)
    pre = BoardLogic()
    pre.grid = [row[:] for row in state.logic.grid]
    pre._cem_white = list(state._cem_white)
    pre._cem_black = list(state._cem_black)

    moving_side  = "white" if cb.turn == chess.WHITE else "black"
    is_cap       = cb.is_capture(move)
    is_ep        = cb.is_en_passant(move)
    san          = cb.san(move)

    fr_sub  = chess_sq_to_sub(move.from_square)
    to_sub  = chess_sq_to_sub(move.to_square)
    fr_glob = sub_to_global(*fr_sub)
    to_glob = sub_to_global(*to_sub)

    robo_pos = fr_glob   # posicao atual do robo (logica global)

    if is_cap:
        # Casa da peca capturada
        if is_ep:
            ep_rank = 4 if moving_side=="white" else 3
            cap_sub  = (ep_rank, to_sub[1])
        else:
            cap_sub = to_sub
        cap_glob = sub_to_global(*cap_sub)

        side_cem = "right" if moving_side=="white" else "left"
        cem_slot = pre.next_cemetery_slot(side_cem)

        if cem_slot:
            # Passo 1: Robo vai ate a peca capturada
            if robo_pos != cap_glob:
                res = path_between(pre, robo_pos, cap_glob, allow_occupied_dest=True)
                if res:
                    animate(pre, cb, res["path"],
                            f"Buscando peca capturada em {notation(*cap_sub)}",
                            log, eval_score, delay)
                    robo_pos = cap_glob

            # Passo 2: Leva peca capturada ao cemiterio
            pre.grid[cap_glob[0]][cap_glob[1]] = EMPTY
            res = path_between(pre, robo_pos, cem_slot, allow_occupied_dest=False)
            if res:
                animate(pre, cb, res["path"],
                        f"Levando ao cemiterio {cem_slot}",
                        log, eval_score, delay)
                robo_pos = cem_slot

            pre.grid[cem_slot[0]][cem_slot[1]] = PIECE
            if moving_side == "white":
                pre._cem_black.append(cem_slot)
            else:
                pre._cem_white.append(cem_slot)

            # Passo 3: Volta para a casa de origem
            if robo_pos != fr_glob:
                res = path_between(pre, robo_pos, fr_glob, allow_occupied_dest=True)
                if res:
                    animate(pre, cb, res["path"],
                            f"Voltando para {notation(*fr_sub)}",
                            log, eval_score, delay)
                    robo_pos = fr_glob

    # Passo 4: Move a peca para o destino
    pre.grid[to_glob[0]][to_glob[1]] = EMPTY
    res = path_between(pre, robo_pos, to_glob, allow_occupied_dest=False)
    if res:
        side_str = f"{WHITE}Brancas{RESET}" if moving_side=="white" else f"{DIM}Negras{RESET}"
        animate(pre, cb, res["path"],
                f"{side_str}: {BOLD}{san}{RESET}  "
                f"({notation(*fr_sub)}→{notation(*to_sub)})",
                log, eval_score, delay)

    # Aplica o lance no estado oficial
    state._cem_white = pre._cem_white
    state._cem_black = pre._cem_black
    state.cb.push(move)
    state._sync()
    # Repoe cemiterios
    for r, c in state._cem_white:
        state.logic.grid[r][c] = PIECE
    for r, c in state._cem_black:
        state.logic.grid[r][c] = PIECE
    state.move_log.append({"san": san, "side": moving_side,
                           "is_capture": is_cap})

    draw(state.logic, state.cb,
         message=f"{BOLD}{GREEN}Lance aplicado: {san}{RESET}",
         move_log=state.move_log, eval_score=eval_score)
    time.sleep(0.4)

# ─── Stockfish ────────────────────────────────────────────────────────────────

def sf_move(engine, board, tempo=0.3):
    result = engine.analyse(board, chess.engine.Limit(time=tempo))
    move   = result["pv"][0] if result.get("pv") else None
    score  = result.get("score")
    eval_s = ""
    if score:
        pov = score.white()
        if pov.is_mate(): eval_s = f"M{pov.mate()}"
        else:
            cp = pov.score()
            eval_s = f"{'+' if cp>=0 else ''}{cp/100:.2f}"
    return move, eval_s

# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stockfish + Navegacao Robotica")
    p.add_argument("--modo", choices=["auto","humano"], default="auto")
    p.add_argument("--lances", type=int, default=40)
    p.add_argument("--delay",  type=float, default=FRAME_DELAY)
    p.add_argument("--tempo",  type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    _enable_ansi(); hide_cursor(); clear_screen()

    print(f"\n {BOLD}{CYAN}Inicializando Stockfish...{RESET}", flush=True)
    sf_path = os.path.normpath(STOCKFISH_PATH)
    if not os.path.isfile(sf_path):
        show_cursor()
        print(f"\n{RED}Stockfish nao encontrado:{RESET} {sf_path}")
        sys.exit(1)

    try:
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    except Exception as e:
        show_cursor(); print(f"\n{RED}Erro ao abrir Stockfish:{RESET} {e}"); sys.exit(1)

    state = GameState(); eval_score = ""; n = 0

    try:
        clear_screen()
        draw(state.logic, state.cb,
             message=f"{BOLD}{GREEN}Partida iniciada! Modo: {args.modo}{RESET}",
             move_log=state.move_log)

        while not state.cb.is_game_over() and n < args.lances:
            n += 1
            is_white = state.cb.turn == chess.WHITE

            if args.modo == "humano" and is_white:
                cursor_home()
                print(render(state.logic, state.cb,
                             message=f"  {BOLD}Seu lance (UCI, ex: e2e4):{RESET} ",
                             move_log=state.move_log, eval_score=eval_score),
                      end="", flush=True)
                show_cursor()
                raw = input("  > ").strip()
                hide_cursor()
                try:
                    move = chess.Move.from_uci(raw)
                    if move not in state.cb.legal_moves:
                        draw(state.logic, state.cb,
                             message=f"{RED}Lance ilegal: {raw}{RESET}",
                             move_log=state.move_log, eval_score=eval_score)
                        time.sleep(1.5); n -= 1; continue
                except ValueError:
                    draw(state.logic, state.cb,
                         message=f"{RED}UCI invalido: '{raw}'{RESET}",
                         move_log=state.move_log, eval_score=eval_score)
                    time.sleep(1.5); n -= 1; continue
            else:
                draw(state.logic, state.cb,
                     message=f"{DIM}Stockfish pensando...{RESET}",
                     move_log=state.move_log, eval_score=eval_score)
                move, eval_score = sf_move(engine, state.cb, args.tempo)
                if move is None: break

            execute_move(state, move, args.delay, eval_score)

        outcome = state.cb.outcome()
        if outcome:
            if outcome.winner == chess.WHITE:
                msg = f"{BOLD}{WHITE}Brancas vencem!{RESET} ({outcome.termination.name})"
            elif outcome.winner == chess.BLACK:
                msg = f"{BOLD}{RED}Negras vencem!{RESET} ({outcome.termination.name})"
            else:
                msg = f"{BOLD}{YELLOW}Empate!{RESET} ({outcome.termination.name})"
        else:
            msg = f"{BOLD}{CYAN}Partida encerrada. Resultado: {state.cb.result()}{RESET}"

        draw(state.logic, state.cb, message=msg,
             move_log=state.move_log, eval_score=eval_score)

    except KeyboardInterrupt:
        pass
    finally:
        engine.quit(); show_cursor(); clear_screen()
        print(f"\n{BOLD}Encerrando.{RESET} Lances: {n}")
        if state.move_log:
            print("Partida:", " ".join(m["san"] for m in state.move_log))
        print()

if __name__ == "__main__":
    main()
