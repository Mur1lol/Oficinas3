r"""
main.py  --  Raiz do projeto (d:/0_ UTFPR/Periodo 10/Oficinas/)
================================================================
Integracao Stockfish + Navegacao Robotica de Xadrez.

Lances especiais:
  Roque      -> Rei move primeiro, depois Torre e posicionada ao lado do Rei
  Promocao   -> Peao vai ao cemiterio, peca de reserva vai ao destino
  En Passant -> Peca move ao destino PRIMEIRO, peao capturado vai depois ao cemiterio

Cemiterios iniciam com reserva de promocao (col 2 para brancas, col 13 para negras):
  2 Rainhas + 1 Torre + 1 Bispo + 1 Cavalo por lado

Uso: python main.py
"""

import os, sys, time

# UTF-8 para o terminal Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import chess, chess.engine

# Adiciona pasta navigation/ ao Python path
_NAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "navigation")
sys.path.insert(0, _NAV)

from board import (
    BoardLogic, BoardPhysical,
    PIECE, EMPTY,
    PHYS_ROWS, PHYS_COLS, LOGIC_ROWS, LOGIC_COLS,
    COL_BOARD_START,
    COL_CEM_ESQ_START, COL_CEM_ESQ_END,
    COL_CEM_DIR_START, COL_CEM_DIR_END,
    ZONE_BOARD, ZONE_CEM_ESQ, ZONE_CEM_DIR, ZONE_VAO,
    zone_of,
)
from pathfinder import find_path_logical

# ─── Caminho do Stockfish ─────────────────────────────────────────────────────

STOCKFISH_EXE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "stockfish-windows-x86-64-universal", "stockfish",
    "stockfish-windows-x86-64-universal.exe"
)

# nivel -> (UCI SkillLevel, ELO ou None, descricao)
SKILL_MAP = {
    1: (0,   800,  "Iniciante"),
    2: (3,   1000, "Basico"),
    3: (6,   1200, "Intermediario"),
    4: (9,   1500, "Avancado"),
    5: (12,  1700, "Expert"),
    6: (15,  2000, "Mestre"),
    7: (18,  2300, "Gran-Mestre"),
    8: (20,  None, "Motor Completo"),
}

# ─── ANSI ─────────────────────────────────────────────────────────────────────

def _a(c): return f"\033[{c}m"
RS=_a(0); BD=_a(1); DM=_a(2)
RD=_a(31); GR=_a(32); YL=_a(33); BL=_a(34); MG=_a(35); CY=_a(36); WH=_a(37)
BG_HO=_a("48;5;239"); BG_PC=_a("48;5;52");  BG_RB=_a("48;5;22")
BG_TR=_a("48;5;17");  BG_CR=_a("48;5;235"); BG_CM=_a("48;5;236")
BG_RV=_a("48;5;58");  BG_CI=_a("48;5;88")

def _enable():
    if os.name == "nt":
        import ctypes; k = ctypes.windll.kernel32
        k.SetConsoleMode(k.GetStdHandle(-11), 7)

clr  = lambda: os.system("cls" if os.name=="nt" else "clear")
hide = lambda: print("\033[?25l", end="", flush=True)
show = lambda: print("\033[?25h", end="", flush=True)
home = lambda: print("\033[H",    end="", flush=True)

FRAME_DELAY = 0.07   # atualizado no setup

# ─── Coordenadas ──────────────────────────────────────────────────────────────

FILE_LABELS = "ABCDEFGH"
_PSYM  = {chess.PAWN:("P","p"), chess.KNIGHT:("N","n"), chess.BISHOP:("B","b"),
          chess.ROOK:("R","r"), chess.QUEEN:("Q","q"), chess.KING:("K","k")}
_PNAME = {chess.PAWN:"Peao", chess.KNIGHT:"Cavalo", chess.BISHOP:"Bispo",
          chess.ROOK:"Torre", chess.QUEEN:"Rainha", chess.KING:"Rei"}

def sq2sub(sq):
    return 7 - chess.square_rank(sq), chess.square_file(sq)

def sub2glob(r, c):
    return r, COL_BOARD_START + c

def sq2glob(sq):
    return sub2glob(*sq2sub(sq))

def note(r, c):
    return f"{FILE_LABELS[c]}{8-r}"

def psym(p):
    w, b = _PSYM.get(p.piece_type, ("?","?"))
    return w if p.color == chess.WHITE else b

# ─── Reservas de Promocao ─────────────────────────────────────────────────────
# CEM_ESQ col=COL_CEM_ESQ_END(2)   linhas 3-7  -> reservas das BRANCAS
# CEM_DIR col=COL_CEM_DIR_START(13) linhas 3-7  -> reservas das NEGRAS

_W_RES_INIT = [
    (chess.QUEEN,  (3, COL_CEM_ESQ_END)),
    (chess.QUEEN,  (4, COL_CEM_ESQ_END)),
    (chess.ROOK,   (5, COL_CEM_ESQ_END)),
    (chess.BISHOP, (6, COL_CEM_ESQ_END)),
    (chess.KNIGHT, (7, COL_CEM_ESQ_END)),
]
_B_RES_INIT = [
    (chess.QUEEN,  (3, COL_CEM_DIR_START)),
    (chess.QUEEN,  (4, COL_CEM_DIR_START)),
    (chess.ROOK,   (5, COL_CEM_DIR_START)),
    (chess.BISHOP, (6, COL_CEM_DIR_START)),
    (chess.KNIGHT, (7, COL_CEM_DIR_START)),
]

# ─── Estado do Jogo ───────────────────────────────────────────────────────────

class GameState:
    def __init__(self):
        self.cb       = chess.Board()
        self.logic    = BoardLogic()
        self.move_log = []
        self.w_res    = list(_W_RES_INIT)
        self.b_res    = list(_B_RES_INIT)
        self.cem_occ  = {}   # (r,c) -> chess.PieceType
        self._rebuild()

    def _rebuild(self):
        self.logic = BoardLogic()
        for sq in chess.SQUARES:
            p = self.cb.piece_at(sq)
            if p:
                gr, gc = sq2glob(sq)
                self.logic.grid[gr][gc] = PIECE
        for _, (r, c) in self.w_res:
            self.logic.grid[r][c] = PIECE
        for _, (r, c) in self.b_res:
            self.logic.grid[r][c] = PIECE
        for (r, c) in self.cem_occ:
            self.logic.grid[r][c] = PIECE

    @property
    def reserve_slots(self):
        return {s for _, s in self.w_res} | {s for _, s in self.b_res}

    def peek_reserve(self, color, ptype):
        lst = self.w_res if color == chess.WHITE else self.b_res
        for i, (pt, s) in enumerate(lst):
            if pt == ptype:
                return i, s
        return None, None

    def consume_reserve(self, color, ptype):
        lst = self.w_res if color == chess.WHITE else self.b_res
        for i, (pt, s) in enumerate(lst):
            if pt == ptype:
                lst.pop(i)
                return s
        return None

    def next_cem_slot(self, side):
        return self.logic.next_cemetery_slot(side)

# ─── Renderizacao ─────────────────────────────────────────────────────────────

def _cell(state, pr, pc, rpos, trail):
    is_real = (pr % 2 == 1) and (pc % 2 == 1)
    lr = (pr - 1) // 2;  lc = (pc - 1) // 2

    if (pr, pc) == rpos:
        return f"{BG_RB}{BD}{GR}[R]{RS}"
    if trail and (pr, pc) in trail:
        return f"{BG_TR}{DM}{CY} .. {RS}"
    if not is_real:
        zc = min(max(pc // 2, 0), LOGIC_COLS - 1)
        z  = zone_of(zc)
        if z == ZONE_VAO:                      return f"{BG_CR}    {RS}"
        if z in (ZONE_CEM_ESQ, ZONE_CEM_DIR):  return f"{BG_CM}    {RS}"
        return f"{BG_CR}    {RS}"

    z   = zone_of(lc)
    val = state.logic.grid[lr][lc]
    rsl = state.reserve_slots

    if z == ZONE_VAO:
        return f"{BG_CR}    {RS}"

    if z in (ZONE_CEM_ESQ, ZONE_CEM_DIR):
        if (lr, lc) in rsl:
            pt = None
            for res_list in (state.w_res, state.b_res):
                for p_type, slot in res_list:
                    if slot == (lr, lc):
                        pt = p_type; break
                if pt: break
            sym = _PNAME.get(pt, "?")[0] if pt else "?"
            return f"{BG_RV}{YL} {sym}R {RS}"
        if val == PIECE:
            return f"{BG_CI}{RD} X  {RS}"
        return f"{BG_CM}{DM}    {RS}"

    # Tabuleiro de jogo
    bc = lc - COL_BOARD_START;  br = lr
    sq = chess.square(bc, 7 - br)
    p  = state.cb.piece_at(sq)
    if p:
        col = RD if p.color == chess.BLACK else WH
        return f"{BG_PC}{BD}{col} {psym(p)}  {RS}"
    return f"{BG_HO}{DM}{WH} .  {RS}"


def render(state, rpos=None, trail=None, msg="", eval_s=""):
    cb = state.cb;  L = []
    L.append(f"\n {BD}{CY}Chess Robot  --  Navegacao Fisica{RS}")
    L.append(f" {DM}{'─'*38}{RS}\n")

    turn_s = f"{WH}Brancas{RS}" if cb.turn==chess.WHITE else f"{DM}Negras{RS}"
    L.append(f"  {BD}Turno:{RS} {turn_s}   {BD}Aval:{RS} {CY}{eval_s or '---'}{RS}   "
             f"{BD}Lance #{RS}{len(state.move_log)+1}")
    L.append("")

    # Tabuleiro 8x8
    L.append("  " + " ".join(f"{BD}{YL}{c}{RS}" for c in "ABCDEFGH"))
    L.append(f"  {'─'*16}")
    for rank in range(7, -1, -1):
        parts = []
        for file in range(8):
            sq = chess.square(file, rank)
            p  = cb.piece_at(sq)
            if p:
                col = RD if p.color==chess.BLACK else WH
                parts.append(f"{BD}{col}{psym(p)}{RS}")
            else:
                parts.append(f"{DM}{'.' if (rank+file)%2==0 else ','}{RS}")
        L.append(f"  {' '.join(parts)} {BD}{YL}{rank+1}{RS}")
    L.append("")

    # Grid fisico
    L.append(f"  {BD}{CY}Grid fisico {PHYS_ROWS}x{PHYS_COLS}{RS}  "
             f"[{MG}CEM{CY}|{YL}VAO{CY}|{GR}TAB{CY}|{YL}VAO{CY}|{MG}CEM{CY}]{RS}")
    hdr = "     "
    for pc in range(PHYS_COLS):
        if pc % 2 == 1:
            lc = (pc-1)//2
            z  = zone_of(lc)
            c  = {ZONE_CEM_ESQ:MG,ZONE_CEM_DIR:MG,ZONE_VAO:YL,ZONE_BOARD:GR}.get(z,WH)
            hdr += f"{c}{lc:2d}{RS} "
        else:
            hdr += "   "
    L.append(hdr)

    for pr in range(PHYS_ROWS):
        row = f"  {BD}{YL}{pr:2d}{RS} "
        row += "".join(_cell(state, pr, pc, rpos, trail) for pc in range(PHYS_COLS))
        L.append(row)
    L.append("")

    if state.move_log:
        recent = state.move_log[-5:]
        L.append(f"  {BD}Historico:{RS}")
        for i, m in enumerate(recent):
            idx = len(state.move_log)-len(recent)+i+1
            sym = f"{WH}P{RS}" if m["side"]=="white" else f"{DM}p{RS}"
            cap = f" {RD}x{RS}" if m.get("is_capture") else ""
            tag = f" {MG}[{m['tag']}]{RS}" if m.get("tag") else ""
            L.append(f"    {DM}{idx:2d}.{RS} {sym} {BD}{m['san']}{RS}{cap}{tag}")
    L.append("")
    L.append(f"  {MG}Res.ESQ{RS}=brancas(2Q+T+B+C cols0-2)  "
             f"{MG}Res.DIR{RS}=negras(2Q+T+B+C cols13-15)")
    L.append(f"  {BG_RV}{YL}XR{RS}=reserva  {BG_CI}{RD}X{RS}=capturado  "
             f"{BG_RB}{GR}[R]{RS}=robo")
    if msg:
        L.append(f"\n  {msg}")
    L.append("")
    return "\n".join(L)


def draw(state, rpos=None, trail=None, msg="", eval_s=""):
    home()
    print(render(state, rpos, trail, msg, eval_s), end="", flush=True)

# ─── Animacao ─────────────────────────────────────────────────────────────────

def animate(state, path, label, eval_s):
    full = set(path);  vis = set()
    for i, pos in enumerate(path):
        vis.add(pos)
        pct = int(100*i/max(len(path)-1, 1))
        f   = int(20*i/max(len(path)-1, 1))
        bar = f"{GR}{'|'*f}{DM}{'-'*(20-f)}{RS}"
        draw(state, rpos=pos, trail=full-vis,
             msg=f"{BD}{CY}{label}{RS}  [{bar}] {pct:3d}%", eval_s=eval_s)
        time.sleep(FRAME_DELAY)
    draw(state, rpos=path[-1], trail=set(),
         msg=f"{BD}{GR}{label}{RS}", eval_s=eval_s)
    time.sleep(FRAME_DELAY * 3)

# ─── Pathfinding helpers ─────────────────────────────────────────────────────

def path_go(state, orig, dest, free_dest=True):
    """Calcula caminho; se free_dest, libera temporariamente a celula de destino."""
    snap = state.logic.grid[dest[0]][dest[1]]
    if free_dest:
        state.logic.grid[dest[0]][dest[1]] = EMPTY
    result = find_path_logical(state.logic, orig, dest)
    state.logic.grid[dest[0]][dest[1]] = snap
    return result

def move_anim(state, orig, dest, label, eval_s, free_dest=True):
    """Calcula e anima movimento entre duas coords logicas globais."""
    res = path_go(state, orig, dest, free_dest=free_dest)
    if res:
        animate(state, res["path"], label, eval_s)
    return res is not None

# ─── Execute Move ─────────────────────────────────────────────────────────────

def execute_move(state, move, eval_s):
    """Executa um lance com animacao completa de todos os casos especiais."""
    cb    = state.cb
    san   = cb.san(move)
    color = cb.turn
    side  = "white" if color==chess.WHITE else "black"
    s_str = "Brancas" if color==chess.WHITE else "Negras"

    is_cap    = cb.is_capture(move)
    is_ep     = cb.is_en_passant(move)
    is_castle = cb.is_castling(move)
    is_promo  = move.promotion is not None

    fr_sub  = sq2sub(move.from_square)
    to_sub  = sq2sub(move.to_square)
    fr_glob = sub2glob(*fr_sub)
    to_glob = sub2glob(*to_sub)
    tag = None

    # ════════════════════════════════════════════════════════
    #  ROQUE: Rei move primeiro, depois Torre e posicionada
    # ════════════════════════════════════════════════════════
    if is_castle:
        tag = "roque"
        ks  = chess.square_file(move.to_square) == 6  # kingside?
        if color == chess.WHITE:
            rk_fr_sq = chess.H1 if ks else chess.A1
            rk_to_sq = chess.F1 if ks else chess.D1
        else:
            rk_fr_sq = chess.H8 if ks else chess.A8
            rk_to_sq = chess.F8 if ks else chess.D8
        rk_fr     = sq2glob(rk_fr_sq)
        rk_to     = sq2glob(rk_to_sq)
        rk_to_sub = sq2sub(rk_to_sq)

        # Passo 1: Rei move para a casa livre
        state.logic.grid[fr_glob[0]][fr_glob[1]] = EMPTY
        move_anim(state, fr_glob, to_glob,
                  f"Roque: Rei {note(*fr_sub)} -> {note(*to_sub)}", eval_s)
        state.logic.grid[to_glob[0]][to_glob[1]] = PIECE
        robo = to_glob

        # Passo 2: Robo vai buscar a Torre
        move_anim(state, robo, rk_fr,
                  f"Roque: buscando Torre em {note(*sq2sub(rk_fr_sq))}", eval_s, free_dest=True)
        robo = rk_fr

        # Passo 3: Torre move para o outro lado do Rei
        state.logic.grid[rk_fr[0]][rk_fr[1]] = EMPTY
        move_anim(state, rk_fr, rk_to,
                  f"Roque: Torre -> {note(*rk_to_sub)}", eval_s, free_dest=True)
        state.logic.grid[rk_to[0]][rk_to[1]] = PIECE

    # ════════════════════════════════════════════════════════
    #  PROMOCAO: Peao ao cemiterio, reserva ao destino
    #  (se tambem e captura, primeiro leva peca capturada)
    # ════════════════════════════════════════════════════════
    elif is_promo:
        tag   = "promocao"
        ptype = move.promotion

        # Se for captura+promocao: leva peca capturada primeiro
        if is_cap:
            cap_cem  = "right" if color==chess.WHITE else "left"
            cap_slot = state.next_cem_slot(cap_cem)
            cap_p    = cb.piece_at(move.to_square)
            if cap_slot:
                move_anim(state, fr_glob, to_glob,
                          f"Captura+Promocao: buscando peca em {note(*to_sub)}", eval_s,
                          free_dest=True)
                state.logic.grid[to_glob[0]][to_glob[1]] = EMPTY
                move_anim(state, to_glob, cap_slot,
                          f"Captura+Promocao: peca ao cemiterio", eval_s, free_dest=False)
                state.logic.grid[cap_slot[0]][cap_slot[1]] = PIECE
                state.cem_occ[cap_slot] = cap_p.piece_type if cap_p else chess.PAWN
                move_anim(state, cap_slot, fr_glob,
                          f"Voltando para {note(*fr_sub)}", eval_s, free_dest=True)

        # Passo P1: Peao vai ao cemiterio do proprio lado
        pawn_cem  = "left" if color==chess.WHITE else "right"
        pawn_slot = state.next_cem_slot(pawn_cem)
        state.logic.grid[fr_glob[0]][fr_glob[1]] = EMPTY
        robo = fr_glob
        if pawn_slot:
            move_anim(state, fr_glob, pawn_slot,
                      f"Promocao: Peao -> cemiterio {pawn_slot}", eval_s, free_dest=False)
            state.logic.grid[pawn_slot[0]][pawn_slot[1]] = PIECE
            state.cem_occ[pawn_slot] = chess.PAWN
            robo = pawn_slot

        # Passo P2: Busca peca de reserva
        _, res_slot = state.peek_reserve(color, ptype)
        if res_slot:
            move_anim(state, robo, res_slot,
                      f"Promocao: buscando {_PNAME.get(ptype,'?')} de reserva",
                      eval_s, free_dest=True)
            robo = res_slot

            # Passo P3: Leva peca de reserva ao destino
            state.logic.grid[res_slot[0]][res_slot[1]] = EMPTY
            state.consume_reserve(color, ptype)
            move_anim(state, robo, to_glob,
                      f"Promocao: {_PNAME.get(ptype,'?')} -> {note(*to_sub)}",
                      eval_s, free_dest=True)
            state.logic.grid[to_glob[0]][to_glob[1]] = PIECE

    # ════════════════════════════════════════════════════════
    #  EN PASSANT: Peca move PRIMEIRO, capturada vai depois
    # ════════════════════════════════════════════════════════
    elif is_ep:
        tag = "en passant"
        ep_sq    = chess.square(chess.square_file(move.to_square),
                                chess.square_rank(move.from_square))
        cap_sub  = sq2sub(ep_sq)
        cap_glob = sub2glob(*cap_sub)
        cem_side = "right" if color==chess.WHITE else "left"

        # Passo 1: Peao move ao destino
        state.logic.grid[fr_glob[0]][fr_glob[1]] = EMPTY
        move_anim(state, fr_glob, to_glob,
                  f"En Passant: {note(*fr_sub)} -> {note(*to_sub)}", eval_s, free_dest=True)
        state.logic.grid[to_glob[0]][to_glob[1]] = PIECE
        robo = to_glob

        # Passo 2: Robo vai buscar peao capturado (posicao diferente!)
        move_anim(state, robo, cap_glob,
                  f"En Passant: buscando peao capturado em {note(*cap_sub)}", eval_s,
                  free_dest=True)
        robo = cap_glob

        # Passo 3: Leva peao capturado ao cemiterio
        cem_slot = state.next_cem_slot(cem_side)
        if cem_slot:
            state.logic.grid[cap_glob[0]][cap_glob[1]] = EMPTY
            move_anim(state, cap_glob, cem_slot,
                      f"En Passant: peao capturado -> cemiterio", eval_s, free_dest=False)
            state.logic.grid[cem_slot[0]][cem_slot[1]] = PIECE
            state.cem_occ[cem_slot] = chess.PAWN

    # ════════════════════════════════════════════════════════
    #  CAPTURA NORMAL
    # ════════════════════════════════════════════════════════
    elif is_cap:
        cem_side  = "right" if color==chess.WHITE else "left"
        cem_slot  = state.next_cem_slot(cem_side)
        cap_piece = cb.piece_at(move.to_square)

        # Passo 1: Vai ate a peca capturada
        move_anim(state, fr_glob, to_glob,
                  f"Captura: buscando {note(*to_sub)}", eval_s, free_dest=True)
        robo = to_glob

        # Passo 2: Leva ao cemiterio
        if cem_slot:
            state.logic.grid[to_glob[0]][to_glob[1]] = EMPTY
            move_anim(state, robo, cem_slot,
                      f"Captura: peca -> cemiterio", eval_s, free_dest=False)
            state.logic.grid[cem_slot[0]][cem_slot[1]] = PIECE
            state.cem_occ[cem_slot] = cap_piece.piece_type if cap_piece else chess.PAWN
            robo = cem_slot

        # Passo 3: Volta para origem
        move_anim(state, robo, fr_glob,
                  f"Voltando para {note(*fr_sub)}", eval_s, free_dest=True)

        # Passo 4: Move peca ao destino
        move_anim(state, fr_glob, to_glob,
                  f"{s_str}: {san}  ({note(*fr_sub)} -> {note(*to_sub)})",
                  eval_s, free_dest=False)

    # ════════════════════════════════════════════════════════
    #  MOVIMENTO SIMPLES
    # ════════════════════════════════════════════════════════
    else:
        state.logic.grid[fr_glob[0]][fr_glob[1]] = EMPTY
        move_anim(state, fr_glob, to_glob,
                  f"{s_str}: {san}  ({note(*fr_sub)} -> {note(*to_sub)})",
                  eval_s, free_dest=True)

    # Aplica o lance no estado oficial
    state.cb.push(move)
    state._rebuild()
    state.move_log.append({"san":san,"side":side,"is_capture":is_cap,"tag":tag})
    draw(state, msg=f"{BD}{GR}Lance concluido: {san}{RS}", eval_s=eval_s)
    time.sleep(0.4)

# ─── Stockfish ───────────────────────────────────────────────────────────────

def sf_best(engine, board, tempo=0.3):
    result = engine.analyse(board, chess.engine.Limit(time=tempo))
    move   = result["pv"][0] if result.get("pv") else None
    score  = result.get("score")
    ev = ""
    if score:
        pov = score.white()
        if pov.is_mate(): ev = f"M{pov.mate()}"
        else:
            cp = pov.score()
            ev = f"{'+' if cp>=0 else ''}{cp/100:.2f}"
    return move, ev

def apply_skill(engine, nivel):
    skill, elo, _ = SKILL_MAP[nivel]
    try:
        engine.configure({"Skill Level": skill})
        if elo:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        else:
            engine.configure({"UCI_LimitStrength": False})
    except Exception:
        pass

# ─── Tela de Configuracao ────────────────────────────────────────────────────

def setup():
    """Exibe menu e retorna (nivel, human_color, frame_delay)."""
    _enable(); hide(); clr()
    print(f"""
 {BD}{CY}╔══════════════════════════════════════════════╗
 ║     Chess Robot  --  Nova Partida           ║
 ╠══════════════════════════════════════════════╣{RS}

 {BD}Nivel do Stockfish:{RS}
   {YL}1{RS} - Iniciante      (ELO ~800)
   {YL}2{RS} - Basico         (ELO ~1000)
   {YL}3{RS} - Intermediario  (ELO ~1200)
   {YL}4{RS} - Avancado       (ELO ~1500)
   {YL}5{RS} - Expert         (ELO ~1700)
   {YL}6{RS} - Mestre         (ELO ~2000)
   {YL}7{RS} - Gran-Mestre    (ELO ~2300)
   {YL}8{RS} - Motor Completo (sem limite)

 {BD}{CY}╠══════════════════════════════════════════════╣{RS}
 {BD}Lance UCI:{RS}  {YL}e2e4{RS}  {YL}e1g1{RS}(roque)  {YL}e7e8q{RS}(promocao)
 {BD}{CY}╚══════════════════════════════════════════════╝{RS}
""")
    show()
    while True:
        try:
            n = int(input(f"  {BD}Nivel (1-8):{RS} ").strip())
            if 1 <= n <= 8: break
            print(f"  {RD}Digite entre 1 e 8.{RS}")
        except ValueError:
            print(f"  {RD}Entrada invalida.{RS}")

    _, _, desc = SKILL_MAP[n]
    print(f"\n  {GR}Nivel: {desc}{RS}\n")

    print(f"  {BD}Voce joga como:{RS}")
    print(f"    {YL}B{RS} - Brancas (voce joga primeiro)")
    print(f"    {YL}N{RS} - Negras  (Stockfish joga primeiro)")
    while True:
        c = input(f"  {BD}Cor (B/N):{RS} ").strip().upper()
        if c in ("B","N"): break
        print(f"  {RD}Digite B ou N.{RS}")
    human_color = chess.WHITE if c=="B" else chess.BLACK
    print(f"\n  {GR}Voce: {'Brancas' if human_color==chess.WHITE else 'Negras'}{RS}\n")

    print(f"  {BD}Velocidade da animacao:{RS}")
    print(f"    {YL}1{RS} - Lenta  (0.15s/frame)")
    print(f"    {YL}2{RS} - Normal (0.07s/frame)")
    print(f"    {YL}3{RS} - Rapida (0.02s/frame)")
    while True:
        try:
            v = int(input(f"  {BD}Velocidade (1-3):{RS} ").strip())
            if 1 <= v <= 3: break
        except ValueError: pass
        print(f"  {RD}Digite 1, 2 ou 3.{RS}")
    hide()
    return n, human_color, {1:0.15, 2:0.07, 3:0.02}[v]

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global FRAME_DELAY
    nivel, human_color, FRAME_DELAY = setup()

    clr()
    print(f"\n {BD}{CY}Inicializando Stockfish...{RS}", flush=True)
    sf_path = os.path.normpath(STOCKFISH_EXE)
    if not os.path.isfile(sf_path):
        show()
        print(f"\n{RD}Stockfish nao encontrado:{RS}\n  {sf_path}")
        return

    try:
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    except Exception as e:
        show(); print(f"\n{RD}Erro ao abrir Stockfish:{RS} {e}"); return

    apply_skill(engine, nivel)
    _, _, desc = SKILL_MAP[nivel]

    state  = GameState()
    eval_s = ""
    tempo  = 0.5

    try:
        clr()
        draw(state,
             msg=f"{BD}{GR}Partida iniciada! Nivel: {desc}  "
                 f"Voce: {'Brancas' if human_color==chess.WHITE else 'Negras'}{RS}")
        time.sleep(1.2)

        while not state.cb.is_game_over():
            is_human = (state.cb.turn == human_color)

            if is_human:
                home()
                print(render(state,
                             msg=f"  {BD}Seu lance (UCI, ex: e2e4 | 'q' para sair):{RS} ",
                             eval_s=eval_s),
                      end="", flush=True)
                show()
                try:
                    raw = input("  > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    break
                hide()

                if raw in ("q","quit","exit","sair"):
                    break

                try:
                    mv = chess.Move.from_uci(raw)
                    if mv not in state.cb.legal_moves:
                        draw(state,
                             msg=f"{RD}Lance ilegal: '{raw}'. Tente novamente.{RS}",
                             eval_s=eval_s)
                        time.sleep(1.5); continue
                except ValueError:
                    draw(state,
                         msg=f"{RD}UCI invalido: '{raw}'.{RS}  "
                             f"Exemplos: {YL}e2e4  e1g1  e7e8q{RS}",
                         eval_s=eval_s)
                    time.sleep(1.5); continue

                _, eval_s = sf_best(engine, state.cb, 0.1)

            else:
                draw(state, msg=f"{DM}Stockfish ({desc}) pensando...{RS}", eval_s=eval_s)
                mv, eval_s = sf_best(engine, state.cb, tempo)
                if mv is None: break

            execute_move(state, mv, eval_s)

        # Resultado final
        outcome = state.cb.outcome()
        if outcome:
            if outcome.winner==chess.WHITE:
                end_msg=f"{BD}{WH}Brancas vencem!{RS} ({outcome.termination.name})"
            elif outcome.winner==chess.BLACK:
                end_msg=f"{BD}{RD}Negras vencem!{RS} ({outcome.termination.name})"
            else:
                end_msg=f"{BD}{YL}Empate!{RS} ({outcome.termination.name})"
        else:
            end_msg=f"{BD}{CY}Partida encerrada. Resultado: {state.cb.result()}{RS}"

        draw(state, msg=end_msg, eval_s=eval_s)
        show()
        input(f"\n  {DM}Pressione Enter para sair...{RS}")

    except KeyboardInterrupt:
        pass
    finally:
        engine.quit(); show(); clr()
        print(f"\n{BD}Lances jogados: {len(state.move_log)}{RS}")
        if state.move_log:
            print("Partida:", " ".join(m["san"] for m in state.move_log))
        print()

if __name__ == "__main__":
    main()
