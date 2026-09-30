r"""
main.py  --  Raiz do projeto
=============================================================================
Integracao Stockfish + Navegacao Robotica de Xadrez.

O robo e tratado como um mecanismo fisico:
  - Sua posicao e rastreada em TODO momento (state.robot_pos)
  - Cada caminho parte SEMPRE da posicao atual do robo
  - Sinal PICKUP e emitido ao pegar uma peca   -> hw_pickup()
  - Sinal RELEASE e emitido ao soltar uma peca  -> hw_release()
  - O robo nunca "teleporta": toda navegacao e animada

Lances especiais:
  Roque      -> Rei move primeiro, depois Torre e posicionada ao lado
  Promocao   -> Peao vai ao cemiterio, peca de reserva vai ao destino
  En Passant -> Peca move ao destino PRIMEIRO, capturada vai depois

Cemiterios iniciam com reserva (col 2 brancas, col 13 negras):
  2 Rainhas + 1 Torre + 1 Bispo + 1 Cavalo por lado

Uso: python main.py
"""

import os, sys, time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import chess, chess.engine

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

# ─── Stockfish ────────────────────────────────────────────────────────────────

STOCKFISH_EXE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "stockfish-windows-x86-64-universal", "stockfish",
    "stockfish-windows-x86-64-universal.exe"
)

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

# ─── Hardware hooks ───────────────────────────────────────────────────────────
# Substitua pelo codigo real de controle do hardware (GPIO, serial, CAN, etc.)

def hw_pickup():
    """Ativa o mecanismo de agarre (eletroima, garra, vacuo, etc.)."""
    pass  # ex: GPIO.output(PIN_MAGNET, GPIO.HIGH)

def hw_release():
    """Desativa o mecanismo de agarre."""
    pass  # ex: GPIO.output(PIN_MAGNET, GPIO.LOW)

# ─── ANSI ─────────────────────────────────────────────────────────────────────

def _a(c): return f"\033[{c}m"
RS=_a(0); BD=_a(1); DM=_a(2)
RD=_a(31); GR=_a(32); YL=_a(33); BL=_a(34); MG=_a(35); CY=_a(36); WH=_a(37)
BG_HO    = _a("48;5;239")   # casa vazia do tabuleiro
BG_PC    = _a("48;5;52")    # casa com peca
BG_ROBOT = _a("48;5;22")    # robo sem peca
BG_CARRY = _a("48;5;130")   # robo carregando peca
BG_TRAIL = _a("48;5;17")    # rastro do caminho
BG_CORR  = _a("48;5;235")   # corredor
BG_CEM   = _a("48;5;236")   # cemiterio vazio
BG_RES   = _a("48;5;58")    # slot de reserva
BG_OCC   = _a("48;5;88")    # cemiterio ocupado
SIG_PU   = _a("48;5;136")   # fundo do sinal pickup
SIG_RL   = _a("48;5;22")    # fundo do sinal release

def _enable():
    if os.name == "nt":
        import ctypes; k = ctypes.windll.kernel32
        k.SetConsoleMode(k.GetStdHandle(-11), 7)

clr  = lambda: os.system("cls" if os.name=="nt" else "clear")
hide = lambda: print("\033[?25l", end="", flush=True)
show = lambda: print("\033[?25h", end="", flush=True)
home = lambda: print("\033[H",    end="", flush=True)

FRAME_DELAY  = 0.07   # atualizado no setup
SIGNAL_PAUSE = 0.5    # pausa durante pickup/release (hardware reage)
ROBOT_HOME   = (0, 0) # posicao logica inicial do robo (CEM_ESQ top-left)

# ─── Coordenadas e simbolos ───────────────────────────────────────────────────

FILE_LABELS = "ABCDEFGH"
_PSYM  = {chess.PAWN:("P","p"), chess.KNIGHT:("N","n"), chess.BISHOP:("B","b"),
          chess.ROOK:("R","r"), chess.QUEEN:("Q","q"), chess.KING:("K","k")}
_PNAME = {chess.PAWN:"Peao", chess.KNIGHT:"Cavalo", chess.BISHOP:"Bispo",
          chess.ROOK:"Torre", chess.QUEEN:"Rainha", chess.KING:"Rei"}

def l2p(r, c):
    """Logico global -> fisico (2r+1, 2c+1)."""
    return (r*2+1, c*2+1)

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

# ─── Layout Fixo do Cemiterio ────────────────────────────────────────────────
#
# SEM pecas extras pre-carregadas. Promocao so e possivel usando uma peca
# JA CAPTURADA do tipo desejado que esteja no cemiterio.
#
# CEM_ESQ (brancas capturadas, cols 0-2):     CEM_DIR (negras capturadas, cols 13-15):
#  Col 0 – Peoes capturados (rows 0-7)        Col 13 – Pecas maiores capturadas
#  Col 1 – Pecas maiores capturadas           Col 14 – Peoes capturados (rows 0-7)
#    row 0-1: Torres (R)                         row 0-1: Torres (R)
#    row 2-3: Cavalos (N)                        row 2-3: Cavalos (N)
#    row 4-5: Bispos (B)                         row 4-5: Bispos (B)
#    row 6:   Rainha (Q)                         row 6:   Rainha (Q)
#    row 7:   Rei    (K)                         row 7:   Rei    (K)
#  Col 2 – livre                              Col 15 – livre

# Colunas de cada zona
W_PAWN_COL  = COL_CEM_ESQ_START          # col 0: peoes brancos capturados
W_MAJ_COL   = COL_CEM_ESQ_START + 1      # col 1: pecas maiores brancas capturadas
B_MAJ_COL   = COL_CEM_DIR_START          # col 13: pecas maiores negras capturadas
B_PAWN_COL  = COL_CEM_DIR_START + 1      # col 14: peoes negros capturados

# Slots fixos para pecas maiores capturadas (ordem = posicao no cemiterio)
W_MAJOR_SLOTS = [
    (chess.ROOK,   (0, W_MAJ_COL)),
    (chess.ROOK,   (1, W_MAJ_COL)),
    (chess.KNIGHT, (2, W_MAJ_COL)),
    (chess.KNIGHT, (3, W_MAJ_COL)),
    (chess.BISHOP, (4, W_MAJ_COL)),
    (chess.BISHOP, (5, W_MAJ_COL)),
    (chess.QUEEN,  (6, W_MAJ_COL)),
    (chess.KING,   (7, W_MAJ_COL)),
]
B_MAJOR_SLOTS = [
    (chess.ROOK,   (0, B_MAJ_COL)),
    (chess.ROOK,   (1, B_MAJ_COL)),
    (chess.KNIGHT, (2, B_MAJ_COL)),
    (chess.KNIGHT, (3, B_MAJ_COL)),
    (chess.BISHOP, (4, B_MAJ_COL)),
    (chess.BISHOP, (5, B_MAJ_COL)),
    (chess.QUEEN,  (6, B_MAJ_COL)),
    (chess.KING,   (7, B_MAJ_COL)),
]

def cem_slot_for(state, piece_type, captured_color):
    """
    Retorna a vaga FIXA no cemiterio para depositar a peca capturada.

    captured_color : cor da peca que foi capturada.
      Brancas capturadas -> CEM_ESQ; Negras capturadas -> CEM_DIR.
    """
    is_white  = (captured_color == chess.WHITE)
    pawn_col  = W_PAWN_COL   if is_white else B_PAWN_COL
    maj_slots = W_MAJOR_SLOTS if is_white else B_MAJOR_SLOTS

    if piece_type == chess.PAWN:
        for row in range(LOGIC_ROWS):
            if state.logic.grid[row][pawn_col] == EMPTY:
                return (row, pawn_col)
        return None

    for pt, slot in maj_slots:
        if pt == piece_type and state.logic.grid[slot[0]][slot[1]] == EMPTY:
            return slot
    return None


def promotable_slot(state, piece_type, promoting_color):
    """
    Retorna o slot de uma peca JA CAPTURADA que pode ser usada para promocao.

    A peca a ser usada deve:
      - Ser do tipo desejado para a promocao
      - Ser da mesma cor que o peao (peca propria que estava capturada)
      - Estar registrada em state.cem_occ

    Retorna (row, col) ou None se nenhuma peca disponivel.
    """
    is_white  = (promoting_color == chess.WHITE)
    maj_slots = W_MAJOR_SLOTS if is_white else B_MAJOR_SLOTS

    for pt, slot in maj_slots:
        if pt == piece_type and state.cem_occ.get(slot) == piece_type:
            return slot
    return None


def is_phys_valid(state, board, move):
    """
    Verifica se um lance e fisicamente executavel.
    Promocoes exigem que haja uma peca capturada do tipo desejado no cemiterio.
    """
    if move.promotion is not None:
        return promotable_slot(state, move.promotion, board.turn) is not None
    return True

# ─── Estado do Jogo ───────────────────────────────────────────────────────────

class GameState:
    """Estado unificado: tabuleiro logico + posicao fisica do robo."""

    def __init__(self):
        self.cb             = chess.Board()
        self.logic          = BoardLogic()
        self.move_log       = []
        self.cem_occ        = {}          # (r,c) -> chess.PieceType  (pecas capturadas)

        # Estado fisico do robo
        self.robot_pos      = ROBOT_HOME  # coordenada logica global atual
        self.robot_carrying = False       # esta carregando uma peca?
        self.signal_log     = []          # historico de sinais para display

        self._rebuild()

    def _rebuild(self):
        """Reconstroi o grid logico: tabuleiro + cemiterio (sem reservas extras)."""
        self.logic = BoardLogic()
        for sq in chess.SQUARES:
            p = self.cb.piece_at(sq)
            if p:
                gr, gc = sq2glob(sq)
                self.logic.grid[gr][gc] = PIECE
        for (r, c) in self.cem_occ:
            self.logic.grid[r][c] = PIECE

# ─── Sinais de Hardware ───────────────────────────────────────────────────────

def sig_pickup(state, desc=""):
    """
    SINAL PICKUP: robo vai pegar a peca na posicao atual.
    Registra no log e chama o hook de hardware.
    """
    pos = state.robot_pos
    entry = (f"PICKUP  | logico={pos}  fisico={l2p(*pos)}"
             + (f"  [{desc}]" if desc else ""))
    state.signal_log.append(("PU", entry))
    hw_pickup()

def sig_release(state, desc=""):
    """
    SINAL RELEASE: robo vai soltar a peca na posicao atual.
    Registra no log e chama o hook de hardware.
    """
    pos = state.robot_pos
    entry = (f"RELEASE | logico={pos}  fisico={l2p(*pos)}"
             + (f"  [{desc}]" if desc else ""))
    state.signal_log.append(("RL", entry))
    hw_release()

# ─── Renderizacao ─────────────────────────────────────────────────────────────

def _cell(state, pr, pc, rpos, trail):
    is_real = (pr % 2 == 1) and (pc % 2 == 1)
    lr = (pr-1)//2;  lc = (pc-1)//2

    if (pr, pc) == rpos:
        if state.robot_carrying:
            return f"{BG_CARRY}{BD}{YL}[P]{RS}"   # carregando peca
        return f"{BG_ROBOT}{BD}{GR}[R]{RS}"        # sem peca

    if trail and (pr, pc) in trail:
        return f"{BG_TRAIL}{DM}{CY} .. {RS}"

    if not is_real:
        zc = min(max(pc//2, 0), LOGIC_COLS-1)
        z  = zone_of(zc)
        if z in (ZONE_CEM_ESQ, ZONE_CEM_DIR): return f"{BG_CEM}    {RS}"
        return f"{BG_CORR}    {RS}"

    z   = zone_of(lc)

    if z == ZONE_VAO:
        return f"{BG_CORR}    {RS}"

    if z in (ZONE_CEM_ESQ, ZONE_CEM_DIR):
        occ = state.cem_occ.get((lr, lc))
        if occ is not None:
            # Mostra o tipo da peca capturada: maiusculo=branca, minusculo=negra
            w_side = (z == ZONE_CEM_ESQ)
            sym = _PSYM.get(occ, ("?","?"))[0 if w_side else 1]
            return f"{BG_OCC}{WH if w_side else RD} {sym}  {RS}"
        return f"{BG_CEM}{DM}    {RS}"

    bc = lc - COL_BOARD_START;  br = lr
    sq = chess.square(bc, 7-br)
    p  = state.cb.piece_at(sq)
    if p:
        col = RD if p.color==chess.BLACK else WH
        return f"{BG_PC}{BD}{col} {psym(p)}  {RS}"
    return f"{BG_HO}{DM}{WH} .  {RS}"


def render(state, rpos=None, trail=None, msg="", eval_s=""):
    cb = state.cb;  L = []
    L.append(f"\n {BD}{CY}Chess Robot  --  Navegacao Fisica{RS}")
    L.append(f" {DM}{'─'*40}{RS}\n")

    # Info de turno e robo
    turn_s   = f"{WH}Brancas{RS}" if cb.turn==chess.WHITE else f"{DM}Negras{RS}"
    carry_s  = f" {BG_CARRY}{YL} CARREGANDO {RS}" if state.robot_carrying else ""
    L.append(f"  {BD}Turno:{RS} {turn_s}   {BD}Aval:{RS} {CY}{eval_s or '---'}{RS}   "
             f"{BD}Lance #{RS}{len(state.move_log)+1}")
    rp = state.robot_pos
    L.append(f"  {BD}Robo:{RS} logico={rp}  fisico={l2p(*rp)}{carry_s}")
    L.append("")

    # Tabuleiro 8x8 compacto
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
            c  = {ZONE_CEM_ESQ:MG,ZONE_CEM_DIR:MG,ZONE_VAO:YL,ZONE_BOARD:GR}.get(zone_of(lc),WH)
            hdr += f"{c}{lc:2d}{RS} "
        else:
            hdr += "   "
    L.append(hdr)
    for pr in range(PHYS_ROWS):
        row = f"  {BD}{YL}{pr:2d}{RS} "
        row += "".join(_cell(state, pr, pc, rpos, trail) for pc in range(PHYS_COLS))
        L.append(row)
    L.append("")

    # Historico de lances
    if state.move_log:
        recent = state.move_log[-4:]
        L.append(f"  {BD}Historico:{RS}")
        for i, m in enumerate(recent):
            idx = len(state.move_log)-len(recent)+i+1
            sym = f"{WH}P{RS}" if m["side"]=="white" else f"{DM}p{RS}"
            cap = f" {RD}x{RS}" if m.get("is_capture") else ""
            tag = f" {MG}[{m['tag']}]{RS}" if m.get("tag") else ""
            L.append(f"    {DM}{idx:2d}.{RS} {sym} {BD}{m['san']}{RS}{cap}{tag}")

    # Log de sinais (ultimos 4)
    if state.signal_log:
        L.append("")
        L.append(f"  {BD}Sinais de hardware:{RS}")
        for kind, entry in state.signal_log[-4:]:
            if kind == "PU":
                L.append(f"    {BG_CARRY}{YL} PICKUP  {RS} {DM}{entry}{RS}")
            else:
                L.append(f"    {BG_ROBOT}{GR} RELEASE {RS} {DM}{entry}{RS}")

    L.append("")
    L.append(f"  {BG_OCC}{WH}P{RS}=branca capturada  "
             f"{BG_OCC}{RD}p{RS}=negra capturada  "
             f"{BG_ROBOT}{GR}[R]{RS}=robo  "
             f"{BG_CARRY}{YL}[P]{RS}=robo+peca")
    if msg:
        L.append(f"\n  {msg}")
    L.append("")
    return "\n".join(L)


def draw(state, rpos=None, trail=None, msg="", eval_s=""):
    home()
    if rpos is None:
        rpos = l2p(*state.robot_pos)
    print(render(state, rpos, trail, msg, eval_s), end="", flush=True)

# ─── Animacao ─────────────────────────────────────────────────────────────────

def animate(state, path, label, eval_s):
    """Anima o robo ao longo de um caminho fisico."""
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
    time.sleep(FRAME_DELAY * 2)

# ─── Controle do Robo ─────────────────────────────────────────────────────────

def robot_navigate(state, dest_glob, label, eval_s):
    """
    Navega o robo da posicao ATUAL (state.robot_pos) ate dest_glob.
    O A* garante que o destino e sempre alcancavel (goal e special-cased).
    Atualiza state.robot_pos ao terminar.
    """
    if state.robot_pos == dest_glob:
        return True
    result = find_path_logical(state.logic, state.robot_pos, dest_glob)
    if result:
        animate(state, result["path"], label, eval_s)
        state.robot_pos = dest_glob
        return True
    # Sem caminho (situacao critica): permanece na posicao atual
    draw(state, msg=f"{RD}[ERRO] Sem caminho: {state.robot_pos} -> {dest_glob}{RS}",
         eval_s=eval_s)
    time.sleep(1.0)
    return False


def robot_pickup(state, desc, eval_s):
    """
    Robo pega a peca na posicao atual:
      1. Remove a peca do grid logico (ela esta sendo carregada)
      2. Emite sinal PICKUP para o hardware
      3. Pausa para o hardware agir
    """
    pos = state.robot_pos
    state.logic.grid[pos[0]][pos[1]] = EMPTY   # peca saiu da casa
    state.robot_carrying = True
    sig_pickup(state, desc)
    draw(state, rpos=l2p(*pos),
         msg=f"{BG_CARRY}{YL} PICKUP {RS}  {BD}{desc}{RS}  @ logico={pos}  fisico={l2p(*pos)}",
         eval_s=eval_s)
    time.sleep(SIGNAL_PAUSE)


def robot_release(state, desc, eval_s):
    """
    Robo solta a peca na posicao atual:
      1. Coloca a peca no grid logico (ela chegou ao destino)
      2. Emite sinal RELEASE para o hardware
      3. Pausa para o hardware agir
    """
    pos = state.robot_pos
    state.logic.grid[pos[0]][pos[1]] = PIECE   # peca chegou ao destino
    state.robot_carrying = False
    sig_release(state, desc)
    draw(state, rpos=l2p(*pos),
         msg=f"{BG_ROBOT}{GR} RELEASE {RS} {BD}{desc}{RS}  @ logico={pos}  fisico={l2p(*pos)}",
         eval_s=eval_s)
    time.sleep(SIGNAL_PAUSE)

# ─── Execucao de Lance ────────────────────────────────────────────────────────

def execute_move(state, move, eval_s):
    """
    Executa um lance com animacao completa.

    Principios:
      - TODA navegacao parte de state.robot_pos (posicao atual)
      - PICKUP: peca sai do grid, hardware e ativado
      - RELEASE: peca entra no grid, hardware e desativado
      - state.robot_pos e atualizado apos cada navegacao
    """
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

    # Descricoes de pecas para os sinais
    fr_piece = cb.piece_at(move.from_square)
    to_piece = cb.piece_at(move.to_square)
    fr_desc  = _PNAME.get(fr_piece.piece_type,"?") if fr_piece else "Peca"
    to_desc  = _PNAME.get(to_piece.piece_type,"?") if to_piece else "Peca"

    # ════════════════════════════════════════════════════════
    #  ROQUE
    # ════════════════════════════════════════════════════════
    if is_castle:
        tag = "roque"
        ks = chess.square_file(move.to_square) == 6
        if color == chess.WHITE:
            rk_fr_sq, rk_to_sq = (chess.H1, chess.F1) if ks else (chess.A1, chess.D1)
        else:
            rk_fr_sq, rk_to_sq = (chess.H8, chess.F8) if ks else (chess.A8, chess.D8)
        rk_fr     = sq2glob(rk_fr_sq)
        rk_to     = sq2glob(rk_to_sq)
        rk_fr_sub = sq2sub(rk_fr_sq)
        rk_to_sub = sq2sub(rk_to_sq)

        # Rei: navega -> pega -> move -> solta
        robot_navigate(state, fr_glob,
                       f"Roque: indo ao Rei em {note(*fr_sub)}", eval_s)
        robot_pickup(state, f"Rei {note(*fr_sub)}", eval_s)
        robot_navigate(state, to_glob,
                       f"Roque: Rei {note(*fr_sub)} -> {note(*to_sub)}", eval_s)
        robot_release(state, f"Rei em {note(*to_sub)}", eval_s)

        # Torre: navega -> pega -> move -> solta
        robot_navigate(state, rk_fr,
                       f"Roque: indo a Torre em {note(*rk_fr_sub)}", eval_s)
        robot_pickup(state, f"Torre {note(*rk_fr_sub)}", eval_s)
        robot_navigate(state, rk_to,
                       f"Roque: Torre -> {note(*rk_to_sub)}", eval_s)
        robot_release(state, f"Torre em {note(*rk_to_sub)}", eval_s)

    # ════════════════════════════════════════════════════════
    #  PROMOCAO (com ou sem captura)
    # ════════════════════════════════════════════════════════
    elif is_promo:
        tag   = "promocao"
        ptype = move.promotion
        pname = _PNAME.get(ptype, "?")

        # Peca capturada disponivel para promocao (validacao ja feita antes de execute_move)
        res_slot = promotable_slot(state, ptype, color)

        # Se for captura+promocao: primeiro leva peca capturada ao cemiterio
        if is_cap:
            cap_ptype = to_piece.piece_type if to_piece else chess.PAWN
            cap_color = chess.BLACK if color==chess.WHITE else chess.WHITE
            cap_slot  = cem_slot_for(state, cap_ptype, cap_color)
            if cap_slot:
                robot_navigate(state, to_glob,
                               f"C+P: indo ao {to_desc} capturado em {note(*to_sub)}", eval_s)
                robot_pickup(state, f"{to_desc} capturado em {note(*to_sub)}", eval_s)
                robot_navigate(state, cap_slot, f"C+P: {to_desc} -> slot {cap_slot}", eval_s)
                robot_release(state, f"{to_desc} no cemiterio {cap_slot}", eval_s)
                state.cem_occ[cap_slot] = cap_ptype

        # Peao promovido vai para a coluna de peoes do proprio lado
        pawn_slot = cem_slot_for(state, chess.PAWN, color)
        robot_navigate(state, fr_glob,
                       f"Promocao: indo ao Peao em {note(*fr_sub)}", eval_s)
        robot_pickup(state, f"Peao {note(*fr_sub)}", eval_s)
        if pawn_slot:
            robot_navigate(state, pawn_slot, f"Promocao: Peao -> slot {pawn_slot}", eval_s)
            robot_release(state, f"Peao no cemiterio {pawn_slot}", eval_s)
            state.cem_occ[pawn_slot] = chess.PAWN

        # Busca peca capturada no cemiterio e leva ao destino (saindo do cemiterio)
        if res_slot:
            robot_navigate(state, res_slot,
                           f"Promocao: buscando {pname} capturado em {res_slot}", eval_s)
            robot_pickup(state, f"{pname} capturado (promocao)", eval_s)
            del state.cem_occ[res_slot]  # peca deixou o cemiterio
            robot_navigate(state, to_glob,
                           f"Promocao: {pname} -> {note(*to_sub)}", eval_s)
            robot_release(state, f"{pname} promovido em {note(*to_sub)}", eval_s)

    # ════════════════════════════════════════════════════════
    #  EN PASSANT: peca move PRIMEIRO, capturada vai depois
    # ════════════════════════════════════════════════════════
    elif is_ep:
        tag = "en passant"
        ep_sq    = chess.square(chess.square_file(move.to_square),
                                chess.square_rank(move.from_square))
        cap_sub  = sq2sub(ep_sq)
        cap_glob = sub2glob(*cap_sub)
        # Peao capturado e da cor oposta
        cap_color = chess.BLACK if color==chess.WHITE else chess.WHITE

        # Peao movente: navega -> pega -> move ao destino -> solta
        robot_navigate(state, fr_glob,
                       f"EP: indo ao Peao em {note(*fr_sub)}", eval_s)
        robot_pickup(state, f"Peao {note(*fr_sub)}", eval_s)
        robot_navigate(state, to_glob,
                       f"EP: Peao {note(*fr_sub)} -> {note(*to_sub)}", eval_s)
        robot_release(state, f"Peao em {note(*to_sub)}", eval_s)

        # Agora vai buscar o peao capturado (posicao diferente!)
        robot_navigate(state, cap_glob,
                       f"EP: indo ao Peao capturado em {note(*cap_sub)}", eval_s)
        robot_pickup(state, f"Peao capturado em {note(*cap_sub)}", eval_s)
        cem_slot = cem_slot_for(state, chess.PAWN, cap_color)
        if cem_slot:
            robot_navigate(state, cem_slot,
                           f"EP: Peao capturado -> slot {cem_slot}", eval_s)
            robot_release(state, f"Peao capturado no cemiterio {cem_slot}", eval_s)
            state.cem_occ[cem_slot] = chess.PAWN

    # ════════════════════════════════════════════════════════
    #  CAPTURA NORMAL
    # ════════════════════════════════════════════════════════
    elif is_cap:
        cap_ptype = to_piece.piece_type if to_piece else chess.PAWN
        cap_color = chess.BLACK if color==chess.WHITE else chess.WHITE
        cem_slot  = cem_slot_for(state, cap_ptype, cap_color)

        # 1. Peca capturada: vai ao slot especifico por tipo no cemiterio
        robot_navigate(state, to_glob,
                       f"Captura: indo ao {to_desc} em {note(*to_sub)}", eval_s)
        robot_pickup(state, f"{to_desc} capturado em {note(*to_sub)}", eval_s)
        if cem_slot:
            robot_navigate(state, cem_slot, f"Captura: {to_desc} -> slot {cem_slot}", eval_s)
            robot_release(state, f"{to_desc} no slot {cem_slot}", eval_s)
            state.cem_occ[cem_slot] = cap_ptype

        # 2. Peca movente: vai buscar na origem, leva ao destino
        robot_navigate(state, fr_glob,
                       f"Captura: indo ao {fr_desc} em {note(*fr_sub)}", eval_s)
        robot_pickup(state, f"{fr_desc} {note(*fr_sub)}", eval_s)
        robot_navigate(state, to_glob,
                       f"{s_str}: {san}  ({note(*fr_sub)} -> {note(*to_sub)})", eval_s)
        robot_release(state, f"{fr_desc} em {note(*to_sub)}", eval_s)

    # ════════════════════════════════════════════════════════
    #  MOVIMENTO SIMPLES
    # ════════════════════════════════════════════════════════
    else:
        robot_navigate(state, fr_glob,
                       f"{s_str}: indo ao {fr_desc} em {note(*fr_sub)}", eval_s)
        robot_pickup(state, f"{fr_desc} {note(*fr_sub)}", eval_s)
        robot_navigate(state, to_glob,
                       f"{s_str}: {san}  ({note(*fr_sub)} -> {note(*to_sub)})", eval_s)
        robot_release(state, f"{fr_desc} em {note(*to_sub)}", eval_s)

    # Aplica o lance e reconstroi o grid
    state.cb.push(move)
    state.robot_carrying = False
    state._rebuild()
    state.move_log.append({"san":san, "side":side, "is_capture":is_cap, "tag":tag})

    draw(state,
         msg=f"{BD}{GR}Lance: {san}  |  Robo em logico={state.robot_pos}{RS}",
         eval_s=eval_s)
    time.sleep(0.4)

# ─── Stockfish ───────────────────────────────────────────────────────────────

def sf_best(engine, board, state=None, tempo=0.3):
    """
    Retorna o melhor lance do Stockfish que seja fisicamente executavel.
    Se state for fornecido, promocoes sem peca disponivel no cemiterio sao filtradas.
    """
    valid = list(board.legal_moves)
    if state:
        valid = [m for m in valid if is_phys_valid(state, board, m)]
    if not valid:
        return None, ""

    def _eval(result):
        score = result.get("score")
        if not score: return ""
        pov = score.white()
        if pov.is_mate(): return f"M{pov.mate()}"
        cp = pov.score()
        return f"{'+' if cp>=0 else ''}{cp/100:.2f}"

    try:
        result = engine.analyse(board, chess.engine.Limit(time=tempo),
                                root_moves=valid)
    except Exception:
        result = engine.analyse(board, chess.engine.Limit(time=tempo))
    move = result["pv"][0] if result.get("pv") else None
    # Se move vier fora dos validos (fallback), pegar primeiro valido
    if move and state and not is_phys_valid(state, board, move):
        move = valid[0] if valid else None
    return move, _eval(result)

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

 {BD}Robo inicia em:{RS} logico={ROBOT_HOME}  fisico={l2p(*ROBOT_HOME)}

 {BD}{CY}╠══════════════════════════════════════════════╣{RS}
 {BD}Sinais:{RS} PICKUP/RELEASE sao emitidos automaticamente.
 {BD}Lance UCI:{RS} {YL}e2e4{RS}  {YL}e1g1{RS}(roque)  {YL}e7e8q{RS}(promocao)
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
    global FRAME_DELAY, SIGNAL_PAUSE
    nivel, human_color, FRAME_DELAY = setup()
    SIGNAL_PAUSE = max(0.15, FRAME_DELAY * 3)

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
             msg=f"{BD}{GR}Partida iniciada!  Nivel: {desc}  "
                 f"Voce: {'Brancas' if human_color==chess.WHITE else 'Negras'}{RS}")
        time.sleep(1.2)

        while not state.cb.is_game_over():
            is_human = (state.cb.turn == human_color)

            if is_human:
                home()
                print(render(state,
                             msg=f"  {BD}Seu lance (UCI | 'q' para sair):{RS} ",
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
                             msg=f"{RD}Lance ilegal: '{raw}'.{RS}", eval_s=eval_s)
                        time.sleep(1.5); continue
                    # Verifica disponibilidade fisica para promocoes
                    if not is_phys_valid(state, state.cb, mv):
                        pname = _PNAME.get(mv.promotion, "?")
                        draw(state,
                             msg=f"{RD}Promocao invalida: nenhum(a) {pname} capturado "
                                 f"disponivel no cemiterio.{RS}", eval_s=eval_s)
                        time.sleep(2); continue
                except ValueError:
                    draw(state,
                         msg=f"{RD}UCI invalido: '{raw}'.{RS}  "
                             f"Ex: {YL}e2e4  e1g1  e7e8q{RS}", eval_s=eval_s)
                    time.sleep(1.5); continue
                _, eval_s = sf_best(engine, state.cb, state=state, tempo=0.1)
            else:
                draw(state, msg=f"{DM}Stockfish ({desc}) pensando...{RS}", eval_s=eval_s)
                mv, eval_s = sf_best(engine, state.cb, state=state, tempo=tempo)
                if mv is None: break

            execute_move(state, mv, eval_s)

        outcome = state.cb.outcome()
        if outcome:
            if outcome.winner==chess.WHITE:
                end_msg=f"{BD}{WH}Brancas vencem!{RS} ({outcome.termination.name})"
            elif outcome.winner==chess.BLACK:
                end_msg=f"{BD}{RD}Negras vencem!{RS} ({outcome.termination.name})"
            else:
                end_msg=f"{BD}{YL}Empate!{RS} ({outcome.termination.name})"
        else:
            end_msg=f"{BD}{CY}Encerrado. Resultado: {state.cb.result()}{RS}"

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
        sig_total = len(state.signal_log)
        print(f"{BD}Sinais emitidos:{RS} {sig_total} "
              f"({sum(1 for k,_ in state.signal_log if k=='PU')} PICKUP + "
              f"{sum(1 for k,_ in state.signal_log if k=='RL')} RELEASE)")
        print()

if __name__ == "__main__":
    main()
