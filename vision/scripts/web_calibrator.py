#!/usr/bin/env python3
"""
vision/scripts/web_calibrator.py
================================
Calibrador web com LIVE VIEW, Configuração de Modo de Jogo (PvE / PvP) e
Laboratório de Validação da Câmera para Raspberry Pi / PC (acesso via navegador).

Funcoes:
  - Video ao vivo da camera com os 4 cantos arrastaveis por cima
  - Video ao vivo do tabuleiro corrigido (grade 8x8)
  - Salvar camera_calibration.json e reference_empty.png
  - Testar ocupacao (casa ocupada, cor e TIPO da peca)
  - Treinar pecas: capturar amostras reais de cada peca direto da camera
    (salvas em vision/pieces/samples/<tipo>_<cor>/)
  - Configurar Modo de Jogo (PvE / PvP, controle por Voz+Gantry ou Manual,
    nivel do Stockfish, verificacao pos-Gantry)
  - Simular e inspecionar as validacoes da camera em tempo real:
      * Modo Manual ("MAGNUS VALIDATE"): detecta qual lance foi feito no
        tabuleiro fisico, valida se e legal ou invalido e mostra a fala (TTS)
      * Modo PvE / Pos-Gantry: verifica se as pecas fisicas batem com o
        tabuleiro logico esperado apos um movimento do Gantry

Uso:
    python vision/scripts/web_calibrator.py
    python vision/scripts/web_calibrator.py --port 5000 --device 0
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import shutil
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import chess
import cv2
import numpy as np

import vision.config as cfg
from vision.board_state import BoardState, SquareState
from vision.board_verifier import (
    GameSettings,
    annotate_validation_image,
    clear_validation_history,
    detect_and_validate_manual_move,
    load_logical_board_snapshot,
    load_validation_history,
    record_validation_event,
    save_logical_board_snapshot,
    verify_board_position,
)
from vision.calibration import CalibrationData, PerspectiveCalibrator
from vision.camera import CameraController
from vision.piece_classifier import PieceClassifier
from vision.piece_detector import PieceDetector
from vision.segmentation import BoardSegmenter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("web_calibrator")

BOARD_SZ = cfg.BOARD_OUTPUT_SIZE_PX
CEM_W = cfg.CEMETERY_REGION_WIDTH_PX
SQ = BOARD_SZ // 8
TOTAL_W = BOARD_SZ + 2 * CEM_W
TOTAL_H = BOARD_SZ
COLS = "ABCDEFGH"
ROWS = "87654321"
PIECE_TYPES = ("pawn", "rook", "knight", "bishop", "queen", "king")
TYPE_LETTER = {"pawn": "P", "rook": "R", "knight": "N", "bishop": "B", "queen": "Q", "king": "K"}
NAME_TO_CHESS_PIECE = {
    "pawn": chess.PAWN,
    "rook": chess.ROOK,
    "knight": chess.KNIGHT,
    "bishop": chess.BISHOP,
    "queen": chess.QUEEN,
    "king": chess.KING,
}
REF_PATH = Path(cfg.VISION_DIR) / "reference_empty.png"
SAMPLES_DIR = Path(cfg.PIECE_SAMPLES_DIR)


def square_display_pos(name: str, orientation: str) -> tuple[int, int]:
    """Casa (ex.: 'E4') -> (x0, y0) em pixels na imagem corrigida."""
    c, r = COLS.index(name[0]), ROWS.index(name[1])
    if orientation == "flipped":
        c, r = 7 - c, 7 - r
    return CEM_W + c * SQ, r * SQ


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VoiceChess — Calibrador, Modos de Jogo & Validação</title>
  <style>
    :root {
      --bg: #0f172a; --card-bg: #1e293b; --card-border: #334155;
      --accent: #38bdf8; --accent-hover: #0284c7; --success: #22c55e;
      --warn: #f59e0b; --danger: #ef4444;
      --text: #f8fafc; --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 1.25rem; }
    header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.25rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--card-border); flex-wrap: wrap; gap: 0.5rem; }
    h1 { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
    h2 { font-size: 0.95rem; font-weight: 600; margin: 0.4rem 0 0.2rem; color: var(--accent); }
    .status-badge { background: #065f46; color: #34d399; font-size: 0.8rem; padding: 0.25rem 0.6rem; border-radius: 9999px; font-weight: 600; display: flex; align-items: center; gap: 0.4rem; }
    .live-dot { width: 8px; height: 8px; background: #ef4444; border-radius: 50%; animation: pulse 1.5s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-bottom: 1.25rem; }
    @media (max-width: 1024px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 0.75rem; padding: 1.25rem; display: flex; flex-direction: column; }
    .card-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 0.75rem; display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
    .view-wrapper { position: relative; width: 100%; border-radius: 0.5rem; overflow: hidden; background: #000; aspect-ratio: 4/3; }
    .preview-wrapper { position: relative; width: 100%; border-radius: 0.5rem; overflow: hidden; background: #000; aspect-ratio: 1174/640; cursor: pointer; }
    .stream-img { width: 100%; height: 100%; object-fit: fill; display: block; }
    #overlayCanvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair; }
    .controls { margin-top: 1rem; display: flex; flex-direction: column; gap: 0.8rem; }
    .point-list { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.4rem; }
    .point-row { display: flex; align-items: center; gap: 0.4rem; font-size: 0.8rem; background: #0f172a80; padding: 0.35rem 0.5rem; border-radius: 0.375rem; }
    .point-color { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
    .point-input { width: 55px; background: #0f172a; border: 1px solid var(--card-border); color: #fff; padding: 0.2rem 0.3rem; border-radius: 0.25rem; text-align: center; font-size: 0.75rem; }
    select.point-input, input.text-input { width: auto; padding: 0.35rem 0.5rem; font-size: 0.82rem; background: #0f172a; border: 1px solid var(--card-border); color: #fff; border-radius: 0.35rem; }
    .btn-group { display: flex; gap: 0.6rem; flex-wrap: wrap; align-items: center; }
    button { background: var(--accent); color: #000; font-weight: 600; border: none; padding: 0.55rem 1rem; border-radius: 0.45rem; cursor: pointer; transition: 0.15s ease; font-size: 0.85rem; }
    button:hover { background: var(--accent-hover); color: #fff; }
    button.btn-success { background: var(--success); color: #000; }
    button.btn-success:hover { background: #16a34a; color: #fff; }
    button.btn-warn { background: var(--warn); color: #000; }
    button.btn-warn:hover { background: #d97706; color: #fff; }
    button.btn-secondary { background: #334155; color: #fff; }
    button.btn-secondary:hover { background: #475569; }
    button.btn-active { background: #0284c7; color: #fff; }
    .alert { padding: 0.6rem 0.9rem; border-radius: 0.45rem; font-size: 0.85rem; display: none; line-height: 1.4; }
    .alert-success { background: #064e3b; border: 1px solid #059669; color: #6ee7b7; }
    .alert-error { background: #7f1d1d; border: 1px solid #dc2626; color: #fca5a5; }
    .slider-box { display: flex; align-items: center; gap: 0.8rem; font-size: 0.85rem; background: #0f172a60; padding: 0.5rem 0.75rem; border-radius: 0.375rem; }
    .slider-box input[type=range] { flex: 1; }
    .instructions { font-size: 0.8rem; color: var(--text-muted); line-height: 1.4; }
    .train-box { background: #0f172a60; border: 1px dashed var(--card-border); padding: 0.75rem; border-radius: 0.5rem; display: flex; flex-direction: column; gap: 0.6rem; }
    .counts { display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.3rem; font-size: 0.72rem; text-align: center; }
    .counts div { background: #0f172a; padding: 0.25rem; border-radius: 0.25rem; }
    .counts .ok { color: #34d399; } .counts .zero { color: #f87171; }
    .info { font-size: 0.85rem; color: var(--text-muted); display: none; line-height: 1.5; background: #0f172a60; padding: 0.6rem; border-radius: 0.4rem; }

    /* Estilos de Modo de Jogo e Validacao */
    .form-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.75rem; }
    @media (max-width: 640px) { .form-grid { grid-template-columns: 1fr; } }
    .field-box { background: #0f172a80; border: 1px solid var(--card-border); padding: 0.65rem 0.8rem; border-radius: 0.5rem; display: flex; flex-direction: column; gap: 0.35rem; }
    .field-box label { font-size: 0.78rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }
    .mode-banner { background: #0f172a90; border-left: 4px solid var(--accent); padding: 0.7rem 0.9rem; border-radius: 0.4rem; font-size: 0.82rem; line-height: 1.45; color: #e2e8f0; }
    .val-layout { display: grid; grid-template-columns: 270px 1fr; gap: 1rem; align-items: start; }
    @media (max-width: 720px) { .val-layout { grid-template-columns: 1fr; } }
    .chessboard-wrap { display: flex; flex-direction: column; align-items: center; gap: 0.4rem; }
    .chessboard { display: grid; grid-template-columns: repeat(8, 30px); grid-template-rows: repeat(8, 30px); border: 2px solid #475569; border-radius: 0.35rem; overflow: hidden; user-select: none; }
    .cb-sq { width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; font-size: 1.25rem; cursor: pointer; position: relative; }
    .cb-light { background: #cbd5e1; color: #0f172a; }
    .cb-dark { background: #475569; color: #f8fafc; }
    .cb-selected { outline: 3px solid #f59e0b; outline-offset: -3px; background: #fef08a !important; color: #000 !important; }
    .cb-lastmove { box-shadow: inset 0 0 0 3px #38bdf8; }
    .cb-coord { position: absolute; font-size: 0.52rem; font-weight: 700; opacity: 0.65; bottom: 1px; right: 2px; line-height: 1; }
    .speech-card { background: #0f172a; border: 1px solid var(--card-border); border-radius: 0.5rem; padding: 0.75rem; display: flex; flex-direction: column; gap: 0.45rem; }
    .badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 999px; font-size: 0.72rem; font-weight: 700; }
    .badge-ok { background: #064e3b; color: #34d399; border: 1px solid #059669; }
    .badge-err { background: #7f1d1d; color: #fca5a5; border: 1px solid #dc2626; }
    .badge-info { background: #0c4a6e; color: #7dd3fc; border: 1px solid #0284c7; }
    .lcd-preview { font-family: "Courier New", monospace; background: #022c22; color: #6ee7b7; border: 1px solid #059669; padding: 0.35rem 0.6rem; border-radius: 0.3rem; font-size: 0.82rem; letter-spacing: 0.08em; }
    .history-list { max-height: 220px; overflow-y: auto; display: flex; flex-direction: column; gap: 0.4rem; margin-top: 0.4rem; }
    .history-item { background: #0f172a90; border: 1px solid var(--card-border); border-radius: 0.4rem; padding: 0.45rem 0.65rem; font-size: 0.78rem; display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; cursor: pointer; transition: 0.12s; }
    .history-item:hover { border-color: var(--accent); background: #1e293b; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>VoiceChess — Calibrador, Modos de Jogo & Validação</h1>
      <p style="color: var(--text-muted); font-size: 0.8rem;">Vídeo ao vivo, calibração de perspectiva, configuração PvE/PvP e laboratório de validação da câmera</p>
    </div>
    <span class="status-badge"><span class="live-dot"></span> <span id="modeText">Ao Vivo</span> <span id="resText" style="color:#94a3b8;"></span></span>
  </header>

  <!-- LINHA 1: MODO DE JOGO + LABORATORIO DE VALIDACAO -->
  <div class="grid">
    <!-- CARD: CONFIGURACAO DO MODO DE JOGO (PvE / PvP) -->
    <div class="card">
      <div class="card-title">
        <span>🎮 Configuração do Modo de Jogo (PvE / PvP)</span>
        <span id="activeModeBadge" class="badge badge-info">PvE</span>
      </div>

      <div class="controls" style="margin-top: 0.2rem;">
        <div class="form-grid">
          <div class="field-box">
            <label for="gameModeSelect">Modo de Partida</label>
            <select id="gameModeSelect" class="point-input" onchange="onGameModeFormChange()">
              <option value="pve">PvE — Jogador vs Computador (Stockfish)</option>
              <option value="pvp">PvP — Jogador vs Jogador</option>
            </select>
          </div>

          <div class="field-box" id="levelBox">
            <label for="levelSelect">Dificuldade Stockfish (PvE)</label>
            <select id="levelSelect" class="point-input">
              <option value="1">Nível 1 — Very Easy (Muito Fácil)</option>
              <option value="2">Nível 2 — Easy (Fácil)</option>
              <option value="3" selected>Nível 3 — Beginner (Iniciante)</option>
              <option value="4">Nível 4 — Casual</option>
              <option value="5">Nível 5 — Intermediate (Intermediário)</option>
              <option value="6">Nível 6 — Club Player</option>
              <option value="7">Nível 7 — Advanced (Avançado)</option>
              <option value="8">Nível 8 — Expert</option>
              <option value="9">Nível 9 — Master (Mestre)</option>
              <option value="10">Nível 10 — Super Hard (Máximo)</option>
            </select>
          </div>

          <div class="field-box">
            <label for="whiteControlSelect">Controle das Brancas (White)</label>
            <select id="whiteControlSelect" class="point-input" onchange="onGameModeFormChange()">
              <option value="voice">🎤 Voz + Gantry (Automático)</option>
              <option value="manual">✋ Manual no Tabuleiro ("MAGNUS VALIDATE")</option>
            </select>
          </div>

          <div class="field-box">
            <label for="blackControlSelect">Controle das Pretas (Black)</label>
            <select id="blackControlSelect" class="point-input" onchange="onGameModeFormChange()">
              <option value="computer">🤖 Computador (Stockfish + Gantry)</option>
              <option value="voice">🎤 Voz + Gantry (Automático)</option>
              <option value="manual">✋ Manual no Tabuleiro ("MAGNUS VALIDATE")</option>
            </select>
          </div>
        </div>

        <label style="display:flex; align-items:center; gap:0.55rem; font-size:0.84rem; background:#0f172a60; padding:0.55rem 0.75rem; border-radius:0.4rem; cursor:pointer;">
          <input type="checkbox" id="verifyGantryCheck" checked onchange="onGameModeFormChange()">
          <span>📷 <strong>Verificar posição pós-Gantry:</strong> capturar foto sempre que o Gantry terminar um movimento para checar se as peças estão nas casas corretas.</span>
        </label>

        <div id="modeBehaviorBanner" class="mode-banner"></div>

        <div class="btn-group" style="justify-content: space-between;">
          <span style="font-size: 0.76rem; color: var(--text-muted);">Salvo em <code>vision/game_settings.json</code> (lido automaticamente pelo <code>main.py</code>)</span>
          <button class="btn-success" onclick="saveGameSettings()">💾 Salvar Modo de Jogo</button>
        </div>
      </div>
    </div>

    <!-- CARD: LABORATORIO DE VALIDACAO DA CAMERA & TABULEIRO LOGICO -->
    <div class="card">
      <div class="card-title">
        <span>🔬 Simulador de Validação da Câmera & Tabuleiro Lógico</span>
        <span id="turnBadge" class="badge badge-info">Vez: Brancas</span>
      </div>

      <div class="val-layout">
        <!-- Mini tabuleiro logico interativo -->
        <div class="chessboard-wrap">
          <div id="logicalBoardGrid" class="chessboard"></div>
          <div style="font-size: 0.73rem; color: var(--text-muted); text-align: center;">
            Clique em 2 casas para mover no tabuleiro lógico
          </div>
          <div class="btn-group" style="justify-content: center; gap: 0.35rem;">
            <button class="btn-secondary" style="padding:0.3rem 0.55rem; font-size:0.75rem;" onclick="logicalBoardAction('undo')" title="Desfazer último lance">↩️ Desfazer</button>
            <button class="btn-secondary" style="padding:0.3rem 0.55rem; font-size:0.75rem;" onclick="logicalBoardAction('reset')" title="Posição inicial padrão">🔄 Início</button>
            <button class="btn-secondary" style="padding:0.3rem 0.55rem; font-size:0.75rem;" onclick="logicalBoardAction('sync_from_camera')" title="Copia as peças atualmente detectadas pela câmera para o tabuleiro lógico">📷 Ler da Câmera</button>
          </div>
        </div>

        <!-- Controles de validacao e resultado -->
        <div style="display:flex; flex-direction:column; gap:0.65rem;">
          <div class="btn-group">
            <button class="btn-warn" onclick="runMagnusValidate()">🎯 Simular "MAGNUS VALIDATE" (Manual)</button>
            <button class="btn-secondary" onclick="runVerifyPosition()">✅ Verificar Posição Pós-Gantry (PvE)</button>
          </div>

          <label style="display:flex; align-items:center; gap:0.45rem; font-size:0.78rem; color:var(--text-muted); cursor:pointer;">
            <input type="checkbox" id="autoApplyCheck" checked>
            <span>Atualizar o tabuleiro lógico automaticamente quando <strong>MAGNUS VALIDATE</strong> detectar um lance válido</span>
          </label>

          <!-- Resultado da ultima validacao -->
          <div class="speech-card" id="validationResultCard">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <strong style="font-size:0.84rem; color:var(--accent);" id="valTitle">Aguardando validação...</strong>
              <span id="valStatusBadge" class="badge badge-info">Pronto</span>
            </div>
            <div style="font-size:0.83rem; line-height:1.4;" id="valMessage">
              Faça um lance físico no tabuleiro e clique em <strong>Simular "MAGNUS VALIDATE"</strong>, ou teste a <strong>Verificação Pós-Gantry</strong>.
            </div>
            <div style="font-size:0.8rem; color:#fde047;" id="valSpeech"></div>
            <div class="lcd-preview" id="valLcd">LCD: VOICE CHESS / READY</div>
          </div>

          <!-- Historico de validacoes -->
          <div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <span style="font-size:0.78rem; font-weight:600; color:var(--text-muted);">📋 Histórico de Validações (clique para ver a foto anotada)</span>
              <div style="display:flex; gap:0.35rem;">
                <button class="btn-secondary" style="padding:0.2rem 0.5rem; font-size:0.7rem;" onclick="loadGameState()">🔄 Atualizar</button>
                <button class="btn-secondary" style="padding:0.2rem 0.5rem; font-size:0.7rem;" onclick="clearHistory()">🗑️ Limpar</button>
              </div>
            </div>
            <div class="history-list" id="historyList"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- LINHA 2: CAMERA AO VIVO + TABULEIRO CORRIGIDO / TREINO -->
  <div class="grid">
    <!-- COLUNA 1: CAMERA AO VIVO + CANTOS -->
    <div class="card">
      <div class="card-title">
        <span>Câmera (arraste os 4 cantos) <span id="camSourceBadge" class="badge badge-info" style="margin-left:0.35rem;">Verificando...</span></span>
        <div style="display: flex; gap: 0.4rem; align-items: center;">
          <select id="camDeviceSelect" class="point-input" style="padding: 0.2rem 0.4rem; font-size: 0.75rem;" title="Índice da câmera USB">
            <option value="0">Cam 0</option>
            <option value="1">Cam 1</option>
            <option value="2">Cam 2</option>
          </select>
          <button class="btn-secondary" style="padding: 0.25rem 0.6rem; font-size: 0.75rem;" onclick="reconnectCamera()">🔌 Reconectar</button>
          <button id="btnToggleLive" class="btn-secondary" style="padding: 0.25rem 0.6rem; font-size: 0.75rem;" onclick="toggleLiveMode()">⏸️ Congelar</button>
          <button class="btn-secondary" style="padding: 0.25rem 0.6rem; font-size: 0.75rem;" onclick="refreshStream()">🔄 Recarregar</button>
        </div>
      </div>

      <div class="view-wrapper" id="viewBox">
        <img id="cameraImg" class="stream-img" src="/stream" alt="Live stream">
        <canvas id="overlayCanvas"></canvas>
      </div>

      <div class="controls">
        <div class="point-list">
          <div class="point-row"><span class="point-color" style="background:#38bdf8;"></span><span>1. Sup. Esq (A8):</span>
            <input type="number" id="x0" class="point-input" onchange="manualCoord(0)"><input type="number" id="y0" class="point-input" onchange="manualCoord(0)"></div>
          <div class="point-row"><span class="point-color" style="background:#4ade80;"></span><span>2. Sup. Dir (H8):</span>
            <input type="number" id="x1" class="point-input" onchange="manualCoord(1)"><input type="number" id="y1" class="point-input" onchange="manualCoord(1)"></div>
          <div class="point-row"><span class="point-color" style="background:#facc15;"></span><span>3. Inf. Dir (H1):</span>
            <input type="number" id="x2" class="point-input" onchange="manualCoord(2)"><input type="number" id="y2" class="point-input" onchange="manualCoord(2)"></div>
          <div class="point-row"><span class="point-color" style="background:#fb923c;"></span><span>4. Inf. Esq (A1):</span>
            <input type="number" id="x3" class="point-input" onchange="manualCoord(3)"><input type="number" id="y3" class="point-input" onchange="manualCoord(3)"></div>
        </div>
        <div class="instructions">
          Coloque cada círculo no <strong>canto externo das casas</strong> (não nas letras/números da borda):
          1=canto de A8, 2=canto de H8, 3=canto de H1, 4=canto de A1. Confira no painel da direita se as linhas verdes caem nas divisões das casas.
        </div>
        <div class="btn-group">
          <button class="btn-secondary" onclick="resetPoints()">Limpar Pontos</button>
          <button class="btn-secondary" onclick="reloadSaved()">Carregar Salvo</button>
        </div>
      </div>
    </div>

    <!-- COLUNA 2: TABULEIRO CORRIGIDO + DETECCAO + TREINO -->
    <div class="card">
      <div class="card-title">
        <span>Tabuleiro Corrigido & Diagnóstico Visual (clique numa casa para selecioná-la)</span>
      </div>

      <div class="preview-wrapper" id="previewBox">
        <img id="previewImg" class="stream-img" src="/stream_warped" alt="Tabuleiro corrigido">
      </div>

      <div class="controls">
        <div class="btn-group" style="justify-content: space-between;">
          <div style="display:flex; align-items:center; gap:0.5rem;">
            <label style="font-size: 0.85rem; color: var(--text-muted);">Orientação:</label>
            <select id="orientationSelect" class="point-input" onchange="syncCorners()">
              <option value="normal">Normal (A1 em baixo à esquerda)</option>
              <option value="flipped">Invertida (A1 no topo à direita)</option>
            </select>
          </div>
          <button class="btn-success" onclick="saveCalibration()">💾 Salvar Calibração</button>
        </div>

        <div class="slider-box">
          <span>Sensibilidade (limiar MAD):</span>
          <input type="range" id="thresholdSlider" min="0.04" max="0.25" step="0.01" value="0.10" oninput="document.getElementById('thresholdVal').textContent=parseFloat(this.value).toFixed(2)">
          <strong id="thresholdVal" style="color: var(--accent);">0.10</strong>
        </div>

        <div class="btn-group">
          <button class="btn-secondary" onclick="saveReference()">📷 Gravar Tabuleiro Vazio</button>
          <button class="btn-secondary" onclick="testDetection()">🔍 Testar Ocupação</button>
          <button class="btn-secondary" onclick="backToLive()">▶️ Voltar ao vivo</button>
        </div>

        <div id="alertBox" class="alert"></div>
        <div id="detectionInfo" class="info"></div>

        <!-- TREINO DAS PECAS -->
        <div class="train-box">
          <h2>🧠 Treinar peças (amostras reais da câmera)</h2>
          <div class="instructions">
            Coloque uma peça numa casa, clique na casa no painel acima (ou escolha abaixo), selecione tipo e cor e clique em <strong>Capturar</strong>.
            Faça 3–5 capturas por peça, em casas claras e escuras e girando a peça.
          </div>
          <div class="btn-group">
            <label style="font-size:0.85rem;">Casa:</label>
            <select id="squareSelect" class="point-input"></select>
            <select id="typeSelect" class="point-input">
              <option value="pawn">Peão</option><option value="rook">Torre</option><option value="knight">Cavalo</option>
              <option value="bishop">Bispo</option><option value="queen">Rainha</option><option value="king">Rei</option>
            </select>
            <select id="colorSelect" class="point-input">
              <option value="white">Branca</option><option value="black">Preta</option>
            </select>
            <button onclick="captureSample()">📸 Capturar</button>
            <button class="btn-secondary" onclick="clearSamples()">🗑️ Apagar deste tipo</button>
          </div>
          <div class="counts" id="countsBox"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const cameraImg = document.getElementById("cameraImg");
    const previewImg = document.getElementById("previewImg");
    const viewBox = document.getElementById("viewBox");
    const previewBox = document.getElementById("previewBox");
    const canvas = document.getElementById("overlayCanvas");
    const ctx = canvas.getContext("2d");

    let isLive = true;
    let corners = [];
    let activeDragIdx = -1;
    const colors = ["#38bdf8", "#4ade80", "#facc15", "#fb923c"];
    let naturalW = 640, naturalH = 480;
    let layout = { cem_w: 267, board_sz: 640, total_w: 1174, total_h: 640 };
    const COLS = "ABCDEFGH", ROWS = "87654321";
    const NAMES = { pawn: "Peão", rook: "Torre", knight: "Cavalo", bishop: "Bispo", queen: "Rainha", king: "Rei" };
    const PIECE_UNICODE = {
      P: "♙", N: "♘", B: "♗", R: "♖", Q: "♕", K: "♔",
      p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚"
    };

    // Estado do mini tabuleiro logico na UI
    let selectedLogicalSquare = null;
    let currentBoardSquares = {};
    let currentLastMoveUci = null;
    let cachedHistory = [];

    // Preenche a lista de casas
    const sqSel = document.getElementById("squareSelect");
    for (const r of ROWS) for (const c of COLS) {
      const o = document.createElement("option"); o.value = o.textContent = c + r; sqSel.appendChild(o);
    }
    sqSel.value = "E4";

    function showAlert(msg, ok = true) {
      const box = document.getElementById("alertBox");
      box.className = "alert " + (ok ? "alert-success" : "alert-error");
      box.innerHTML = msg;
      box.style.display = "block";
      clearTimeout(window._alertT);
      window._alertT = setTimeout(() => { box.style.display = "none"; }, 6000);
    }

    function applyFrameSize(w, h) {
      if (!w || !h) return;
      naturalW = w; naturalH = h;
      viewBox.style.aspectRatio = w + " / " + h;
      document.getElementById("resText").textContent = "· " + w + "x" + h;
      resizeOverlay();
    }

    function toggleLiveMode() {
      const btn = document.getElementById("btnToggleLive");
      const dot = document.querySelector(".live-dot");
      const text = document.getElementById("modeText");
      isLive = !isLive;
      if (!isLive) {
        cameraImg.src = "/api/snapshot?t=" + Date.now();
        previewImg.src = "/api/snapshot_warped?t=" + Date.now();
        btn.textContent = "▶️ Retomar"; btn.className = "btn-secondary btn-active";
        dot.style.background = "#94a3b8"; text.textContent = "Pausado";
      } else {
        cameraImg.src = "/stream?t=" + Date.now();
        previewImg.src = "/stream_warped?t=" + Date.now();
        btn.textContent = "⏸️ Congelar"; btn.className = "btn-secondary";
        dot.style.background = "#ef4444"; text.textContent = "Ao Vivo";
      }
    }

    function refreshStream() {
      cameraImg.src = (isLive ? "/stream?t=" : "/api/snapshot?t=") + Date.now();
      previewImg.src = (isLive ? "/stream_warped?t=" : "/api/snapshot_warped?t=") + Date.now();
    }

    function backToLive() {
      previewImg.src = (isLive ? "/stream_warped?t=" : "/api/snapshot_warped?t=") + Date.now();
    }

    // ── Overlay dos cantos ──
    function resizeOverlay() {
      const rect = viewBox.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      renderOverlay();
    }
    window.addEventListener("resize", resizeOverlay);

    function renderOverlay() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const sx = canvas.width / naturalW, sy = canvas.height / naturalH;
      if (corners.length >= 2) {
        ctx.beginPath();
        ctx.moveTo(corners[0][0] * sx, corners[0][1] * sy);
        for (let i = 1; i < corners.length; i++) ctx.lineTo(corners[i][0] * sx, corners[i][1] * sy);
        if (corners.length === 4) { ctx.closePath(); ctx.fillStyle = "rgba(56,189,248,0.15)"; ctx.fill(); }
        ctx.lineWidth = 2; ctx.strokeStyle = "#38bdf8"; ctx.stroke();
      }
      corners.forEach((pt, idx) => {
        const cx = pt[0] * sx, cy = pt[1] * sy;
        ctx.strokeStyle = colors[idx]; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(cx - 14, cy); ctx.lineTo(cx + 14, cy); ctx.moveTo(cx, cy - 14); ctx.lineTo(cx, cy + 14); ctx.stroke();
        ctx.beginPath(); ctx.arc(cx, cy, 7, 0, 2 * Math.PI);
        ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.stroke();
        ctx.fillStyle = colors[idx]; ctx.font = "bold 13px sans-serif";
        ctx.fillText((idx + 1).toString(), cx + 10, cy - 8);
        document.getElementById("x" + idx).value = Math.round(pt[0]);
        document.getElementById("y" + idx).value = Math.round(pt[1]);
      });
    }

    function toNatural(e) {
      const rect = canvas.getBoundingClientRect();
      return [(e.clientX - rect.left) * naturalW / rect.width,
              (e.clientY - rect.top) * naturalH / rect.height];
    }

    canvas.addEventListener("mousedown", (e) => {
      const [x, y] = toNatural(e);
      const tol = 25 * naturalW / canvas.getBoundingClientRect().width;
      activeDragIdx = corners.findIndex(p => Math.hypot(p[0] - x, p[1] - y) < tol);
      if (activeDragIdx === -1 && corners.length < 4) {
        corners.push([x, y]);
        renderOverlay();
        if (corners.length === 4) syncCorners();
      }
    });
    window.addEventListener("mousemove", (e) => {
      if (activeDragIdx === -1) return;
      const [x, y] = toNatural(e);
      corners[activeDragIdx] = [Math.max(0, Math.min(naturalW, x)), Math.max(0, Math.min(naturalH, y))];
      renderOverlay();
    });
    window.addEventListener("mouseup", () => {
      if (activeDragIdx !== -1) { activeDragIdx = -1; if (corners.length === 4) syncCorners(); }
    });

    function manualCoord(idx) {
      if (!corners[idx]) return;
      const x = parseFloat(document.getElementById("x" + idx).value);
      const y = parseFloat(document.getElementById("y" + idx).value);
      if (!isNaN(x) && !isNaN(y)) { corners[idx] = [x, y]; renderOverlay(); if (corners.length === 4) syncCorners(); }
    }

    function resetPoints() {
      corners = [];
      for (let i = 0; i < 4; i++) { document.getElementById("x" + i).value = ""; document.getElementById("y" + i).value = ""; }
      renderOverlay(); syncCorners();
    }

    function syncCorners() {
      const orientation = document.getElementById("orientationSelect").value;
      fetch("/api/set_corners", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ corners, orientation }) })
      .then(() => { if (!isLive) previewImg.src = "/api/snapshot_warped?t=" + Date.now(); });
    }

    // ── Clique no tabuleiro corrigido -> seleciona a casa ──
    previewBox.addEventListener("click", (e) => {
      const rect = previewImg.getBoundingClientRect();
      const x = (e.clientX - rect.left) * layout.total_w / rect.width;
      const y = (e.clientY - rect.top) * layout.total_h / rect.height;
      const sq = layout.board_sz / 8;
      let c = Math.floor((x - layout.cem_w) / sq), r = Math.floor(y / sq);
      if (c < 0 || c > 7 || r < 0 || r > 7) return;
      if (document.getElementById("orientationSelect").value === "flipped") { c = 7 - c; r = 7 - r; }
      sqSel.value = COLS[c] + ROWS[r];
      showAlert("Casa selecionada: <strong>" + sqSel.value + "</strong>");
    });

    function post(url, body) {
      return fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                          body: JSON.stringify(body || {}) }).then(r => r.json());
    }

    function saveCalibration() {
      if (corners.length !== 4) { showAlert("Selecione os 4 cantos antes de salvar.", false); return; }
      post("/api/save", { corners, orientation: document.getElementById("orientationSelect").value })
      .then(res => res.success
        ? showAlert("Calibração salva! <strong>Agora grave o tabuleiro vazio novamente</strong> (a referência antiga usa a grade antiga).")
        : showAlert("Erro ao salvar: " + res.error, false))
      .catch(err => showAlert("Falha de rede: " + err, false));
    }

    function saveReference() {
      post("/api/save_reference").then(res => res.success
        ? showAlert("Referência do tabuleiro vazio gravada em <strong>reference_empty.png</strong>!")
        : showAlert("Erro: " + res.error, false));
    }

    function testDetection() {
      const info = document.getElementById("detectionInfo");
      const thresh = parseFloat(document.getElementById("thresholdSlider").value);
      info.style.display = "block";
      info.innerHTML = "<em>Analisando...</em>";
      post("/api/detect", { threshold: thresh }).then(res => {
        if (!res.success) { info.innerHTML = "<span style='color:#fca5a5;'>Erro: " + res.error + "</span>"; return; }
        if (res.annotated_image) previewImg.src = "data:image/jpeg;base64," + res.annotated_image;
        if (!isLive) cameraImg.src = "/api/snapshot?t=" + Date.now();
        const list = res.squares.length
          ? res.squares.map(s => `${s.square}: ${s.color === "white" ? "branca" : "preta"} ${s.piece ? NAMES[s.piece] : "<span style='color:#fbbf24'>?</span>"}`).join(" · ")
          : "<strong style='color:#34d399;'>Nenhuma peça (tabuleiro vazio)</strong>";
        const hint = res.classifier === "samples" ? "amostras reais"
                   : res.classifier === "icons" ? "ícones genéricos (treine as peças para melhorar!)" : "desligado";
        info.innerHTML = `<strong>Peças (${res.squares.length}):</strong> ${list}<br>
          <strong>Classificador:</strong> ${hint} | <strong>Limiar:</strong> ${thresh.toFixed(2)} | <strong>Confiança:</strong> ${(res.confidence * 100).toFixed(1)}%`;
      }).catch(err => { info.innerHTML = "Erro de conexão: " + err; });
    }

    // ── Treino ──
    function loadCounts() {
      fetch("/api/samples").then(r => r.json()).then(data => {
        const box = document.getElementById("countsBox");
        box.innerHTML = "";
        for (const color of ["white", "black"]) for (const t of ["pawn","rook","knight","bishop","queen","king"]) {
          const n = data[t + "_" + color] || 0;
          const d = document.createElement("div");
          d.className = n ? "ok" : "zero";
          d.textContent = (color === "white" ? "⚪ " : "⚫ ") + NAMES[t] + ": " + n;
          box.appendChild(d);
        }
      });
    }

    function captureSample() {
      const body = { square: sqSel.value, type: document.getElementById("typeSelect").value,
                     color: document.getElementById("colorSelect").value };
      post("/api/capture_sample", body).then(res => {
        if (res.success) {
          showAlert(`Amostra salva: ${NAMES[body.type]} ${body.color === "white" ? "branca" : "preta"} em ${body.square} (total: ${res.count})`);
          loadCounts();
        } else showAlert("Erro: " + res.error, false);
      });
    }

    function clearSamples() {
      const t = document.getElementById("typeSelect").value, c = document.getElementById("colorSelect").value;
      if (!confirm(`Apagar todas as amostras de ${NAMES[t]} ${c === "white" ? "branca" : "preta"}?`)) return;
      post("/api/clear_samples", { type: t, color: c }).then(() => { showAlert("Amostras apagadas."); loadCounts(); });
    }

    // ── Modo de Jogo (PvE / PvP) ──
    function onGameModeFormChange() {
      const mode = document.getElementById("gameModeSelect").value;
      const blackSel = document.getElementById("blackControlSelect");
      const whiteSel = document.getElementById("whiteControlSelect");
      const levelBox = document.getElementById("levelBox");
      const verifyCheck = document.getElementById("verifyGantryCheck").checked;
      const badge = document.getElementById("activeModeBadge");

      if (mode === "pve") {
        blackSel.value = "computer";
        blackSel.disabled = true;
        levelBox.style.opacity = "1";
        badge.textContent = "PvE (vs Stockfish)";
      } else {
        if (blackSel.value === "computer") blackSel.value = "manual";
        blackSel.disabled = false;
        levelBox.style.opacity = "0.45";
        badge.textContent = "PvP (2 Jogadores)";
      }

      const wCtrl = whiteSel.value === "manual" ? "Manual (MAGNUS VALIDATE)" : "Voz + Gantry";
      const bCtrl = mode === "pve"
        ? "Computador (Stockfish + Gantry)"
        : (blackSel.value === "manual" ? "Manual (MAGNUS VALIDATE)" : "Voz + Gantry");

      const banner = document.getElementById("modeBehaviorBanner");
      if (mode === "pve") {
        banner.innerHTML = `<strong>Comportamento da Câmera no Modo PvE:</strong><br>`
          + `• <strong>Brancas (${wCtrl}) vs Pretas (${bCtrl}):</strong> `
          + (verifyCheck
             ? `Sempre que o <strong>Gantry</strong> concluir um movimento, a câmera captura uma foto automaticamente para verificar se as peças físicas estão nas casas esperadas.`
             : `Verificação pós-Gantry desativada.`)
          + (whiteSel.value === "manual"
             ? `<br>• Como as Brancas estão em modo <strong>Manual</strong>, diga <code>MAGNUS VALIDATE</code> após mover sua peça no tabuleiro.`
             : ``);
      } else {
        banner.innerHTML = `<strong>Comportamento da Câmera no Modo PvP:</strong><br>`
          + `• <strong>Brancas:</strong> ${wCtrl} &nbsp;|&nbsp; <strong>Pretas:</strong> ${bCtrl}<br>`
          + `• Nos turnos em modo <strong>Manual</strong>, o jogador move a peça fisicamente e diz <code>MAGNUS VALIDATE</code>. A câmera tira uma foto, valida se o lance é legal ou inválido e anuncia em voz alta qual foi o lance jogado!`;
      }
    }

    function saveGameSettings() {
      const payload = {
        game_mode: document.getElementById("gameModeSelect").value,
        level: parseInt(document.getElementById("levelSelect").value, 10),
        white_control: document.getElementById("whiteControlSelect").value,
        black_control: document.getElementById("blackControlSelect").value,
        verify_gantry_moves: document.getElementById("verifyGantryCheck").checked,
        diff_threshold: parseFloat(document.getElementById("thresholdSlider").value)
      };
      post("/api/game_settings", payload).then(res => {
        if (res.success) {
          showAlert("Configuração de modo de jogo salva em <strong>game_settings.json</strong>!");
          applyGameSettingsToForm(res.settings);
        } else {
          showAlert("Erro ao salvar modo de jogo: " + res.error, false);
        }
      });
    }

    function applyGameSettingsToForm(s) {
      if (!s) return;
      document.getElementById("gameModeSelect").value = s.game_mode || "pve";
      document.getElementById("levelSelect").value = s.level || 4;
      document.getElementById("whiteControlSelect").value = s.white_control || "voice";
      document.getElementById("blackControlSelect").value = s.black_control || (s.game_mode === "pve" ? "computer" : "manual");
      document.getElementById("verifyGantryCheck").checked = s.verify_gantry_moves !== false;
      if (s.diff_threshold) {
        document.getElementById("thresholdSlider").value = s.diff_threshold;
        document.getElementById("thresholdVal").textContent = parseFloat(s.diff_threshold).toFixed(2);
      }
      onGameModeFormChange();
    }

    // ── Tabuleiro Logico & Validacoes da Camera ──
    function renderLogicalBoard(boardData) {
      if (!boardData) return;
      currentBoardSquares = boardData.squares || {};
      currentLastMoveUci = boardData.last_move_uci || null;
      const grid = document.getElementById("logicalBoardGrid");
      grid.innerHTML = "";

      const turnBadge = document.getElementById("turnBadge");
      turnBadge.textContent = boardData.turn === "white" ? "Vez: Brancas ⚪" : "Vez: Pretas ⚫";

      const lastFrom = currentLastMoveUci ? currentLastMoveUci.slice(0, 2).toUpperCase() : null;
      const lastTo = currentLastMoveUci ? currentLastMoveUci.slice(2, 4).toUpperCase() : null;

      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const sq = COLS[c] + ROWS[r];
          const div = document.createElement("div");
          const isLight = (r + c) % 2 === 0;
          div.className = "cb-sq " + (isLight ? "cb-light" : "cb-dark");
          if (selectedLogicalSquare === sq) div.classList.add("cb-selected");
          if (sq === lastFrom || sq === lastTo) div.classList.add("cb-lastmove");

          const p = currentBoardSquares[sq];
          if (p && PIECE_UNICODE[p.symbol]) {
            div.textContent = PIECE_UNICODE[p.symbol];
          }
          if (c === 7 || r === 7) {
            const coord = document.createElement("span");
            coord.className = "cb-coord";
            coord.textContent = sq;
            div.appendChild(coord);
          }
          div.onclick = () => onLogicalSquareClick(sq);
          grid.appendChild(div);
        }
      }
    }

    function onLogicalSquareClick(sq) {
      if (!selectedLogicalSquare) {
        if (!currentBoardSquares[sq]) return;
        selectedLogicalSquare = sq;
        renderLogicalBoard({ squares: currentBoardSquares, turn: document.getElementById("turnBadge").textContent.includes("Brancas") ? "white" : "black", last_move_uci: currentLastMoveUci });
        return;
      }
      if (selectedLogicalSquare === sq) {
        selectedLogicalSquare = null;
        loadGameState();
        return;
      }
      const uci = (selectedLogicalSquare + sq).toLowerCase();
      selectedLogicalSquare = null;
      post("/api/logical_board", { action: "push", uci }).then(res => {
        if (!res.success) {
          showAlert("Lance lógico inválido (" + uci.toUpperCase() + "): " + res.error, false);
        }
        if (res.board) renderLogicalBoard(res.board);
      });
    }

    function logicalBoardAction(action) {
      selectedLogicalSquare = null;
      const thresh = parseFloat(document.getElementById("thresholdSlider").value);
      post("/api/logical_board", { action, threshold: thresh }).then(res => {
        if (res.success) {
          if (res.message) showAlert(res.message);
          if (res.board) renderLogicalBoard(res.board);
        } else {
          showAlert("Erro: " + res.error, false);
        }
      });
    }

    function displayValidationCard(res, titlePrefix) {
      const title = document.getElementById("valTitle");
      const badge = document.getElementById("valStatusBadge");
      const msg = document.getElementById("valMessage");
      const speech = document.getElementById("valSpeech");
      const lcd = document.getElementById("valLcd");

      const ok = res.valid !== undefined ? res.valid : res.matches;
      title.textContent = titlePrefix + (res.san ? ` — ${res.san} (${res.move_uci})` : "");
      badge.className = "badge " + (ok ? "badge-ok" : "badge-err");
      badge.textContent = ok ? "VÁLIDO / OK" : (res.error_code || "INVÁLIDO");
      msg.innerHTML = res.message || "";
      speech.innerHTML = res.speech ? `🔊 <strong>Fala (TTS):</strong> "${res.speech}"` : "";
      if (res.lcd_lines && res.lcd_lines.length === 2) {
        lcd.textContent = `LCD: ${res.lcd_lines[0]}  |  ${res.lcd_lines[1]}`;
      }
      if (res.annotated_image) {
        previewImg.src = "data:image/jpeg;base64," + res.annotated_image;
      }
      if (!isLive) {
        cameraImg.src = "/api/snapshot?t=" + Date.now();
      }
    }

    function runMagnusValidate() {
      const thresh = parseFloat(document.getElementById("thresholdSlider").value);
      const autoApply = document.getElementById("autoApplyCheck").checked;
      document.getElementById("valTitle").textContent = "Analisando câmera (MAGNUS VALIDATE)...";
      post("/api/validate_manual", { threshold: thresh, auto_apply: autoApply }).then(res => {
        if (!res.success) {
          showAlert("Erro na validação: " + res.error, false);
          return;
        }
        displayValidationCard(res, "MAGNUS VALIDATE");
        if (res.board) renderLogicalBoard(res.board);
        if (res.history) renderHistory(res.history);
      });
    }

    function runVerifyPosition() {
      const thresh = parseFloat(document.getElementById("thresholdSlider").value);
      document.getElementById("valTitle").textContent = "Verificando posição pós-Gantry...";
      post("/api/verify_position", { threshold: thresh }).then(res => {
        if (!res.success) {
          showAlert("Erro na verificação: " + res.error, false);
          return;
        }
        displayValidationCard(res, "Verificação Pós-Gantry");
        if (res.history) renderHistory(res.history);
      });
    }

    function renderHistory(items) {
      cachedHistory = items || [];
      const box = document.getElementById("historyList");
      box.innerHTML = "";
      if (!cachedHistory.length) {
        box.innerHTML = `<div style="font-size:0.76rem; color:var(--text-muted); padding:0.4rem;">Nenhuma validação registrada ainda.</div>`;
        return;
      }
      cachedHistory.forEach((it, idx) => {
        const row = document.createElement("div");
        row.className = "history-item";
        const isManual = it.type === "manual_validate";
        const typeLabel = isManual ? "MAGNUS VALIDATE" : "PÓS-GANTRY";
        const badgeCls = it.valid ? "badge-ok" : "badge-err";
        const statusTxt = it.valid ? (it.san || it.move_uci || "OK") : (it.error_code || "ERRO");
        row.innerHTML = `
          <div>
            <strong>[${it.timestamp || "--:--"}] ${typeLabel}</strong>
            <div style="color:var(--text-muted); font-size:0.73rem;">${it.speech || it.message || ""}</div>
          </div>
          <span class="badge ${badgeCls}">${statusTxt}</span>
        `;
        row.onclick = () => {
          displayValidationCard(it, typeLabel);
        };
        box.appendChild(row);
      });
    }

    function clearHistory() {
      post("/api/clear_history").then(() => renderHistory([]));
    }

    let lastHistorySignature = "";
    function loadGameState(isPoll) {
      fetch("/api/game_state").then(r => r.json()).then(data => {
        if (!isPoll && data.settings) applyGameSettingsToForm(data.settings);
        if (data.board && !selectedLogicalSquare) renderLogicalBoard(data.board);
        if (data.history) {
          const top = data.history[0];
          const sig = top ? `${top.timestamp}|${top.type}|${top.move_uci}|${top.error_code}` : "empty";
          const changed = sig !== lastHistorySignature;
          lastHistorySignature = sig;
          renderHistory(data.history);
          if (isPoll && changed && top) {
            const label = top.type === "manual_validate" ? "MAGNUS VALIDATE" : "Verificação Pós-Gantry";
            displayValidationCard(top, label);
          }
        }
      }).catch(() => {});
    }

    function updateCameraBadge(hasRealCamera, deviceIdx) {
      const b = document.getElementById("camSourceBadge");
      if (!b) return;
      if (hasRealCamera) {
        b.className = "badge badge-ok";
        b.textContent = "📷 Câmera USB Ativa (Cam " + (deviceIdx ?? 0) + ")";
      } else {
        b.className = "badge badge-err";
        b.textContent = "⚠️ Sem Câmera USB (Usando Referência Salva)";
      }
    }

    function reconnectCamera() {
      const dev = parseInt(document.getElementById("camDeviceSelect").value, 10);
      post("/api/reconnect_camera", { device_index: dev }).then(res => {
        updateCameraBadge(res.camera_connected, res.device_index);
        if (res.frame_size) applyFrameSize(res.frame_size[0], res.frame_size[1]);
        refreshStream();
        if (res.camera_connected) {
          showAlert("Câmera USB conectada com sucesso no dispositivo " + res.device_index + "!");
        } else {
          showAlert("Não foi possível abrir a câmera USB (verifique se ela está conectada ou se outro terminal está usando /dev/video" + dev + ").", false);
        }
      });
    }

    function reloadSaved(silent) {
      fetch("/api/status").then(r => r.json()).then(data => {
        updateCameraBadge(data.camera_connected, data.device_index);
        if (data.device_index !== undefined) {
          document.getElementById("camDeviceSelect").value = String(data.device_index);
        }
        if (data.frame_size) applyFrameSize(data.frame_size[0], data.frame_size[1]);
        if (data.layout) { layout = data.layout; previewBox.style.aspectRatio = layout.total_w + " / " + layout.total_h; }
        const cal = data.calibration;
        if (cal && cal.board_corners && cal.board_corners.length === 4) {
          corners = cal.board_corners;
          document.getElementById("orientationSelect").value = cal.board_orientation || "normal";
          renderOverlay(); syncCorners();
          if (!silent) showAlert("Calibração salva carregada.");
        } else if (!silent) showAlert("Nenhuma calibração salva encontrada.", false);
      });
    }

    window.onload = function () {
      reloadSaved(true);
      loadCounts();
      loadGameState(false);
      setInterval(() => loadGameState(true), 1500);
    };
  </script>
</body>
</html>
"""


class WebCalibratorApp:
    def __init__(
        self,
        device_index: int = cfg.CAMERA_DEVICE_INDEX,
        static_image_path: Optional[str] = None,
        output_file: str = cfg.CALIBRATION_FILE,
    ) -> None:
        self.device_index = device_index
        self.output_file = Path(output_file)
        self.lock = threading.Lock()
        self.last_frame: Optional[np.ndarray] = None
        self.camera: Optional[CameraController] = None

        self.active_corners: list[list[float]] = []
        self.active_orientation: str = "normal"
        self.calibrated_resolution: list[int] = [640, 480]

        # Estado logico do tabuleiro para testes/simulacao de MAGNUS VALIDATE e Pos-Gantry
        self.logical_board: chess.Board = chess.Board()
        self.last_move_uci: Optional[str] = None
        self._last_snapshot_ts: Optional[str] = None
        self._cached_detector: Optional[PieceDetector] = None
        self._cached_detector_key: Optional[tuple] = None
        snap = load_logical_board_snapshot()
        if snap and snap.get("fen"):
            try:
                self.logical_board = chess.Board(snap["fen"])
                self.last_move_uci = snap.get("last_move_uci")
                self._last_snapshot_ts = snap.get("updated_at")
            except Exception:
                self.logical_board = chess.Board()

        if self.output_file.is_file():
            try:
                cal = CalibrationData.load(self.output_file)
                if cal.is_valid():
                    self.active_corners = cal.board_corners
                    self.active_orientation = cal.board_orientation
                    if cal.capture_resolution and len(cal.capture_resolution) == 2:
                        self.calibrated_resolution = list(cal.capture_resolution)
            except Exception:
                pass

        if static_image_path and Path(static_image_path).is_file():
            img = cv2.imread(static_image_path)
            if img is not None:
                self.last_frame = img
                log.info("Usando imagem estática: %s", static_image_path)

        if self.last_frame is None:
            self._init_camera()

    # ── camera ──────────────────────────────────────────────────────────────

    def _init_camera(self) -> None:
        try:
            self.camera = CameraController(
                device_index=self.device_index,
                allow_web_bridge=False,
            )
            self.camera.open()
            self.device_index = self.camera._device_index
            log.info("Câmera aberta no dispositivo %d", self.device_index)
        except Exception as e:
            log.warning("Não foi possível abrir a câmera (%s). Usando referência salva em modo simulado.", e)
            self.camera = None
            self.last_frame = self._generate_synthetic_board()

    def reconnect_camera(self, device_index: Optional[int] = None) -> dict:
        with self.lock:
            if self.camera is not None:
                self.camera.close()
                self.camera = None
            if device_index is not None:
                self.device_index = int(device_index)
            self.last_frame = None
        self._init_camera()
        return {
            "success": True,
            "camera_connected": self.has_real_camera(),
            "device_index": self.device_index,
            "frame_size": self.frame_size(),
        }

    def has_real_camera(self) -> bool:
        return self.camera is not None and self.camera.is_ready

    def _generate_synthetic_board(self) -> np.ndarray:
        """
        Gera um quadro simulado na MESMA resolucao da calibracao salva (ex: 640x480)
        e projeta a imagem real de referencia (reference_empty.png) exatamente dentro
        dos 4 cantos calibrados para que a grade 8x8 fique 100% alinhada mesmo sem camera USB.
        """
        w, h = self.calibrated_resolution if len(self.calibrated_resolution) == 2 else (640, 480)
        img = np.full((h, w, 3), 32, dtype=np.uint8)

        corners = self.active_corners if len(self.active_corners) == 4 else [
            [int(w * 0.24), int(h * 0.06)],
            [int(w * 0.90), int(h * 0.06)],
            [int(w * 0.90), int(h * 0.94)],
            [int(w * 0.24), int(h * 0.94)],
        ]
        dst_pts = np.array(corners, dtype=np.float32)

        if REF_PATH.is_file():
            ref = cv2.imread(str(REF_PATH))
            if ref is not None and ref.shape[1] >= CEM_W + BOARD_SZ:
                src_pts = np.array([
                    [CEM_W,            0],
                    [CEM_W + BOARD_SZ, 0],
                    [CEM_W + BOARD_SZ, BOARD_SZ],
                    [CEM_W,            BOARD_SZ],
                ], dtype=np.float32)
                M_inv = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped_back = cv2.warpPerspective(ref, M_inv, (w, h))
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
                img[mask > 0] = warped_back[mask > 0]
                return img

        # Fallback caso nao exista reference_empty.png
        board_tex = np.zeros((BOARD_SZ, BOARD_SZ, 3), dtype=np.uint8)
        sq = BOARD_SZ // 8
        for r in range(8):
            for c in range(8):
                color = (200, 200, 200) if (r + c) % 2 == 0 else (60, 60, 60)
                cv2.rectangle(board_tex, (c * sq, r * sq), ((c + 1) * sq, (r + 1) * sq), color, -1)
        src_pts = np.array([[0, 0], [BOARD_SZ, 0], [BOARD_SZ, BOARD_SZ], [0, BOARD_SZ]], dtype=np.float32)
        M_inv = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_back = cv2.warpPerspective(board_tex, M_inv, (w, h))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
        img[mask > 0] = warped_back[mask > 0]
        return img

    def get_frame(self, fresh: bool = False) -> np.ndarray:
        """
        Retorna um frame da câmera. Quando ``fresh=True`` (usado em validações,
        detecção e snapshots), drena os frames antigos do buffer do driver USB
        (V4L2/DirectShow) antes da leitura para garantir a imagem em tempo real.
        """
        with self.lock:
            if self.camera is not None and self.camera._cap is not None:
                if fresh:
                    for _ in range(4):
                        self.camera._cap.grab()
                ret, frame = self.camera._cap.read()
                if ret and frame is not None:
                    self.last_frame = frame
            if self.last_frame is None:
                self.last_frame = self._generate_synthetic_board()
            return self.last_frame.copy()

    def frame_size(self) -> list[int]:
        with self.lock:
            f = self.last_frame
        if f is None:
            f = self.get_frame()
        return [int(f.shape[1]), int(f.shape[0])]

    # ── calibracao ──────────────────────────────────────────────────────────

    def _saved_calibration(self) -> Optional[CalibrationData]:
        if not self.output_file.is_file():
            return None
        cal = CalibrationData.load(self.output_file)
        return cal if cal.is_valid() else None

    def _warp(self, frame: np.ndarray, corners, orientation: str) -> np.ndarray:
        cal = CalibrationData(board_corners=corners, board_orientation=orientation,
                              capture_resolution=[frame.shape[1], frame.shape[0]])
        return PerspectiveCalibrator(cal).correct_perspective(frame)

    def compute_live_warped(self) -> np.ndarray:
        frame = self.get_frame()
        with self.lock:
            corners = list(self.active_corners)
            orientation = self.active_orientation
        if len(corners) == 4:
            warped = self._warp(frame, corners, orientation)
            return BoardSegmenter(board_orientation=orientation).draw_grid(warped)
        display = cv2.resize(frame, (TOTAL_W, TOTAL_H))
        cv2.putText(display, "Selecione os 4 cantos", (TOTAL_W // 2 - 180, TOTAL_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return display

    def save_calibration(self, corners: list[list[float]], orientation: str = "normal") -> bool:
        frame = self.get_frame()
        with self.lock:
            self.active_corners = corners
            self.active_orientation = orientation
        cal = CalibrationData(board_corners=corners, board_orientation=orientation,
                              camera_device=self.device_index,
                              capture_resolution=[frame.shape[1], frame.shape[0]])
        cal.save(self.output_file)
        return True

    def save_reference_empty(self) -> dict:
        cal = self._saved_calibration()
        if cal is None:
            return {"success": False, "error": "Salve a calibração antes de gravar a referência."}
        warped = PerspectiveCalibrator(cal).correct_perspective(self.get_frame(fresh=True))
        cv2.imwrite(str(REF_PATH), warped)
        self._invalidate_detector()
        log.info("Referência de tabuleiro vazio gravada em %s", REF_PATH)
        return {"success": True}

    # ── deteccao ────────────────────────────────────────────────────────────

    def _invalidate_detector(self) -> None:
        self._cached_detector = None
        self._cached_detector_key = None

    def _get_detector(self, threshold: float, orientation: str) -> PieceDetector:
        ref_mtime = REF_PATH.stat().st_mtime_ns if REF_PATH.is_file() else 0
        key = (round(float(threshold), 4), orientation, ref_mtime)
        if self._cached_detector is None or self._cached_detector_key != key:
            ref_img = cv2.imread(str(REF_PATH)) if REF_PATH.is_file() else None
            self._cached_detector = PieceDetector(
                reference_image=ref_img,
                occupied_diff_threshold=threshold,
                board_orientation=orientation,
            )
            self._cached_detector_key = key
        return self._cached_detector

    def _sync_logical_board_from_disk(self) -> None:
        snap = load_logical_board_snapshot()
        if not snap or not snap.get("fen"):
            return
        ts = snap.get("updated_at")
        if ts and ts != self._last_snapshot_ts:
            try:
                self.logical_board = chess.Board(snap["fen"])
                self.last_move_uci = snap.get("last_move_uci")
                self._last_snapshot_ts = ts
            except Exception:
                pass

    def _save_logical_snapshot(self, note: str) -> None:
        save_logical_board_snapshot(self.logical_board, self.last_move_uci, note)
        snap = load_logical_board_snapshot()
        if snap:
            self._last_snapshot_ts = snap.get("updated_at")

    def _capture_board_state(
        self,
        threshold: float = cfg.OCCUPIED_DIFF_THRESHOLD,
        prior_mode: str = "auto",
        unbiased_squares: Optional[set[str] | list[str] | tuple[str, ...]] = None,
    ) -> tuple[Optional[np.ndarray], Optional[BoardState], Optional[str], Optional[str]]:
        """Captura o frame atual (drenando buffer antigo), corrige a perspectiva e retorna (warped, BoardState, orientation, error)."""
        self._sync_logical_board_from_disk()
        cal = self._saved_calibration()
        if cal is None:
            return None, None, None, "Falta salvar a calibração da câmera."
        warped = PerspectiveCalibrator(cal).correct_perspective(self.get_frame(fresh=True))
        orientation = cal.board_orientation
        detector = self._get_detector(threshold, orientation)
        segmenter = BoardSegmenter(board_orientation=orientation)
        squares = segmenter.segment_board(warped)
        prior_b = None if prior_mode == "cap_only" else self.logical_board
        board_analyses = detector.analyze_all(
            squares,
            prior_board=prior_b,
            prior_mode=prior_mode,
            unbiased_squares=unbiased_squares,
        )

        board_dict = {
            sq: SquareState(
                occupied=a.occupied,
                color=a.color,
                piece=a.piece,
                confidence=a.confidence,
            )
            for sq, a in board_analyses.items()
        }
        all_confs = [a.confidence for a in board_analyses.values()]
        overall = round(float(np.mean(all_confs)), 4) if all_confs else 1.0
        state = BoardState(
            board=board_dict,
            cemeteries={"white": [], "black": []},
            timestamp=time.monotonic(),
            overall_confidence=overall,
        )
        return warped, state, orientation, None

    def detect_occupied(self, threshold: float = cfg.OCCUPIED_DIFF_THRESHOLD) -> dict:
        self._sync_logical_board_from_disk()
        cal = self._saved_calibration()
        if cal is None:
            return {"success": False, "error": "Falta salvar a calibração."}

        warped = PerspectiveCalibrator(cal).correct_perspective(self.get_frame(fresh=True))
        orientation = cal.board_orientation
        detector = self._get_detector(threshold, orientation)
        segmenter = BoardSegmenter(board_orientation=orientation)
        squares = segmenter.segment_board(warped)
        annotated = segmenter.draw_grid(warped)

        clf = detector._classifier
        mode = "off" if clf is None else ("samples" if clf.has_samples else "icons")

        board_analyses = detector.analyze_all(
            squares,
            prior_board=self.logical_board,
            prior_mode="auto",
        )
        found, confidences = [], []
        for name, a in board_analyses.items():
            confidences.append(a.confidence)
            if not a.occupied:
                continue
            found.append({"square": name, "color": a.color, "piece": a.piece})
            x0, y0 = square_display_pos(name, orientation)
            cx, cy = getattr(a, "center", (40, 40))
            if orientation == "flipped":
                cx, cy = SQ - cx, SQ - cy
            box_color = (0, 255, 0) if a.piece else (0, 200, 255)
            cv2.rectangle(annotated, (x0 + 3, y0 + 3), (x0 + SQ - 3, y0 + SQ - 3), box_color, 3)
            cv2.circle(annotated, (x0 + cx, y0 + cy), 17, (255, 220, 0), 1)
            cv2.circle(annotated, (x0 + cx, y0 + cy), 2, (0, 255, 255), -1)
            tag = (a.color or "?")[0].upper() + (TYPE_LETTER.get(a.piece, "?"))
            cv2.rectangle(annotated, (x0 + 4, y0 + 4), (x0 + 38, y0 + 26), (0, 0, 0), -1)
            cv2.putText(annotated, tag, (x0 + 7, y0 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return {
            "success": True,
            "squares": found,
            "confidence": round(float(np.mean(confidences)) if confidences else 1.0, 3),
            "classifier": mode,
            "annotated_image": base64.b64encode(buf).decode("utf-8") if ok else None,
        }

    # ── modo de jogo, tabuleiro logico e validacoes da camera ───────────────

    def serialize_logical_board(self) -> dict:
        squares = {}
        for sq_idx in chess.SQUARES:
            piece = self.logical_board.piece_at(sq_idx)
            if piece is not None:
                name = chess.square_name(sq_idx).upper()
                squares[name] = {
                    "symbol": piece.symbol(),
                    "color": "white" if piece.color == chess.WHITE else "black",
                }
        return {
            "fen": self.logical_board.fen(),
            "turn": "white" if self.logical_board.turn == chess.WHITE else "black",
            "squares": squares,
            "last_move_uci": self.last_move_uci,
            "is_check": self.logical_board.is_check(),
            "is_game_over": self.logical_board.is_game_over(),
        }

    def get_game_state(self) -> dict:
        # Sincroniza caso o jogo principal (main.py) tenha atualizado o estado logico
        self._sync_logical_board_from_disk()
        settings = GameSettings.load()
        return {
            "settings": settings.to_dict(),
            "board": self.serialize_logical_board(),
            "history": load_validation_history(),
        }

    def save_game_settings(self, data: dict) -> dict:
        try:
            current = GameSettings.load()
            settings = GameSettings(
                game_mode=data.get("game_mode", current.game_mode),
                level=int(data.get("level", current.level)),
                white_control=data.get("white_control", current.white_control),
                black_control=data.get("black_control", current.black_control),
                verify_gantry_moves=bool(data.get("verify_gantry_moves", current.verify_gantry_moves)),
                diff_threshold=float(data.get("diff_threshold", current.diff_threshold)),
            ).normalize()
            settings.save()
            return {"success": True, "settings": settings.to_dict()}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def manipulate_logical_board(self, data: dict) -> dict:
        action = data.get("action", "")
        if action == "reset":
            self.logical_board = chess.Board()
            self.last_move_uci = None
            self._save_logical_snapshot("Reset via Web UI")
            return {"success": True, "board": self.serialize_logical_board(), "message": "Tabuleiro lógico reiniciado na posição inicial."}

        if action == "undo":
            if self.logical_board.move_stack:
                self.logical_board.pop()
                self.last_move_uci = (
                    self.logical_board.peek().uci() if self.logical_board.move_stack else None
                )
                self._save_logical_snapshot("Undo via Web UI")
                return {"success": True, "board": self.serialize_logical_board(), "message": "Último lance desfeito."}
            return {"success": False, "error": "Nenhum lance para desfazer.", "board": self.serialize_logical_board()}

        if action == "push":
            uci = (data.get("uci") or "").strip().lower()
            try:
                move = chess.Move.from_uci(uci)
                if move not in self.logical_board.legal_moves and len(uci) == 4:
                    promo = chess.Move.from_uci(uci + "q")
                    if promo in self.logical_board.legal_moves:
                        move = promo
                if move in self.logical_board.legal_moves:
                    self.logical_board.push(move)
                    self.last_move_uci = move.uci()
                    self._save_logical_snapshot(f"Move {move.uci()}")
                    return {"success": True, "board": self.serialize_logical_board()}
                # Caso o desenvolvedor esteja montando uma posicao livre no tabuleiro logico:
                piece = self.logical_board.remove_piece_at(move.from_square)
                if piece is not None:
                    self.logical_board.set_piece_at(move.to_square, piece)
                    self.logical_board.turn = not self.logical_board.turn
                    self.last_move_uci = move.uci()
                    self._save_logical_snapshot(f"Free move {move.uci()}")
                    return {"success": True, "board": self.serialize_logical_board()}
                return {"success": False, "error": "Casa de origem vazia.", "board": self.serialize_logical_board()}
            except Exception as exc:
                return {"success": False, "error": str(exc), "board": self.serialize_logical_board()}

        if action == "sync_from_camera":
            thresh = float(data.get("threshold", cfg.OCCUPIED_DIFF_THRESHOLD))
            _, state, _, err = self._capture_board_state(thresh, prior_mode="cap_only")
            if err or state is None:
                return {"success": False, "error": err or "Falha na captura."}
            new_board = chess.Board(None)
            count = 0
            for sq_name, sq_st in state.board.items():
                if not sq_st.occupied:
                    continue
                c_color = chess.WHITE if sq_st.color != "black" else chess.BLACK
                c_type = NAME_TO_CHESS_PIECE.get(sq_st.piece or "pawn", chess.PAWN)
                new_board.set_piece_at(chess.parse_square(sq_name.lower()), chess.Piece(c_type, c_color))
                count += 1
            new_board.turn = chess.WHITE
            self.logical_board = new_board
            self.last_move_uci = None
            self._save_logical_snapshot("Synced from camera")
            return {
                "success": True,
                "board": self.serialize_logical_board(),
                "message": f"Tabuleiro lógico sincronizado com a câmera ({count} peças detectadas). Agora mova uma peça física e clique em Simular MAGNUS VALIDATE!",
            }

        return {"success": False, "error": f"Ação desconhecida: {action}"}

    def simulate_magnus_validate(
        self, threshold: float = cfg.OCCUPIED_DIFF_THRESHOLD, auto_apply: bool = True
    ) -> dict:
        warped, state, orientation, err = self._capture_board_state(
            threshold, prior_mode="pre_move"
        )
        if err or warped is None or state is None:
            return {"success": False, "error": err or "Falha ao capturar imagem."}

        res = detect_and_validate_manual_move(self.logical_board, state)
        annotated = annotate_validation_image(
            warped, state, orientation=orientation or "normal", manual_result=res
        )
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf).decode("utf-8") if ok else None

        if res.valid and auto_apply and res.move is not None:
            self.logical_board.push(res.move)
            self.last_move_uci = res.move.uci()
            self._save_logical_snapshot(f"Validated {res.san}")

        event = {
            "type": "manual_validate",
            **res.to_dict(),
            "annotated_image": b64,
        }
        record_validation_event(event)
        return {
            **res.to_dict(),
            "success": True,
            "annotated_image": b64,
            "board": self.serialize_logical_board(),
            "history": load_validation_history(),
        }

    def simulate_verify_position(
        self, threshold: float = cfg.OCCUPIED_DIFF_THRESHOLD
    ) -> dict:
        self._sync_logical_board_from_disk()
        last_move = chess.Move.from_uci(self.last_move_uci) if self.last_move_uci else None
        unbiased = None
        if last_move is not None:
            unbiased = {
                chess.square_name(last_move.from_square).upper(),
                chess.square_name(last_move.to_square).upper(),
            }
        warped, state, orientation, err = self._capture_board_state(
            threshold, prior_mode="post_move", unbiased_squares=unbiased
        )
        res = verify_board_position(self.logical_board, state, last_move=last_move)
        annotated = annotate_validation_image(
            warped, state, orientation=orientation or "normal", verify_result=res
        )
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf).decode("utf-8") if ok else None

        event = {
            "type": "gantry_verify",
            **res.to_dict(),
            "valid": res.success,
            "annotated_image": b64,
        }
        record_validation_event(event)
        return {
            **res.to_dict(),
            "success": True,
            "valid": res.success,
            "annotated_image": b64,
            "board": self.serialize_logical_board(),
            "history": load_validation_history(),
        }

    # ── treino ──────────────────────────────────────────────────────────────

    def capture_sample(self, square: str, ptype: str, color: str) -> dict:
        square = (square or "").upper()
        if ptype not in PIECE_TYPES or color not in ("white", "black"):
            return {"success": False, "error": "Tipo ou cor inválidos."}
        if len(square) != 2 or square[0] not in COLS or square[1] not in ROWS:
            return {"success": False, "error": "Casa inválida."}
        cal = self._saved_calibration()
        if cal is None:
            return {"success": False, "error": "Salve a calibração antes de capturar amostras."}

        warped = PerspectiveCalibrator(cal).correct_perspective(self.get_frame(fresh=True))
        roi = BoardSegmenter(board_orientation=cal.board_orientation).segment_board(warped)[square]

        folder = SAMPLES_DIR / f"{ptype}_{color}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{square}_{int(time.time() * 1000)}.png"
        cv2.imwrite(str(path), roi)
        self._invalidate_detector()
        count = len(list(folder.glob("*.png")))
        log.info("Amostra salva: %s (total %d)", path, count)
        return {"success": True, "count": count}

    def clear_samples(self, ptype: str, color: str) -> dict:
        folder = SAMPLES_DIR / f"{ptype}_{color}"
        if ptype in PIECE_TYPES and color in ("white", "black") and folder.is_dir():
            shutil.rmtree(folder)
        self._invalidate_detector()
        return {"success": True}

    @staticmethod
    def sample_counts() -> dict:
        counts = {}
        if SAMPLES_DIR.is_dir():
            for sub in SAMPLES_DIR.iterdir():
                if sub.is_dir():
                    counts[sub.name] = len(list(sub.glob("*.png"))) + len(list(sub.glob("*.jpg")))
        return counts


def make_handler(app: WebCalibratorApp):
    class RequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _json(self, obj, status: int = 200) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _jpeg(self, img: np.ndarray, quality: int = 80) -> None:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                self.send_error(500, "Erro ao codificar imagem")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(buf.tobytes())

        def _mjpeg(self, producer, delay: float, quality: int) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    ok, buf = cv2.imencode(".jpg", producer(), [cv2.IMWRITE_JPEG_QUALITY, quality])
                    if ok:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(buf)}\r\n\r\n".encode())
                        self.wfile.write(buf.tobytes())
                        self.wfile.write(b"\r\n")
                    time.sleep(delay)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                data = HTML_TEMPLATE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/snapshot":
                self._jpeg(app.get_frame(fresh=True))
            elif path == "/api/raw_camera":
                if app.has_real_camera():
                    self._jpeg(app.get_frame(fresh=True), 85)
                else:
                    self.send_error(503, "No physical camera connected")
            elif path == "/api/snapshot_warped":
                self._jpeg(app.compute_live_warped(), 85)
            elif path == "/api/status":
                cal = app._saved_calibration()
                self._json({
                    "calibration": cal.to_dict() if cal else None,
                    "frame_size": app.frame_size(),
                    "camera_connected": app.has_real_camera(),
                    "device_index": app.device_index,
                    "layout": {"cem_w": CEM_W, "board_sz": BOARD_SZ,
                               "total_w": TOTAL_W, "total_h": TOTAL_H},
                })
            elif path == "/api/samples":
                self._json(app.sample_counts())
            elif path == "/api/game_state":
                self._json(app.get_game_state())
            elif path == "/stream":
                self._mjpeg(app.get_frame, 0.05, 70)
            elif path == "/stream_warped":
                self._mjpeg(app.compute_live_warped, 0.08, 75)
            else:
                self.send_error(404, "Not Found")

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            except Exception:
                data = {}

            if path == "/api/set_corners":
                with app.lock:
                    app.active_corners = data.get("corners", [])
                    app.active_orientation = data.get("orientation", "normal")
                self._json({"success": True})
            elif path == "/api/save":
                corners = data.get("corners", [])
                if len(corners) != 4:
                    self._json({"success": False, "error": "Requer 4 cantos"}, 400)
                else:
                    self._json({"success": app.save_calibration(corners, data.get("orientation", "normal"))})
            elif path == "/api/save_reference":
                self._json(app.save_reference_empty())
            elif path == "/api/detect":
                self._json(app.detect_occupied(float(data.get("threshold", cfg.OCCUPIED_DIFF_THRESHOLD))))
            elif path == "/api/capture_sample":
                self._json(app.capture_sample(data.get("square"), data.get("type"), data.get("color")))
            elif path == "/api/clear_samples":
                self._json(app.clear_samples(data.get("type"), data.get("color")))
            elif path == "/api/game_settings":
                self._json(app.save_game_settings(data))
            elif path == "/api/logical_board":
                self._json(app.manipulate_logical_board(data))
            elif path == "/api/validate_manual":
                self._json(app.simulate_magnus_validate(
                    threshold=float(data.get("threshold", cfg.OCCUPIED_DIFF_THRESHOLD)),
                    auto_apply=bool(data.get("auto_apply", True)),
                ))
            elif path == "/api/verify_position":
                self._json(app.simulate_verify_position(
                    threshold=float(data.get("threshold", cfg.OCCUPIED_DIFF_THRESHOLD)),
                ))
            elif path == "/api/clear_history":
                clear_validation_history()
                self._json({"success": True})
            elif path == "/api/reconnect_camera":
                self._json(app.reconnect_camera(data.get("device_index")))
            else:
                self.send_error(404, "Not Found")

    return RequestHandler


def get_ip_addresses() -> list[str]:
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    ips.append("127.0.0.1")
    return list(dict.fromkeys(ips))


def main():
    parser = argparse.ArgumentParser(description="VoiceChess Web Calibrator & Validation Lab")
    parser.add_argument("--port", "-p", type=int, default=5000, help="Porta HTTP (padrao: 5000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host bind (padrao: 0.0.0.0)")
    parser.add_argument("--device", "-d", type=int, default=cfg.CAMERA_DEVICE_INDEX, help="Indice da camera")
    parser.add_argument("--image", "-i", default=None, help="Imagem estatica em vez da camera (opcional)")
    parser.add_argument("--output", "-o", default=cfg.CALIBRATION_FILE, help="Arquivo JSON de calibracao")
    args = parser.parse_args()

    app = WebCalibratorApp(device_index=args.device, static_image_path=args.image,
                           output_file=args.output)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    server.daemon_threads = True

    w, h = app.frame_size()
    print("\n" + "=" * 65)
    print("  VoiceChess — Calibrador Web, Modos de Jogo & Validacao")
    print("=" * 65)
    print(f"  Resolucao real da camera: {w}x{h}")
    print("  Abra no navegador do seu PC:")
    for ip in get_ip_addresses():
        print(f"     -> http://{ip}:{args.port}")
    print("\n  Ctrl+C para encerrar.")
    print("=" * 65 + "\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando servidor web...")
    finally:
        server.server_close()
        if app.camera is not None:
            app.camera.close()


if __name__ == "__main__":
    main()
