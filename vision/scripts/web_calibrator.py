#!/usr/bin/env python3
"""
vision/scripts/web_calibrator.py
================================
Calibrador web com LIVE VIEW para Raspberry Pi headless (acesso via navegador).

Funcoes:
  - Video ao vivo da camera com os 4 cantos arrastaveis por cima
  - Video ao vivo do tabuleiro corrigido (grade 8x8)
  - Salvar camera_calibration.json e reference_empty.png
  - Testar ocupacao (casa ocupada, cor e TIPO da peca)
  - Treinar pecas: capturar amostras reais de cada peca direto da camera
    (salvas em vision/pieces/samples/<tipo>_<cor>/)

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

import cv2
import numpy as np

import vision.config as cfg
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
  <title>VoiceChess — Calibrador & Live View</title>
  <style>
    :root {
      --bg: #0f172a; --card-bg: #1e293b; --card-border: #334155;
      --accent: #38bdf8; --accent-hover: #0284c7; --success: #22c55e;
      --text: #f8fafc; --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 1.25rem; }
    header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.25rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--card-border); }
    h1 { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
    h2 { font-size: 0.95rem; font-weight: 600; margin: 0.4rem 0 0.2rem; color: var(--accent); }
    .status-badge { background: #065f46; color: #34d399; font-size: 0.8rem; padding: 0.25rem 0.6rem; border-radius: 9999px; font-weight: 600; display: flex; align-items: center; gap: 0.4rem; }
    .live-dot { width: 8px; height: 8px; background: #ef4444; border-radius: 50%; animation: pulse 1.5s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }
    @media (max-width: 1024px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 0.75rem; padding: 1.25rem; display: flex; flex-direction: column; }
    .card-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 0.75rem; display: flex; justify-content: space-between; align-items: center; }
    /* A proporcao do quadro e ajustada via JS para a resolucao REAL da camera
       (sem barras pretas), assim o clique cai exatamente no pixel certo. */
    .view-wrapper { position: relative; width: 100%; border-radius: 0.5rem; overflow: hidden; background: #000; aspect-ratio: 4/3; }
    .preview-wrapper { position: relative; width: 100%; border-radius: 0.5rem; overflow: hidden; background: #000; aspect-ratio: 1174/640; cursor: pointer; }
    .stream-img { width: 100%; height: 100%; object-fit: fill; display: block; }
    #overlayCanvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair; }
    .controls { margin-top: 1rem; display: flex; flex-direction: column; gap: 0.8rem; }
    .point-list { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.4rem; }
    .point-row { display: flex; align-items: center; gap: 0.4rem; font-size: 0.8rem; background: #0f172a80; padding: 0.35rem 0.5rem; border-radius: 0.375rem; }
    .point-color { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
    .point-input { width: 55px; background: #0f172a; border: 1px solid var(--card-border); color: #fff; padding: 0.2rem 0.3rem; border-radius: 0.25rem; text-align: center; font-size: 0.75rem; }
    select.point-input { width: auto; padding: 0.25rem 0.4rem; }
    .btn-group { display: flex; gap: 0.6rem; flex-wrap: wrap; align-items: center; }
    button { background: var(--accent); color: #000; font-weight: 600; border: none; padding: 0.55rem 1rem; border-radius: 0.45rem; cursor: pointer; transition: 0.15s ease; font-size: 0.85rem; }
    button:hover { background: var(--accent-hover); color: #fff; }
    button.btn-success { background: var(--success); color: #000; }
    button.btn-success:hover { background: #16a34a; color: #fff; }
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
  </style>
</head>
<body>
  <header>
    <div>
      <h1>VoiceChess — Calibrador & Live View</h1>
      <p style="color: var(--text-muted); font-size: 0.8rem;">Vídeo ao vivo, calibração da perspectiva e treino das peças</p>
    </div>
    <span class="status-badge"><span class="live-dot"></span> <span id="modeText">Ao Vivo</span> <span id="resText" style="color:#94a3b8;"></span></span>
  </header>

  <div class="grid">
    <!-- COLUNA 1: CAMERA AO VIVO + CANTOS -->
    <div class="card">
      <div class="card-title">
        <span>Câmera (arraste os 4 cantos)</span>
        <div style="display: flex; gap: 0.4rem;">
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
        <span>Tabuleiro Corrigido (clique numa casa para selecioná-la)</span>
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
    // Resolucao REAL do frame (vem do servidor em /api/status)
    let naturalW = 640, naturalH = 480;
    let layout = { cem_w: 267, board_sz: 640, total_w: 1174, total_h: 640 };
    const COLS = "ABCDEFGH", ROWS = "87654321";
    const NAMES = { pawn: "Peão", rook: "Torre", knight: "Cavalo", bishop: "Bispo", queen: "Rainha", king: "Rei" };

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

    // ── Overlay dos cantos (sempre do mesmo tamanho da imagem exibida) ──
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
        // mira fina para precisao
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
        const list = res.squares.length
          ? res.squares.map(s => `${s.square}: ${s.color === "white" ? "branca" : "preta"} ${s.piece ? NAMES[s.piece] : "<span style='color:#fbbf24'>?</span>"}`).join(" · ")
          : "<strong style='color:#34d399;'>Nenhuma peça (tabuleiro vazio)</strong>";
        const hint = res.classifier === "samples" ? "amostras reais"
                   : res.classifier === "icons" ? "ícones genéricos (treine as peças para melhorar!)" : "desligado";
        let handNotice = "";
        if (res.hands && res.hands.length) {
          handNotice = `<br><span style='color:#f87171;'>⚠️ <strong>Mão/Obstrução detectada</strong> nas casas: ${res.hands.join(", ")} (desconsideradas)</span>`;
        }
        info.innerHTML = `<strong>Peças (${res.squares.length}):</strong> ${list}${handNotice}<br>
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

    function reloadSaved(silent) {
      fetch("/api/status").then(r => r.json()).then(data => {
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

    window.onload = function () { reloadSaved(true); loadCounts(); };
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

        if self.output_file.is_file():
            try:
                cal = CalibrationData.load(self.output_file)
                if cal.is_valid():
                    self.active_corners = cal.board_corners
                    self.active_orientation = cal.board_orientation
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
            self.camera = CameraController(device_index=self.device_index)
            self.camera.open()
            log.info("Câmera aberta no dispositivo %d", self.device_index)
        except Exception as e:
            log.warning("Não foi possível abrir a câmera (%s). Usando frame simulado.", e)
            self.camera = None
            self.last_frame = self._generate_synthetic_board()

    @staticmethod
    def _generate_synthetic_board() -> np.ndarray:
        w, h = cfg.CAMERA_RESOLUTION
        img = np.full((h, w, 3), 40, dtype=np.uint8)
        cv2.putText(img, "Sem Camera Real - Frame Simulado", (w // 2 - 250, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        bx, by, bw = w // 2 - 250, h // 2 - 250, 500
        sz = bw // 8
        for r in range(8):
            for c in range(8):
                color = (200, 200, 200) if (r + c) % 2 == 0 else (60, 60, 60)
                cv2.rectangle(img, (bx + c * sz, by + r * sz),
                              (bx + (c + 1) * sz, by + (r + 1) * sz), color, -1)
        return img

    def get_frame(self) -> np.ndarray:
        with self.lock:
            if self.camera is not None and self.camera._cap is not None:
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
        warped = PerspectiveCalibrator(cal).correct_perspective(self.get_frame())
        cv2.imwrite(str(REF_PATH), warped)
        log.info("Referência de tabuleiro vazio gravada em %s", REF_PATH)
        return {"success": True}

    # ── deteccao ────────────────────────────────────────────────────────────

    def detect_occupied(self, threshold: float = cfg.OCCUPIED_DIFF_THRESHOLD) -> dict:
        cal = self._saved_calibration()
        if cal is None:
            return {"success": False, "error": "Falta salvar a calibração."}
        ref_img = cv2.imread(str(REF_PATH)) if REF_PATH.is_file() else None
        if ref_img is None:
            return {"success": False,
                    "error": "Sem referência! Tire as peças e clique em 'Gravar Tabuleiro Vazio'."}

        warped = PerspectiveCalibrator(cal).correct_perspective(self.get_frame())
        orientation = cal.board_orientation
        detector = PieceDetector(reference_image=ref_img, occupied_diff_threshold=threshold,
                                 board_orientation=orientation)
        segmenter = BoardSegmenter(board_orientation=orientation)
        squares = segmenter.segment_board(warped)
        annotated = segmenter.draw_grid(warped)

        clf = detector._classifier
        mode = "off" if clf is None else ("samples" if clf.has_samples else "icons")

        analyses = detector.analyze_all(squares)
        found, confidences = [], []
        for name, a in analyses.items():
            confidences.append(a.confidence)
            if not a.occupied or not getattr(a, "is_piece", True):
                continue
            found.append({"square": name, "color": a.color, "piece": a.piece})
            x0, y0 = square_display_pos(name, orientation)
            box_color = (0, 255, 0) if a.piece else (0, 200, 255)
            cv2.rectangle(annotated, (x0 + 3, y0 + 3), (x0 + SQ - 3, y0 + SQ - 3), box_color, 3)
            tag = (a.color or "?")[0].upper() + (TYPE_LETTER.get(a.piece, "?"))
            cv2.rectangle(annotated, (x0 + 4, y0 + 4), (x0 + 38, y0 + 26), (0, 0, 0), -1)
            cv2.putText(annotated, tag, (x0 + 7, y0 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return {
            "success": True,
            "squares": found,
            "hands": [],
            "confidence": round(float(np.mean(confidences)) if confidences else 1.0, 3),
            "classifier": mode,
            "annotated_image": base64.b64encode(buf).decode("utf-8") if ok else None,
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

        warped = PerspectiveCalibrator(cal).correct_perspective(self.get_frame())
        roi = BoardSegmenter(board_orientation=cal.board_orientation).segment_board(warped)[square]

        folder = SAMPLES_DIR / f"{ptype}_{color}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{square}_{int(time.time() * 1000)}.png"
        cv2.imwrite(str(path), roi)
        count = len(list(folder.glob("*.png")))
        log.info("Amostra salva: %s (total %d)", path, count)
        return {"success": True, "count": count}

    @staticmethod
    def clear_samples(ptype: str, color: str) -> dict:
        folder = SAMPLES_DIR / f"{ptype}_{color}"
        if ptype in PIECE_TYPES and color in ("white", "black") and folder.is_dir():
            shutil.rmtree(folder)
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
                self._jpeg(app.get_frame())
            elif path == "/api/snapshot_warped":
                self._jpeg(app.compute_live_warped(), 85)
            elif path == "/api/status":
                cal = app._saved_calibration()
                self._json({
                    "calibration": cal.to_dict() if cal else None,
                    "frame_size": app.frame_size(),
                    "layout": {"cem_w": CEM_W, "board_sz": BOARD_SZ,
                               "total_w": TOTAL_W, "total_h": TOTAL_H},
                })
            elif path == "/api/samples":
                self._json(app.sample_counts())
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
    parser = argparse.ArgumentParser(description="VoiceChess Web Calibrator")
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
    print("  VoiceChess — Calibrador Web (LIVE VIEW + treino de pecas)")
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
