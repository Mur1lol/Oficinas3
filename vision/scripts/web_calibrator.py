#!/usr/bin/env python3
"""
vision/scripts/web_calibrator.py
================================
Web-based Camera & Board Calibration Tool with LIVE VIEW for Headless Raspberry Pi.

Provides a responsive web interface to:
  - View Live Camera Stream (MJPEG) with real-time draggable corner overlay
  - View Live Warped Board Preview (perspective corrected with 8x8 grid)
  - Freeze / Unfreeze frame (Live / Snapshot toggle)
  - Fine-tune corners with pixel inputs
  - Adjust detection sensitivity threshold live
  - Run visual piece occupancy detection with highlighted squares
  - Save camera_calibration.json and reference_empty.png directly on the Pi

Usage:
------
    python vision/scripts/web_calibrator.py
    python vision/scripts/web_calibrator.py --port 5000 --device 0
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
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
from vision.piece_detector import PieceDetector
from vision.segmentation import BoardSegmenter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("web_calibrator")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VoiceChess — Calibrador & Live View</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --success: #22c55e;
      --danger: #ef4444;
      --text: #f8fafc;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 1.25rem; }
    header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.25rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--card-border); }
    h1 { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
    .status-badge { background: #065f46; color: #34d399; font-size: 0.8rem; padding: 0.25rem 0.6rem; border-radius: 9999px; font-weight: 600; display: flex; align-items: center; gap: 0.4rem; }
    .live-dot { width: 8px; height: 8px; background: #ef4444; border-radius: 50%; animation: pulse 1.5s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }
    @media (max-width: 1024px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 0.75rem; padding: 1.25rem; display: flex; flex-direction: column; }
    .card-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 0.75rem; display: flex; justify-content: space-between; align-items: center; }
    .view-wrapper { position: relative; width: 100%; border-radius: 0.5rem; overflow: hidden; background: #000; aspect-ratio: 16/9; display: flex; align-items: center; justify-content: center; }
    .preview-wrapper { position: relative; width: 100%; border-radius: 0.5rem; overflow: hidden; background: #000; aspect-ratio: 16/10; display: flex; align-items: center; justify-content: center; }
    .stream-img { width: 100%; height: 100%; object-fit: contain; display: block; }
    #overlayCanvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair; }
    .controls { margin-top: 1rem; display: flex; flex-direction: column; gap: 0.8rem; }
    .point-list { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.4rem; }
    .point-row { display: flex; align-items: center; gap: 0.4rem; font-size: 0.8rem; background: #0f172a80; padding: 0.35rem 0.5rem; border-radius: 0.375rem; }
    .point-color { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
    .point-input { width: 55px; background: #0f172a; border: 1px solid var(--card-border); color: #fff; padding: 0.2rem 0.3rem; border-radius: 0.25rem; text-align: center; font-size: 0.75rem; }
    .btn-group { display: flex; gap: 0.6rem; flex-wrap: wrap; }
    button { background: var(--accent); color: #000; font-weight: 600; border: none; padding: 0.55rem 1rem; border-radius: 0.45rem; cursor: pointer; transition: 0.15s ease; font-size: 0.85rem; }
    button:hover { background: var(--accent-hover); color: #fff; }
    button.btn-success { background: var(--success); color: #000; }
    button.btn-success:hover { background: #16a34a; color: #fff; }
    button.btn-secondary { background: #334155; color: #fff; }
    button.btn-secondary:hover { background: #475569; }
    button.btn-active { background: #0284c7; color: #fff; }
    .alert { padding: 0.6rem 0.9rem; border-radius: 0.45rem; margin-top: 0.8rem; font-size: 0.85rem; display: none; line-height: 1.4; }
    .alert-success { background: #064e3b; border: 1px solid #059669; color: #6ee7b7; }
    .alert-error { background: #7f1d1d; border: 1px solid #dc2626; color: #fca5a5; }
    .slider-box { display: flex; align-items: center; gap: 0.8rem; font-size: 0.85rem; background: #0f172a60; padding: 0.5rem 0.75rem; border-radius: 0.375rem; }
    .slider-box input[type=range] { flex: 1; }
    .instructions { font-size: 0.8rem; color: var(--text-muted); line-height: 1.4; margin-top: 0.4rem; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>VoiceChess — Calibrador & Live View</h1>
      <p style="color: var(--text-muted); font-size: 0.8rem;">Vídeo ao vivo da câmera e ajuste da perspectiva do tabuleiro</p>
    </div>
    <div style="display: flex; gap: 0.75rem; align-items: center;">
      <span class="status-badge"><span class="live-dot"></span> <span id="modeText">Ao Vivo</span></span>
    </div>
  </header>

  <div class="grid">
    <!-- COLUNA 1: VÍDEO DA CÂMERA AO VIVO COM OVERLAY DE CANTOS -->
    <div class="card">
      <div class="card-title">
        <span>Câmera (Arraste ou clique nos 4 cantos)</span>
        <div style="display: flex; gap: 0.4rem;">
          <button id="btnToggleLive" class="btn-secondary" style="padding: 0.25rem 0.6rem; font-size: 0.75rem;" onclick="toggleLiveMode()">⏸️ Congelar Frame</button>
          <button class="btn-secondary" style="padding: 0.25rem 0.6rem; font-size: 0.75rem;" onclick="refreshStream()">🔄 Recarregar</button>
        </div>
      </div>

      <div class="view-wrapper" id="viewBox">
        <img id="cameraImg" class="stream-img" src="/stream" alt="Live stream">
        <canvas id="overlayCanvas"></canvas>
      </div>

      <div class="controls">
        <div class="point-list">
          <div class="point-row">
            <span class="point-color" style="background: #38bdf8;"></span>
            <span>1. Sup. Esq (A8):</span>
            <input type="number" id="x0" class="point-input" onchange="manualCoord(0)">
            <input type="number" id="y0" class="point-input" onchange="manualCoord(0)">
          </div>
          <div class="point-row">
            <span class="point-color" style="background: #4ade80;"></span>
            <span>2. Sup. Dir (H8):</span>
            <input type="number" id="x1" class="point-input" onchange="manualCoord(1)">
            <input type="number" id="y1" class="point-input" onchange="manualCoord(1)">
          </div>
          <div class="point-row">
            <span class="point-color" style="background: #facc15;"></span>
            <span>3. Inf. Dir (H1):</span>
            <input type="number" id="x2" class="point-input" onchange="manualCoord(2)">
            <input type="number" id="y2" class="point-input" onchange="manualCoord(2)">
          </div>
          <div class="point-row">
            <span class="point-color" style="background: #fb923c;"></span>
            <span>4. Inf. Esq (A1):</span>
            <input type="number" id="x3" class="point-input" onchange="manualCoord(3)">
            <input type="number" id="y3" class="point-input" onchange="manualCoord(3)">
          </div>
        </div>

        <div class="instructions">
          Arraste os círculos numerados até os 4 cantos da grade do tabuleiro (1=A8, 2=H8, 3=H1, 4=A1).
        </div>

        <div class="btn-group">
          <button class="btn-secondary" onclick="resetPoints()">Limpar Pontos</button>
          <button class="btn-secondary" onclick="reloadSaved()">Carregar Salvo</button>
        </div>
      </div>
    </div>

    <!-- COLUNA 2: PREVIEW RETIFICADO AO VIVO E DETECÇÃO -->
    <div class="card">
      <div class="card-title">
        <span>Tabuleiro Corrigido (Live View 8x8)</span>
        <span id="warpStatus" style="font-size: 0.8rem; color: var(--text-muted);">Grade ativa</span>
      </div>

      <div class="preview-wrapper">
        <img id="previewImg" class="stream-img" src="/stream_warped" alt="Preview da calibração">
      </div>

      <div class="controls">
        <div style="display: flex; gap: 1rem; align-items: center; justify-content: space-between;">
          <div style="display: flex; align-items: center; gap: 0.5rem;">
            <label style="font-size: 0.85rem; color: var(--text-muted);">Orientação:</label>
            <select id="orientationSelect" class="point-input" style="width: auto; padding: 0.25rem 0.5rem;" onchange="syncCorners()">
              <option value="normal">Normal (A1 em baixo à esquerda)</option>
              <option value="flipped">Invertida (A1 no topo à direita)</option>
            </select>
          </div>
          <button class="btn-success" onclick="saveCalibration()">💾 Salvar Calibração</button>
        </div>

        <!-- Controle de Sensibilidade da Detecção -->
        <div class="slider-box">
          <span>Sensibilidade (Limiar MAD):</span>
          <input type="range" id="thresholdSlider" min="0.04" max="0.25" step="0.01" value="0.10" oninput="updateSliderVal(this.value)">
          <strong id="thresholdVal" style="color: var(--accent);">0.10</strong>
        </div>

        <div class="btn-group" style="margin-top: 0.4rem;">
          <button class="btn-secondary" onclick="saveReference()">📷 Gravar Tabuleiro Vazio</button>
          <button class="btn-secondary" onclick="testDetection()">🔍 Testar Ocupação</button>
        </div>

        <div id="alertBox" class="alert"></div>
        <div id="detectionInfo" style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-muted); display: none; line-height: 1.5; background: #0f172a60; padding: 0.6rem; border-radius: 0.4rem;"></div>
      </div>
    </div>
  </div>

  <script>
    const cameraImg = document.getElementById("cameraImg");
    const previewImg = document.getElementById("previewImg");
    const canvas = document.getElementById("overlayCanvas");
    const ctx = canvas.getContext("2d");

    let isLive = true;
    let corners = [];
    let activeDragIdx = -1;
    const colors = ["#38bdf8", "#4ade80", "#facc15", "#fb923c"];
    let naturalW = 1280;
    let naturalH = 720;

    function showAlert(msg, isSuccess = true) {
      const box = document.getElementById("alertBox");
      box.className = "alert " + (isSuccess ? "alert-success" : "alert-error");
      box.innerHTML = msg;
      box.style.display = "block";
      setTimeout(() => { box.style.display = "none"; }, 6000);
    }

    function updateSliderVal(v) {
      document.getElementById("thresholdVal").textContent = parseFloat(v).toFixed(2);
    }

    function toggleLiveMode() {
      const btn = document.getElementById("btnToggleLive");
      const dot = document.querySelector(".live-dot");
      const text = document.getElementById("modeText");
      if (isLive) {
        // Congela: muda src para um snapshot estático
        isLive = false;
        cameraImg.src = "/api/snapshot?t=" + Date.now();
        previewImg.src = "/api/snapshot_warped?t=" + Date.now();
        btn.textContent = "▶️ Retomar Ao Vivo";
        btn.className = "btn-secondary btn-active";
        dot.style.background = "#94a3b8";
        text.textContent = "Pausado";
      } else {
        // Retoma stream
        isLive = true;
        cameraImg.src = "/stream?t=" + Date.now();
        previewImg.src = "/stream_warped?t=" + Date.now();
        btn.textContent = "⏸️ Congelar Frame";
        btn.className = "btn-secondary";
        dot.style.background = "#ef4444";
        text.textContent = "Ao Vivo";
      }
    }

    function refreshStream() {
      if (isLive) {
        cameraImg.src = "/stream?t=" + Date.now();
        previewImg.src = "/stream_warped?t=" + Date.now();
      } else {
        cameraImg.src = "/api/snapshot?t=" + Date.now();
      }
    }

    function resizeOverlay() {
      const rect = cameraImg.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      renderOverlay();
    }

    window.addEventListener("resize", resizeOverlay);
    cameraImg.onload = function() {
      if (cameraImg.naturalWidth) {
        naturalW = cameraImg.naturalWidth;
        naturalH = cameraImg.naturalHeight;
      }
      resizeOverlay();
    };

    function renderOverlay() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (corners.length === 0) return;

      const scaleX = canvas.width / naturalW;
      const scaleY = canvas.height / naturalH;

      if (corners.length >= 2) {
        ctx.beginPath();
        ctx.moveTo(corners[0][0] * scaleX, corners[0][1] * scaleY);
        for (let i = 1; i < corners.length; i++) {
          ctx.lineTo(corners[i][0] * scaleX, corners[i][1] * scaleY);
        }
        if (corners.length === 4) {
          ctx.closePath();
          ctx.fillStyle = "rgba(56, 189, 248, 0.18)";
          ctx.fill();
        }
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = "#38bdf8";
        ctx.stroke();
      }

      corners.forEach((pt, idx) => {
        const cx = pt[0] * scaleX;
        const cy = pt[1] * scaleY;

        ctx.beginPath();
        ctx.arc(cx, cy, 9, 0, 2 * Math.PI);
        ctx.fillStyle = colors[idx];
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#ffffff";
        ctx.stroke();

        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 13px sans-serif";
        ctx.fillText((idx + 1).toString(), cx + 12, cy + 4);

        const xi = document.getElementById("x" + idx);
        const yi = document.getElementById("y" + idx);
        if (xi) xi.value = Math.round(pt[0]);
        if (yi) yi.value = Math.round(pt[1]);
      });
    }

    function getCanvasCoords(e) {
      const rect = canvas.getBoundingClientRect();
      const scaleX = naturalW / rect.width;
      const scaleY = naturalH / rect.height;
      return [
        (e.clientX - rect.left) * scaleX,
        (e.clientY - rect.top) * scaleY
      ];
    }

    canvas.addEventListener("mousedown", (e) => {
      const [cx, cy] = getCanvasCoords(e);
      // Checa se clicou perto de um ponto para arrastar
      activeDragIdx = corners.findIndex(pt => Math.hypot(pt[0] - cx, pt[1] - cy) < 40);
      if (activeDragIdx === -1) {
        if (corners.length < 4) {
          corners.push([cx, cy]);
          renderOverlay();
          if (corners.length === 4) syncCorners();
        }
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (activeDragIdx !== -1) {
        const [cx, cy] = getCanvasCoords(e);
        corners[activeDragIdx] = [
          Math.max(0, Math.min(naturalW, cx)),
          Math.max(0, Math.min(naturalH, cy))
        ];
        renderOverlay();
      }
    });

    window.addEventListener("mouseup", () => {
      if (activeDragIdx !== -1) {
        activeDragIdx = -1;
        if (corners.length === 4) syncCorners();
      }
    });

    function manualCoord(idx) {
      if (!corners[idx]) return;
      const x = parseFloat(document.getElementById("x" + idx).value);
      const y = parseFloat(document.getElementById("y" + idx).value);
      if (!isNaN(x) && !isNaN(y)) {
        corners[idx] = [x, y];
        renderOverlay();
        if (corners.length === 4) syncCorners();
      }
    }

    function resetPoints() {
      corners = [];
      for (let i = 0; i < 4; i++) {
        document.getElementById("x" + i).value = "";
        document.getElementById("y" + i).value = "";
      }
      renderOverlay();
      syncCorners();
    }

    function syncCorners() {
      const orientation = document.getElementById("orientationSelect").value;
      fetch("/api/set_corners", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ corners, orientation })
      }).then(() => {
        if (!isLive) {
          previewImg.src = "/api/snapshot_warped?t=" + Date.now();
        }
      });
    }

    function saveCalibration() {
      if (corners.length !== 4) {
        showAlert("Por favor, selecione os 4 cantos antes de salvar.", false);
        return;
      }
      const orientation = document.getElementById("orientationSelect").value;
      fetch("/api/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ corners, orientation })
      })
      .then(r => r.json())
      .then(res => {
        if (res.success) {
          showAlert("Calibração salva com sucesso em camera_calibration.json!");
        } else {
          showAlert("Erro ao salvar: " + res.error, false);
        }
      })
      .catch(err => showAlert("Falha de rede: " + err, false));
    }

    function saveReference() {
      if (corners.length !== 4) {
        showAlert("É necessário calibrar os 4 cantos antes de salvar a referência.", false);
        return;
      }
      fetch("/api/save_reference", { method: "POST" })
      .then(r => r.json())
      .then(res => {
        if (res.success) {
          showAlert("Referência do tabuleiro vazio gravada em <strong>reference_empty.png</strong>!");
        } else {
          showAlert("Erro ao gravar referência: " + res.error, false);
        }
      })
      .catch(err => showAlert("Erro: " + err, false));
    }

    function testDetection() {
      const info = document.getElementById("detectionInfo");
      const thresh = parseFloat(document.getElementById("thresholdSlider").value);
      info.style.display = "block";
      info.innerHTML = "<em>Analisando imagem em relação ao tabuleiro vazio...</em>";

      fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ threshold: thresh })
      })
      .then(r => r.json())
      .then(res => {
        if (res.success) {
          if (res.annotated_image) {
            // Mostra o frame com as caixas verdes anotadas no painel da direita
            previewImg.src = "data:image/jpeg;base64," + res.annotated_image;
          }
          const occ = res.occupied_squares.length > 0
            ? res.occupied_squares.join(", ")
            : "<strong style='color:#34d399;'>Nenhuma peça detectada (Tabuleiro 100% Vazio)</strong>";
          info.innerHTML = `
            <strong>Casas Ocupadas:</strong> ${occ}<br>
            <strong>Total:</strong> ${res.occupied_squares.length} peça(s) detectada(s)<br>
            <strong>Limiar MAD usado:</strong> ${thresh.toFixed(2)} | <strong>Confiança:</strong> ${(res.confidence * 100).toFixed(1)}%
          `;
          showAlert(res.occupied_squares.length === 0
            ? "Tabuleiro reconhecido como VAZIO com sucesso!"
            : `Detectadas ${res.occupied_squares.length} peças! Veja as marcações na imagem à direita.`);
        } else {
          info.innerHTML = "<span style='color:#fca5a5;'>Erro: " + res.error + "</span>";
          showAlert(res.error, false);
        }
      })
      .catch(err => {
        info.innerHTML = "<span style='color:#fca5a5;'>Erro de conexão: " + err + "</span>";
      });
    }

    function reloadSaved() {
      fetch("/api/status")
      .then(r => r.json())
      .then(data => {
        if (data.calibration && data.calibration.board_corners && data.calibration.board_corners.length === 4) {
          corners = data.calibration.board_corners;
          document.getElementById("orientationSelect").value = data.calibration.board_orientation || "normal";
          renderOverlay();
          syncCorners();
          showAlert("Calibração existente carregada do JSON.");
        } else {
          showAlert("Nenhuma calibração anterior encontrada.", false);
        }
      });
    }

    window.onload = function() {
      reloadSaved();
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
        self.static_image_path = static_image_path
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

        if self.static_image_path and Path(self.static_image_path).is_file():
            img = cv2.imread(self.static_image_path)
            if img is not None:
                self.last_frame = img
                log.info("Usando imagem estática: %s", self.static_image_path)

        if self.last_frame is None:
            self._init_camera()

    def _init_camera(self) -> None:
        try:
            self.camera = CameraController(device_index=self.device_index)
            self.camera.open()
            log.info("Câmera aberta no dispositivo %d", self.device_index)
        except Exception as e:
            log.warning("Não foi possível abrir a câmera (%s). Criando frame simulado.", e)
            self.last_frame = self._generate_synthetic_board()

    def _generate_synthetic_board(self) -> np.ndarray:
        w, h = cfg.CAMERA_RESOLUTION
        img = np.full((h, w, 3), 40, dtype=np.uint8)
        cv2.putText(
            img,
            "Sem Camera Real - Frame Simulado",
            (w // 2 - 250, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )
        bx, by, bw = w // 2 - 250, h // 2 - 250, 500
        for r in range(8):
            for c in range(8):
                color = (200, 200, 200) if (r + c) % 2 == 0 else (60, 60, 60)
                sz = bw // 8
                cv2.rectangle(
                    img,
                    (bx + c * sz, by + r * sz),
                    (bx + (c + 1) * sz, by + (r + 1) * sz),
                    color,
                    -1,
                )
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

    def compute_preview(
        self, corners: list[list[float]], orientation: str = "normal"
    ) -> Optional[np.ndarray]:
        frame = self.get_frame()
        if len(corners) != 4:
            return None

        cal_data = CalibrationData(
            board_corners=corners,
            board_orientation=orientation,
            capture_resolution=[frame.shape[1], frame.shape[0]],
        )
        calibrator = PerspectiveCalibrator(cal_data)
        warped = calibrator.correct_perspective(frame)
        segmenter = BoardSegmenter(board_orientation=orientation)
        annotated = segmenter.draw_grid(warped)
        return annotated

    def compute_live_warped(self) -> np.ndarray:
        frame = self.get_frame()
        with self.lock:
            corners = list(self.active_corners)
            orientation = self.active_orientation

        if len(corners) == 4:
            cal_data = CalibrationData(
                board_corners=corners,
                board_orientation=orientation,
                capture_resolution=[frame.shape[1], frame.shape[0]],
            )
            calibrator = PerspectiveCalibrator(cal_data)
            warped = calibrator.correct_perspective(frame)
            segmenter = BoardSegmenter(board_orientation=orientation)
            return segmenter.draw_grid(warped)
        else:
            # Mostra o frame original com aviso
            display = cv2.resize(frame, (640, 640))
            cv2.putText(
                display,
                "Selecione os 4 cantos",
                (50, 320),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            return display

    def save_calibration(self, corners: list[list[float]], orientation: str = "normal") -> bool:
        frame = self.get_frame()
        with self.lock:
            self.active_corners = corners
            self.active_orientation = orientation

        cal = CalibrationData(
            board_corners=corners,
            board_orientation=orientation,
            camera_device=self.device_index,
            capture_resolution=[frame.shape[1], frame.shape[0]],
        )
        cal.save(self.output_file)
        return True

    def save_reference_empty(self) -> bool:
        if not self.output_file.is_file():
            return False
        cal = CalibrationData.load(self.output_file)
        frame = self.get_frame()
        calibrator = PerspectiveCalibrator(cal)
        warped = calibrator.correct_perspective(frame)
        ref_path = Path(cfg.VISION_DIR) / "reference_empty.png"
        cv2.imwrite(str(ref_path), warped)
        log.info("Referência de tabuleiro vazio gravada em %s", ref_path)
        return True

    def detect_occupied(self, threshold: float = cfg.OCCUPIED_DIFF_THRESHOLD) -> dict:
        if not self.output_file.is_file():
            return {"success": False, "error": "Falta calibração prévia"}

        cal = CalibrationData.load(self.output_file)
        frame = self.get_frame()
        calibrator = PerspectiveCalibrator(cal)
        warped = calibrator.correct_perspective(frame)

        ref_path = Path(cfg.VISION_DIR) / "reference_empty.png"
        ref_img = cv2.imread(str(ref_path)) if ref_path.is_file() else None
        if ref_img is None:
            return {
                "success": False,
                "error": "Imagem de referência não encontrada! Clique primeiro em 'Gravar Tabuleiro Vazio' com o tabuleiro sem peças.",
            }

        detector = PieceDetector(
            reference_image=ref_img,
            occupied_diff_threshold=threshold,
            board_orientation=cal.board_orientation,
        )
        segmenter = BoardSegmenter(board_orientation=cal.board_orientation)
        squares = segmenter.segment_board(warped)

        annotated_board = segmenter.draw_grid(warped)
        occupied = []
        confidences = []

        cols = "ABCDEFGH"
        rows = "87654321"

        for name, roi in squares.items():
            analysis = detector.analyze_square(name, roi)
            if analysis.occupied:
                color_label = analysis.color or "?"
                occupied.append(f"{name} ({color_label})")

                # Desenha destaque verde na casa ocupada
                col_idx = cols.index(name[0])
                row_idx = rows.index(name[1])
                x0 = cfg.CEMETERY_REGION_WIDTH_PX + col_idx * 80
                y0 = row_idx * 80
                cv2.rectangle(
                    annotated_board,
                    (x0 + 4, y0 + 4),
                    (x0 + 76, y0 + 76),
                    (0, 255, 0),
                    3,
                )
                tag = f"{color_label[0].upper()}"
                cv2.putText(
                    annotated_board,
                    tag,
                    (x0 + 26, y0 + 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
            confidences.append(analysis.confidence)

        avg_conf = float(np.mean(confidences)) if confidences else 1.0

        # Codifica imagem anotada em base64 para exibir no browser
        ret, buf = cv2.imencode(".jpg", annotated_board, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_annotated = base64.b64encode(buf).decode("utf-8") if ret else None

        return {
            "success": True,
            "occupied_squares": occupied,
            "confidence": round(avg_conf, 3),
            "annotated_image": b64_annotated,
        }


def make_handler(app: WebCalibratorApp):
    class RequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/" or parsed.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

            elif parsed.path == "/api/snapshot":
                frame = app.get_frame()
                ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(buf.tobytes())
                else:
                    self.send_error(500, "Erro ao codificar frame")

            elif parsed.path == "/api/snapshot_warped":
                preview = app.compute_live_warped()
                ret, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(buf.tobytes())
                else:
                    self.send_error(500, "Erro ao codificar preview")

            elif parsed.path == "/api/status":
                cal_data = None
                if app.output_file.is_file():
                    try:
                        cal_data = CalibrationData.load(app.output_file).to_dict()
                    except Exception:
                        pass
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "calibration": cal_data,
                            "device": app.device_index,
                            "output_file": str(app.output_file),
                        }
                    ).encode("utf-8")
                )

            elif parsed.path == "/stream":
                # Stream ao vivo da câmera
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        frame = app.get_frame()
                        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if ret:
                            self.wfile.write(b"--frame\r\n")
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(buf)))
                            self.end_headers()
                            self.wfile.write(buf.tobytes())
                            self.wfile.write(b"\r\n")
                        time.sleep(0.04)  # ~25 FPS
                except (BrokenPipeError, ConnectionResetError):
                    pass

            elif parsed.path == "/stream_warped":
                # Stream ao vivo do tabuleiro retificado
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        warped = app.compute_live_warped()
                        ret, buf = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ret:
                            self.wfile.write(b"--frame\r\n")
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(buf)))
                            self.end_headers()
                            self.wfile.write(buf.tobytes())
                            self.wfile.write(b"\r\n")
                        time.sleep(0.07)  # ~14 FPS
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404, "Not Found")

        def do_POST(self):
            parsed = urlparse(self.path)
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"

            try:
                data = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                data = {}

            if parsed.path == "/api/set_corners":
                corners = data.get("corners", [])
                orientation = data.get("orientation", "normal")
                with app.lock:
                    app.active_corners = corners
                    app.active_orientation = orientation
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"success": true}')

            elif parsed.path == "/api/save":
                corners = data.get("corners", [])
                orientation = data.get("orientation", "normal")
                if len(corners) == 4:
                    success = app.save_calibration(corners, orientation)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": success}).encode("utf-8"))
                else:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"success": false, "error": "Requer 4 cantos"}')

            elif parsed.path == "/api/save_reference":
                success = app.save_reference_empty()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": success}).encode("utf-8"))

            elif parsed.path == "/api/detect":
                threshold = float(data.get("threshold", cfg.OCCUPIED_DIFF_THRESHOLD))
                res = app.detect_occupied(threshold=threshold)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
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
    parser.add_argument("--image", "-i", default=None, help="Caminho para imagem estatica (opcional)")
    parser.add_argument("--output", "-o", default=cfg.CALIBRATION_FILE, help="Arquivo de destino do JSON")
    args = parser.parse_args()

    app = WebCalibratorApp(
        device_index=args.device,
        static_image_path=args.image,
        output_file=args.output,
    )

    handler = make_handler(app)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    ips = get_ip_addresses()
    print("\n" + "=" * 65)
    print("  🎯 VoiceChess — Calibrador de Câmera Web (LIVE VIEW)")
    print("=" * 65)
    print("  Servidor iniciado com sucesso!")
    print("\n  Abra o navegador no seu PC Windows:")
    for ip in ips:
        print(f"     👉 http://{ip}:{args.port}")
    print("\n  Pressione Ctrl+C no terminal para encerrar.")
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
