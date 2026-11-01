#!/usr/bin/env python3
"""
vision/scripts/web_calibrator.py
================================
Web-based Camera & Board Calibration Tool for Headless Raspberry Pi.

Provides a lightweight web interface to calibrate the chessboard corners,
view live streams, preview perspective correction, save camera_calibration.json,
and record empty board reference images — all accessible from a PC browser
without requiring X11/GUI or extra pip dependencies (uses Python standard library).

Usage:
------
    python vision/scripts/web_calibrator.py
    python vision/scripts/web_calibrator.py --port 5000 --device 0
    python vision/scripts/web_calibrator.py --image path/to/sample.jpg
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

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
  <title>VoiceChess — Calibrador de Câmera</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --success: #22c55e;
      --text: #f8fafc;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 1.5rem; }
    header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid var(--card-border); }
    h1 { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
    .status-badge { background: #065f46; color: #34d399; font-size: 0.85rem; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 600; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
    @media (max-width: 1024px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 0.75rem; padding: 1.25rem; }
    .card-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; }
    .canvas-container { position: relative; width: 100%; border-radius: 0.5rem; overflow: hidden; background: #000; cursor: crosshair; }
    canvas { display: block; width: 100%; height: auto; }
    .preview-img { width: 100%; height: auto; border-radius: 0.5rem; background: #000; aspect-ratio: 1; object-fit: contain; }
    .controls { margin-top: 1.25rem; display: flex; flex-direction: column; gap: 1rem; }
    .point-list { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; }
    .point-row { display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; background: #0f172a80; padding: 0.4rem 0.6rem; border-radius: 0.375rem; }
    .point-color { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
    .point-input { width: 60px; background: #0f172a; border: 1px solid var(--card-border); color: #fff; padding: 0.2rem 0.4rem; border-radius: 0.25rem; text-align: center; font-size: 0.8rem; }
    .btn-group { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 1rem; }
    button { background: var(--accent); color: #000; font-weight: 600; border: none; padding: 0.6rem 1.2rem; border-radius: 0.5rem; cursor: pointer; transition: 0.15s ease; font-size: 0.9rem; }
    button:hover { background: var(--accent-hover); color: #fff; }
    button.btn-success { background: var(--success); color: #000; }
    button.btn-success:hover { background: #16a34a; color: #fff; }
    button.btn-secondary { background: #334155; color: #fff; }
    button.btn-secondary:hover { background: #475569; }
    .alert { padding: 0.75rem 1rem; border-radius: 0.5rem; margin-top: 1rem; font-size: 0.9rem; display: none; }
    .alert-success { background: #064e3b; border: 1px solid #059669; color: #6ee7b7; }
    .alert-error { background: #7f1d1d; border: 1px solid #dc2626; color: #fca5a5; }
    .instructions { font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; margin-top: 0.75rem; }
    .instructions strong { color: var(--text); }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>VoiceChess — Calibrador Web</h1>
      <p style="color: var(--text-muted); font-size: 0.85rem;">Calibração de perspectiva do tabuleiro para Raspberry Pi (Modo Headless)</p>
    </div>
    <span class="status-badge" id="cameraStatus">Câmera Conectada</span>
  </header>

  <div class="grid">
    <!-- COLUNA 1: IMAGEM DA CÂMERA E SELEÇÃO DE CANTOS -->
    <div class="card">
      <div class="card-title">
        <span>Imagem da Câmera (Clique nos 4 Cantos)</span>
        <button class="btn-secondary" style="padding: 0.3rem 0.7rem; font-size: 0.8rem;" onclick="fetchSnapshot()">Atualizar Frame</button>
      </div>

      <div class="canvas-container" id="canvasBox">
        <canvas id="canvas"></canvas>
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
          <strong>Como usar:</strong> Clique em sentido horário nos 4 cantos da grade do tabuleiro:
          1 (A8 / Sup-Esq) &rarr; 2 (H8 / Sup-Dir) &rarr; 3 (H1 / Inf-Dir) &rarr; 4 (A1 / Inf-Esq).
          Você também pode arrastar os círculos ou ajustar os valores numéricos.
        </div>

        <div class="btn-group">
          <button class="btn-secondary" onclick="resetPoints()">Limpar Pontos</button>
          <button class="btn-secondary" onclick="reloadSaved()">Carregar Salvo</button>
        </div>
      </div>
    </div>

    <!-- COLUNA 2: PREVIEW RETIFICADO E AÇÕES -->
    <div class="card">
      <div class="card-title">
        <span>Tabuleiro Corrigido (Vista Superior 8x8)</span>
        <button class="btn-secondary" style="padding: 0.3rem 0.7rem; font-size: 0.8rem;" onclick="updatePreview()">Recalcular</button>
      </div>

      <img id="previewImg" class="preview-img" alt="Preview da calibração aparecerá aqui">

      <div class="controls">
        <div style="display: flex; gap: 1rem; align-items: center;">
          <label style="font-size: 0.9rem; color: var(--text-muted);">Orientação:</label>
          <select id="orientationSelect" class="point-input" style="width: auto; padding: 0.3rem 0.6rem;" onchange="updatePreview()">
            <option value="normal">Normal (A1 em baixo à esquerda)</option>
            <option value="flipped">Invertida (A1 no topo à direita)</option>
          </select>
        </div>

        <div class="btn-group" style="margin-top: 1rem;">
          <button class="btn-success" onclick="saveCalibration()">💾 Salvar Calibração</button>
          <button class="btn-secondary" onclick="saveReference()">📷 Gravar Tabuleiro Vazio</button>
          <button class="btn-secondary" onclick="testDetection()">🔍 Testar Ocupação</button>
        </div>

        <div id="alertBox" class="alert"></div>

        <div id="detectionInfo" style="margin-top: 1rem; font-size: 0.85rem; color: var(--text-muted); display: none;"></div>
      </div>
    </div>
  </div>

  <script>
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    let currentImage = new Image();
    let corners = []; // [[x,y], ...]
    let activeDragIdx = -1;
    const colors = ["#38bdf8", "#4ade80", "#facc15", "#fb923c"];

    function showAlert(msg, isSuccess = true) {
      const box = document.getElementById("alertBox");
      box.className = "alert " + (isSuccess ? "alert-success" : "alert-error");
      box.textContent = msg;
      box.style.display = "block";
      setTimeout(() => { box.style.display = "none"; }, 5000);
    }

    function fetchSnapshot() {
      currentImage.src = "/api/snapshot?t=" + new Date().getTime();
    }

    currentImage.onload = function() {
      canvas.width = currentImage.naturalWidth;
      canvas.height = currentImage.naturalHeight;
      renderCanvas();
    };

    function renderCanvas() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(currentImage, 0, 0);

      if (corners.length >= 2) {
        ctx.beginPath();
        ctx.moveTo(corners[0][0], corners[0][1]);
        for (let i = 1; i < corners.length; i++) {
          ctx.lineTo(corners[i][0], corners[i][1]);
        }
        if (corners.length === 4) {
          ctx.closePath();
          ctx.fillStyle = "rgba(56, 189, 248, 0.15)";
          ctx.fill();
        }
        ctx.lineWidth = 3;
        ctx.strokeStyle = "#38bdf8";
        ctx.stroke();
      }

      corners.forEach((pt, idx) => {
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], 10, 0, 2 * Math.PI);
        ctx.fillStyle = colors[idx];
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#ffffff";
        ctx.stroke();

        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 14px sans-serif";
        ctx.fillText((idx + 1).toString(), pt[0] + 14, pt[1] + 5);

        const xi = document.getElementById("x" + idx);
        const yi = document.getElementById("y" + idx);
        if (xi) xi.value = Math.round(pt[0]);
        if (yi) yi.value = Math.round(pt[1]);
      });
    }

    function getCanvasCoords(e) {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      return [
        (e.clientX - rect.left) * scaleX,
        (e.clientY - rect.top) * scaleY
      ];
    }

    canvas.addEventListener("mousedown", (e) => {
      const [cx, cy] = getCanvasCoords(e);
      // Checa se clicou perto de um ponto existente para arrastar
      activeDragIdx = corners.findIndex(pt => Math.hypot(pt[0] - cx, pt[1] - cy) < 25);
      if (activeDragIdx === -1) {
        if (corners.length < 4) {
          corners.push([cx, cy]);
          renderCanvas();
          if (corners.length === 4) updatePreview();
        }
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (activeDragIdx !== -1) {
        const [cx, cy] = getCanvasCoords(e);
        corners[activeDragIdx] = [
          Math.max(0, Math.min(canvas.width, cx)),
          Math.max(0, Math.min(canvas.height, cy))
        ];
        renderCanvas();
      }
    });

    window.addEventListener("mouseup", () => {
      if (activeDragIdx !== -1) {
        activeDragIdx = -1;
        if (corners.length === 4) updatePreview();
      }
    });

    function manualCoord(idx) {
      if (!corners[idx]) return;
      const x = parseFloat(document.getElementById("x" + idx).value);
      const y = parseFloat(document.getElementById("y" + idx).value);
      if (!isNaN(x) && !isNaN(y)) {
        corners[idx] = [x, y];
        renderCanvas();
        if (corners.length === 4) updatePreview();
      }
    }

    function resetPoints() {
      corners = [];
      for (let i = 0; i < 4; i++) {
        document.getElementById("x" + i).value = "";
        document.getElementById("y" + i).value = "";
      }
      renderCanvas();
      document.getElementById("previewImg").src = "";
    }

    function updatePreview() {
      if (corners.length !== 4) return;
      const orientation = document.getElementById("orientationSelect").value;
      fetch("/api/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ corners, orientation })
      })
      .then(r => r.json())
      .then(data => {
        if (data.image) {
          document.getElementById("previewImg").src = "data:image/jpeg;base64," + data.image;
        }
      })
      .catch(err => console.error("Erro no preview:", err));
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
        body: JSON.stringify({
          corners: corners,
          orientation: orientation
        })
      })
      .then(r => r.json())
      .then(res => {
        if (res.success) {
          showAlert("Calibração salva com sucesso em camera_calibration.json!");
        } else {
          showAlert("Erro ao salvar: " + res.error, false);
        }
      })
      .catch(err => showAlert("Falha de rede ao salvar: " + err, false));
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
          showAlert("Imagem de referência (tabuleiro vazio) salva em reference_empty.png!");
        } else {
          showAlert("Erro ao gravar referência: " + res.error, false);
        }
      })
      .catch(err => showAlert("Erro: " + err, false));
    }

    function testDetection() {
      const info = document.getElementById("detectionInfo");
      info.style.display = "block";
      info.textContent = "Analisando ocupação das casas...";
      fetch("/api/detect", { method: "POST" })
      .then(r => r.json())
      .then(res => {
        if (res.success) {
          const occ = res.occupied_squares.join(", ") || "Nenhuma peça detectada";
          info.innerHTML = `<strong>Casas Ocupadas:</strong> ${occ}<br><strong>Confiança:</strong> ${(res.confidence * 100).toFixed(1)}%`;
          showAlert("Detecção de ocupação concluída!");
        } else {
          info.textContent = "Erro na detecção: " + res.error;
          showAlert("Falha na detecção: " + res.error, false);
        }
      })
      .catch(err => {
        info.textContent = "Erro de rede: " + err;
      });
    }

    function reloadSaved() {
      fetch("/api/status")
      .then(r => r.json())
      .then(data => {
        if (data.calibration && data.calibration.board_corners && data.calibration.board_corners.length === 4) {
          corners = data.calibration.board_corners;
          document.getElementById("orientationSelect").value = data.calibration.board_orientation || "normal";
          renderCanvas();
          updatePreview();
          showAlert("Calibração anterior carregada.");
        } else {
          showAlert("Nenhuma calibração anterior encontrada.", false);
        }
      });
    }

    window.onload = function() {
      fetchSnapshot();
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

        if self.static_image_path and Path(self.static_image_path).is_file():
            img = cv2.imread(self.static_image_path)
            if img is not None:
                self.last_frame = img
                log.info("Usando imagem estatica: %s", self.static_image_path)

        if self.last_frame is None:
            self._init_camera()

    def _init_camera(self) -> None:
        try:
            self.camera = CameraController(device_index=self.device_index)
            self.camera.open()
            log.info("Camera aberta no dispositivo %d", self.device_index)
        except Exception as e:
            log.warning("Nao foi possivel abrir a camera (%s). Criando frame simulado.", e)
            # Imagem de fallback caso a webcam nao esteja conectada
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
        # Desenha tabuleiro generico centralizado
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

        # Desenha a grade com labels no tabuleiro
        segmenter = BoardSegmenter(board_orientation=orientation)
        annotated = segmenter.draw_grid(warped)
        return annotated

    def save_calibration(self, corners: list[list[float]], orientation: str = "normal") -> bool:
        frame = self.get_frame()
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
        log.info("Referencia de tabuleiro vazio gravada em %s", ref_path)
        return True

    def detect_occupied(self) -> dict:
        if not self.output_file.is_file():
            return {"success": False, "error": "Falta calibracao previa"}
        cal = CalibrationData.load(self.output_file)
        frame = self.get_frame()
        calibrator = PerspectiveCalibrator(cal)
        warped = calibrator.correct_perspective(frame)

        ref_path = Path(cfg.VISION_DIR) / "reference_empty.png"
        ref_img = cv2.imread(str(ref_path)) if ref_path.is_file() else None
        if ref_img is None:
            return {
                "success": False,
                "error": "Imagem de referência não encontrada! Clique primeiro em 'Gravar Tabuleiro Vazio' com o tabuleiro limpo.",
            }

        detector = PieceDetector(
            reference_image=ref_img,
            board_orientation=cal.board_orientation,
        )
        segmenter = BoardSegmenter(board_orientation=cal.board_orientation)
        squares = segmenter.segment_board(warped)

        occupied = []
        confidences = []
        for name, roi in squares.items():
            analysis = detector.analyze_square(name, roi)
            if analysis.occupied:
                occupied.append(f"{name} ({analysis.color or '?'})")
            confidences.append(analysis.confidence)

        avg_conf = float(np.mean(confidences)) if confidences else 1.0
        return {
            "success": True,
            "occupied_squares": occupied,
            "confidence": round(avg_conf, 3),
            "reference_used": True,
        }


def make_handler(app: WebCalibratorApp):
    class RequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Silencia logs repetitivos de polling
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
                ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(buf.tobytes())
                else:
                    self.send_error(500, "Falha na codificacao da imagem")

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
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.end_headers()
                try:
                    while True:
                        frame = app.get_frame()
                        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ret:
                            self.wfile.write(b"--frame\r\n")
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(buf)))
                            self.end_headers()
                            self.wfile.write(buf.tobytes())
                            self.wfile.write(b"\r\n")
                        time.sleep(0.06)
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

            if parsed.path == "/api/preview":
                corners = data.get("corners", [])
                orientation = data.get("orientation", "normal")
                preview = app.compute_preview(corners, orientation)
                if preview is not None:
                    ret, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    b64 = base64.b64encode(buf).decode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"image": b64}).encode("utf-8"))
                else:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error": "Cantos invalidos"}')

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
                res = app.detect_occupied()
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
        # Obtem IP de saida
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
    print("  🎯 VoiceChess — Calibrador de Câmera Web (Modo Headless)")
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
