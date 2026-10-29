// Comunication.h - Protocolo de comunicação serial JSON para o ESP32 (Voice Chess)
#pragma once
#include <Arduino.h>
#include <ArduinoJson.h>
#include "Gantry.h"

class Comunication {
 public:
  Comunication(unsigned long baudrate = 115200) : baudrate_(baudrate) {}

  void iniciar() {
    Serial.begin(baudrate_);
    bufferSerial_.reserve(256);
  }

  // Envia resposta de sucesso padrão {"comando": "ok"}
  void responderOk(const char* mensagem = nullptr) {
    StaticJsonDocument<128> doc;
    doc["comando"] = "ok";
    if (mensagem) {
      doc["mensagem"] = mensagem;
    }
    serializeJson(doc, Serial);
    Serial.print('\n');
  }

  // Envia resposta de erro
  void responderErro(const char* motivo) {
    StaticJsonDocument<192> doc;
    doc["comando"] = "erro";
    doc["motivo"] = motivo;
    serializeJson(doc, Serial);
    Serial.print('\n');
  }

  // Envia status atual do gantry
  void responderStatus(const Gantry& gantry) {
    StaticJsonDocument<256> doc;
    doc["comando"] = "status";
    doc["x"] = gantry.posX();
    doc["y"] = gantry.posY();
    doc["homed"] = gantry.estaHomed();
    doc["drivers"] = gantry.estaHabilitado();
    doc["eletroima"] = gantry.eletroima().estaLigado();
    doc["fim_x"] = gantry.fimX().acionado();
    doc["fim_y"] = gantry.fimY().acionado();
    serializeJson(doc, Serial);
    Serial.print('\n');
  }

  // Processa dados seriais recebidos. Chame frequentemente no loop().
  void atualizar(Gantry& gantry) {
    while (Serial.available()) {
      char c = Serial.read();
      if (c == '\n' || c == '\r') {
        if (bufferSerial_.length() > 0) {
          processarLinha(bufferSerial_, gantry);
          bufferSerial_ = "";
        }
      } else {
        bufferSerial_ += c;
      }
    }
  }

 private:
  void processarLinha(const String& linha, Gantry& gantry) {
    String str = linha;
    str.trim();
    if (str.length() == 0) return;

    // Se a linha começar com '{', trata como comando JSON
    if (str.startsWith("{")) {
      processarJson(str, gantry);
      return;
    }

    // Comandos de texto rápido / modo interativo via monitor serial:
    str.toLowerCase();
    if (str == "on") {
      gantry.habilitar(true);
      responderOk("drivers ligados");
    } else if (str == "off") {
      gantry.habilitar(false);
      responderOk("drivers desligados");
    } else if (str == "h" || str == "homing") {
      if (gantry.homing()) {
        responderOk("homing concluido");
      } else {
        responderErro("falha no homing");
      }
    } else if (str == "m on") {
      gantry.eletroima().ligar(100);
      responderOk("eletroima ligado");
    } else if (str == "m off") {
      gantry.eletroima().desligar();
      responderOk("eletroima desligado");
    } else if (str == "p" || str == "status") {
      responderStatus(gantry);
    } else if (str.startsWith("x ") || str.startsWith("y ")) {
      char eixo = toupper(str.charAt(0));
      float dist = str.substring(2).toFloat();
      if (!gantry.estaHabilitado()) gantry.habilitar(true);
      ResultadoMovimento res = (eixo == 'X') ? gantry.mover(dist, 0.0f) : gantry.mover(0.0f, dist);
      if (res == MOV_OK) {
        responderOk("movimento concluido");
      } else if (res == MOV_BATEU_FIM_CURSO) {
        responderErro("fim de curso acionado");
      } else {
        responderErro("erro no movimento");
      }
    } else {
      responderErro("comando texto desconhecido");
    }
  }

  void processarJson(const String& jsonStr, Gantry& gantry) {
    // Trajetorias de captura chegam a ~60 passos (3 listas); 6 KB e suficiente
    DynamicJsonDocument doc(6144);
    DeserializationError err = deserializeJson(doc, jsonStr);
    if (err) {
      responderErro("json invalido");
      return;
    }

    const char* comando = doc["comando"] | "";

    // ── Comando "gantry": movimenta o robô e controla o eletroímã ──
    if (strcmp(comando, "gantry") == 0) {
      JsonObject resposta = doc["resposta"];
      JsonArray arrX = resposta["x"];
      JsonArray arrY = resposta["y"];
      JsonArray arrIma = resposta["eletroima"];

      size_t n = arrX.size();
      if (n == 0 || arrY.size() != n || arrIma.size() != n) {
        responderErro("listas x, y e eletroima devem ter o mesmo tamanho e > 0");
        return;
      }

      // Aloca buffers temporários para os passos
      float* posX = new float[n];
      float* posY = new float[n];
      int* posIma = new int[n];

      for (size_t i = 0; i < n; i++) {
        posX[i] = arrX[i].as<float>();
        posY[i] = arrY[i].as<float>();
        posIma[i] = arrIma[i].as<int>();
      }

      ResultadoMovimento res = gantry.executarTrajetoria(posX, posY, posIma, n);

      delete[] posX;
      delete[] posY;
      delete[] posIma;

      // Desliga o eletroima por seguranca
      gantry.eletroima().desligar();

      // Antes de responder que o movimento foi finalizado, retorna o gantry para o Homing (0,0)
      if (res == MOV_OK) {
        res = gantry.irPara(0.0f, 0.0f);
      }

      if (res == MOV_OK) {
        responderOk();
      } else if (res == MOV_BATEU_FIM_CURSO) {
        responderErro("fim de curso acionado durante o movimento");
      } else if (res == MOV_ABORTADO) {
        responderErro("movimento abortado pela serial");
      } else {
        responderErro("erro na execucao da trajetoria");
      }
      return;
    }

    // ── Comando "homing" em JSON ──
    if (strcmp(comando, "homing") == 0) {
      if (gantry.homing()) {
        responderOk("homing concluido");
      } else {
        responderErro("falha ao realizar homing");
      }
      return;
    }

    // ── Comando "eletroima" em JSON ──
    if (strcmp(comando, "eletroima") == 0) {
      int estado = doc["resposta"] | 0;
      if (estado > 0) {
        gantry.eletroima().ligar(100);
      } else {
        gantry.eletroima().desligar();
      }
      responderOk();
      return;
    }

    // ── Comando "status" em JSON ──
    if (strcmp(comando, "status") == 0) {
      responderStatus(gantry);
      return;
    }

    // ── Comando "mover" em JSON: desloca no eixo X ou Y em milímetros ──
    if (strcmp(comando, "mover") == 0) {
      float dx = 0.0f;
      float dy = 0.0f;
      if (doc["resposta"].is<JsonObject>()) {
        JsonObject rsp = doc["resposta"];
        if (rsp.containsKey("eixo")) {
          const char* eixo = rsp["eixo"] | "X";
          float dist = rsp["distancia"] | 0.0f;
          if (toupper(eixo[0]) == 'X') dx = dist;
          else if (toupper(eixo[0]) == 'Y') dy = dist;
        } else {
          dx = rsp["dx"] | 0.0f;
          dy = rsp["dy"] | 0.0f;
        }
      }
      if (!gantry.estaHabilitado()) {
        gantry.habilitar(true);
      }
      ResultadoMovimento res = gantry.mover(dx, dy);
      if (res == MOV_OK) {
        responderOk("movimento concluido");
      } else if (res == MOV_BATEU_FIM_CURSO) {
        responderErro("fim de curso acionado");
      } else {
        responderErro("erro no movimento");
      }
      return;
    }

    responderErro("comando json desconhecido");
  }

  unsigned long baudrate_;
  String bufferSerial_;
};
