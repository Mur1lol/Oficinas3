// Eletroima.h - Acionamento e proteção do eletroímã 12 V via TIP31C (Voice Chess)
#pragma once
#include <Arduino.h>

class Eletroima {
 public:
  Eletroima(uint8_t pino, uint32_t tempoMaxMs = 20000, uint32_t freqHz = 1000)
      : pino_(pino), tempoMaxMs_(tempoMaxMs), freqHz_(freqHz) {}

  void iniciar() {
    digitalWrite(pino_, LOW);
    pinMode(pino_, OUTPUT);
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcAttach(pino_, freqHz_, RESOLUCAO_BITS);
#else
    ledcSetup(CANAL, freqHz_, RESOLUCAO_BITS);
    ledcAttachPin(pino_, CANAL);
#endif
    escrever(0);
  }

  // Liga o eletroímã com potência de 0 a 100% (padrão 100%)
  void ligar(uint8_t potencia = 100) {
    potencia_ = constrain(potencia, 0, 100);
    escrever(potencia_);
    ligadoDesde_ = millis();
    ligado_ = potencia_ > 0;
  }

  // Desliga o eletroímã
  void desligar() {
    escrever(0);
    ligado_ = false;
  }

  // Chame periodicamente no loop(). Desliga automaticamente se estourar o tempo máximo.
  // Retorna true se acabou de desligar por proteção de tempo.
  bool atualizar() {
    if (ligado_ && tempoMaxMs_ > 0 && millis() - ligadoDesde_ >= tempoMaxMs_) {
      desligar();
      return true;
    }
    return false;
  }

  bool estaLigado() const { return ligado_; }
  uint8_t potencia() const { return ligado_ ? potencia_ : 0; }
  uint32_t ligadoHaMs() const { return ligado_ ? millis() - ligadoDesde_ : 0; }
  void definirTempoMax(uint32_t ms) { tempoMaxMs_ = ms; }
  uint8_t pino() const { return pino_; }

 private:
  static const uint8_t RESOLUCAO_BITS = 8;
  static const uint8_t CANAL = 0;

  void escrever(uint8_t pct) {
    uint32_t duty = (uint32_t)pct * ((1u << RESOLUCAO_BITS) - 1) / 100;
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcWrite(pino_, duty);
#else
    ledcWrite(CANAL, duty);
#endif
  }

  uint8_t pino_;
  uint32_t tempoMaxMs_;
  uint32_t freqHz_;
  uint8_t potencia_ = 100;
  bool ligado_ = false;
  uint32_t ligadoDesde_ = 0;
};
