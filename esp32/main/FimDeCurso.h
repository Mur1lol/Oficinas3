// FimDeCurso.h - Controle e leitura de fins de curso (endstops) do Gantry
#pragma once
#include <Arduino.h>

class FimDeCurso {
 public:
  FimDeCurso(uint8_t pino, bool ativoEmAlto = true)
      : pino_(pino), ativoEmAlto_(ativoEmAlto) {}

  void iniciar() {
    pinMode(pino_, INPUT);
  }

  // Retorna true se o fim de curso estiver acionado no momento
  bool acionado() const {
    return digitalRead(pino_) == (ativoEmAlto_ ? HIGH : LOW);
  }

  // Leitura com filtro de ruído (amostras consecutivas)
  bool acionadoFiltrado(uint8_t amostras = 3, uint16_t delayUs = 10) const {
    for (uint8_t i = 0; i < amostras; i++) {
      if (!acionado()) return false;
      if (delayUs > 0) delayMicroseconds(delayUs);
    }
    return true;
  }

  uint8_t pino() const { return pino_; }
  bool ativoEmAlto() const { return ativoEmAlto_; }

 private:
  uint8_t pino_;
  bool ativoEmAlto_;
};
