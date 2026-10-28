// Motores.h - Controle dos drivers de passo para cinemática CoreXY (Voice Chess)
#pragma once
#include <Arduino.h>

class Motores {
 public:
  Motores(uint8_t aStep = 26, uint8_t aDir = 27,
          uint8_t bStep = 18, uint8_t bDir = 19,
          uint8_t enMot = 23,
          bool inverterA = false, bool inverterB = false, bool trocarAB = false)
      : aStepPin_(aStep), aDirPin_(aDir),
        bStepPin_(bStep), bDirPin_(bDir),
        enMotPin_(enMot),
        inverterA_(inverterA), inverterB_(inverterB), trocarAB_(trocarAB) {}

  void iniciar() {
    // Define pinos físicos de acordo com trocarAB
    pinoStepA_ = trocarAB_ ? bStepPin_ : aStepPin_;
    pinoDirA_  = trocarAB_ ? bDirPin_  : aDirPin_;
    pinoStepB_ = trocarAB_ ? aStepPin_ : bStepPin_;
    pinoDirB_  = trocarAB_ ? aDirPin_  : bDirPin_;

    // Drivers iniciam DESLIGADOS (EN_MOT em HIGH) por segurança
    digitalWrite(enMotPin_, HIGH);
    pinMode(enMotPin_, OUTPUT);

    for (uint8_t pino : {pinoStepA_, pinoDirA_, pinoStepB_, pinoDirB_}) {
      pinMode(pino, OUTPUT);
      digitalWrite(pino, LOW);
    }
  }

  // Liga (true, LOW no EN_MOT) ou desliga (false, HIGH no EN_MOT) os drivers
  void habilitar(bool ligar) {
    habilitado_ = ligar;
    digitalWrite(enMotPin_, ligar ? LOW : HIGH);
  }

  bool estaHabilitado() const { return habilitado_; }

  // Ajusta a direção física dos dois motores
  void definirDirecoes(bool dirA_positiva, bool dirB_positiva) {
    digitalWrite(pinoDirA_, dirA_positiva != inverterA_ ? HIGH : LOW);
    digitalWrite(pinoDirB_, dirB_positiva != inverterB_ ? HIGH : LOW);
    delayMicroseconds(5); // tempo de setup do sinal DIR no driver
  }

  // Envia pulso de micropasso para os motores selecionados
  void pulsarPasso(bool passoA, bool passoB, uint16_t pulsoUs = 2) {
    if (passoA) digitalWrite(pinoStepA_, HIGH);
    if (passoB) digitalWrite(pinoStepB_, HIGH);
    delayMicroseconds(pulsoUs);
    digitalWrite(pinoStepA_, LOW);
    digitalWrite(pinoStepB_, LOW);
  }

  // Configurações de inversão em tempo de execução se necessário
  void setInversoes(bool invA, bool invB, bool trocar) {
    inverterA_ = invA;
    inverterB_ = invB;
    trocarAB_ = trocar;
    iniciar();
  }

 private:
  uint8_t aStepPin_, aDirPin_;
  uint8_t bStepPin_, bDirPin_;
  uint8_t enMotPin_;
  bool inverterA_, inverterB_, trocarAB_;

  uint8_t pinoStepA_ = 0, pinoDirA_ = 0;
  uint8_t pinoStepB_ = 0, pinoDirB_ = 0;
  bool habilitado_ = false;
};
