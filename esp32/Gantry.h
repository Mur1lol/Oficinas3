// Gantry.h - Classe unificada do Gantry CoreXY (Motores + Eletroímã + Fins de Curso)
#pragma once
#include <Arduino.h>
#include "Motores.h"
#include "Eletroima.h"
#include "FimDeCurso.h"

enum ResultadoMovimento {
  MOV_OK = 0,
  MOV_ABORTADO,
  MOV_BATEU_FIM_CURSO,
  MOV_ERRO_DRIVERS_DESLIGADOS,
  MOV_ERRO_SEM_HOME
};

class Gantry {
 public:
  static constexpr float PASSOS_MM   = 80.0f;   // 200 * 16 / (polia 20 dentes * 2 mm)
  static constexpr float CURSO_X     = 800.0f;  // Limite útil X em mm
  static constexpr float CURSO_Y     = 400.0f;  // Limite útil Y em mm
  static constexpr float V_INICIAL   = 5.0f;    // mm/s no início/fim da rampa
  static constexpr float MARGEM_HOME = 2.0f;    // Zero fica 2 mm antes dos fins de curso

  Gantry(uint8_t aStep = 26, uint8_t aDir = 27,
         uint8_t bStep = 18, uint8_t bDir = 19,
         uint8_t enMot = 23,
         uint8_t pinoIma = 25,
         uint8_t pinoFimX = 32, uint8_t pinoFimY = 33,
         bool fimAtivoAlto = true)
      : motores_(aStep, aDir, bStep, bDir, enMot),
        eletroima_(pinoIma),
        fimX_(pinoFimX, fimAtivoAlto),
        fimY_(pinoFimY, fimAtivoAlto) {}

  void iniciar() {
    motores_.iniciar();
    eletroima_.iniciar();
    fimX_.iniciar();
    fimY_.iniciar();
    posA_ = 0;
    posB_ = 0;
    homed_ = false;
  }

  // Liga ou desliga os drivers de passo
  void habilitar(bool ligar) {
    motores_.habilitar(ligar);
    if (!ligar) homed_ = false; // Desligar o motor perde o alinhamento
  }

  bool estaHabilitado() const { return motores_.estaHabilitado(); }
  bool estaHomed() const { return homed_; }

  float posX() const { return (posA_ + posB_) / (2.0f * PASSOS_MM); }
  float posY() const { return (posA_ - posB_) / (2.0f * PASSOS_MM); }

  void definirPosicao(float x, float y) {
    posA_ = lroundf((x + y) * PASSOS_MM);
    posB_ = lroundf((x - y) * PASSOS_MM);
  }

  void zerarSemHoming() {
    posA_ = 0;
    posB_ = 0;
  }

  void setVelocidade(float mm_s) { vel_ = constrain(mm_s, 1.0f, 150.0f); }
  void setAceleracao(float mm_s2) { acel_ = constrain(mm_s2, 10.0f, 3000.0f); }
  float velocidade() const { return vel_; }
  float aceleracao() const { return acel_; }

  // Acesso direto aos periféricos
  Eletroima& eletroima() { return eletroima_; }
  Motores& motores() { return motores_; }
  const FimDeCurso& fimX() const { return fimX_; }
  const FimDeCurso& fimY() const { return fimY_; }

  // Chame no loop(): verifica timeouts do eletroímã
  void atualizar() {
    eletroima_.atualizar();
  }

  // Move dx, dy em relação à posição atual com rampa de aceleração trapezoidal.
  // 'procurar' = 'X' ou 'Y' para homing.
  ResultadoMovimento mover(float dx, float dy, char procurar = 0, float vMax = 0, bool checarSerial = true) {
    if (!motores_.estaHabilitado()) {
      return MOV_ERRO_DRIVERS_DESLIGADOS;
    }

    // Cinemática CoreXY: da = dx + dy, db = dx - dy
    long da = lroundf((dx + dy) * PASSOS_MM);
    long db = lroundf((dx - dy) * PASSOS_MM);
    long na = labs(da);
    long nb = labs(db);
    long n  = max(na, nb);
    if (n == 0) return MOV_OK;

    motores_.definirDirecoes(da >= 0, db >= 0);

    float L = sqrtf(dx * dx + dy * dy);
    float mmPorPasso = L / n;
    float v = (vMax > 0) ? vMax : vel_;
    float v0q = V_INICIAL * V_INICIAL;

    long ea = n / 2, eb = n / 2;
    int sa = (da >= 0) ? 1 : -1;
    int sb = (db >= 0) ? 1 : -1;

    uint8_t contaFim = 0;
    ResultadoMovimento r = MOV_OK;
    uint32_t t = micros();

    for (long i = 0; i < n; i++) {
      if (checarSerial && Serial.available()) {
        r = MOV_ABORTADO;
        break;
      }

      bool fx = fimX_.acionado();
      bool fy = fimY_.acionado();
      bool bate = (procurar == 'X') ? fx : (procurar == 'Y') ? fy : ((fx && dx < 0) || (fy && dy < 0));
      contaFim = bate ? (contaFim + 1) : 0;
      if (contaFim >= 3) {
        r = MOV_BATEU_FIM_CURSO;
        break;
      }

      float s = i * mmPorPasso;
      float vs = min(v, min(sqrtf(v0q + 2.0f * acel_ * s), sqrtf(v0q + 2.0f * acel_ * (L - s))));
      uint32_t intervalo = (uint32_t)(mmPorPasso / vs * 1e6f);

      bool passoA = false, passoB = false;
      ea -= na; if (ea < 0) { ea += n; passoA = true; }
      eb -= nb; if (eb < 0) { eb += n; passoB = true; }

      while ((int32_t)(micros() - t) < (int32_t)intervalo) {}
      t += intervalo;

      motores_.pulsarPasso(passoA, passoB, 2);

      if (passoA) posA_ += sa;
      if (passoB) posB_ += sb;
    }

    return r;
  }

  // Vai para a coordenada absoluta (x, y) em mm. Trava dentro do curso útil.
  ResultadoMovimento irPara(float x, float y) {
    if (homed_) {
      x = constrain(x, 0.0f, CURSO_X);
      y = constrain(y, 0.0f, CURSO_Y);
    }
    return mover(x - posX(), y - posY());
  }

  // Rotina completa de Homing
  bool homing() {
    if (!motores_.estaHabilitado()) {
      motores_.habilitar(true);
    }
    homed_ = false;

    float ax = 0, ay = 0;
    if (!procurarFim('X', ax) || !procurarFim('Y', ay)) {
      return false;
    }

    definirPosicao(-MARGEM_HOME, -MARGEM_HOME);
    if (mover(MARGEM_HOME, MARGEM_HOME, 0, 15, false) != MOV_OK) {
      return false;
    }

    homed_ = true;
    return true;
  }

  // Executa uma sequência completa de movimentos e controle do eletroímã (ex: vinda do Python)
  // x e y são posições absolutas ou relativas conforme a lógica do jogo (padrão absoluto em mm)
  ResultadoMovimento executarTrajetoria(const float* x, const float* y, const int* ima, size_t tamanho) {
    if (!motores_.estaHabilitado()) {
      motores_.habilitar(true);
    }

    for (size_t i = 0; i < tamanho; i++) {
      // Ajusta o eletroímã para o passo atual
      if (ima[i] > 0) {
        eletroima_.ligar(100);
      } else {
        eletroima_.desligar();
      }

      // Desloca para a posição do passo
      ResultadoMovimento res = irPara(x[i], y[i]);
      if (res != MOV_OK) {
        return res;
      }
    }
    return MOV_OK;
  }

 private:
  bool procurarFim(char eixo, float& achou) {
    const FimDeCurso& f = (eixo == 'X') ? fimX_ : fimY_;
    float ux = (eixo == 'X') ? 1.0f : 0.0f;
    float uy = (eixo == 'Y') ? 1.0f : 0.0f;
    float maxMm = (eixo == 'X') ? (CURSO_X + 50.0f) : (CURSO_Y + 50.0f);

    // Se já estiver acionado, dá um pequeno passo à frente para desencostar
    if (f.acionado()) mover(5.0f * ux, 5.0f * uy, 0, 10, false);
    if (f.acionado()) return false;

    // Busca rápida (15 mm/s) até acionar
    ResultadoMovimento r = mover(-maxMm * ux, -maxMm * uy, eixo, 15, false);
    if (r != MOV_BATEU_FIM_CURSO) return false;

    // Recua 3 mm
    mover(3.0f * ux, 3.0f * uy, 0, 5, false);

    // Busca lenta de precisão (3 mm/s)
    if (mover(-6.0f * ux, -6.0f * uy, eixo, 3, false) != MOV_BATEU_FIM_CURSO) {
      return false;
    }

    achou = (eixo == 'X') ? posX() : posY();
    return true;
  }

  Motores motores_;
  Eletroima eletroima_;
  FimDeCurso fimX_;
  FimDeCurso fimY_;

  float vel_  = 30.0f;   // mm/s
  float acel_ = 300.0f;  // mm/s²
  long posA_  = 0;
  long posB_  = 0;
  bool homed_ = false;
};
