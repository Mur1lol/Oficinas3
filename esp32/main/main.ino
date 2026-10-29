// main.ino - Firmware principal do ESP32 para o Voice Chess
// Integração: Gantry (CoreXY + Motores + Eletroímã + Fins de Curso) + Comunicação Serial JSON
//
// Placa: ESP32 Dev Module
// Comunicação: 115200 baud via USB
//
// Pinagem da PCB A:
//   Motor A: STEP = D26, DIR = D27
//   Motor B: STEP = D18, DIR = D19
//   EN_MOT:  D23 (ativo em LOW)
//   Eletroímã: D25 (TIP31C)
//   Fim de curso X: D32 (J4), no lado direito
//   Fim de curso Y: D33 (J5), na frente
//
// Área: X 0..350 mm, Y 0..650 mm. Home = (0, 0), com recuo pós-chave de 5 mm em X e 10 mm em Y.

#include <Arduino.h>
#include "Gantry.h"
#include "Comunication.h"

// Instâncias do sistema
Gantry gantry(
    26, 27,  // Motor A: Step, Dir
    18, 19,  // Motor B: Step, Dir
    23,      // Enable dos motores (ativo em LOW)
    25,      // Pino do eletroímã
    32, 33,  // Fins de curso: X (D32, lado direito), Y (D33, frente)
    false,   // Módulos de fim de curso: ativo em nível BAIXO
    false, false, false  // inverterA, inverterB, trocarAB: os mesmos do teste_gantry
);

Comunication comunicacao(115200);

void setup() {
  // Inicializa comunicação serial
  comunicacao.iniciar();
  delay(200);

  // Inicializa motores, eletroímã e fins de curso
  gantry.iniciar();

  // Liga drivers inicialmente para manter o carro travado ou aguarda comando
  // gantry.habilitar(true);

  // Envia confirmação de boot pronto via JSON
  comunicacao.responderOk("ESP32 VoiceChess inicializado");
}

void loop() {
  // 1. Atualiza proteções de segurança (timeout de superaquecimento do eletroímã)
  gantry.atualizar();

  // 2. Processa comandos recebidos da Raspberry Pi ou comandos manuais
  comunicacao.atualizar(gantry);
}
