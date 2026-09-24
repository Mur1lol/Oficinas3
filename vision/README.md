# VoiceChess — Módulo de Visão

Sistema de captura e processamento de imagens para o VoiceChess. Identifica o estado físico do tabuleiro de xadrez via câmera USB posicionada acima da estrutura.

> **Módulo independente** — não depende do módulo de voz (`voice/`). Pode ser desenvolvido, testado e executado separadamente.

---

## Estrutura de Arquivos

```
vision/
├── config.py              # Todas as constantes configuráveis
├── requirements.txt       # Dependências Python (opencv + numpy)
├── smoke_test.py          # Testa todos os módulos SEM câmera física
│
├── __init__.py            # CameraSystem — fachada pública do módulo
├── camera.py              # Controle da câmera USB (captura, warmup, retry)
├── calibration.py         # Correção de perspectiva (4 cantos do tabuleiro)
├── segmentation.py        # Divide a imagem em 64 casas + cemitérios
├── piece_detector.py      # Detecta ocupação e cor das peças
├── board_state.py         # Estrutura de dados do estado do tabuleiro
├── state_comparator.py    # Compara dois estados (valida movimento)
├── move_validator.py      # Orquestra captura → processo → validação
├── diag_manager.py        # Salva imagens de diagnóstico (opcional)
│
└── scripts/
    └── calibrate_camera.py  # Assistente interativo de calibração
```

---

## Descrição de Cada Arquivo

### `config.py`
Centraliza **todas as constantes** do módulo de visão. Edite este arquivo para ajustar o sistema ao seu hardware sem tocar em nenhum módulo de lógica.

Principais parâmetros:
| Constante | Padrão | Descrição |
|-----------|--------|-----------|
| `CAMERA_DEVICE_INDEX` | `0` | Índice da câmera USB no OpenCV |
| `CAMERA_RESOLUTION` | `(1280, 960)` | Resolução de captura |
| `CAMERA_WARMUP_SECONDS` | `2.0` | Tempo de espera para estabilização |
| `BOARD_OUTPUT_SIZE_PX` | `640` | Tamanho da imagem corrigida (px) |
| `OCCUPIED_DIFF_THRESHOLD` | `0.12` | Diferença mínima para "casa ocupada" |
| `PIECE_BRIGHTNESS_THRESHOLD` | `160` | Brilho HSV separando peças brancas/pretas |
| `MIN_CONFIDENCE` | `0.70` | Confiança mínima para aceitar resultado |
| `MAX_CAPTURE_ATTEMPTS` | `3` | Máximo de tentativas de captura |
| `SAVE_DIAGNOSTIC_IMAGES` | `True` | Salva imagens para debug |

---

### `camera.py` — `CameraController`
Controla a câmera USB usando OpenCV (`cv2.VideoCapture`).

- Abre a câmera com a resolução configurada
- Descarta frames durante o período de warmup (auto-exposição)
- Valida se o frame capturado não está escuro/obstruído
- Tenta novamente (`capture_with_retry`) até `MAX_CAPTURE_ATTEMPTS` vezes

---

### `calibration.py` — `PerspectiveCalibrator` + `InteractiveCalibrator`
Corrige a perspectiva da câmera posicionada em ângulo.

- **`CalibrationData`**: armazena os 4 cantos do tabuleiro + metadados, salvo em `camera_calibration.json`
- **`PerspectiveCalibrator`**: aplica a transformação de perspectiva a cada frame capturado, produzindo uma imagem retangular vista de cima
- **`InteractiveCalibrator`**: janela OpenCV onde o usuário clica nos 4 cantos do tabuleiro para gerar a calibração

A calibração precisa ser feita **uma única vez** por configuração de câmera.

---

### `segmentation.py` — `BoardSegmenter`
Divide a imagem corrigida em regiões individuais.

- **64 casas do tabuleiro**: `A1` até `H8`, cada uma um recorte de 80×80 px
- **Cemitérios**: 8 slots por lado (esquerdo = peças brancas capturadas, direito = pretas)
- Método `draw_grid()` sobrepõe o grid na imagem para debug visual

---

### `piece_detector.py` — `PieceDetector`
Analisa o recorte de cada casa para determinar:

1. **Ocupada ou vazia** — compara o recorte atual com a imagem de referência (tabuleiro vazio) usando diferença absoluta média (MAD). Sem referência, usa variância de pixels.
2. **Cor da peça** — analisa o canal V (brilho) do espaço HSV. Peças brancas têm brilho alto; pretas têm brilho baixo.
3. **Tipo da peça** — *não implementado ainda* (Opção B futura, requer modelo CNN).

---

### `board_state.py` — `BoardState` + `SquareState`
Estrutura de dados que representa o estado físico completo do jogo.

```python
state = BoardState(
    board={
        "E2": SquareState(occupied=True, color="white", confidence=0.96),
        "E4": SquareState(occupied=False, color=None, confidence=0.99),
    },
    cemeteries={"white": [...], "black": [...]},
    overall_confidence=0.94,
)
print(state.to_dict())   # serializa para JSON
```

---

### `state_comparator.py` — `StateComparator`
Compara dois `BoardState` (antes e depois de um movimento) e verifica:

- A casa de origem ficou vazia
- A casa de destino ficou ocupada
- A cor da peça no destino confere com a origem
- Se houve captura: a peça capturada apareceu no cemitério correto

Retorna `CompareResult` com `success`, `error`, `message` e métricas de diagnóstico.

---

### `move_validator.py` — `MoveValidator`
Orquestra o ciclo completo de validação visual de um movimento:

```
IDLE → CAMERA_CAPTURING → IMAGE_PROCESSING → BOARD_VALIDATING → CONFIRMED/FAILED
```

- Captura o estado **antes** do movimento
- Aguarda o sinal do gantry XY (ou um tempo fixo em modo de teste)
- Captura o estado **depois** do movimento
- Compara e retorna o resultado
- Repete até `MAX_CAPTURE_ATTEMPTS` em caso de falha

---

### `diag_manager.py` — `DiagManager`
Salva imagens de diagnóstico em `vision/diags/` durante o desenvolvimento.

- Controle por `SAVE_DIAGNOSTIC_IMAGES` em `config.py`
- Rotação automática: apaga os arquivos mais antigos ao atingir `MAX_DIAGNOSTIC_FILES`
- Método `save_squares()` gera uma folha de contato com todas as 64 casas anotadas

---

### `__init__.py` — `CameraSystem`
Fachada pública que conecta todos os componentes. É o único ponto de contato que outros módulos precisam conhecer.

```python
from vision import CameraSystem

cam = CameraSystem()
cam.initialize()

state = cam.capture_board_state()
result = cam.validate_move({"from": "E2", "to": "E4"})
cam.shutdown()
```

---

### `scripts/calibrate_camera.py`
Script standalone para calibração interativa. Exibe a imagem da câmera e permite clicar nos 4 cantos do tabuleiro para gerar `camera_calibration.json`.

---

## Instalação

```bash
cd Oficinas/

# Instalar dependências de visão
pip install opencv-python numpy

# Ou usando o arquivo de requirements
pip install -r vision/requirements.txt
```

> **Nota**: Use `opencv-python` para ter a interface gráfica (janela OpenCV).  
> Em produção headless, use `opencv-python-headless` (mais leve).

---

## Como Testar

### Passo 1 — Testar sem câmera (smoke test)

Verifica se todos os módulos importam e as lógicas básicas funcionam.
Não precisa de câmera física:

```bash
cd Oficinas/
python vision/smoke_test.py
```

Saída esperada:
```
All imports OK
BoardState: 64 squares, confidence=1.0
Segmented 64 squares
Cemetery slots: white=8, black=8
Analysis E2: occupied=False, color=None, conf=0.80
Compare: success=False, error=MOVE_NOT_CONFIRMED
CalibrationData round-trip: valid=True
Corrected shape: (640, 960, 3)

All smoke tests PASSED.
```

---

### Passo 2 — Calibrar a câmera (primeira vez)

Com a câmera USB conectada e o tabuleiro posicionado:

```bash
cd Oficinas/
python vision/scripts/calibrate_camera.py
```

Uma janela abrirá com a imagem da câmera. **Clique APENAS nos 4 cantos do tabuleiro 8x8 central** em ordem horária:
1. Canto superior esquerdo (próximo a A8)
2. Canto superior direito (próximo a H8)
3. Canto inferior direito (próximo a H1)
4. Canto inferior esquerdo (próximo a A1)

> **Importante**: Não inclua os cemitérios na marcação. O sistema calculará automaticamente a área dos cemitérios laterais baseando-se nas medidas do tabuleiro central.

Pressione **ENTER** para confirmar. O arquivo `vision/camera_calibration.json` será gerado.

Opções úteis:
```bash
# Câmera diferente do padrão (índice 0)
python vision/scripts/calibrate_camera.py --device 1

# Testar calibração existente com overlay do grid
python vision/scripts/calibrate_camera.py --test-only --show-grid

# Calibrar usando uma foto estática (sem câmera)
python vision/scripts/calibrate_camera.py --image foto_tabuleiro.jpg
```

---

### Passo 3 — Testar captura com câmera real

```python
# teste_camera.py (criar na pasta Oficinas/)
import sys
sys.path.insert(0, ".")

from vision import CameraSystem

cam = CameraSystem()
if cam.initialize():
    print("Câmera OK!")
    state = cam.capture_board_state()
    if state:
        print(f"Estado capturado: {len(state.occupied_squares())} casas ocupadas")
        print(f"Confiança: {state.overall_confidence:.2f}")
    cam.shutdown()
else:
    print("Falha ao inicializar câmera")
```

```bash
cd Oficinas/
python teste_camera.py
```

---

### Passo 4 — Testar validação de movimento

```python
# teste_movimento.py (criar na pasta Oficinas/)
import sys
sys.path.insert(0, ".")

from vision import CameraSystem

cam = CameraSystem()
cam.initialize()

print("Posicione a peça em E2. Pressione ENTER...")
input()

result = cam.validate_move({"from": "E2", "to": "E4"})

print(f"Sucesso: {result.success}")
print(f"Origem vazia: {result.origin_empty}")
print(f"Destino ocupado: {result.dest_occupied}")
print(f"Mensagem: {result.message}")

cam.shutdown()
```

> **Atenção**: Por padrão, o validador aguarda `POST_MOVE_SETTLE_SECONDS` (1.5 s) após ser chamado antes de capturar a imagem pós-movimento. Mova a peça fisicamente nesse intervalo ou aumente o valor em `config.py`.

---

### Passo 5 — Inspecionar imagens de diagnóstico

Com `SAVE_DIAGNOSTIC_IMAGES = True` (padrão) em `vision/config.py`, as imagens são salvas em `vision/diags/`. Cada captura gera:

| Arquivo | Conteúdo |
|---------|----------|
| `*_raw_attempt1.jpg` | Frame bruto da câmera |
| `*_corrected_attempt1.jpg` | Após correção de perspectiva |
| `reference_corrected.jpg` | Imagem de referência (tabuleiro vazio) |
| `calibration_corrected.jpg` | Preview gerado pela calibração |

Para **desligar** o salvamento em produção:
```python
# vision/config.py
SAVE_DIAGNOSTIC_IMAGES: bool = False
```

---

## Fluxo Completo

```
Iniciar
  └─► CameraSystem.initialize()
        ├─ Abre câmera (warmup 2 s)
        └─ Captura imagem de referência (tabuleiro vazio)

Capturar estado atual
  └─► CameraSystem.capture_board_state()
        ├─ Captura frame
        ├─ Corrige perspectiva
        ├─ Segmenta 64 casas + cemitérios
        └─ Detecta ocupação e cor de cada casa → BoardState

Validar movimento
  └─► CameraSystem.validate_move({"from": "E2", "to": "E4"})
        ├─ Captura estado ANTES
        ├─ Aguarda 1.5 s (futuro: sinal do gantry)
        ├─ Captura estado DEPOIS
        ├─ Compara estados
        └─ Retorna CompareResult (success, mensagem, diagnóstico)
```

---

## Próximos Passos (Opção B — Classificação de Tipo de Peça)

A classificação do tipo da peça (peão, torre, cavalo...) está reservada para uma etapa futura. A implementação prevista é:

1. Coletar dataset de imagens de cada peça sobre o tabuleiro
2. Treinar um modelo leve (MobileNetV3 ou EfficientNet-Lite) com 12 classes (6 peças × 2 cores)
3. Carregar o modelo **uma única vez** no `PieceDetector.__init__()`
4. Chamar a inferência apenas para casas detectadas como ocupadas

O campo `piece` em `SquareState` já está reservado para isso e sempre retorna `None` até que o modelo seja integrado.
