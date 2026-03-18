# Soft Alpha Matting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adicionar soft alpha matting ao pipeline de remoção de fundo para preservar efeitos semi-transparentes como fogo, névoa de gelo e glow.

**Architecture:** Duas novas funções (`estimate_background_color` e `compute_soft_alpha`) são adicionadas ao `main.py`. A função `remove_background` é modificada para chamar essas funções, substituindo o GaussianBlur simples da etapa final por um cálculo de alpha proporcional à distância de cor em relação ao fundo.

**Tech Stack:** Python 3.12, OpenCV 4.13 (`cv2`), NumPy 2.4 — sem dependências novas.

---

## File Map

| Arquivo | Ação | Responsabilidade |
|---|---|---|
| `main.py` | Modificar | Adicionar 2 funções; ajustar `remove_background` |
| `tests/test_soft_alpha.py` | Criar | Testes unitários das 2 novas funções |
| `requirements.txt` | Criar | Pinnar dependências + pytest |

---

## Setup

### Task 0: Ambiente de testes

**Files:**
- Create: `requirements.txt`
- Create: `tests/test_soft_alpha.py` (arquivo vazio por enquanto)

- [ ] **Step 1: Criar requirements.txt com dependências e pytest**

```
numpy==2.4.3
opencv-python==4.13.0.92
pytest>=8.0
```

Salvar em `requirements.txt` na raiz do projeto.

- [ ] **Step 2: Instalar pytest no venv**

```bash
source venv/bin/activate
pip install pytest
```

Esperado: `Successfully installed pytest-...`

- [ ] **Step 3: Criar diretório de testes e arquivo vazio**

```bash
mkdir -p tests
touch tests/__init__.py
touch tests/test_soft_alpha.py
```

- [ ] **Step 4: Verificar que pytest encontra o projeto**

```bash
source venv/bin/activate && python -m pytest tests/ --collect-only
```

Esperado: `no tests ran` (ainda sem testes, mas sem erros de importação).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/
git commit -m "chore: add pytest and test scaffold"
```

---

## Task 1: `estimate_background_color`

**Files:**
- Modify: `tests/test_soft_alpha.py`
- Modify: `main.py`

### 1.1 — Testes para `estimate_background_color`

- [ ] **Step 1: Escrever os testes**

Adicionar ao `tests/test_soft_alpha.py`:

```python
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import estimate_background_color


def make_solid_image(h, w, bgr_color):
    """Cria imagem BGR sólida com a cor dada."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bgr_color
    return img


def test_solid_white_returns_white():
    img = make_solid_image(100, 100, (255, 255, 255))
    result = estimate_background_color(img)
    np.testing.assert_array_almost_equal(result, [255.0, 255.0, 255.0], decimal=1)


def test_solid_gray_returns_gray():
    img = make_solid_image(100, 100, (128, 128, 128))
    result = estimate_background_color(img)
    np.testing.assert_array_almost_equal(result, [128.0, 128.0, 128.0], decimal=1)


def test_solid_color_returns_that_color():
    img = make_solid_image(100, 100, (50, 120, 200))
    result = estimate_background_color(img)
    np.testing.assert_array_almost_equal(result, [50.0, 120.0, 200.0], decimal=1)


def test_checkerboard_border_returns_white():
    """Borda com alta variância (simula xadrez) → retorna branco."""
    img = make_solid_image(100, 100, (200, 200, 200))
    # Alterna pixels pretos e brancos na borda
    for i in range(20):
        for j in range(100):
            if (i + j) % 2 == 0:
                img[i, j] = (0, 0, 0)
                img[99 - i, j] = (0, 0, 0)
                img[j, i] = (0, 0, 0)
                img[j, 99 - i] = (0, 0, 0)
    result = estimate_background_color(img)
    np.testing.assert_array_equal(result, [255.0, 255.0, 255.0])


def test_returns_float32():
    img = make_solid_image(100, 100, (100, 100, 100))
    result = estimate_background_color(img)
    assert result.dtype == np.float32


def test_returns_shape_3():
    img = make_solid_image(100, 100, (100, 100, 100))
    result = estimate_background_color(img)
    assert result.shape == (3,)
```

- [ ] **Step 2: Rodar testes — confirmar que falham**

```bash
source venv/bin/activate && python -m pytest tests/test_soft_alpha.py -v -k "estimate"
```

Esperado: `ImportError` ou `AttributeError: module 'main' has no attribute 'estimate_background_color'`

- [ ] **Step 3: Implementar `estimate_background_color` em `main.py`**

Adicionar **antes** da função `remove_background` existente:

```python
def estimate_background_color(bgr, border_size=20):
    """Estima a cor do fundo amostrando a borda da imagem.

    Retorna np.ndarray shape=(3,) dtype=float32 em ordem BGR.
    Se a borda tiver alta variância (fundo quadriculado), retorna branco.
    """
    top    = bgr[:border_size, :].reshape(-1, 3)
    bottom = bgr[-border_size:, :].reshape(-1, 3)
    left   = bgr[:, :border_size, :].reshape(-1, 3)
    right  = bgr[:, -border_size:, :].reshape(-1, 3)

    border_pixels = np.concatenate([top, bottom, left, right], axis=0)  # shape (N, 3)

    import cv2 as _cv2
    border_gray = _cv2.cvtColor(
        border_pixels.reshape(1, -1, 3),
        _cv2.COLOR_BGR2GRAY
    ).flatten()

    if border_gray.std() > 30:
        return np.array([255.0, 255.0, 255.0], dtype=np.float32)

    return np.median(border_pixels, axis=0).astype(np.float32)
```

> Nota: `import cv2 as _cv2` dentro da função não é necessário — `cv2` já está importado no topo de `main.py`. Use apenas `cv2` diretamente.

A versão final correta (usando o `cv2` já importado):

```python
def estimate_background_color(bgr, border_size=20):
    top    = bgr[:border_size, :].reshape(-1, 3)
    bottom = bgr[-border_size:, :].reshape(-1, 3)
    left   = bgr[:, :border_size, :].reshape(-1, 3)
    right  = bgr[:, -border_size:, :].reshape(-1, 3)

    border_pixels = np.concatenate([top, bottom, left, right], axis=0)

    border_gray = cv2.cvtColor(
        border_pixels.reshape(1, -1, 3),
        cv2.COLOR_BGR2GRAY
    ).flatten()

    if border_gray.std() > 30:
        return np.array([255.0, 255.0, 255.0], dtype=np.float32)

    return np.median(border_pixels, axis=0).astype(np.float32)
```

- [ ] **Step 4: Rodar testes — confirmar que passam**

```bash
source venv/bin/activate && python -m pytest tests/test_soft_alpha.py -v -k "estimate"
```

Esperado: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_soft_alpha.py
git commit -m "feat: add estimate_background_color with tests"
```

---

## Task 2: `compute_soft_alpha`

**Files:**
- Modify: `tests/test_soft_alpha.py`
- Modify: `main.py`

### 2.1 — Testes para `compute_soft_alpha`

- [ ] **Step 1: Escrever os testes**

Adicionar ao `tests/test_soft_alpha.py` (após os testes existentes):

```python
import cv2
from main import compute_soft_alpha


def make_test_inputs(h=100, w=100, bg_color=(255, 255, 255)):
    """Helper: retorna bgr, coarse_mask, definite_bg_mask, bg_color para testes."""
    bgr = np.full((h, w, 3), bg_color, dtype=np.uint8)
    coarse_mask = np.zeros((h, w), dtype=np.uint8)
    definite_bg_mask = np.zeros((h, w), dtype=bool)
    bg = np.array(bg_color, dtype=np.float32)
    return bgr, coarse_mask, definite_bg_mask, bg


def test_guard_all_zero_coarse_mask_returns_zeros():
    bgr, coarse_mask, definite_bg_mask, bg = make_test_inputs()
    # coarse_mask é tudo zero → guarda retorna zeros
    result = compute_soft_alpha(bgr, coarse_mask, definite_bg_mask, bg)
    assert result.dtype == np.uint8
    assert result.shape == (100, 100)
    assert np.all(result == 0)


def test_definite_bg_pixels_are_transparent():
    bgr, coarse_mask, definite_bg_mask, bg = make_test_inputs()
    coarse_mask[40:60, 40:60] = 1        # objeto no centro
    definite_bg_mask[0:10, :] = True      # fundo certo no topo
    result = compute_soft_alpha(bgr, coarse_mask, definite_bg_mask, bg)
    # Pixels de fundo certo devem ser 0
    assert np.all(result[0:10, :] == 0)


def test_definite_fg_pixels_are_opaque():
    """Pixels no núcleo do objeto (pós-erosão) devem ter alpha 255."""
    h, w = 100, 100
    # Objeto grande para que erosão(iter=3) ainda tenha pixels
    bgr = np.full((h, w, 3), (0, 0, 255), dtype=np.uint8)  # vermelho (fire)
    coarse_mask = np.zeros((h, w), dtype=np.uint8)
    coarse_mask[20:80, 20:80] = 1   # objeto 60×60 — erosão 3px deixa núcleo 54×54
    definite_bg_mask = np.zeros((h, w), dtype=bool)
    bg = np.array([255.0, 255.0, 255.0], dtype=np.float32)

    result = compute_soft_alpha(bgr, coarse_mask, definite_bg_mask, bg)
    # Centro do objeto deve ser opaco
    assert result[50, 50] == 255


def test_fire_pixel_in_transition_gets_high_alpha():
    """Pixel de fogo (0, 80, 255) em zona de transição com fundo branco → alpha alto."""
    h, w = 60, 60
    bgr = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)  # fundo branco
    # Pixel de fogo na borda do objeto
    bgr[30, 30] = (0, 80, 255)  # B=0, G=80, R=255 → |0-255|=255 → alpha=255

    coarse_mask = np.zeros((h, w), dtype=np.uint8)
    coarse_mask[28:33, 28:33] = 1   # objeto pequeno — pixel 30,30 está na transição

    definite_bg_mask = np.zeros((h, w), dtype=bool)
    bg = np.array([255.0, 255.0, 255.0], dtype=np.float32)

    result = compute_soft_alpha(bgr, coarse_mask, definite_bg_mask, bg)
    # Pixel de fogo deve ter alpha alto (≥ 200)
    assert result[30, 30] >= 200, f"Expected alpha >= 200, got {result[30, 30]}"


def test_near_white_pixel_in_transition_gets_low_alpha():
    """Pixel quase-branco (240, 240, 240) em zona de transição → alpha baixo."""
    h, w = 60, 60
    bgr = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)
    bgr[30, 30] = (240, 240, 240)  # max(15,15,15) = 15 → alpha ~15

    coarse_mask = np.zeros((h, w), dtype=np.uint8)
    coarse_mask[28:33, 28:33] = 1

    definite_bg_mask = np.zeros((h, w), dtype=bool)
    bg = np.array([255.0, 255.0, 255.0], dtype=np.float32)

    result = compute_soft_alpha(bgr, coarse_mask, definite_bg_mask, bg)
    assert result[30, 30] <= 30, f"Expected alpha <= 30, got {result[30, 30]}"


def test_returns_uint8():
    bgr, coarse_mask, definite_bg_mask, bg = make_test_inputs()
    coarse_mask[40:60, 40:60] = 1
    result = compute_soft_alpha(bgr, coarse_mask, definite_bg_mask, bg)
    assert result.dtype == np.uint8


def test_returns_correct_shape():
    bgr, coarse_mask, definite_bg_mask, bg = make_test_inputs(80, 120)
    coarse_mask[30:50, 50:70] = 1
    result = compute_soft_alpha(bgr, coarse_mask, definite_bg_mask, bg)
    assert result.shape == (80, 120)
```

- [ ] **Step 2: Rodar testes — confirmar que falham**

```bash
source venv/bin/activate && python -m pytest tests/test_soft_alpha.py -v -k "compute"
```

Esperado: `ImportError` ou `AttributeError: module 'main' has no attribute 'compute_soft_alpha'`

- [ ] **Step 3: Implementar `compute_soft_alpha` em `main.py`**

Adicionar **após** `estimate_background_color` e **antes** de `remove_background`:

```python
def compute_soft_alpha(bgr, coarse_mask, definite_bg_mask, bg_color):
    """Gera canal alpha com semi-transparência para efeitos (fogo, névoa, glow).

    Parâmetros:
        bgr              : uint8  (H,W,3)  — imagem BGR
        coarse_mask      : uint8  (H,W)    — máscara 0/1 pós-GrabCut+MORPH_OPEN
        definite_bg_mask : bool   (H,W)    — True = fundo certo (flood fill)
        bg_color         : float32 (3,)    — cor do fundo estimada, ordem BGR

    Retorno: uint8 (H,W), valores 0–255.
    """
    h, w = bgr.shape[:2]

    # Guarda: sem foreground detectado
    if not np.any(coarse_mask):
        return np.zeros((h, w), dtype=np.uint8)

    # ── Passo 1: Trimap ──────────────────────────────────────────────────────
    kernel = np.ones((3, 3), np.uint8)
    definite_fg = cv2.erode(coarse_mask, kernel, iterations=3)  # uint8, 0/1

    # ── Passo 2: Alpha na zona de transição ──────────────────────────────────
    float_bgr = bgr.astype(np.float32)
    diff = float_bgr - bg_color  # broadcast (3,) → (H,W,3); B=0, G=1, R=2

    if np.all(bg_color > 230):
        alpha_float = np.max(np.abs(diff), axis=2) / 255.0
    else:
        alpha_float = np.sqrt(np.sum(diff ** 2, axis=2)) / (np.sqrt(3.0) * 255.0)

    alpha_float = np.clip(alpha_float, 0.0, 1.0)

    # ── Passo 3: Montar alpha final ───────────────────────────────────────────
    transition_mask = (~definite_bg_mask) & (definite_fg == 0)

    final_alpha = np.zeros((h, w), dtype=np.float32)
    final_alpha[(definite_fg == 1) & (~definite_bg_mask)] = 1.0
    final_alpha[transition_mask] = alpha_float[transition_mask]
    # definite_bg_mask permanece 0.0

    # ── Passo 4: Suavização na fronteira ─────────────────────────────────────
    coarse_u8 = coarse_mask.astype(np.uint8) * 255
    dilated_fg = cv2.dilate(coarse_u8, kernel, iterations=2)
    eroded_fg  = cv2.erode(coarse_u8,  kernel, iterations=2)
    frontier_binary = ((dilated_fg.astype(np.int16) - eroded_fg.astype(np.int16)) > 0)

    frontier = cv2.dilate(frontier_binary.astype(np.uint8), kernel, iterations=1) > 0

    blurred = cv2.GaussianBlur(final_alpha, (3, 3), 0)
    final_alpha[frontier] = blurred[frontier]

    # Restaura zonas certas
    final_alpha[definite_bg_mask] = 0.0
    final_alpha[(definite_fg == 1) & (~definite_bg_mask)] = 1.0

    return np.clip(final_alpha * 255.0, 0, 255).astype(np.uint8)
```

> **Nota importante:** `dilated_fg - eroded_fg` com uint8 pode sofrer underflow. Por isso o cast para `int16` antes da subtração garante resultado correto.

- [ ] **Step 4: Rodar testes — confirmar que passam**

```bash
source venv/bin/activate && python -m pytest tests/test_soft_alpha.py -v -k "compute"
```

Esperado: `7 passed`

- [ ] **Step 5: Rodar todos os testes**

```bash
source venv/bin/activate && python -m pytest tests/ -v
```

Esperado: todos os testes passando (`13 passed` no total).

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_soft_alpha.py
git commit -m "feat: add compute_soft_alpha with tests"
```

---

## Task 3: Integração em `remove_background`

**Files:**
- Modify: `main.py` (função `remove_background`)

- [ ] **Step 1: Localizar os dois pontos de mudança em `remove_background`**

No `main.py` atual, a função `remove_background` tem:
1. Linhas que extraem `bgr` da imagem — adicionar chamada a `estimate_background_color` logo após
2. A linha `bg_mask = temp_mask[1:-1, 1:-1] > 0` — preservar essa variável (será passada a `compute_soft_alpha`)
3. O bloco `# ── PASSO 3: Refinamento de bordas` — substituir inteiramente por `compute_soft_alpha`

- [ ] **Step 2: Modificar `remove_background`**

Substituir o início da função (após `bgr = img[:, :, :3].copy()`) para adicionar `bg_color`:

```python
    bgr = img[:, :, :3].copy()

    # ── PASSO 0: Estimativa da cor do fundo ────────────────────────────────
    bg_color = estimate_background_color(bgr)
```

Garantir que `bg_mask` seja convertida para `bool` após o flood fill:

```python
    bg_mask = temp_mask[1:-1, 1:-1] > 0   # bool (H,W) — fundo certo
```

(Esta linha já existe — verificar que o tipo é bool, não uint8. A expressão `> 0` já produz bool em NumPy.)

Substituir o **PASSO 3** completo:

**Remover** (linhas ~58–64):
```python
    # ── PASSO 3: Refinamento de bordas ──────────────────────────────────────
    # Limpeza morfológica para remover ruídos
    kernel = np.ones((3, 3), np.uint8)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_final = cv2.GaussianBlur(mask_final.astype(np.float32) * 255, (3, 3), 0)

    alpha = np.clip(mask_final, 0, 255).astype(np.uint8)
```

**Substituir por:**
```python
    # ── PASSO 3: Alpha suave preservando efeitos semi-transparentes ─────────
    kernel = np.ones((3, 3), np.uint8)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel, iterations=1)

    alpha = compute_soft_alpha(bgr, mask_final, bg_mask, bg_color)
```

A função `remove_background` completa final deve ficar:

```python
def remove_background(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load {input_path}")
        return

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    h, w = img.shape[:2]
    bgr = img[:, :, :3].copy()

    # ── PASSO 0: Estimativa da cor do fundo ────────────────────────────────
    bg_color = estimate_background_color(bgr)

    # ── PASSO 1: Máscara Inicial (Flood Fill + Brancos) ─────────────────────
    mask_init = np.zeros((h, w), np.uint8)

    pure_white = (bgr[:,:,0] > 248) & (bgr[:,:,1] > 248) & (bgr[:,:,2] > 248)
    mask_init[pure_white] = cv2.GC_PR_BGD

    temp_mask = np.zeros((h + 2, w + 2), np.uint8)
    diff = (4, 4, 4)
    for x in [0, w-1]:
        for y in range(h):
            if temp_mask[y+1, x+1] == 0:
                cv2.floodFill(bgr, temp_mask, (x, y), 255, diff, diff, 4 | cv2.FLOODFILL_MASK_ONLY)
    for y in [0, h-1]:
        for x in range(w):
            if temp_mask[y+1, x+1] == 0:
                cv2.floodFill(bgr, temp_mask, (x, y), 255, diff, diff, 4 | cv2.FLOODFILL_MASK_ONLY)

    bg_mask = temp_mask[1:-1, 1:-1] > 0   # bool (H,W)
    mask_init[bg_mask] = cv2.GC_BGD

    center_rect = (int(w*0.05), int(h*0.05), int(w*0.9), int(h*0.9))
    cv2.rectangle(mask_init, (center_rect[0], center_rect[1]),
                 (center_rect[0]+center_rect[2], center_rect[1]+center_rect[3]),
                 cv2.GC_PR_FGD, -1)

    # ── PASSO 2: GrabCut ────────────────────────────────────────────────────
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask_init, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    mask_final = np.where((mask_init == 2) | (mask_init == 0), 0, 1).astype('uint8')

    # ── PASSO 3: Alpha suave preservando efeitos semi-transparentes ─────────
    kernel = np.ones((3, 3), np.uint8)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel, iterations=1)

    alpha = compute_soft_alpha(bgr, mask_final, bg_mask, bg_color)

    # Montagem do resultado
    result = bgr.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = alpha

    cv2.imwrite(output_path, result)
    print(f"Saved: {os.path.basename(output_path)}")
```

- [ ] **Step 3: Rodar todos os testes para confirmar que nada quebrou**

```bash
source venv/bin/activate && python -m pytest tests/ -v
```

Esperado: todos passando.

- [ ] **Step 4: Testar manualmente com uma imagem real**

```bash
source venv/bin/activate && python main.py
```

Verificar as imagens em `outputs/` — os efeitos semi-transparentes devem aparecer preservados.

- [ ] **Step 5: Commit final**

```bash
git add main.py
git commit -m "feat: integrate soft alpha matting into remove_background pipeline"
```

---

## Verificação Final

```bash
source venv/bin/activate && python -m pytest tests/ -v
```

Todos os testes devem passar. A saída esperada é algo como:

```
tests/test_soft_alpha.py::test_solid_white_returns_white PASSED
tests/test_soft_alpha.py::test_solid_gray_returns_gray PASSED
tests/test_soft_alpha.py::test_solid_color_returns_that_color PASSED
tests/test_soft_alpha.py::test_checkerboard_border_returns_white PASSED
tests/test_soft_alpha.py::test_returns_float32 PASSED
tests/test_soft_alpha.py::test_returns_shape_3 PASSED
tests/test_soft_alpha.py::test_guard_all_zero_coarse_mask_returns_zeros PASSED
tests/test_soft_alpha.py::test_definite_bg_pixels_are_transparent PASSED
tests/test_soft_alpha.py::test_definite_fg_pixels_are_opaque PASSED
tests/test_soft_alpha.py::test_fire_pixel_in_transition_gets_high_alpha PASSED
tests/test_soft_alpha.py::test_near_white_pixel_in_transition_gets_low_alpha PASSED
tests/test_soft_alpha.py::test_returns_uint8 PASSED
tests/test_soft_alpha.py::test_returns_correct_shape PASSED
13 passed
```
