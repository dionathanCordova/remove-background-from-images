# Soft Alpha Matting para Efeitos Semi-Transparentes

**Data:** 2026-03-18
**Projeto:** removerBgImages
**Arquivo alvo:** `main.py`

---

## Problema

O pipeline atual (Flood Fill → GrabCut → blur gaussiano) produz uma máscara **binária**: cada pixel é totalmente opaco (255) ou totalmente transparente (0). Isso funciona bem para objetos sólidos, mas falha com efeitos semi-transparentes como:

- Fogo/chamas (laranja/vermelho com bordas feathered)
- Névoa de gelo / fumaça (azul claro, muito semi-transparente)
- Glow / halo luminoso ao redor do objeto

---

## Solução: Alpha Matting por Distância de Cor

Mantém o pipeline de detecção existente e adiciona uma etapa de **soft alpha** na zona de transição entre fundo certo e objeto certo. Nenhuma dependência nova é necessária.

---

## Pipeline Revisado

```
Passo 0 — estimate_background_color(bgr)
            → bg_color: np.ndarray shape=(3,) dtype=float32, ordem de canais BGR

Passo 1 — Flood Fill + pure_white               (mantido)
            → bg_mask: np.ndarray shape=(H,W) dtype=bool    (True = fundo certo)
            → mask_init: np.ndarray shape=(H,W) dtype=uint8 (valores GC_*)

Passo 2 — GrabCut(bgr, mask_init)               (mantido, muta mask_init in-place)
           mask_final = where(mask_init ∈ {0,2}, 0, 1).astype(uint8)
           MORPH_OPEN(mask_final, kernel=3×3, iter=1)  (mantido)
            → mask_final: np.ndarray shape=(H,W) dtype=uint8, valores 0/1, já pós-MORPH_OPEN

Passo 3 — compute_soft_alpha(bgr, mask_final, bg_mask, bg_color)
            → alpha: np.ndarray shape=(H,W) dtype=uint8 valores 0–255
            substitui completamente o GaussianBlur simples do passo 3 atual

Montagem — result[:,:,3] = alpha                (mantido)
```

---

## `estimate_background_color(bgr, border_size=20)`

**Retorno:** `np.ndarray shape=(3,) dtype=float32`, canais em ordem **BGR** (igual a `bgr`).

**Amostragem da borda:**
```
topo:     bgr[:border_size, :]          shape (border_size, W, 3)
fundo:    bgr[-border_size:, :]         shape (border_size, W, 3)
esquerda: bgr[:, :border_size, :]       shape (H, border_size, 3)
direita:  bgr[:, -border_size:, :]      shape (H, border_size, 3)
```
Todos concatenados em shape `(N, 3)` — os cantos aparecem duplicados, efeito insignificante.
Retorna `np.median(pixels, axis=0).astype(np.float32)`.

**Caso especial — fundo quadriculado:**
```python
# border_bgr_strip: concatenação de todas as 4 tiras da borda, shape (N, 3) → reshape para imagem
# Converte para escala de cinza para detectar variação de patches
border_pixels_bgr = np.concatenate([...])          # mesma concatenação descrita acima, shape (N,3)
border_gray = cv2.cvtColor(
    border_pixels_bgr.reshape(1, -1, 3),            # shape (1, N, 3)
    cv2.COLOR_BGR2GRAY                               # → shape (1, N)
).flatten()                                          # shape (N,)
if border_gray.std() > 30:
    return np.array([255.0, 255.0, 255.0], dtype=np.float32)
```
Nota: gradientes suaves também podem ter std > 30; nesse caso retornar branco é conservador e aceitável — o pipeline funciona corretamente, apenas estimando o fundo como branco.

---

## `compute_soft_alpha(bgr, coarse_mask, definite_bg_mask, bg_color)`

**Parâmetros:**
| Nome | dtype | shape | Valores | Observação |
|---|---|---|---|---|
| `bgr` | uint8 | (H,W,3) | 0–255 | canais BGR |
| `coarse_mask` | uint8 | (H,W) | 0 ou 1 | já pós-GrabCut + MORPH_OPEN |
| `definite_bg_mask` | bool | (H,W) | True/False | True = fundo certo do flood fill |
| `bg_color` | float32 | (3,) | 0.0–255.0 | canais BGR, mesma ordem que `bgr` |

**Retorno:** `np.ndarray shape=(H,W) dtype=uint8`, valores 0–255.

**Guarda:** Se `coarse_mask` for inteiramente zero (GrabCut não encontrou foreground), retorna imediatamente `np.zeros((H, W), dtype=np.uint8)`.

---

### Passo 1 — Trimap

```python
kernel = np.ones((3, 3), np.uint8)
definite_fg = cv2.erode(coarse_mask, kernel, iterations=3)   # uint8, valores 0/1
```

Prioridade (da maior para menor — quando há sobreposição, a de maior prioridade vence):

| Pri | Zona | Critério | Alpha |
|---|---|---|---|
| 1 | Fundo certo | `definite_bg_mask == True` | 0 |
| 2 | Objeto certo | `definite_fg == 1` AND `definite_bg_mask == False` | 255 |
| 3 | Transição | todos os outros pixels | calculado por cor |

A sobreposição entre prioridade 1 e 2 (pixel marcado como fundo certo E foreground erodido) é resolvida por prioridade: **fundo certo ganha, alpha = 0**.

---

### Passo 2 — Alpha na zona de transição

```python
float_bgr   = bgr.astype(np.float32)          # cast explícito; shape (H,W,3)
diff        = float_bgr - bg_color             # broadcast bg_color shape (3,) → (H,W,3)
                                               # canais alinhados: B=0, G=1, R=2 em ambos
```

Seleção da fórmula:

```python
if np.all(bg_color > 230):   # fundo claro (branco, cinza muito claro)
    alpha_float = np.max(np.abs(diff), axis=2) / 255.0
else:                        # fundo colorido ou gradiente
    alpha_float = np.sqrt(np.sum(diff ** 2, axis=2)) / (np.sqrt(3.0) * 255.0)

alpha_float = np.clip(alpha_float, 0.0, 1.0)   # aplica antes da montagem
```

O `clip` em `[0, 1]` é essencial para fundos escuros onde `diff` pode ser negativo, inflando valores se não truncado.

---

### Passo 3 — Montar alpha final

```python
transition_mask = (~definite_bg_mask) & (definite_fg == 0)   # bool (H,W)

final_alpha = np.zeros((H, W), dtype=np.float32)
# Parênteses explícitos: & tem precedência maior que == em Python/NumPy
final_alpha[(definite_fg == 1) & (~definite_bg_mask)] = 1.0  # objeto certo (P2 implica ~P1)
final_alpha[transition_mask]                          = alpha_float[transition_mask]
# definite_bg_mask permanece 0.0
# Nota: a condição (definite_fg==1) & (~definite_bg_mask) já exclui pixels de fundo certo,
# implementando a prioridade P1 implicitamente.
```

---

### Passo 4 — Suavização na fronteira

A fronteira é a região de borda ao redor das zonas certas, definida **antes** de ser usada:

```python
kernel = np.ones((3, 3), np.uint8)

# frontier_binary: pixels na borda entre zona certa e zona de transição
coarse_u8    = coarse_mask.astype(np.uint8) * 255
dilated_fg   = cv2.dilate(coarse_u8, kernel, iterations=2)   # uint8 0/255
eroded_fg    = cv2.erode(coarse_u8,  kernel, iterations=2)   # uint8 0/255
frontier_binary = ((dilated_fg - eroded_fg) > 0)             # bool (H,W)

# frontier final: dilata levemente a frontier_binary
frontier = cv2.dilate(frontier_binary.astype(np.uint8), kernel, iterations=1) > 0

# aplica blur apenas na frontier
blurred = cv2.GaussianBlur(final_alpha, (3, 3), 0)
final_alpha[frontier] = blurred[frontier]

# restaura zonas certas (blur não deve alterá-las)
final_alpha[definite_bg_mask]                           = 0.0
final_alpha[(definite_fg == 1) & (~definite_bg_mask)]   = 1.0
```

---

### Retorno

```python
return np.clip(final_alpha * 255.0, 0, 255).astype(np.uint8)
```

---

## Verificação das Fórmulas

Todos os exemplos assumem fundo branco (255, 255, 255) — fórmula "fundo claro":
`alpha_uint8 = max(|B-255|, |G-255|, |R-255|)` onde colunas do cálculo seguem ordem B, G, R.

| Efeito | Cor BGR | Cálculo \|B-255\|, \|G-255\|, \|R-255\| | alpha uint8 |
|---|---|---|---|
| Fogo núcleo | (0, 80, 255) | max(255, 175, 0) | 255 |
| Fogo borda | (100, 200, 255) | max(155, 55, 0) | 155 |
| Névoa gelo | (240, 220, 200) | max(15, 35, 55) | 55 |
| Glow amarelo | (180, 255, 255) | max(75, 0, 0) | 75 |
| Branco puro | (255, 255, 255) | max(0, 0, 0) | 0 |
| Cinza médio em fundo cinza (120,120,120) | bg=(120,120,120) → fórmula Euclidiana | dist=0 → 0 | 0 |

---

## Restrições e Trade-offs

- **Efeitos com cor igual ao fundo** → alpha 0, inevitável sem informação adicional.
- **Gradientes irregulares** → `estimate_background_color` pode estimar mal; a mediana de borda mitiga.
- **Std > 30 em gradientes suaves** → heurística xadrez pode falhar; retornar branco é conservador e não prejudica o resultado.
- **Prioridade de design:** equilíbrio entre preservar efeitos e bordas limpas.

---

## Arquivos Modificados

| Arquivo | Mudança |
|---|---|
| `main.py` | Adicionar `estimate_background_color()` e `compute_soft_alpha()`; modificar `remove_background()` |

Nenhuma dependência nova — apenas `cv2` e `numpy`.
