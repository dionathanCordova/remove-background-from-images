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

Esses efeitos são cortados ou removidos completamente porque o GrabCut os classifica como fundo (cor diferente do objeto principal) ou os inclui como totalmente opacos (perdendo a semi-transparência).

---

## Solução: Alpha Matting por Distância de Cor

Mantém o pipeline de detecção existente e adiciona uma etapa de **soft alpha** na zona de transição entre fundo certo e objeto certo. Nenhuma dependência nova é necessária.

---

## Pipeline Revisado

```
Passo 0 — Estimar cor do fundo (NOVO)
Passo 1 — Máscara inicial: Flood Fill + brancos (mantido)
Passo 2 — GrabCut (mantido)
Passo 3 — Gerar Trimap: 3 zonas (NOVO, substitui blur simples)
Passo 4 — Calcular alpha suave por distância de cor (NOVO)
Montar resultado BGRA (mantido)
```

---

## Componentes

### `estimate_background_color(bgr, border_size=20) → np.ndarray`

Amostra os pixels da borda (20px ao redor da imagem) e retorna a **mediana** como cor de fundo estimada. A mediana é robusta contra cantos que possam conter parte do objeto.

**Caso especial — fundo quadriculado:** Detectado verificando se há variação de patches claros/escuros (desvio padrão local > 30) nas bordas. Se detectado, retorna `(255, 255, 255)` — tratado como branco para fins de cálculo de alpha.

---

### `compute_soft_alpha(bgr, coarse_mask, definite_bg_mask, bg_color) → np.ndarray`

Gera o canal alpha final com semi-transparência preservada.

**Entrada:**
- `bgr`: imagem original
- `coarse_mask`: máscara binária do GrabCut (0=fundo, 1=objeto)
- `definite_bg_mask`: máscara booleana do flood fill (fundo certo)
- `bg_color`: cor do fundo estimada pelo Passo 0

**Trimap — 3 zonas:**

| Zona | Critério | Alpha |
|---|---|---|
| Fundo certo | `definite_bg_mask == True` | 0 |
| Objeto certo | `erosão(coarse_mask, iter=3)` | 255 |
| Transição | Tudo entre os dois | calculado por cor |

**Cálculo do alpha na zona de transição:**

Para fundo claro (`bg_color > 230` em todos os canais):
```
alpha = max(|R - bg_R|, |G - bg_G|, |B - bg_B|) / 255
```

Para fundos coloridos / gradiente:
```
dist  = sqrt((R-bg_R)² + (G-bg_G)² + (B-bg_B)²)
alpha = dist / (sqrt(3) × 255)
```

**Pós-processamento:** Gaussian blur 3×3 apenas na fronteira entre zonas para suavizar a transição sem afetar o interior do objeto.

---

## Fluxo de Dados em `remove_background()`

```
img (BGRA)
  └─ bgr = img[:,:,:3]
       ├─ bg_color = estimate_background_color(bgr)
       ├─ mask_init, bg_mask = [flood fill + pure white detection]
       ├─ GrabCut(bgr, mask_init) → coarse_mask
       └─ alpha = compute_soft_alpha(bgr, coarse_mask, bg_mask, bg_color)
            └─ result[:,:,3] = alpha → salva PNG
```

---

## Comportamento Esperado por Tipo de Efeito

| Efeito | Cor típica | bg branco | Alpha esperado |
|---|---|---|---|
| Fogo (núcleo) | (255, 80, 0) | sim | ~255 (opaco) |
| Fogo (borda) | (255, 200, 100) | sim | ~200 (semi) |
| Névoa de gelo | (200, 220, 240) | sim | ~55 (translúcido) |
| Glow amarelo | (255, 255, 180) | sim | ~75 (translúcido) |
| Pixel branco puro | (255, 255, 255) | sim | 0 (transparente) |

---

## Restrições e Trade-offs

- **Efeitos com cor igual ao fundo:** Se o efeito tiver a mesma cor do fundo (ex: névoa branca sobre fundo branco), será transparente — isso é matematicamente inevitável sem informação adicional.
- **Fundo muito heterogêneo:** Gradientes muito irregulares podem causar estimativa imprecisa de `bg_color`. A mediana sobre 20px de borda mitiga mas não elimina o problema.
- **Prioridade:** Equilíbrio entre preservar efeitos e ter bordas limpas (não prioriza um extremo).

---

## Arquivos Modificados

| Arquivo | Tipo de mudança |
|---|---|
| `main.py` | Adicionar 2 funções; modificar `remove_background()` |

Nenhuma dependência nova. Requer apenas `cv2` e `numpy` (já instalados).
