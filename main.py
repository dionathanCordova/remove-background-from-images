import cv2
import numpy as np
import os

def remove_background(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load {input_path}")
        return

    # Converte para BGRA (adiciona canal alpha) se necessário
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    h, w = img.shape[:2]
    bgr = img[:, :, :3].copy()
    b_ch = bgr[:, :, 0].astype(np.int32)
    g_ch = bgr[:, :, 1].astype(np.int32)
    r_ch = bgr[:, :, 2].astype(np.int32)

    # ── PASSO 1: Flood Fill com FLOODFILL_FIXED_RANGE ──────────────────────────
    # Compara cada pixel com a COR DA SEMENTE (não com o vizinho).
    # Seeds densas a cada 4px nas 4 bordas.
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    flood_img = bgr.copy()

    lo, hi = 35, 35
    flags = (
        4
        | cv2.FLOODFILL_FIXED_RANGE
        | cv2.FLOODFILL_MASK_ONLY
        | (255 << 8)
    )

    step = 4
    for x in range(0, w, step):
        cv2.floodFill(flood_img, mask_ff, (x, 0),     0, loDiff=(lo,lo,lo), upDiff=(hi,hi,hi), flags=flags)
        cv2.floodFill(flood_img, mask_ff, (x, h - 1), 0, loDiff=(lo,lo,lo), upDiff=(hi,hi,hi), flags=flags)
    for y in range(0, h, step):
        cv2.floodFill(flood_img, mask_ff, (0,     y), 0, loDiff=(lo,lo,lo), upDiff=(hi,hi,hi), flags=flags)
        cv2.floodFill(flood_img, mask_ff, (w - 1, y), 0, loDiff=(lo,lo,lo), upDiff=(hi,hi,hi), flags=flags)

    bg_flood = mask_ff[1:-1, 1:-1]

    # ── PASSO 2: Remoção de branco puro residual ────────────────────────────────
    # Pixels com TODOS os canais > 245 são fundo branco com certeza.
    # Isso captura:
    #   - Regiões brancas internas NÃO alcançadas pelo flood fill (ex: espaço
    #     entre itens que se cruzam, como as manoplas em X)
    #   - Pontos isolados próximos às bordas do item
    #
    # SEGURANÇA: pixels de efeitos mágicos (faíscas, brilhos coloridos) têm
    # saturação ou matiz suficientes para NÃO atingir esse threshold (>245 em
    # todos os canais simultaneamente). Ex: uma faísca azul-cyan tem B alto mas
    # R e G menores → não é removida.
    pure_white = (
        (b_ch > 245) & (g_ch > 245) & (r_ch > 245)
    ).astype(np.uint8) * 255

    # ── PASSO 3: Combinação e limpeza morfológica ───────────────────────────────
    combined_bg = cv2.bitwise_or(bg_flood, pure_white)

    kernel = np.ones((5, 5), np.uint8)
    combined_bg = cv2.morphologyEx(combined_bg, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── PASSO 4: Máscara do item ────────────────────────────────────────────────
    item_mask = cv2.bitwise_not(combined_bg)

    # ── PASSO 5: Suavizar bordas (anti-aliasing leve) ───────────────────────────
    alpha_smooth = cv2.GaussianBlur(item_mask.astype(np.float32), (3, 3), 0)
    alpha_final  = np.clip(alpha_smooth, 0, 255).astype(np.uint8)

    result = img.copy()
    result[:, :, 3] = alpha_final
    cv2.imwrite(output_path, result)
    print(f"Saved: {os.path.basename(output_path)}")


# ── USO ───────────────────────────────────────────────────────────────────────
input_dir  = "uploads"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
for fname in files:
    inp = os.path.join(input_dir, fname)
    out = os.path.join(output_dir, fname)
    remove_background(inp, out)