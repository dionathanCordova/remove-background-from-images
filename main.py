import cv2
import numpy as np
import os

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


def remove_background(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load {input_path}")
        return

    # Converte para BGRA se necessário
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    h, w = img.shape[:2]
    bgr = img[:, :, :3].copy()

    # ── PASSO 1: Máscara Inicial (Flood Fill + Brancos) ──────────────────────────
    # Criamos uma máscara inicial para "ajudar" o GrabCut
    mask_init = np.zeros((h, w), np.uint8)
    
    # Detecção de fundo branco PURO (mais restrito para não pegar cordões cinzas)
    pure_white = (bgr[:,:,0] > 248) & (bgr[:,:,1] > 248) & (bgr[:,:,2] > 248)
    mask_init[pure_white] = cv2.GC_PR_BGD  # Provável Fundo

    # Flood Fill nas bordas (tolerância menor para não invadir o objeto)
    temp_mask = np.zeros((h + 2, w + 2), np.uint8)
    # Tolerância de cor reduzida (de 10 para 4) para ser mais conservador
    diff = (4, 4, 4)
    for x in [0, w-1]:
        for y in range(h):
            if temp_mask[y+1, x+1] == 0:
                cv2.floodFill(bgr, temp_mask, (x, y), 255, diff, diff, 4 | cv2.FLOODFILL_MASK_ONLY)
    for y in [0, h-1]:
        for x in range(w):
            if temp_mask[y+1, x+1] == 0:
                cv2.floodFill(bgr, temp_mask, (x, y), 255, diff, diff, 4 | cv2.FLOODFILL_MASK_ONLY)
    
    bg_mask = temp_mask[1:-1, 1:-1] > 0
    mask_init[bg_mask] = cv2.GC_BGD  # Fundo com certeza

    # Área central maior como "objeto com certeza" para proteger cordões internos
    center_rect = (int(w*0.05), int(h*0.05), int(w*0.9), int(h*0.9))
    cv2.rectangle(mask_init, (center_rect[0], center_rect[1]), 
                 (center_rect[0]+center_rect[2], center_rect[1]+center_rect[3]), 
                 cv2.GC_PR_FGD, -1)

    # ── PASSO 2: GrabCut ────────────────────────────────────────────────────────
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Rodamos o GrabCut usando a máscara inicial
    cv2.grabCut(bgr, mask_init, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # Máscara final: 0 e 2 são fundo, 1 e 3 são objeto
    mask_final = np.where((mask_init == 2) | (mask_init == 0), 0, 1).astype('uint8')

    # ── PASSO 3: Refinamento de bordas ──────────────────────────────────────────
    # Limpeza morfológica para remover ruídos
    kernel = np.ones((3, 3), np.uint8)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_final = cv2.GaussianBlur(mask_final.astype(np.float32) * 255, (3, 3), 0)
    
    alpha = np.clip(mask_final, 0, 255).astype(np.uint8)

    # Montagem do resultado
    result = bgr.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = alpha

    cv2.imwrite(output_path, result)
    print(f"Saved (Improved): {os.path.basename(output_path)}")


# ── USO ───────────────────────────────────────────────────────────────────────
input_dir  = "uploads"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
for fname in files:
    inp = os.path.join(input_dir, fname)
    out = os.path.join(output_dir, fname)
    remove_background(inp, out)