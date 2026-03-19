import cv2
import numpy as np
import os

def estimate_background_color(bgr, border_size=20):
    """Estima a cor do fundo amostrando a borda da imagem.

    Retorna np.ndarray shape=(3,) dtype=float32 em ordem BGR.
    Se a borda tiver alta variância (fundo quadriculado), retorna branco.
    """
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
    assert coarse_mask.max() <= 1, "coarse_mask must contain values 0/1, not 0/255"

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
    transition_mask = (~definite_bg_mask) & (definite_fg == 0) & (coarse_mask > 0)

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
    # Use maximum to preserve strong color signals (e.g. fire pixels) that blur would attenuate
    final_alpha[frontier] = np.maximum(blurred[frontier], alpha_float[frontier])

    # Restaura zonas certas
    final_alpha[definite_bg_mask] = 0.0
    final_alpha[(definite_fg == 1) & (~definite_bg_mask)] = 1.0

    return np.clip(final_alpha * 255.0, 0, 255).astype(np.uint8)


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
if __name__ == "__main__":
    input_dir  = "uploads"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    for fname in files:
        inp = os.path.join(input_dir, fname)
        out = os.path.join(output_dir, fname)
        remove_background(inp, out)
