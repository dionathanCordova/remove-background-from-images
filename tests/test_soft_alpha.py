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
