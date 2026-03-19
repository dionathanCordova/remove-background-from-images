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
