"""La galería se queda con la mejor foto de la visita, salvo que se pida la última."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calidad_foto import calidad_foto, conviene_reemplazar


def _foto(alto, ancho, semilla=0):
    rng = np.random.default_rng(semilla)
    return rng.integers(0, 255, (alto, ancho, 3), dtype=np.uint8)


def test_una_foto_mas_grande_y_nitida_puntua_mas():
    assert calidad_foto(_foto(200, 100)) > calidad_foto(_foto(100, 50))


def test_una_foto_movida_pierde_contra_una_mas_chica_pero_nitida():
    grande_movida = cv2.GaussianBlur(_foto(200, 100), (31, 31), 15)
    chica_nitida = _foto(120, 60)
    assert calidad_foto(chica_nitida) > calidad_foto(grande_movida)


def test_una_foto_vacia_no_rompe():
    assert calidad_foto(np.zeros((0, 10, 3), np.uint8)) == 0.0


def test_modo_mejor_solo_reemplaza_si_no_empeora():
    assert conviene_reemplazar("mejor", None, 1.0)
    assert conviene_reemplazar("mejor", 5.0, 6.0)
    assert not conviene_reemplazar("mejor", 5.0, 4.0)


def test_modo_ultima_siempre_reemplaza():
    assert conviene_reemplazar("ultima", 5.0, 1.0)
