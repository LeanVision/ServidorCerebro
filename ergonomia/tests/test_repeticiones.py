"""Conteo de ciclos y nivel de actividad manual."""

import numpy as np
import pytest

from ergonomia.repeticiones import analizar_repeticiones, nam_radwin


def _seno(frecuencia_hz, segundos=60, fps=10, ruido=0.0, amplitud=40.0):
    t = np.arange(int(segundos * fps)) / fps
    rng = np.random.default_rng(0)
    return list(80 + amplitud * np.sin(2 * np.pi * frecuencia_hz * t) + rng.normal(0, ruido, len(t)))


def test_cuenta_los_ciclos_aunque_haya_ruido():
    r = analizar_repeticiones("codo", _seno(0.5, ruido=4.0), fps=10)
    assert r.ciclos == pytest.approx(30, abs=1)
    assert r.ciclos_por_minuto == pytest.approx(30, abs=1)
    assert r.segundos_por_ciclo == pytest.approx(2.0, abs=0.1)


def test_un_temblor_chico_no_son_ciclos():
    r = analizar_repeticiones("codo", _seno(1.0, amplitud=3.0), fps=10)
    assert r.ciclos == 0
    assert r.nam_estimado == 0.0


def test_tolera_cuadros_sin_postura():
    angulos = _seno(0.5)
    for i in range(0, len(angulos), 7):
        angulos[i] = None
    assert analizar_repeticiones("codo", angulos, fps=10).ciclos == pytest.approx(30, abs=1)


def test_nam_crece_con_frecuencia_y_movimiento():
    assert nam_radwin(0.0, 0.5) == 0.0
    assert nam_radwin(0.5, 0.5) < nam_radwin(1.0, 0.5) < nam_radwin(1.0, 0.9) <= 10.0
