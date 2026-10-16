"""Puntajes REBA por segmento y total."""

import pytest

from ergonomia.esqueleto import Angulos
from ergonomia.reba import (Condiciones, calcular_reba, nivel_riesgo, puntaje_antebrazo,
                            puntaje_brazo, puntaje_piernas, puntaje_tronco)


@pytest.mark.parametrize("angulo,esperado", [(0, 1), (15, 2), (-15, 2), (40, 3), (-30, 3), (70, 4)])
def test_tronco(angulo, esperado):
    assert puntaje_tronco(angulo) == esperado


@pytest.mark.parametrize("angulo,esperado", [(0, 1), (30, 2), (-30, 2), (60, 3), (120, 4)])
def test_brazo(angulo, esperado):
    assert puntaje_brazo(angulo) == esperado


def test_antebrazo_y_piernas():
    assert puntaje_antebrazo(80) == 1
    assert puntaje_antebrazo(30) == 2
    assert puntaje_piernas(10, False) == 1
    assert puntaje_piernas(45, False) == 2
    assert puntaje_piernas(70, True) == 4


def test_postura_neutra_es_riesgo_inapreciable():
    r = calcular_reba(Angulos(tronco=0, cuello=5, brazo=10, antebrazo=80, rodilla=0), Condiciones())
    assert (r.puntaje, r.nivel) == (1, "inapreciable")


def test_postura_conocida():
    # Tronco 30° (3), cuello 25° (2), piernas 1 -> A=4; brazo 60° (3),
    # antebrazo 80° (1), muñeca 1 -> B=3; tabla C[4][3] = 4.
    a = Angulos(tronco=30, cuello=25, brazo=60, antebrazo=80, rodilla=0)
    assert calcular_reba(a, Condiciones()).puntaje == 4
    # Con 12 kg (+2 en A) y agarre malo (+2 en B): C[6][5] = 8, más 1 de actividad.
    r = calcular_reba(a, Condiciones(carga_kg=12, agarre="malo"), actividad=1)
    assert (r.puntaje, r.nivel) == (9, "alto")


def test_sin_tronco_no_hay_puntaje():
    assert calcular_reba(Angulos(brazo=10), Condiciones()) is None


def test_niveles():
    assert [nivel_riesgo(p) for p in (1, 3, 7, 10, 11)] == ["inapreciable", "bajo", "medio", "alto", "muy alto"]
