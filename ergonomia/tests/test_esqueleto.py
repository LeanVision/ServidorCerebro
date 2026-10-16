"""Los ángulos salen bien de un esqueleto de perfil, mire hacia donde mire."""

import pytest
from esqueletos import perfil

from ergonomia.esqueleto import calcular_angulos, elegir_lado


@pytest.mark.parametrize("mira_derecha", [True, False])
def test_recupera_los_angulos_de_la_postura(mira_derecha):
    e = perfil(tronco=30, cuello=10, brazo=60, codo=80, muneca=25, rodilla=40, mira_derecha=mira_derecha)
    a = calcular_angulos(e, "der")
    assert a.tronco == pytest.approx(30, abs=0.5)
    assert a.cuello == pytest.approx(10, abs=0.5)
    assert a.brazo == pytest.approx(60, abs=0.5)
    assert a.antebrazo == pytest.approx(80, abs=0.5)
    assert a.muneca == pytest.approx(25, abs=0.5)
    assert a.rodilla == pytest.approx(40, abs=0.5)
    assert a.sentado is False


@pytest.mark.parametrize("mira_derecha", [True, False])
def test_extension_es_negativa(mira_derecha):
    a = calcular_angulos(perfil(tronco=-15, brazo=-30, mira_derecha=mira_derecha), "der")
    assert a.tronco == pytest.approx(-15, abs=0.5)
    assert a.brazo == pytest.approx(-30, abs=0.5)


def test_manos_arriba_de_los_hombros():
    a = calcular_angulos(perfil(brazo=170, codo=0), "der")
    assert a.altura_muneca > 1.0


def test_elige_el_lado_que_mejor_se_ve():
    assert elegir_lado(perfil(lado="izq")) == "izq"
    assert elegir_lado(perfil(lado="der")) == "der"


def test_sin_cadera_no_inventa_angulos():
    e = perfil()
    del e.puntos["cadera_der"]
    assert calcular_angulos(e, "der").tronco is None
