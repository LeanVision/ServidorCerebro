"""Del video al informe, con un modelo de pose falso (sin bajar pesos)."""

import json
import math

import cv2
import numpy as np
from esqueletos import perfil

from ergonomia import analizar
from ergonomia.esqueleto import calcular_angulos
from ergonomia.informe import Cuadro, armar_informe, informe_markdown
from ergonomia.reba import Condiciones


def test_informe_de_postura_quieta_agachada():
    a = calcular_angulos(perfil(tronco=70, brazo=30), "der")
    cuadros = [Cuadro(i / 10, a) for i in range(900)]  # 90 s
    inf = armar_informe(cuadros, 10, Condiciones())
    assert inf["posturas_porcentaje_tiempo"]["tronco_flexion_mas_60"] == 100.0
    assert inf["postura_estatica_mas_larga_s"] == 90.0
    assert inf["reba"]["actividad_sumada"] == 1  # estática más de 1 minuto
    texto = informe_markdown(inf)
    assert "planilla 2.F" in texto and "La muñeca no se midió" in texto


class _PoseFalsa:
    """Brazo que sube y baja una vez cada 2 segundos."""

    def estimar(self, cuadro, tiempo_ms):
        codo = 80 + 40 * math.sin(2 * math.pi * 0.5 * tiempo_ms / 1000)
        return perfil(tronco=10, brazo=20, codo=codo)


def test_de_punta_a_punta(tmp_path, monkeypatch):
    video = tmp_path / "puesto.mp4"
    escritor = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20, (320, 240))
    for _ in range(20 * 30):  # 30 s a 20 fps
        escritor.write(np.zeros((240, 320, 3), np.uint8))
    escritor.release()

    monkeypatch.setattr(analizar, "crear_modelo", lambda nombre, ruta: _PoseFalsa())
    salida = tmp_path / "salida"
    analizar.main([str(video), "--salida", str(salida), "--carga-kg", "3", "--video-esqueleto"])

    inf = json.loads((salida / "informe.json").read_text(encoding="utf-8"))
    assert inf["cuadros_analizados"] == 300  # 10 por segundo
    codo = next(r for r in inf["repeticiones"] if r["segmento"] == "codo")
    assert 14 <= codo["ciclos"] <= 15
    assert inf["reba"]["actividad_sumada"] == 1  # más de 4 por minuto
    assert (salida / "informe.md").exists()
    assert (salida / "esqueleto.mp4").stat().st_size > 0
    assert len((salida / "angulos.csv").read_text().splitlines()) == 301
