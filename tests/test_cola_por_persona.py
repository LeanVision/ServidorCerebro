"""La cola de Re-ID: un recorte pendiente por persona y, si se llena, se pierde el viejo."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cola_por_persona import NUEVO, NUEVO_DESCARTANDO_VIEJO, REEMPLAZADO, ColaPorPersona


def test_un_recorte_nuevo_reemplaza_al_pendiente_de_la_misma_persona():
    cola = ColaPorPersona(10)
    assert cola.poner(("cam1", "7"), "viejo") == NUEVO
    assert cola.poner(("cam1", "7"), "nuevo") == REEMPLAZADO
    assert cola.qsize() == 1
    assert asyncio.run(cola.tomar(8)) == ["nuevo"]


def test_reemplazar_no_le_hace_perder_el_turno():
    cola = ColaPorPersona(10)
    cola.poner(("cam1", "1"), "a1")
    cola.poner(("cam1", "2"), "b1")
    cola.poner(("cam1", "1"), "a2")
    assert asyncio.run(cola.tomar(8)) == ["a2", "b1"]


def test_la_misma_persona_en_otra_camara_es_otro_trabajo():
    cola = ColaPorPersona(10)
    cola.poner(("cam1", "7"), "x")
    cola.poner(("cam2", "7"), "y")
    assert cola.qsize() == 2


def test_sin_tracker_no_se_agrupa():
    cola = ColaPorPersona(10)
    cola.poner(None, "x")
    cola.poner(None, "y")
    assert asyncio.run(cola.tomar(8)) == ["x", "y"]


def test_llena_descarta_el_mas_viejo_y_no_el_nuevo():
    cola = ColaPorPersona(2)
    cola.poner(("cam1", "1"), "a")
    cola.poner(("cam1", "2"), "b")
    assert cola.poner(("cam1", "3"), "c") == NUEVO_DESCARTANDO_VIEJO
    assert asyncio.run(cola.tomar(8)) == ["b", "c"]


def test_tomar_respeta_el_tamano_del_lote():
    cola = ColaPorPersona(10)
    for i in range(5):
        cola.poner(("cam1", str(i)), i)
    assert asyncio.run(cola.tomar(3)) == [0, 1, 2]
    assert cola.qsize() == 2


def test_tomar_espera_hasta_que_llegue_algo():
    async def escenario():
        cola = ColaPorPersona(10)
        tarea = asyncio.create_task(cola.tomar(8))
        await asyncio.sleep(0.01)
        assert not tarea.done()
        cola.poner(("cam1", "1"), "llego")
        return await asyncio.wait_for(tarea, 1)

    assert asyncio.run(escenario()) == ["llego"]
