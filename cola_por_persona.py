"""Cola de Re-ID que guarda un solo recorte pendiente por persona.

Con una cola FIFO común, cada persona seguida por una cámara manda recortes
más rápido de lo que OSNet los procesa cuando hay mucha gente: la cola se
llena de recortes viejos de las mismas personas, el worker los procesa en
orden y los resultados llegan con decenas de segundos de atraso. Y cuando se
llenaba se rechazaba el recorte nuevo, justo el que más vale.

Acá cada persona (cámara + tracker_id) tiene a lo sumo un trabajo esperando:
si llega un recorte nuevo de alguien que ya tiene uno en cola, lo reemplaza
y conserva el lugar en la fila. Así la cola nunca tiene más trabajos que
gente en cuadro, y lo que se procesa es siempre lo más reciente. Si aun así
se llena (cámaras sin tracker_id, o más gente que capacidad), se descarta el
más viejo y no el nuevo.

No importa nada del Cerebro ni de torch para poder testearla sola.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any, Hashable

NUEVO = "nuevo"
REEMPLAZADO = "reemplazado"
NUEVO_DESCARTANDO_VIEJO = "nuevo_descartando_viejo"


class ColaPorPersona:
    def __init__(self, capacidad: int) -> None:
        self.capacidad = max(1, capacidad)
        self._trabajos: OrderedDict[Hashable, Any] = OrderedDict()
        self._hay_trabajo = asyncio.Event()

    def poner(self, clave: Hashable | None, trabajo: Any) -> str:
        """Encola un trabajo. `clave` None = no se agrupa con ningún otro."""
        if clave is None:
            clave = object()
        if clave in self._trabajos:
            # Se reemplaza en el lugar: la persona no pierde su turno por
            # haber mandado un recorte más nuevo.
            self._trabajos[clave] = trabajo
            return REEMPLAZADO
        resultado = NUEVO
        if len(self._trabajos) >= self.capacidad:
            self._trabajos.popitem(last=False)
            resultado = NUEVO_DESCARTANDO_VIEJO
        self._trabajos[clave] = trabajo
        self._hay_trabajo.set()
        return resultado

    async def tomar(self, maximo: int) -> list[Any]:
        """Espera a que haya trabajo y devuelve hasta `maximo`, en orden."""
        while not self._trabajos:
            self._hay_trabajo.clear()
            await self._hay_trabajo.wait()
        lote = []
        while self._trabajos and len(lote) < max(1, maximo):
            _, trabajo = self._trabajos.popitem(last=False)
            lote.append(trabajo)
        return lote

    def qsize(self) -> int:
        return len(self._trabajos)
