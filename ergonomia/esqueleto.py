"""Esqueleto de una persona en un cuadro y los ángulos por segmento.

Los modelos de pose devuelven puntos en coordenadas de imagen (x a la
derecha, y hacia ABAJO). Todo lo de acá trabaja en 2D y supone la cámara de
COSTADO al trabajador: de frente, la flexión de tronco, cuello y brazo casi
no se ve en la imagen y los ángulos salen chicos aunque la postura sea mala.

Convención de signos: positivo = flexión (hacia adelante, hacia donde mira la
persona); negativo = extensión (hacia atrás).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Nombres comunes a todos los modelos. YOLO-pose trae los 17 de COCO;
# MediaPipe trae 33, de los que se mapean estos más los dedos índice (que
# permiten medir la muñeca).
PUNTOS = (
    "nariz", "oreja_izq", "oreja_der",
    "hombro_izq", "hombro_der", "codo_izq", "codo_der",
    "muneca_izq", "muneca_der", "indice_izq", "indice_der",
    "cadera_izq", "cadera_der", "rodilla_izq", "rodilla_der",
    "tobillo_izq", "tobillo_der",
)

# Por debajo de esta confianza el punto se toma como no visto.
CONFIANZA_MINIMA = 0.3


@dataclass
class Esqueleto:
    """Puntos de una persona: nombre -> (x, y, confianza) en píxeles."""

    puntos: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    def punto(self, nombre: str) -> tuple[float, float] | None:
        p = self.puntos.get(nombre)
        if p is None or p[2] < CONFIANZA_MINIMA:
            return None
        return p[0], p[1]

    def confianza_lado(self, lado: str) -> float:
        sufijo = "_izq" if lado == "izq" else "_der"
        nombres = [n for n in PUNTOS if n.endswith(sufijo)]
        return sum(self.puntos.get(n, (0, 0, 0.0))[2] for n in nombres) / len(nombres)


@dataclass
class Angulos:
    """Ángulos en grados de un lado del cuerpo. None = no se pudo medir."""

    tronco: float | None = None      # flexión respecto de la vertical
    cuello: float | None = None      # flexión respecto del tronco
    brazo: float | None = None       # flexión del hombro respecto del tronco
    antebrazo: float | None = None   # flexión del codo (0 = brazo estirado)
    muneca: float | None = None      # desvío de la mano respecto del antebrazo
    rodilla: float | None = None     # flexión de rodilla (0 = pierna estirada)
    altura_muneca: float | None = None  # muñeca sobre la cadera, en largos de tronco
    sentado: bool | None = None


def _resta(a, b):
    return a[0] - b[0], a[1] - b[1]


def _angulo_firmado(desde, hacia) -> float:
    """Ángulo en grados para girar el vector `desde` hasta `hacia` (-180..180)."""
    return math.degrees(math.atan2(desde[0] * hacia[1] - desde[1] * hacia[0],
                                   desde[0] * hacia[0] + desde[1] * hacia[1]))


def _flexion(a, vertice, b) -> float:
    """180 menos el ángulo interior en `vertice`: 0 = segmentos alineados."""
    return 180.0 - abs(_angulo_firmado(_resta(a, vertice), _resta(b, vertice)))


def elegir_lado(esqueleto: Esqueleto) -> str:
    """El lado más visto por la cámara (el otro queda tapado por el cuerpo)."""
    return "izq" if esqueleto.confianza_lado("izq") >= esqueleto.confianza_lado("der") else "der"


def _mira_a_la_derecha(e: Esqueleto, hombro, cadera, s: str) -> bool | None:
    """Hacia dónde mira la persona en la imagen, por la nariz o los pies."""
    nariz = e.punto("nariz")
    if nariz is not None and abs(nariz[0] - hombro[0]) > 1:
        return nariz[0] > hombro[0]
    # Sin cara visible: los pies apuntan hacia adelante, pero en 2D sólo
    # tenemos tobillos; usamos la rodilla, que al flexionar va hacia adelante.
    rodilla = e.punto("rodilla" + s)
    if rodilla is not None and abs(rodilla[0] - cadera[0]) > 1:
        return rodilla[0] > cadera[0]
    return None


def calcular_angulos(e: Esqueleto, lado: str) -> Angulos:
    s = "_izq" if lado == "izq" else "_der"
    hombro, cadera = e.punto("hombro" + s), e.punto("cadera" + s)
    if hombro is None or cadera is None:
        return Angulos()
    derecha = _mira_a_la_derecha(e, hombro, cadera, s)
    if derecha is None:
        return Angulos()
    # Si mira a la izquierda, se espeja en x: así "adelante" siempre es +x y
    # una flexión es siempre un giro en el mismo sentido.
    signo_x = 1.0 if derecha else -1.0

    def p(nombre):
        q = e.punto(nombre + s)
        return None if q is None else (q[0] * signo_x, q[1])

    hombro, cadera = p("hombro"), p("cadera")
    codo, muneca, indice = p("codo"), p("muneca"), p("indice")
    rodilla, tobillo, oreja = p("rodilla"), p("tobillo"), p("oreja")

    arriba = (0.0, -1.0)
    tronco_vec = _resta(hombro, cadera)
    largo_tronco = math.hypot(*tronco_vec)
    a = Angulos()
    # Con y hacia abajo y "adelante" = +x, un giro desde arriba hacia
    # adelante da ángulo positivo: es la flexión.
    a.tronco = _angulo_firmado(arriba, tronco_vec)
    if oreja is not None:
        a.cuello = _angulo_firmado(tronco_vec, _resta(oreja, hombro))
    if codo is not None:
        abajo_tronco = _resta(cadera, hombro)
        # Desde "colgando a lo largo del tronco" hacia adelante es negativo
        # en este giro, por eso el signo.
        a.brazo = -_angulo_firmado(abajo_tronco, _resta(codo, hombro))
        if muneca is not None:
            a.antebrazo = _flexion(hombro, codo, muneca)
            if indice is not None:
                a.muneca = _flexion(codo, muneca, indice)
    if muneca is not None and largo_tronco > 0:
        a.altura_muneca = (cadera[1] - muneca[1]) / largo_tronco
    if rodilla is not None and tobillo is not None:
        a.rodilla = _flexion(cadera, rodilla, tobillo)
        muslo = _resta(rodilla, cadera)
        # Sentado: el muslo queda más cerca de la horizontal que de la vertical.
        a.sentado = abs(muslo[0]) > abs(muslo[1])
    return a
