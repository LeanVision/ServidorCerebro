"""Puntaje REBA (Rapid Entire Body Assessment, Hignett y McAtamney, 2000).

La Res. SRT 886/2015 admite métodos reconocidos internacionalmente para
evaluar posturas forzadas; REBA es uno de los más usados.

Lo que una cámara de costado NO ve, y por eso no suma acá: giros e
inclinaciones laterales de tronco y cuello, abducción y rotación del brazo,
hombro elevado, brazo apoyado, desvío lateral de la muñeca. El peso de la
carga, el agarre y el apoyo de las piernas los informa quien evalúa.
"""

from __future__ import annotations

from dataclasses import dataclass

from .esqueleto import Angulos

# Tabla A: [tronco 1-5][cuello 1-3][piernas 1-4]
_TABLA_A = (
    ((1, 2, 3, 4), (1, 2, 3, 4), (3, 3, 5, 6)),
    ((2, 3, 4, 5), (3, 4, 5, 6), (4, 5, 6, 7)),
    ((2, 4, 5, 6), (4, 5, 6, 7), (5, 6, 7, 8)),
    ((3, 5, 6, 7), (5, 6, 7, 8), (6, 7, 8, 9)),
    ((4, 6, 7, 8), (6, 7, 8, 9), (7, 8, 9, 9)),
)

# Tabla B: [brazo 1-6][antebrazo 1-2][muñeca 1-3]
_TABLA_B = (
    ((1, 2, 2), (1, 2, 3)),
    ((1, 2, 3), (2, 3, 4)),
    ((3, 4, 5), (4, 5, 5)),
    ((4, 5, 5), (5, 6, 7)),
    ((6, 7, 8), (7, 8, 8)),
    ((7, 8, 8), (8, 9, 9)),
)

# Tabla C: [puntaje A 1-12][puntaje B 1-12]
_TABLA_C = (
    (1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7),
    (1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8),
    (2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8),
    (3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9),
    (4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9),
    (6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10),
    (7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11),
    (8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11),
    (9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12),
    (10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12),
    (11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12),
    (12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12),
)

AGARRES = {"bueno": 0, "regular": 1, "malo": 2, "inaceptable": 3}

NIVELES = (
    (1, "inapreciable", "no es necesaria actuación"),
    (3, "bajo", "puede ser necesaria la actuación"),
    (7, "medio", "es necesaria la actuación"),
    (10, "alto", "es necesaria la actuación cuanto antes"),
    (15, "muy alto", "es necesaria la actuación de inmediato"),
)


@dataclass
class Condiciones:
    """Lo que la cámara no mide y lo informa quien evalúa."""

    carga_kg: float = 0.0
    golpes_o_fuerza_brusca: bool = False
    agarre: str = "bueno"
    apoyo_un_pie: bool = False


@dataclass
class ResultadoReba:
    puntaje: int
    nivel: str
    tronco: int
    cuello: int
    piernas: int
    brazo: int
    antebrazo: int
    muneca: int
    muneca_medida: bool


def puntaje_tronco(flexion: float) -> int:
    if flexion < -20:
        return 3
    if abs(flexion) <= 5:
        return 1
    if flexion <= 20:
        return 2
    return 3 if flexion <= 60 else 4


def puntaje_cuello(flexion: float) -> int:
    # Con 5° de tolerancia para la extensión: el ruido del modelo de pose
    # haría saltar a 2 un cuello derecho.
    return 1 if -5 <= flexion <= 20 else 2


def puntaje_piernas(rodilla: float | None, un_pie: bool) -> int:
    base = 2 if un_pie else 1
    if rodilla is None or rodilla < 30:
        return base
    return base + (1 if rodilla <= 60 else 2)


def puntaje_brazo(flexion: float) -> int:
    if -20 <= flexion <= 20:
        return 1
    if flexion < -20 or flexion <= 45:
        return 2
    return 3 if flexion <= 90 else 4


def puntaje_antebrazo(flexion: float) -> int:
    return 1 if 60 <= flexion <= 100 else 2


def puntaje_muneca(desvio: float) -> int:
    return 1 if desvio <= 15 else 2


def puntaje_carga(c: Condiciones) -> int:
    base = 0 if c.carga_kg < 5 else 1 if c.carga_kg <= 10 else 2
    return base + (1 if c.golpes_o_fuerza_brusca else 0)


def nivel_riesgo(puntaje: int) -> str:
    for tope, nombre, _ in NIVELES:
        if puntaje <= tope:
            return nombre
    return NIVELES[-1][1]


def calcular_reba(a: Angulos, c: Condiciones, actividad: int = 0) -> ResultadoReba | None:
    """REBA de un cuadro. `actividad` (0-3) se suma al final: postura estática
    más de 1 minuto, movimientos repetidos más de 4 veces por minuto, cambios
    bruscos de postura. None si no se ve el tronco o el brazo."""
    if a.tronco is None or a.brazo is None:
        return None
    tronco = puntaje_tronco(a.tronco)
    cuello = puntaje_cuello(a.cuello) if a.cuello is not None else 1
    piernas = puntaje_piernas(a.rodilla, c.apoyo_un_pie)
    brazo = puntaje_brazo(a.brazo)
    antebrazo = puntaje_antebrazo(a.antebrazo) if a.antebrazo is not None else 1
    muneca = puntaje_muneca(a.muneca) if a.muneca is not None else 1

    puntaje_a = _TABLA_A[tronco - 1][cuello - 1][piernas - 1] + puntaje_carga(c)
    puntaje_b = _TABLA_B[brazo - 1][antebrazo - 1][muneca - 1] + AGARRES[c.agarre]
    puntaje_c = _TABLA_C[min(puntaje_a, 12) - 1][min(puntaje_b, 12) - 1]
    total = puntaje_c + actividad
    return ResultadoReba(total, nivel_riesgo(total), tronco, cuello, piernas,
                         brazo, antebrazo, muneca, a.muneca is not None)
