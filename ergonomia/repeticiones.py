"""Movimientos repetitivos de miembros superiores (planilla 2.E).

Cuenta ciclos sobre una señal que sube y baja con cada movimiento (el ángulo
del codo o del hombro), con histéresis para que el temblor del modelo de pose
no cuente ciclos de más. Con la frecuencia y la fracción de tiempo en
movimiento estima el Nivel de Actividad Manual (NAM / HAL) con la ecuación de
Radwin et al. (2015). Es una ESTIMACIÓN para orientar al evaluador: la
planilla pide además la fuerza percibida (escala de Borg), que la cámara no ve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Por debajo de esta amplitud (grados) no hay un movimiento que contar.
AMPLITUD_MINIMA = 15.0
# Velocidad angular (grados/s) a partir de la cual el segmento "está en movimiento".
VELOCIDAD_MOVIMIENTO = 20.0


@dataclass
class Repeticiones:
    segmento: str
    ciclos: int
    ciclos_por_minuto: float
    segundos_por_ciclo: float | None
    fraccion_en_movimiento: float
    nam_estimado: float | None


def _suavizar(senal: np.ndarray, ventana: int) -> np.ndarray:
    if ventana <= 1 or len(senal) < ventana:
        return senal
    nucleo = np.ones(ventana) / ventana
    relleno = np.pad(senal, (ventana // 2, ventana - 1 - ventana // 2), mode="edge")
    return np.convolve(relleno, nucleo, mode="valid")


def _interpolar_huecos(senal: np.ndarray) -> np.ndarray:
    validos = ~np.isnan(senal)
    if validos.sum() < 2:
        return senal
    x = np.arange(len(senal))
    return np.interp(x, x[validos], senal[validos])


def contar_ciclos(senal: np.ndarray) -> int:
    """Ciclos completos: cada vez que la señal pasa de la zona baja a la alta."""
    if len(senal) == 0:
        return 0
    bajo, alto = np.percentile(senal, 10), np.percentile(senal, 90)
    amplitud = alto - bajo
    if amplitud < AMPLITUD_MINIMA:
        return 0
    umbral_bajo, umbral_alto = bajo + 0.35 * amplitud, bajo + 0.65 * amplitud
    ciclos, estado = 0, None
    for valor in senal:
        if valor <= umbral_bajo:
            estado = "bajo"
        elif valor >= umbral_alto:
            if estado == "bajo":
                ciclos += 1
            estado = "alto"
    return ciclos


def nam_radwin(frecuencia_hz: float, fraccion_movimiento: float) -> float:
    """NAM = 6,56 · ln(D) · F^1,31 / (1 + 3,18 · F^1,31), D en %, F en Hz."""
    d = fraccion_movimiento * 100.0
    if frecuencia_hz <= 0 or d <= 1:
        return 0.0
    f = frecuencia_hz ** 1.31
    return float(min(10.0, max(0.0, 6.56 * math.log(d) * f / (1 + 3.18 * f))))


def analizar_repeticiones(segmento: str, angulos: list[float | None], fps: float) -> Repeticiones:
    senal = _interpolar_huecos(np.array([np.nan if a is None else a for a in angulos], float))
    duracion = len(senal) / fps if fps > 0 else 0.0
    if duracion <= 0 or np.isnan(senal).any():
        return Repeticiones(segmento, 0, 0.0, None, 0.0, None)
    senal = _suavizar(senal, max(1, int(round(fps * 0.2))))
    ciclos = contar_ciclos(senal)
    velocidad = np.abs(np.diff(senal)) * fps
    fraccion = float((velocidad > VELOCIDAD_MOVIMIENTO).mean()) if len(velocidad) else 0.0
    frecuencia = ciclos / duracion
    return Repeticiones(
        segmento=segmento,
        ciclos=ciclos,
        ciclos_por_minuto=frecuencia * 60.0,
        segundos_por_ciclo=duracion / ciclos if ciclos else None,
        fraccion_en_movimiento=fraccion,
        nam_estimado=nam_radwin(frecuencia, fraccion),
    )
