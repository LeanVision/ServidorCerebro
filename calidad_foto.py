"""Qué foto de cada persona se muestra en la galería del panel.

La cámara manda cada 6 s el recorte con más confianza de esa espera. Antes la
galería mostraba siempre el último que llegaba, aunque fuera de lejos, movido
o cortado. Ahora, por defecto, se queda con la mejor foto de la visita: la más
grande (la persona más cerca, con más detalle) entre las que están nítidas.
"""

from __future__ import annotations

import cv2
import numpy as np

# Varianza del Laplaciano a partir de la cual una foto se considera nítida.
# Por debajo, su puntaje baja en proporción: una foto movida pierde contra
# una más chica pero nítida.
NITIDEZ_REFERENCIA = 100.0


def calidad_foto(imagen_bgr: np.ndarray) -> float:
    """Puntaje para comparar fotos de la misma persona: más alto es mejor."""
    alto, ancho = imagen_bgr.shape[:2]
    if alto == 0 or ancho == 0:
        return 0.0
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())
    return alto * ancho * min(1.0, nitidez / NITIDEZ_REFERENCIA)


def conviene_reemplazar(modo: str, calidad_actual: float | None, calidad_nueva: float) -> bool:
    """modo "ultima": siempre la que llega. "mejor": solo si es al menos igual de buena."""
    if modo == "ultima" or calidad_actual is None:
        return True
    return calidad_nueva >= calidad_actual
