"""Esqueletos sintéticos de perfil, con ángulos conocidos, para las pruebas."""

import math

from ergonomia.esqueleto import Esqueleto


def _rotar(v, grados):
    """Gira hacia adelante (+x) en coordenadas de imagen (y hacia abajo)."""
    t = math.radians(grados)
    return v[0] * math.cos(t) + v[1] * math.sin(t), -v[0] * math.sin(t) + v[1] * math.cos(t)


def _mas(p, v, largo):
    return p[0] + v[0] * largo, p[1] + v[1] * largo


def perfil(tronco=0.0, cuello=0.0, brazo=0.0, codo=90.0, muneca=None, rodilla=0.0,
           mira_derecha=True, lado="der"):
    cadera = (300.0, 400.0)
    arriba = _rotar((0.0, -1.0), -tronco)  # flexión de tronco: hombro hacia adelante
    hombro = _mas(cadera, arriba, 200)
    oreja = _mas(hombro, _rotar(arriba, -cuello), 50)
    nariz = _mas(oreja, (1.0, 0.0), 20)
    abajo = (-arriba[0], -arriba[1])
    dir_brazo = _rotar(abajo, brazo)
    codo_p = _mas(hombro, dir_brazo, 120)
    dir_antebrazo = _rotar(dir_brazo, codo)
    muneca_p = _mas(codo_p, dir_antebrazo, 110)
    indice_p = _mas(muneca_p, _rotar(dir_antebrazo, muneca or 0.0), 30)
    rodilla_p = _mas(cadera, (0.0, 1.0), 180)
    tobillo_p = _mas(rodilla_p, _rotar((0.0, 1.0), -rodilla), 180)

    puntos = {"nariz": nariz, "oreja": oreja, "hombro": hombro, "codo": codo_p, "muneca": muneca_p,
              "cadera": cadera, "rodilla": rodilla_p, "tobillo": tobillo_p}
    if muneca is not None:
        puntos["indice"] = indice_p
    s = "_" + lado
    otro = "_izq" if lado == "der" else "_der"
    resultado = {}
    for nombre, (x, y) in puntos.items():
        if not mira_derecha:
            x = 1000.0 - x
        if nombre == "nariz":
            resultado[nombre] = (x, y, 0.9)
        else:
            resultado[nombre + s] = (x, y, 0.9)
            resultado[nombre + otro] = (x, y, 0.2)  # el lado tapado se ve peor
    return Esqueleto(resultado)
