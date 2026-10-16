"""Analiza un video de un puesto de trabajo y genera el informe ergonómico.

Uso (desde la raíz del repo):

    python -m ergonomia.analizar video.mp4 --carga-kg 8 --agarre regular

Deja en la carpeta de salida informe.md (para leer), informe.json (para
procesar) y angulos.csv (ángulos cuadro por cuadro). Con --video-esqueleto
también un video con SOLO el esqueleto sobre fondo negro: sirve para
mostrar el análisis sin exponer la cara ni el cuerpo del trabajador. Los
cuadros del video original no se guardan.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, fields
from pathlib import Path

import cv2
import numpy as np

from .esqueleto import Angulos, Esqueleto, calcular_angulos, elegir_lado
from .informe import Cuadro, armar_informe, informe_markdown
from .pose import crear_modelo
from .reba import AGARRES, Condiciones

_HUESOS = (
    ("hombro", "codo"), ("codo", "muneca"), ("muneca", "indice"),
    ("hombro", "cadera"), ("cadera", "rodilla"), ("rodilla", "tobillo"), ("hombro", "oreja"),
)


def _dibujar_esqueleto(e: Esqueleto | None, lado: str | None, alto: int, ancho: int, texto: str) -> np.ndarray:
    lienzo = np.zeros((alto, ancho, 3), np.uint8)
    if e is not None and lado is not None:
        s = "_izq" if lado == "izq" else "_der"
        for a, b in _HUESOS:
            pa, pb = e.punto(a + s), e.punto(b + s)
            if pa and pb:
                cv2.line(lienzo, tuple(map(int, pa)), tuple(map(int, pb)), (0, 220, 255), 3)
    cv2.putText(lienzo, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return lienzo


def analizar_video(ruta: str, modelo, lado: str = "auto", cuadros_por_segundo: float = 10.0,
                   video_esqueleto: str | None = None) -> tuple[list[Cuadro], float, str | None]:
    """Devuelve los cuadros analizados, los cuadros por segundo efectivos y el lado usado."""
    captura = cv2.VideoCapture(ruta)
    if not captura.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {ruta}")
    fps_video = captura.get(cv2.CAP_PROP_FPS) or 30.0
    paso = max(1, int(round(fps_video / cuadros_por_segundo)))
    fps_efectivo = fps_video / paso

    escritor, cuadros, lado_usado = None, [], None if lado == "auto" else lado
    indice = 0
    while True:
        ok, imagen = captura.read()
        if not ok:
            break
        if indice % paso:
            indice += 1
            continue
        tiempo_ms = int(indice * 1000 / fps_video)
        esqueleto = modelo.estimar(imagen, tiempo_ms)
        angulos = None
        if esqueleto is not None:
            # El lado se fija con el primer cuadro con persona: cambiarlo en
            # medio del video mezclaría los dos brazos en la misma serie.
            lado_usado = lado_usado or elegir_lado(esqueleto)
            angulos = calcular_angulos(esqueleto, lado_usado)
        cuadros.append(Cuadro(tiempo_ms / 1000.0, angulos))

        if video_esqueleto:
            alto, ancho = imagen.shape[:2]
            if escritor is None:
                escritor = cv2.VideoWriter(video_esqueleto, cv2.VideoWriter_fourcc(*"mp4v"),
                                           fps_efectivo, (ancho, alto))
            tronco = f"tronco {angulos.tronco:.0f}" if angulos and angulos.tronco is not None else "sin postura"
            escritor.write(_dibujar_esqueleto(esqueleto, lado_usado, alto, ancho,
                                              f"{tiempo_ms / 1000:.1f}s  {tronco}"))
        indice += 1
    captura.release()
    if escritor is not None:
        escritor.release()
    return cuadros, fps_efectivo, lado_usado


def guardar_csv(cuadros: list[Cuadro], ruta: Path) -> None:
    columnas = [f.name for f in fields(Angulos)]
    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        w = csv.writer(archivo)
        w.writerow(["tiempo_s", *columnas])
        for c in cuadros:
            valores = asdict(c.angulos) if c.angulos else {}
            w.writerow([f"{c.tiempo_s:.2f}", *[
                "" if valores.get(k) is None else (f"{valores[k]:.1f}" if isinstance(valores[k], float) else valores[k])
                for k in columnas
            ]])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--salida", default=None, help="carpeta de salida (por defecto, <video>_ergonomia)")
    p.add_argument("--modelo", choices=("yolo", "mediapipe"), default="yolo")
    p.add_argument("--ruta-modelo", default=None, help="pesos YOLO (.pt) o archivo .task de MediaPipe")
    p.add_argument("--lado", choices=("auto", "izq", "der"), default="auto",
                   help="lado del cuerpo que mira a la cámara")
    p.add_argument("--cuadros-por-segundo", type=float, default=10.0)
    p.add_argument("--carga-kg", type=float, default=0.0, help="peso que manipula (lo informa el evaluador)")
    p.add_argument("--golpes", action="store_true", help="fuerza brusca o golpes al manipular")
    p.add_argument("--agarre", choices=tuple(AGARRES), default="bueno")
    p.add_argument("--un-pie", action="store_true", help="trabaja apoyado en un solo pie")
    p.add_argument("--puesto", default="", help="nombre del puesto, para el informe")
    p.add_argument("--video-esqueleto", action="store_true",
                   help="guardar un video con sólo el esqueleto sobre fondo negro")
    args = p.parse_args(argv)

    video = Path(args.video)
    salida = Path(args.salida or f"{video.with_suffix('')}_ergonomia")
    salida.mkdir(parents=True, exist_ok=True)

    modelo = crear_modelo(args.modelo, args.ruta_modelo)
    cuadros, fps, lado = analizar_video(
        str(video), modelo, args.lado, args.cuadros_por_segundo,
        str(salida / "esqueleto.mp4") if args.video_esqueleto else None,
    )
    condiciones = Condiciones(args.carga_kg, args.golpes, args.agarre, args.un_pie)
    datos = {"Video": video.name, "Puesto": args.puesto or "(sin nombre)",
             "Modelo de pose": args.modelo, "Lado analizado": lado or "sin persona detectada"}
    informe = armar_informe(cuadros, fps, condiciones, datos)

    (salida / "informe.json").write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    (salida / "informe.md").write_text(informe_markdown(informe), encoding="utf-8")
    guardar_csv(cuadros, salida / "angulos.csv")
    print(f"Informe en {salida / 'informe.md'}")


if __name__ == "__main__":
    main()
