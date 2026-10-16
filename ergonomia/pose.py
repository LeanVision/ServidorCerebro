"""Modelos de estimación de pose: de un cuadro BGR a un Esqueleto.

Dos opciones, las dos corren local (el video no sale de la máquina):

- "yolo": Ultralytics YOLO-pose (17 puntos COCO). Baja los pesos solo la
  primera vez. No trae los dedos, así que la muñeca no se mide.
- "mediapipe": MediaPipe Pose Landmarker (33 puntos, incluye dedos índice).
  Necesita el archivo .task descargado aparte (ver README).

Si en el cuadro hay varias personas se toma la más grande: se supone que la
cámara apunta al trabajador del puesto evaluado.
"""

from __future__ import annotations

import numpy as np

from .esqueleto import Esqueleto

_COCO = (
    "nariz", None, None, "oreja_izq", "oreja_der",
    "hombro_izq", "hombro_der", "codo_izq", "codo_der",
    "muneca_izq", "muneca_der", "cadera_izq", "cadera_der",
    "rodilla_izq", "rodilla_der", "tobillo_izq", "tobillo_der",
)

_MEDIAPIPE = {
    0: "nariz", 7: "oreja_izq", 8: "oreja_der",
    11: "hombro_izq", 12: "hombro_der", 13: "codo_izq", 14: "codo_der",
    15: "muneca_izq", 16: "muneca_der", 19: "indice_izq", 20: "indice_der",
    23: "cadera_izq", 24: "cadera_der", 25: "rodilla_izq", 26: "rodilla_der",
    27: "tobillo_izq", 28: "tobillo_der",
}


class PoseYolo:
    def __init__(self, pesos: str = "yolo11n-pose.pt"):
        from ultralytics import YOLO

        self.modelo = YOLO(pesos)

    def estimar(self, cuadro_bgr: np.ndarray, tiempo_ms: int) -> Esqueleto | None:
        resultado = self.modelo(cuadro_bgr, verbose=False)[0]
        if resultado.keypoints is None or len(resultado.boxes) == 0:
            return None
        cajas = resultado.boxes.xywh.cpu().numpy()
        i = int(np.argmax(cajas[:, 2] * cajas[:, 3]))
        xy = resultado.keypoints.xy[i].cpu().numpy()
        conf = resultado.keypoints.conf
        conf = conf[i].cpu().numpy() if conf is not None else np.ones(len(xy))
        return Esqueleto({nombre: (float(xy[j][0]), float(xy[j][1]), float(conf[j]))
                          for j, nombre in enumerate(_COCO) if nombre})


class PoseMediaPipe:
    def __init__(self, modelo_task: str):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        opciones = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=modelo_task),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.detector = vision.PoseLandmarker.create_from_options(opciones)

    def estimar(self, cuadro_bgr: np.ndarray, tiempo_ms: int) -> Esqueleto | None:
        import cv2
        import mediapipe as mp

        alto, ancho = cuadro_bgr.shape[:2]
        imagen = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(cuadro_bgr, cv2.COLOR_BGR2RGB))
        resultado = self.detector.detect_for_video(imagen, tiempo_ms)
        if not resultado.pose_landmarks:
            return None
        puntos = resultado.pose_landmarks[0]
        return Esqueleto({nombre: (puntos[j].x * ancho, puntos[j].y * alto,
                                   float(puntos[j].visibility or 0.0))
                          for j, nombre in _MEDIAPIPE.items()})


def crear_modelo(nombre: str, ruta_modelo: str | None):
    if nombre == "yolo":
        return PoseYolo(ruta_modelo or "yolo11n-pose.pt")
    if nombre == "mediapipe":
        if not ruta_modelo:
            raise SystemExit("MediaPipe necesita --ruta-modelo con el archivo pose_landmarker_*.task (ver README).")
        return PoseMediaPipe(ruta_modelo)
    raise SystemExit(f"Modelo de pose desconocido: {nombre}")
