"""Cerebro LeanVision (CerebroLocal): recepción en cola, Re-ID robusto y heatmap nativo 640x480.

Portado desde mi-proyecto/cerebro_server.py el 2026-08-04, adaptado para
mantener compatibilidad con la configuración de CerebroLocal (Supabase vía
.env local, demografia.onnx y haarcascade_frontalface_default.xml en esta
carpeta).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from torchreid.utils import FeatureExtractor

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None


def _cargar_dotenv_local(ruta: str = ".env") -> None:
    """Carga variables desde un .env local sin depender de python-dotenv.

    No pisa variables ya presentes en el entorno (permite overrides desde
    la shell). El archivo .env no se versiona (ver .gitignore).
    """
    if not os.path.exists(ruta):
        return
    with open(ruta, "r", encoding="utf-8-sig") as archivo:
        for numero_linea, linea in enumerate(archivo, start=1):
            linea = linea.strip()
            if not linea or linea.startswith("#") or "=" not in linea:
                continue
            clave, _, valor = linea.partition("=")
            clave = clave.strip()
            valor = valor.strip().strip('"').strip("'")
            try:
                os.environ.setdefault(clave, valor)
            except (ValueError, OSError) as error:
                # Una línea con caracteres inválidos (ej. copiado con algún
                # byte invisible) no debe tumbar el arranque del servidor.
                print(f"Aviso: .env línea {numero_linea} inválida, se omite ({error!r}).")


_cargar_dotenv_local()

HEATMAP_WIDTH = 640
HEATMAP_HEIGHT = 480
# Heatmap por grilla, acumulativo (sin decaimiento): se mantiene estable
# durante todo el día para poder comparar qué zonas fueron más concurridas al
# cierre. Se reinicia manualmente con /api/reset.
HEATMAP_CELDA_PX = 20
IP_CAMARA = os.getenv("LEANVISION_CAMERA_URL", "http://172.31.99.7:8002")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Plano único multicámara: cada cámara se calibra con un cuadrilátero que
# mapea su cuadro completo (4 esquinas) a una región del plano compartido.
# Sin calibrar, las coordenadas de esa cámara pasan sin cambios (fallback
# idéntico al comportamiento de una sola cámara). No se versiona (deployment-
# specific), pero SÍ sobrevive al auto-pull al estar en .gitignore.
PLANO_CONFIG_PATH = os.getenv("LEANVISION_PLANO_CONFIG", "plano_config.json")
PUNTOS_CAMARA_ORIGEN = np.float32(
    [[0, 0], [HEATMAP_WIDTH, 0], [HEATMAP_WIDTH, HEATMAP_HEIGHT], [0, HEATMAP_HEIGHT]]
)
# Imagen real del plano (opcional): se guarda como archivo aparte, no adentro
# del JSON. Su existencia en disco ES el estado ("¿hay imagen?"), no hace
# falta duplicarlo en plano_config.json. Mismo criterio de no-versionar.
PLANO_IMAGEN_PATH = os.getenv("LEANVISION_PLANO_IMAGEN", "plano_imagen.jpg")
PLANO_IMAGEN_MAX_BYTES = 10 * 1024 * 1024
PLANO_IMAGEN_LADO_MAXIMO_PX = 1600  # redimensiona si es más grande, para no servir fotos enormes en cada carga de página.

# La recepción HTTP no ejecuta IA: sólo encola JPEGs. Así el loop de la cámara
# nunca queda esperando al modelo ni se acumulan threads sin límite.
REID_QUEUE_SIZE = 300
REID_WORKERS = 1  # OSNet sobre CPU es más estable con una inferencia a la vez.
SUPABASE_WORKERS = 1

# Re-ID: match flexible para cambios sentado/parado, actualización estricta
# para impedir que un recorte incorrecto contamine el centroide de una persona.
UMBRAL_SIMILITUD = 0.36
# Entre cámaras distintas no hay guardias de posición/zona que ayuden a
# descartar falsos positivos (no tiene sentido comparar píxeles de encuadres
# distintos): la similitud visual queda como único filtro. Luz y ángulo
# varían más entre cámaras que dentro de una sola, así que exigimos más
# certeza antes de fusionar dos identidades de cámaras distintas.
# Punto de partida razonado, no validado aún con cámaras reales en paralelo.
UMBRAL_SIMILITUD_CROSS_CAMARA = 0.55
UMBRAL_ACTUALIZACION_ALBUM = 0.52
MAX_FOTOS_ALBUM = 8
INTERVALO_ACTUALIZACION_ALBUM = 1.0
TIEMPO_TELETRANSPORTACION = 3.0
TIEMPO_INACTIVIDAD_SEGUNDOS = 60.0
TIEMPO_TRACKER_ACTIVO = 3.0
DISTANCIA_REID_LOCAL_PX = 110.0
TIEMPO_CONTINUIDAD_POSTURA = 30.0
TOLERANCIA_POSTURA_X_PX = 100.0
TOLERANCIA_POSTURA_Y_PX = 180.0
UMBRAL_SIMILITUD_CONTINUIDAD = 0.20

# Estabilidad de IDs (menos parpadeo): un cliente recién creado es "provisional"
# hasta acumular varias apariciones; así un recorte espurio no genera una tarjeta
# fantasma. La similitud mostrada se suaviza con una media móvil exponencial.
MIN_APARICIONES_VISIBLE = 2
ALFA_SIMILITUD_EMA = 0.5

# Edad/género con InsightFace: durante la sesión sólo se guardan fotos (sin
# analizarlas); el análisis real corre una única vez al cerrar la sesión, así
# nunca compite por CPU con el worker de Re-ID en tiempo real. Más fotos no
# cuesta CPU en vivo (sólo copiar bytes); da más chances de agarrar un ángulo
# de cara detectable, sobre todo con la cámara lejos o en gran angular.
DEMO_MAX_FOTOS = 12
DEMO_INTERVALO_SEGUNDOS = 2.5
DEMO_MIN_DET_SCORE = 0.55

logging.basicConfig(
    level=os.getenv("LEANVISION_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("leanvision.cerebro")

app = FastAPI(title="LeanVision Cerebro (CerebroLocal)", version="2.0.0")
state_lock = threading.RLock()
clientes_globales: dict[str, dict] = {}
traductor_camaras: dict[str, dict] = {}
# Grilla del heatmap: (col, fila) -> {"valor": float, "ultimo": timestamp}.
heatmap_celdas: dict[tuple[int, int], dict] = {}
# Plano único: cámaras registradas/calibradas y zonas de negocio (ver arriba).
calibraciones_camaras: dict[str, dict] = {}
zonas_negocio: list[dict] = []
# Contorno del local (paredes) en coordenadas del plano. Vacío = rectángulo
# completo, que es como se comportaba antes de poder dibujarlo.
contorno_local: list[list[float]] = []
_homografias_cache: dict[str, np.ndarray] = {}
contador_global_ids = 1
cola_reid: asyncio.Queue[DetectionJob] | None = None
reid_executor = ThreadPoolExecutor(max_workers=REID_WORKERS, thread_name_prefix="reid")
supabase_executor = ThreadPoolExecutor(max_workers=SUPABASE_WORKERS, thread_name_prefix="supabase")
metricas = {
    "accepted": 0,
    "processed": 0,
    "rejected_full": 0,
    "failed": 0,
    "last_processing_ms": 0.0,
}


@dataclass(frozen=True)
class DetectionJob:
    image_bytes: bytes
    zona: str
    tracker_id: str | None
    camara_id: str
    branch_id: str
    pos_x: float
    pos_y: float


class CamaraIn(BaseModel):
    camara_id: str
    nombre: str = ""
    video_url: str


class PosicionIn(BaseModel):
    posicion: list[float]


class CalibracionIn(BaseModel):
    puntos_plano: list[list[float]]
    # Los 4 puntos equivalentes en la imagen de la cámara. Sin esto se asumen
    # las esquinas del cuadro completo, que sólo es correcto si la cámara mira
    # el piso perfectamente de arriba hacia abajo.
    puntos_camara: list[list[float]] | None = None


class ZonaIn(BaseModel):
    name: str
    color: str = "#00ff88"
    polygon: list[list[float]]


class ContornoIn(BaseModel):
    polygon: list[list[float]]


print("Cargando modelo Re-ID OSNet-IBN...")
extractor_ia = FeatureExtractor(model_name="osnet_ibn_x1_0", device="cpu")
print("Modelo Re-ID listo.")

# Edad/género con InsightFace (modelos pre-entrenados: detección SCRFD +
# genderage). La primera ejecución descarga los pesos a ~/.insightface/models;
# luego funciona 100% offline y en CPU.
try:
    if FaceAnalysis is None:
        raise RuntimeError("insightface no está instalado (pip install insightface)")
    print("Cargando modelo de edad/género (InsightFace)...")
    analizador_rostros = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "genderage"])
    # 320x320 se usó mientras esto corría en tiempo real (competía por CPU con
    # el worker de Re-ID). Ahora el análisis corre una única vez al cerrar la
    # sesión, así que se puede subir a 480x480: mejor detección para personas
    # cerca de la cámara (~150-200ms por foto, tolerable como evento puntual).
    analizador_rostros.prepare(ctx_id=-1, det_size=(480, 480))  # ctx_id=-1 => CPU
    tiene_demografia = True
    print("Modelo de edad/género listo.")
except Exception as error:
    analizador_rostros = None
    tiene_demografia = False
    logger.warning("Edad/género no disponible (%s); las sesiones se guardarán sin demografía.", error)

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.warning(
        "SUPABASE_URL / SUPABASE_KEY no configurados (esperados en .env o variables de entorno); "
        "las sesiones no se guardarán en Supabase."
    )


def _normalizar_huella(huella: torch.Tensor) -> torch.Tensor:
    return F.normalize(huella.detach().cpu(), p=2, dim=0)


def _similitud(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b))


def _distancia(a: tuple[float, float] | None, b: tuple[float, float]) -> float:
    if a is None:
        return float("inf")
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _misma_ubicacion_sentado_parado(a: tuple[float, float] | None, b: tuple[float, float]) -> bool:
    """Tolera el desplazamiento vertical normal al sentarse o ponerse de pie."""
    if a is None:
        return False
    return abs(a[0] - b[0]) <= TOLERANCIA_POSTURA_X_PX and abs(a[1] - b[1]) <= TOLERANCIA_POSTURA_Y_PX


def _recorte_para_reid(imagen: np.ndarray) -> np.ndarray:
    """Reduce bordes y parte baja: suelen contener sillas u otras personas."""
    alto, ancho = imagen.shape[:2]
    margen_x = int(ancho * 0.08)
    limite_inferior = max(1, int(alto * 0.90))
    recorte = imagen[:limite_inferior, margen_x:ancho - margen_x]
    return recorte if recorte.size else imagen


def _agregar_huella_confiable(datos: dict, huella: torch.Tensor, ahora: float) -> bool:
    """Actualiza álbum/centroide sólo con muestras coherentes con la identidad."""
    album: list[torch.Tensor] = datos["historial"]
    centroide: torch.Tensor = datos["centroide"]
    sim_centroide = _similitud(huella, centroide)
    sim_mejor_foto = max((_similitud(huella, anterior) for anterior in album), default=-1.0)

    if ahora - datos.get("ultimo_update_album", 0.0) < INTERVALO_ACTUALIZACION_ALBUM:
        return False
    if sim_centroide < UMBRAL_ACTUALIZACION_ALBUM or sim_mejor_foto < UMBRAL_ACTUALIZACION_ALBUM:
        return False

    album.append(huella)
    if len(album) > MAX_FOTOS_ALBUM:
        album.pop(0)
    # Todas las muestras llegaron por el filtro anterior, por lo que el
    # promedio no mezcla hashes de personas distintas.
    datos["centroide"] = _normalizar_huella(torch.stack(album).mean(dim=0))
    datos["ultimo_update_album"] = ahora
    return True


def _puntaje_identidad(huella: torch.Tensor, datos: dict) -> float:
    similitudes = sorted((_similitud(huella, h) for h in datos["historial"]), reverse=True)
    if not similitudes:
        return -1.0
    # Promediar las tres mejores vistas reduce el efecto de una sola foto mala.
    mejores = similitudes[: min(3, len(similitudes))]
    sim_album = sum(mejores) / len(mejores)
    sim_centroide = _similitud(huella, datos["centroide"])
    return 0.65 * sim_album + 0.35 * sim_centroide


def _ids_bloqueados_en_misma_camara(
    camara_id: str, id_local_actual: str | None, posicion: tuple[float, float], ahora: float,
) -> set[str]:
    """Evita unir dos personas simultáneas, pero permite recuperar una cercana.

    Si ByteTrack cambia el ID de una persona sentada, el identificador viejo
    queda activo unos segundos. Se permite esa asociación sólo si está cerca de
    la posición actual; identificadores activos lejanos siguen bloqueados.
    """
    bloqueados: set[str] = set()
    prefijo = f"{camara_id}_"
    for clave, info in traductor_camaras.items():
        if clave == id_local_actual or not clave.startswith(prefijo):
            continue
        if ahora - info.get("ultimo_update", 0.0) > TIEMPO_TRACKER_ACTIVO:
            continue
        if _distancia(info.get("posicion"), posicion) > DISTANCIA_REID_LOCAL_PX:
            bloqueados.add(info["id_global"])
    return bloqueados


def _registrar_heatmap(pos_x: float, pos_y: float) -> None:
    """Acumula presencia en la celda de la grilla, sin decaimiento."""
    if not (0 <= pos_x < HEATMAP_WIDTH and 0 <= pos_y < HEATMAP_HEIGHT):
        return
    celda_id = (int(pos_x // HEATMAP_CELDA_PX), int(pos_y // HEATMAP_CELDA_PX))
    celda = heatmap_celdas.setdefault(celda_id, {"valor": 0.0})
    celda["valor"] += 1.0


def _snapshot_heatmap() -> tuple[list[dict], float]:
    """Devuelve los puntos acumulados hasta ahora y el valor máximo (para
    escalar colores en el dashboard)."""
    puntos: list[dict] = []
    maximo = 0.0
    for (col, fila), celda in heatmap_celdas.items():
        valor = celda["valor"]
        puntos.append({
            "x": int(col * HEATMAP_CELDA_PX + HEATMAP_CELDA_PX // 2),
            "y": int(fila * HEATMAP_CELDA_PX + HEATMAP_CELDA_PX // 2),
            "value": round(valor, 3),
        })
        maximo = max(maximo, valor)
    return puntos, maximo


def _recalcular_homografia(camara_id: str) -> None:
    """Recalcula y cachea la matriz de homografía de una cámara calibrada."""
    registro = calibraciones_camaras.get(camara_id, {})
    puntos_plano = registro.get("puntos_plano")
    if not puntos_plano:
        _homografias_cache.pop(camara_id, None)
        return
    puntos_camara = registro.get("puntos_camara")
    origen = np.float32(puntos_camara) if puntos_camara else PUNTOS_CAMARA_ORIGEN
    _homografias_cache[camara_id] = cv2.getPerspectiveTransform(
        origen, np.float32(puntos_plano)
    )


def _cargar_plano_config(ruta: str = PLANO_CONFIG_PATH) -> None:
    """Carga cámaras/zonas del plano desde disco. Nunca tumba el arranque:
    ante cualquier error, sigue con el estado vacío (mismo criterio que
    _cargar_dotenv_local)."""
    if not os.path.exists(ruta):
        return
    try:
        with open(ruta, "r", encoding="utf-8-sig") as archivo:
            datos = json.load(archivo)
        calibraciones_camaras.update(datos.get("camaras", {}))
        zonas_negocio.extend(datos.get("zonas", []))
        contorno_local[:] = datos.get("contorno", [])
        for camara_id in calibraciones_camaras:
            _recalcular_homografia(camara_id)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        logger.warning("No se pudo cargar %s, se ignora (%r).", ruta, error)


def _guardar_plano_config(ruta: str = PLANO_CONFIG_PATH) -> None:
    """Escritura atómica (mismo patrón que app_limpia.py: .tmp + replace)."""
    payload = {
        "camaras": calibraciones_camaras,
        "zonas": zonas_negocio,
        "contorno": contorno_local,
    }
    ruta_tmp = f"{ruta}.tmp"
    with open(ruta_tmp, "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, ensure_ascii=False, indent=2)
    os.replace(ruta_tmp, ruta)


def _punto_en_poligono(px: float, py: float, polygon: list) -> bool:
    """Ray-casting even-odd, portado de mi-proyecto/leanvision/zone.py."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _zona_en_punto(x: float, y: float) -> str | None:
    """Nombre de la primera zona de negocio que contiene el punto, o None."""
    for zona in zonas_negocio:
        if _punto_en_poligono(x, y, zona["polygon"]):
            return zona["name"]
    return None


def _area_poligono(polygon: list) -> float:
    """Fórmula de shoelace: cerca de 0 implica puntos colineales/duplicados
    (cuadrilátero degenerado, homografía inválida)."""
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _transformar_a_plano(camara_id: str, x: float, y: float) -> tuple[float, float]:
    """Convierte coordenadas nativas de una cámara a espacio de plano.

    Sin calibración, devuelve el punto sin cambios: preserva el
    comportamiento actual para cámaras no calibradas.
    """
    matriz = _homografias_cache.get(camara_id)
    if matriz is None:
        return x, y
    punto = np.array([[[x, y]]], dtype=np.float32)
    nx, ny = cv2.perspectiveTransform(punto, matriz)[0][0]
    return float(nx), float(ny)


def _actualizar_zona_y_tiempo(datos: dict, zona_nueva: str, ahora: float) -> None:
    """Acumula tiempo por zona (en memoria, sin decaimiento). Se llama antes
    de sobreescribir zona_actual, para cerrar el intervalo de la zona previa.
    """
    zona_anterior = datos.get("zona_actual")
    if zona_anterior is None:
        datos["zona_desde"] = ahora
    elif zona_anterior != zona_nueva:
        tiempos: dict[str, float] = datos.setdefault("zona_tiempos", {})
        tiempos[zona_anterior] = tiempos.get(zona_anterior, 0.0) + (ahora - datos.get("zona_desde", ahora))
        datos["zona_desde"] = ahora
    # zona_anterior == zona_nueva: el intervalo sigue abierto, no se toca zona_desde.


def _cerrar_intervalo_zona(datos: dict, ahora: float) -> dict[str, float]:
    """Tiempos por zona incluyendo el intervalo todavía abierto (para exponer
    en /api/clientes o al cerrar la sesión)."""
    tiempos = dict(datos.get("zona_tiempos", {}))
    zona_actual = datos.get("zona_actual")
    if zona_actual is not None:
        tiempos[zona_actual] = tiempos.get(zona_actual, 0.0) + (ahora - datos.get("zona_desde", ahora))
    return tiempos


def _guardar_foto_para_demografia(datos: dict, ahora: float, image_bytes: bytes) -> None:
    """Guarda una muestra para analizar recién al cerrar la sesión.

    No llama a InsightFace acá: sólo copia bytes bajo el lock (costo
    insignificante). Así la demografía nunca compite por CPU con el worker de
    Re-ID en tiempo real; el análisis pesado se hace una única vez, al final.
    """
    if not tiene_demografia:
        return
    fotos: list[bytes] = datos.setdefault("fotos_demografia", [])
    if ahora - datos.get("ultimo_guardado_demo", 0.0) < DEMO_INTERVALO_SEGUNDOS:
        return
    datos["ultimo_guardado_demo"] = ahora
    fotos.append(image_bytes)
    if len(fotos) > DEMO_MAX_FOTOS:
        fotos.pop(0)


def _rango_edad(edad: int) -> str:
    return "-18" if edad < 18 else "18-25" if edad <= 25 else "26-35" if edad <= 35 else "36-45" if edad <= 45 else "46+"


def _analizar_demografia_al_cierre(fotos: list[bytes]) -> tuple[str, str]:
    """Corre InsightFace sobre las fotos guardadas durante la sesión, todas
    juntas y una sola vez, cuando la persona ya se está por dar de baja. Nunca
    se ejecuta mientras la sesión está activa.
    """
    if not tiene_demografia or analizador_rostros is None or not fotos:
        return "No definido", "No definido"
    votos_genero: list[str] = []
    votos_edad: list[int] = []
    for image_bytes in fotos:
        try:
            imagen = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if imagen is None:
                continue
            rostros = analizador_rostros.get(imagen)
            if not rostros:
                continue
            mejor = max(rostros, key=lambda r: float(getattr(r, "det_score", 0.0)))
            if float(getattr(mejor, "det_score", 0.0)) < DEMO_MIN_DET_SCORE:
                continue
            votos_genero.append("Mujer" if mejor.sex == "F" else "Hombre")
            votos_edad.append(int(mejor.age))
        except Exception as error:
            logger.warning("No se pudo inferir edad/género de una muestra: %s", error)
    if not votos_edad:
        return "No definido", "No definido"
    genero = max(("Hombre", "Mujer"), key=lambda g: votos_genero.count(g))
    edad_mediana = int(round(float(np.median(votos_edad))))
    return genero, _rango_edad(edad_mediana)


def _procesar_deteccion(job: DetectionJob) -> str:
    """Trabajo CPU serializado: decodifica, extrae Re-ID y actualiza estado."""
    global contador_global_ids
    started_at = time.perf_counter()
    imagen_cv2 = cv2.imdecode(np.frombuffer(job.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if imagen_cv2 is None:
        raise ValueError("JPEG inválido recibido desde la cámara.")

    imagen_rgb = cv2.cvtColor(_recorte_para_reid(imagen_cv2), cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        huella_nueva = _normalizar_huella(extractor_ia([imagen_rgb])[0])

    ahora = time.time()
    posicion = (job.pos_x, job.pos_y)
    foto_b64 = base64.b64encode(job.image_bytes).decode("utf-8")
    id_local = f"{job.camara_id}_{job.tracker_id}" if job.tracker_id else None

    with state_lock:
        # posicion (píxeles nativos de la cámara) sigue usándose tal cual en
        # toda la lógica de Re-ID de abajo (distancias/umbrales tuneados en
        # esa escala). posicion_plano es sólo para heatmap/zona de negocio;
        # nunca se mezclan.
        posicion_plano = _transformar_a_plano(job.camara_id, *posicion)
        _registrar_heatmap(*posicion_plano)
        zona_calculada = _zona_en_punto(*posicion_plano) or job.zona

        # La ruta local es rápida, pero no actualiza el álbum ciegamente: un
        # recorte erróneo de ese tracker no debe alterar la identidad global.
        if id_local and id_local in traductor_camaras:
            id_global = traductor_camaras[id_local]["id_global"]
            if id_global in clientes_globales:
                datos = clientes_globales[id_global]
                _agregar_huella_confiable(datos, huella_nueva, ahora)
                datos["apariciones"] = datos.get("apariciones", 1) + 1
                _actualizar_zona_y_tiempo(datos, zona_calculada, ahora)
                datos.update({
                    "foto_b64": foto_b64,
                    "zona_actual": zona_calculada,
                    "timestamp": ahora,
                    "hora_legible": time.strftime("%H:%M:%S"),
                    "camara_id": job.camara_id,
                    "posicion": posicion,
                })
                traductor_camaras[id_local].update(ultimo_update=ahora, posicion=posicion)
                _guardar_foto_para_demografia(datos, ahora, job.image_bytes)
                _actualizar_metricas_procesado(started_at)
                return id_global

        bloqueados = _ids_bloqueados_en_misma_camara(job.camara_id, id_local, posicion, ahora)
        mejor_id_global: str | None = None
        mejor_puntaje = -1.0
        mejor_misma_camara = False
        candidatos_continuidad: list[tuple[str, float]] = []

        for persona_id, datos in clientes_globales.items():
            if datos.get("branch_id") != job.branch_id or persona_id in bloqueados:
                continue
            misma_camara = datos.get("camara_id") == job.camara_id
            # El salto de zona "imposible en tan poco tiempo" sólo tiene
            # sentido dentro de la misma cámara: cruzar de una cámara a otra
            # con áreas adyacentes es precisamente la transición más rápida
            # y más legítima que existe (hand-off entre cámaras).
            if (
                misma_camara
                and zona_calculada != datos.get("zona_actual")
                and ahora - datos.get("timestamp", 0.0) < TIEMPO_TELETRANSPORTACION
            ):
                continue

            puntaje = _puntaje_identidad(huella_nueva, datos)
            # Bonus pequeño para recuperar una persona que se mantuvo en el
            # mismo asiento cuando el tracker local cambia de ID. Sólo aplica
            # dentro de la misma cámara: comparar píxeles entre encuadres
            # distintos no tiene relación espacial real.
            if misma_camara and _distancia(datos.get("posicion"), posicion) <= DISTANCIA_REID_LOCAL_PX:
                puntaje += 0.04
            continuidad_postura = (
                misma_camara
                and ahora - datos.get("timestamp", 0.0) <= TIEMPO_CONTINUIDAD_POSTURA
                and _misma_ubicacion_sentado_parado(datos.get("posicion"), posicion)
            )
            if continuidad_postura:
                candidatos_continuidad.append((persona_id, puntaje))
            if puntaje > mejor_puntaje:
                mejor_puntaje, mejor_id_global, mejor_misma_camara = puntaje, persona_id, misma_camara

        continuidad_unica = len(candidatos_continuidad) == 1
        puede_recuperar_por_continuidad = (
            continuidad_unica
            and candidatos_continuidad[0][1] >= UMBRAL_SIMILITUD_CONTINUIDAD
        )
        umbral_aplicable = UMBRAL_SIMILITUD if mejor_misma_camara else UMBRAL_SIMILITUD_CROSS_CAMARA
        if mejor_id_global is not None and (
            mejor_puntaje >= umbral_aplicable or puede_recuperar_por_continuidad
        ):
            if puede_recuperar_por_continuidad:
                id_global, similitud_asignada = candidatos_continuidad[0]
            else:
                id_global, similitud_asignada = mejor_id_global, mejor_puntaje
            datos = clientes_globales[id_global]
            _agregar_huella_confiable(datos, huella_nueva, ahora)
            datos["apariciones"] = datos.get("apariciones", 1) + 1
        else:
            id_global = f"Cliente_Global_{contador_global_ids}"
            contador_global_ids += 1
            datos = {
                "historial": [huella_nueva],
                "centroide": huella_nueva.clone(),
                "ultimo_update_album": ahora,
                "hora_entrada": ahora,
                "zona_entrada": zona_calculada,
                "branch_id": job.branch_id,
                "apariciones": 1,
                "similitud_ema": 0.0,
                "fotos_demografia": [],
                "ultimo_guardado_demo": 0.0,
            }
            clientes_globales[id_global] = datos
            similitud_asignada = 0.0

        if id_local:
            traductor_camaras[id_local] = {
                "id_global": id_global,
                "ultimo_update": ahora,
                "posicion": posicion,
            }
        similitud_actual = max(0.0, similitud_asignada)
        # EMA para que el % mostrado no salte entre frames buenos y malos.
        datos["similitud_ema"] = (
            ALFA_SIMILITUD_EMA * similitud_actual
            + (1 - ALFA_SIMILITUD_EMA) * datos.get("similitud_ema", similitud_actual)
        )
        _actualizar_zona_y_tiempo(datos, zona_calculada, ahora)
        datos.update({
            "foto_b64": foto_b64,
            "zona_actual": zona_calculada,
            "similitud": datos["similitud_ema"],
            "timestamp": ahora,
            "hora_legible": time.strftime("%H:%M:%S"),
            "branch_id": job.branch_id,
            "camara_id": job.camara_id,
            "posicion": posicion,
        })
        _guardar_foto_para_demografia(datos, ahora, job.image_bytes)
        _actualizar_metricas_procesado(started_at)
        return id_global


def _actualizar_metricas_procesado(started_at: float) -> None:
    metricas["processed"] += 1
    metricas["last_processing_ms"] = round((time.perf_counter() - started_at) * 1000, 1)


async def _worker_reid() -> None:
    assert cola_reid is not None
    loop = asyncio.get_running_loop()
    while True:
        job = await cola_reid.get()
        try:
            await loop.run_in_executor(reid_executor, _procesar_deteccion, job)
        except Exception:
            logger.exception("No se pudo procesar una detección")
            with state_lock:
                metricas["failed"] += 1
        finally:
            cola_reid.task_done()


@app.post("/identificar")
async def identificar_persona(
    file: UploadFile = File(...),
    zona: str = Form("Desconocida"),
    tracker_id: str | None = Form(None),
    camara_id: str = Form("camara_default"),
    branch_id: str = Form("SUC-001"),
    pos_x: str = Form("0"),
    pos_y: str = Form("0"),
):
    """Acepta rápido; el procesamiento real ocurre en el worker acotado."""
    if cola_reid is None:
        raise HTTPException(status_code=503, detail="El servidor aún está iniciando.")
    try:
        posicion_x, posicion_y = float(pos_x), float(pos_y)
    except ValueError as error:
        raise HTTPException(status_code=422, detail="pos_x y pos_y deben ser números.") from error

    contenido = await file.read()
    if not contenido:
        raise HTTPException(status_code=422, detail="La imagen está vacía.")
    trabajo = DetectionJob(contenido, zona, tracker_id, camara_id, branch_id, posicion_x, posicion_y)
    try:
        cola_reid.put_nowait(trabajo)
    except asyncio.QueueFull as error:
        with state_lock:
            metricas["rejected_full"] += 1
        raise HTTPException(status_code=429, detail="Cola de Re-ID llena; reintentar.") from error
    with state_lock:
        metricas["accepted"] += 1
    return {"status": "queued", "queue_size": cola_reid.qsize()}


def procesar_y_enviar_supabase(pid: str, datos: dict, tiempo_adentro: int) -> None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase no configurado; no se guarda la sesión %s.", pid)
        return
    genero, rango_edad = _analizar_demografia_al_cierre(datos.get("fotos_demografia", []))
    payload = {
        "branch_id": datos.get("branch_id", "SUC-001"),
        "tracker_id": pid,
        "gender": genero,
        "age_range": rango_edad,
        "entered_at": datetime.fromtimestamp(datos["hora_entrada"], timezone.utc).isoformat(),
        "exited_at": datetime.fromtimestamp(datos["timestamp"], timezone.utc).isoformat(),
        "dwell_time_seconds": tiempo_adentro,
    }
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    try:
        response = requests.post(SUPABASE_URL, json=payload, headers=headers, timeout=5)
        response.raise_for_status()
    except requests.RequestException as error:
        cuerpo = getattr(error.response, "text", "")
        logger.warning("No se pudo guardar la sesión en Supabase: %s | body=%s", error, cuerpo)
        return
    logger.info("Sesión %s guardada en Supabase (%ds, %s, %s).", pid, tiempo_adentro, genero, rango_edad)


async def reloj_limpiador_background() -> None:
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(20)
        ahora = time.time()
        with state_lock:
            vencidos = [pid for pid, datos in clientes_globales.items() if ahora - datos.get("timestamp", ahora) > TIEMPO_INACTIVIDAD_SEGUNDOS]
            sesiones = [(pid, clientes_globales.pop(pid)) for pid in vencidos]
            for _pid, datos in sesiones:
                # Cierra el intervalo de la última zona antes de que la sesión
                # se pierda; si no, el tiempo desde "zona_desde" hasta ahora
                # nunca se contabiliza en zona_tiempos.
                datos["zona_tiempos"] = _cerrar_intervalo_zona(datos, ahora)
            for clave in [k for k, v in traductor_camaras.items() if ahora - v.get("ultimo_update", 0.0) > TIEMPO_INACTIVIDAD_SEGUNDOS]:
                traductor_camaras.pop(clave, None)
        for pid, datos in sesiones:
            tiempo_adentro = int(datos["timestamp"] - datos["hora_entrada"])
            if tiempo_adentro > 5:
                await loop.run_in_executor(supabase_executor, procesar_y_enviar_supabase, pid, datos, tiempo_adentro)


@app.on_event("startup")
async def iniciar_servicios() -> None:
    global cola_reid
    _cargar_plano_config()
    cola_reid = asyncio.Queue(maxsize=REID_QUEUE_SIZE)
    for _ in range(REID_WORKERS):
        asyncio.create_task(_worker_reid())
    asyncio.create_task(reloj_limpiador_background())


@app.get("/health")
def health() -> dict:
    with state_lock:
        return {
            **metricas,
            "queue_size": cola_reid.qsize() if cola_reid else None,
            "queue_capacity": REID_QUEUE_SIZE,
            "active_global_ids": len(clientes_globales),
            "heatmap_size": {"width": HEATMAP_WIDTH, "height": HEATMAP_HEIGHT},
        }


@app.get("/api/clientes")
def obtener_clientes() -> dict:
    ahora = time.time()
    with state_lock:
        clientes = []
        for persona_id, datos in clientes_globales.items():
            # Los clientes provisionales (una sola aparición) no se muestran
            # todavía: evita tarjetas fantasma por un recorte espurio.
            if datos.get("apariciones", 1) < MIN_APARICIONES_VISIBLE:
                continue
            clientes.append({
                "id": persona_id,
                "zona": datos.get("zona_actual", "Desconocida"),
                "similitud": f"{datos.get('similitud', 0.0) * 100:.1f}%",
                "ultima_vista": datos.get("hora_legible", "--:--:--"),
                # El género/edad se calculan recién al cerrar la sesión (así
                # nunca compiten por CPU con el Re-ID en vivo); mientras la
                # persona sigue activa no hay nada que mostrar todavía.
                "genero": "Pendiente",
                "edad": "Pendiente",
                "foto": datos.get("foto_b64", ""),
                "tiempos_zona": {k: round(v, 1) for k, v in _cerrar_intervalo_zona(datos, ahora).items()},
            })
    return {"clientes": clientes}


@app.get("/api/heatmap")
def obtener_heatmap() -> dict:
    with state_lock:
        puntos, maximo = _snapshot_heatmap()
    return {"width": HEATMAP_WIDTH, "height": HEATMAP_HEIGHT, "puntos": puntos, "max": round(maximo, 3)}


@app.get("/api/zonas_camara")
def proxy_zonas() -> dict:
    """Las zonas ya están en las mismas coordenadas nativas del heatmap."""
    try:
        response = requests.get(f"{IP_CAMARA}/config", timeout=2)
        response.raise_for_status()
        return {"zones": response.json().get("zones", [])}
    except requests.RequestException as error:
        logger.warning("No se pudieron leer las zonas de la cámara: %s", error)
        return {"zones": []}


@app.post("/api/reset")
def resetear_memoria() -> dict:
    global contador_global_ids
    with state_lock:
        clientes_globales.clear()
        traductor_camaras.clear()
        heatmap_celdas.clear()
        contador_global_ids = 1
    return {"ok": True}


# --- Plano único multicámara: registro/calibración de cámaras y zonas de
# negocio. Ver plano_config.json / _transformar_a_plano / _zona_en_punto. ---


@app.get("/api/camaras")
def listar_camaras() -> dict:
    with state_lock:
        camaras = [
            {**datos, "calibrada": bool(datos.get("puntos_plano"))}
            for datos in calibraciones_camaras.values()
        ]
    return {"camaras": camaras}


@app.post("/api/camaras")
def registrar_camara(camara: CamaraIn) -> dict:
    with state_lock:
        existente = calibraciones_camaras.get(camara.camara_id, {})
        registro = {
            "camara_id": camara.camara_id,
            "nombre": camara.nombre,
            "video_url": camara.video_url,
            "puntos_plano": existente.get("puntos_plano"),
            "puntos_camara": existente.get("puntos_camara"),
            "posicion": existente.get("posicion"),
        }
        calibraciones_camaras[camara.camara_id] = registro
        _guardar_plano_config()
    return registro


@app.post("/api/camaras/{camara_id}/posicion")
def posicionar_camara(camara_id: str, datos: PosicionIn) -> dict:
    """Dónde está físicamente la cámara en el plano. Es sólo para mostrarla en
    el mapa: la proyección de lo que ve sale de la homografía, no de acá."""
    if camara_id not in calibraciones_camaras:
        raise HTTPException(status_code=404, detail="Cámara no registrada.")
    if len(datos.posicion) != 2:
        raise HTTPException(status_code=422, detail="La posición debe ser un punto [x, y].")
    with state_lock:
        calibraciones_camaras[camara_id]["posicion"] = datos.posicion
        _guardar_plano_config()
        registro = calibraciones_camaras[camara_id]
    return {"ok": True, **registro}


@app.delete("/api/camaras/{camara_id}")
def borrar_camara(camara_id: str) -> dict:
    with state_lock:
        calibraciones_camaras.pop(camara_id, None)
        _homografias_cache.pop(camara_id, None)
        _guardar_plano_config()
    return {"ok": True}


@app.post("/api/camaras/{camara_id}/calibracion")
def calibrar_camara(camara_id: str, calibracion: CalibracionIn) -> dict:
    if camara_id not in calibraciones_camaras:
        raise HTTPException(status_code=404, detail="Cámara no registrada. Registrala primero con POST /api/camaras.")
    if len(calibracion.puntos_plano) != 4:
        raise HTTPException(status_code=422, detail="Se necesitan exactamente 4 puntos (TL, TR, BR, BL).")
    if _area_poligono(calibracion.puntos_plano) < 1.0:
        raise HTTPException(status_code=422, detail="Los 4 puntos son colineales o están duplicados; el cuadrilátero es inválido.")
    if calibracion.puntos_camara is not None:
        if len(calibracion.puntos_camara) != 4:
            raise HTTPException(status_code=422, detail="Se necesitan exactamente 4 puntos sobre la imagen de la cámara.")
        if _area_poligono(calibracion.puntos_camara) < 1.0:
            raise HTTPException(status_code=422, detail="Los 4 puntos marcados sobre el video son colineales o están duplicados.")
    with state_lock:
        calibraciones_camaras[camara_id]["puntos_plano"] = calibracion.puntos_plano
        calibraciones_camaras[camara_id]["puntos_camara"] = calibracion.puntos_camara
        _recalcular_homografia(camara_id)
        _guardar_plano_config()
        registro = calibraciones_camaras[camara_id]
    return {"ok": True, **registro}


@app.delete("/api/camaras/{camara_id}/calibracion")
def borrar_calibracion(camara_id: str) -> dict:
    if camara_id not in calibraciones_camaras:
        raise HTTPException(status_code=404, detail="Cámara no registrada.")
    with state_lock:
        calibraciones_camaras[camara_id]["puntos_plano"] = None
        calibraciones_camaras[camara_id]["puntos_camara"] = None
        _homografias_cache.pop(camara_id, None)
        _guardar_plano_config()
    return {"ok": True}


@app.get("/api/plano")
def obtener_plano() -> dict:
    with state_lock:
        return {
            "width": HEATMAP_WIDTH,
            "height": HEATMAP_HEIGHT,
            "contorno": list(contorno_local),
            "tiene_imagen": os.path.exists(PLANO_IMAGEN_PATH),
        }


@app.post("/api/plano/contorno")
def guardar_contorno(contorno: ContornoIn) -> dict:
    if len(contorno.polygon) < 3:
        raise HTTPException(status_code=422, detail="El contorno del local necesita al menos 3 puntos.")
    if _area_poligono(contorno.polygon) < 1.0:
        raise HTTPException(status_code=422, detail="El contorno es degenerado (puntos colineales o duplicados).")
    with state_lock:
        contorno_local[:] = contorno.polygon
        _guardar_plano_config()
    return {"ok": True, "contorno": list(contorno_local)}


@app.delete("/api/plano/contorno")
def borrar_contorno() -> dict:
    with state_lock:
        contorno_local.clear()
        _guardar_plano_config()
    return {"ok": True}


@app.post("/api/plano/imagen")
async def subir_imagen_plano(file: UploadFile = File(...)) -> dict:
    """Sube una imagen real del plano (foto, plano dibujado, etc.) para
    mostrar de fondo en vez de la grilla abstracta. Se guarda como archivo
    aparte (no adentro del JSON); la coordenadas del plano (640x480) no
    dependen de su resolución real — se muestra recortada para llenar el
    mismo espacio donde ya se calibran cámaras y zonas.
    """
    contenido = await file.read()
    if not contenido:
        raise HTTPException(status_code=422, detail="La imagen está vacía.")
    if len(contenido) > PLANO_IMAGEN_MAX_BYTES:
        raise HTTPException(status_code=422, detail=f"La imagen pesa más de {PLANO_IMAGEN_MAX_BYTES // (1024*1024)}MB.")
    imagen = cv2.imdecode(np.frombuffer(contenido, np.uint8), cv2.IMREAD_COLOR)
    if imagen is None:
        raise HTTPException(status_code=422, detail="No se pudo leer el archivo como imagen (¿es un JPG/PNG válido?).")
    alto, ancho = imagen.shape[:2]
    lado_mayor = max(alto, ancho)
    if lado_mayor > PLANO_IMAGEN_LADO_MAXIMO_PX:
        factor = PLANO_IMAGEN_LADO_MAXIMO_PX / lado_mayor
        imagen = cv2.resize(imagen, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", imagen, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo procesar la imagen.")
    ruta_tmp = f"{PLANO_IMAGEN_PATH}.tmp"
    with open(ruta_tmp, "wb") as archivo:
        archivo.write(buf.tobytes())
    os.replace(ruta_tmp, PLANO_IMAGEN_PATH)
    return {"ok": True}


@app.get("/api/plano/imagen")
def obtener_imagen_plano() -> FileResponse:
    if not os.path.exists(PLANO_IMAGEN_PATH):
        raise HTTPException(status_code=404, detail="No hay imagen del plano cargada todavía.")
    return FileResponse(PLANO_IMAGEN_PATH, media_type="image/jpeg")


@app.delete("/api/plano/imagen")
def borrar_imagen_plano() -> dict:
    try:
        os.remove(PLANO_IMAGEN_PATH)
    except FileNotFoundError:
        pass
    return {"ok": True}


@app.get("/api/zonas")
def listar_zonas() -> dict:
    with state_lock:
        return {"zones": list(zonas_negocio)}


@app.post("/api/zonas")
def crear_zona(zona: ZonaIn) -> dict:
    if len(zona.polygon) < 3:
        raise HTTPException(status_code=422, detail="La zona necesita al menos 3 puntos.")
    with state_lock:
        nueva = {
            "id": f"zona_{len(zonas_negocio) + 1}_{int(time.time())}",
            "name": zona.name,
            "color": zona.color,
            "polygon": zona.polygon,
        }
        zonas_negocio.append(nueva)
        _guardar_plano_config()
    return nueva


@app.put("/api/zonas/{zona_id}")
def editar_zona(zona_id: str, zona: ZonaIn) -> dict:
    if len(zona.polygon) < 3:
        raise HTTPException(status_code=422, detail="La zona necesita al menos 3 puntos.")
    with state_lock:
        for existente in zonas_negocio:
            if existente["id"] == zona_id:
                existente.update(name=zona.name, color=zona.color, polygon=zona.polygon)
                _guardar_plano_config()
                return existente
    raise HTTPException(status_code=404, detail="Zona no encontrada.")


@app.delete("/api/zonas/{zona_id}")
def borrar_zona(zona_id: str) -> dict:
    with state_lock:
        restantes = [z for z in zonas_negocio if z["id"] != zona_id]
        if len(restantes) == len(zonas_negocio):
            raise HTTPException(status_code=404, detail="Zona no encontrada.")
        zonas_negocio[:] = restantes
        _guardar_plano_config()
    return {"ok": True}


@app.get("/calibrar", response_class=HTMLResponse)
def panel_calibracion() -> str:
    return """
    <!doctype html><html lang="es"><head><meta charset="utf-8">
    <title>LeanVision Cerebro — Calibración</title>
    <style>
      body{margin:0;padding:24px;background:#0f172a;color:#e2e8f0;font:15px system-ui,sans-serif}
      header,.layout{max-width:1280px;margin:auto}
      header{display:flex;align-items:center;gap:16px;margin-bottom:20px}
      header a{color:#7dd3fc}
      .layout{display:flex;gap:24px;align-items:start;flex-wrap:wrap}
      .panel{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px}
      .lado{min-width:260px;flex:0 0 260px}
      .map{width:640px;max-width:100%;box-sizing:border-box}
      #map{position:relative;width:640px;max-width:100%;aspect-ratio:640/480;overflow:hidden;background:#111827;background-image:linear-gradient(#33415555 1px,transparent 1px),linear-gradient(90deg,#33415555 1px,transparent 1px);background-size:40px 40px}
      #plano{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}
      input,select{width:100%;box-sizing:border-box;padding:7px 8px;margin:4px 0;background:#0f172a;border:1px solid #334155;border-radius:6px;color:#e2e8f0}
      button{padding:8px 12px;background:#2563eb;color:white;border:0;border-radius:6px;cursor:pointer;margin:3px 3px 3px 0}
      button.rojo{background:#ef4444}button.gris{background:#475569}
      .fila{display:flex;gap:6px}.fila input{flex:1}
      .item{background:#334155;border-radius:8px;padding:8px 10px;margin-bottom:8px;font-size:13px}
      .item b{display:block}
      .item .acciones{margin-top:6px}
      .item button{padding:5px 8px;font-size:12px}
      .previa-wrap{position:relative;width:100%;border-radius:8px;overflow:hidden;background:#111827;aspect-ratio:640/480}
      .previa-wrap img{width:100%;height:100%;object-fit:contain;display:block}
      #video-canvas{position:absolute;inset:0;width:100%;height:100%}
      #video-canvas.activo{cursor:crosshair;box-shadow:inset 0 0 0 3px #facc15}
      #instrucciones{min-height:20px;color:#facc15;font-size:14px}
      .badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;margin-left:6px}
      .badge.si{background:#16653477;color:#86efac}.badge.no{background:#7f1d1d77;color:#fca5a5}
      .badge.pos{background:#1e3a8a77;color:#93c5fd}
      .paso{font-size:12px;color:#94a3b8;margin:6px 0 0}
    </style></head><body>
    <header><h1 style="margin:0">Calibración — plano único</h1><a href="/">&larr; Volver al dashboard</a></header>
    <main class="layout">

      <section class="panel lado">
        <h2>Cámaras</h2>
        <div class="fila"><input id="c_id" placeholder="ID (ej: camara_zero_1)"></div>
        <div class="fila"><input id="c_nombre" placeholder="Nombre"></div>
        <div class="fila"><input id="c_url" placeholder="http://IP:8002/video"></div>
        <button onclick="registrarCamara()">Registrar / actualizar</button>
        <div id="lista-camaras" style="margin-top:12px"></div>
        <h2>Vista en vivo</h2>
        <div class="previa-wrap">
          <img id="video-previa" src="">
          <canvas id="video-canvas" width="640" height="480"></canvas>
        </div>
        <p class="paso" id="paso-video">Seleccioná una cámara para ver su video.</p>
      </section>

      <section class="panel map">
        <h2>Plano del local</h2>
        <div class="fila" style="align-items:center">
          <input id="imagen_input" type="file" accept="image/*">
          <button onclick="subirImagenPlano()">Subir imagen del plano</button>
          <button class="rojo" id="btn-quitar-imagen" onclick="quitarImagenPlano()" style="display:none">Quitar imagen</button>
        </div>
        <p class="paso">Se recorta para llenar el recuadro de 640×480 (mejor si es apaisada, proporción similar a 4:3).</p>
        <div>
          <button onclick="modoContorno()">1. Dibujar local</button>
          <button onclick="modoPosicion()">2. Ubicar cámara</button>
          <button onclick="modoCalibrar()">3. Calibrar cámara</button>
          <button onclick="modoZona()">4. Dibujar zona</button>
        </div>
        <p id="instrucciones">Empezá dibujando la forma del local con el paso 1.</p>
        <div id="map"><canvas id="plano" width="640" height="480"></canvas></div>
        <div id="controles-contorno" style="display:none">
          <button onclick="guardarContorno()">Guardar forma del local</button>
          <button class="gris" onclick="limpiarDibujo()">Limpiar</button>
          <button class="rojo" onclick="borrarContorno()">Borrar forma guardada</button>
        </div>
        <div id="controles-calibracion" style="display:none">
          <button onclick="confirmarCalibracion()">Confirmar calibración</button>
          <button class="gris" onclick="limpiarDibujo()">Rehacer</button>
        </div>
        <div id="controles-zona" style="display:none">
          <div class="fila">
            <input id="z_nombre" placeholder="Nombre de la zona">
            <input id="z_color" type="color" value="#00ff88" style="width:44px;padding:2px">
          </div>
          <button onclick="guardarZona()">Guardar zona (doble clic para cerrar el polígono)</button>
          <button class="gris" onclick="limpiarDibujo()">Limpiar</button>
        </div>
      </section>

      <section class="panel lado">
        <h2>Zonas de negocio</h2>
        <div id="lista-zonas"></div>
      </section>

    </main>
    <script>
      const $=s=>document.querySelector(s);
      const canvas=$('#plano'), ctx=canvas.getContext('2d');
      const vcanvas=$('#video-canvas'), vctx=vcanvas.getContext('2d');
      const PALETA=['#38bdf8','#f472b6','#a78bfa','#fb923c','#4ade80'];
      let camaras=[], zonas=[], contorno=[], camaraSel=null, modo=null;
      let puntosCalibracion=[], poligonoZona=[], poligonoContorno=[], puntosVideo=[];
      let faseCalibracion='video';

      function pointFrom(event,el,w,h){
        const rect=el.getBoundingClientRect();
        return [
          Math.round((event.clientX-rect.left)*(w/rect.width)),
          Math.round((event.clientY-rect.top)*(h/rect.height)),
        ];
      }
      const pointFromEvent=e=>pointFrom(e,canvas,canvas.width,canvas.height);

      function colorCamara(camara_id){
        const idx=camaras.findIndex(c=>c.camara_id===camara_id);
        return PALETA[idx>=0?idx%PALETA.length:0];
      }

      function dibujarPoligono(puntos,color,relleno,etiqueta,punteado){
        if(!puntos||puntos.length<2) return;
        ctx.setLineDash(punteado?[6,4]:[]);
        ctx.beginPath();ctx.moveTo(...puntos[0]);puntos.slice(1).forEach(p=>ctx.lineTo(...p));
        if(puntos.length>=3){ctx.closePath();}
        ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();
        if(relleno&&puntos.length>=3){ctx.fillStyle=color+'33';ctx.fill();}
        ctx.setLineDash([]);
        if(etiqueta) {ctx.fillStyle='white';ctx.fillText(etiqueta,puntos[0][0]+5,puntos[0][1]-5);}
      }

      function marcarPuntos(contexto,puntos,color,numerar){
        puntos.forEach((p,i)=>{
          contexto.fillStyle=color;contexto.beginPath();contexto.arc(p[0],p[1],6,0,7);contexto.fill();
          if(numerar){contexto.fillStyle='#111827';contexto.font='bold 11px sans-serif';contexto.fillText(i+1,p[0]-3,p[1]+4);}
        });
      }

      function dibujarCamara(c){
        if(!c.posicion) return;
        const [x,y]=c.posicion, color=colorCamara(c.camara_id);
        ctx.beginPath();ctx.arc(x,y,11,0,7);
        ctx.fillStyle=color;ctx.fill();
        ctx.strokeStyle='#0f172a';ctx.lineWidth=2;ctx.stroke();
        ctx.fillStyle='#0f172a';ctx.font='bold 12px sans-serif';ctx.fillText('C',x-4,y+4);
        ctx.fillStyle=color;ctx.font='12px sans-serif';
        ctx.fillText(c.nombre||c.camara_id,x+15,y+4);
      }

      function redraw(){
        ctx.clearRect(0,0,canvas.width,canvas.height);
        if(contorno.length>=3){
          ctx.beginPath();ctx.moveTo(...contorno[0]);contorno.slice(1).forEach(p=>ctx.lineTo(...p));ctx.closePath();
          ctx.fillStyle='#1e293b88';ctx.fill();
          ctx.strokeStyle='#94a3b8';ctx.lineWidth=3;ctx.stroke();
        }
        zonas.forEach(z=>dibujarPoligono(z.polygon,z.color||'#00ff88',true,z.name,false));
        camaras.forEach(c=>{if(c.puntos_plano) dibujarPoligono(c.puntos_plano,colorCamara(c.camara_id),false,null,true);});
        camaras.forEach(dibujarCamara);
        if(modo==='calibrar'&&faseCalibracion==='plano'){
          dibujarPoligono(puntosCalibracion,'#facc15',false,null,false);
          marcarPuntos(ctx,puntosCalibracion,'#facc15',true);
        } else if(modo==='zona'){
          dibujarPoligono(poligonoZona,'#facc15',poligonoZona.length>=3,null,false);
          marcarPuntos(ctx,poligonoZona,'#facc15',false);
        } else if(modo==='contorno'){
          dibujarPoligono(poligonoContorno,'#38bdf8',poligonoContorno.length>=3,null,false);
          marcarPuntos(ctx,poligonoContorno,'#38bdf8',false);
        }
      }

      function redrawVideo(){
        vctx.clearRect(0,0,vcanvas.width,vcanvas.height);
        if(modo!=='calibrar') return;
        dibujarEn(vctx,puntosVideo,'#facc15');
        marcarPuntos(vctx,puntosVideo,'#facc15',true);
      }
      function dibujarEn(contexto,puntos,color){
        if(puntos.length<2) return;
        contexto.beginPath();contexto.moveTo(...puntos[0]);puntos.slice(1).forEach(p=>contexto.lineTo(...p));
        if(puntos.length>=3) contexto.closePath();
        contexto.strokeStyle=color;contexto.lineWidth=2;contexto.stroke();
      }

      function ocultarControles(){
        $('#controles-calibracion').style.display='none';
        $('#controles-zona').style.display='none';
        $('#controles-contorno').style.display='none';
        vcanvas.classList.remove('activo');
      }

      function modoContorno(){
        modo='contorno'; poligonoContorno=[];
        ocultarControles();
        $('#controles-contorno').style.display='block';
        $('#instrucciones').textContent='Dibujá la forma del local: un clic por esquina (puede ser en L o irregular), doble clic para cerrar.';
        redraw(); redrawVideo();
      }

      function modoPosicion(){
        if(!camaraSel) return alert('Elegí primero una cámara de la lista.');
        modo='posicion';
        ocultarControles();
        $('#instrucciones').textContent=`Hacé clic en el plano donde está físicamente la cámara "${camaraSel}".`;
        redraw(); redrawVideo();
      }

      function modoCalibrar(){
        if(!camaraSel) return alert('Elegí primero una cámara de la lista.');
        modo='calibrar'; faseCalibracion='video'; puntosVideo=[]; puntosCalibracion=[];
        ocultarControles();
        vcanvas.classList.add('activo');
        $('#instrucciones').textContent='Paso A — marcá 4 puntos del PISO sobre el video (izquierda). Elegí referencias reconocibles: esquinas de una baldosa, del mostrador, etc.';
        $('#paso-video').textContent='Clic 1 de 4 sobre el video.';
        redraw(); redrawVideo();
      }

      function modoZona(){
        modo='zona'; poligonoZona=[];
        ocultarControles();
        $('#controles-zona').style.display='block';
        $('#instrucciones').textContent='Clic para agregar vértices de la zona (mínimo 3), doble clic para cerrar.';
        redraw(); redrawVideo();
      }

      function limpiarDibujo(){
        puntosCalibracion=[]; poligonoZona=[]; poligonoContorno=[]; puntosVideo=[];
        if(modo==='calibrar') modoCalibrar();
        else if(modo==='contorno') modoContorno();
        else {redraw(); redrawVideo();}
      }

      vcanvas.addEventListener('click',(event)=>{
        if(modo!=='calibrar'||faseCalibracion!=='video') return;
        if(puntosVideo.length>=4) return;
        puntosVideo.push(pointFrom(event,vcanvas,640,480));
        redrawVideo();
        if(puntosVideo.length<4){
          $('#paso-video').textContent=`Clic ${puntosVideo.length+1} de 4 sobre el video.`;
        } else {
          faseCalibracion='plano';
          vcanvas.classList.remove('activo');
          $('#paso-video').textContent='4 puntos marcados en el video.';
          $('#instrucciones').textContent='Paso B — ahora marcá esos MISMOS 4 puntos en el plano, en el mismo orden (1, 2, 3, 4).';
        }
      });

      canvas.addEventListener('click',(event)=>{
        if(!modo) return;
        const p=pointFromEvent(event);
        if(modo==='posicion'){
          guardarPosicion(p);
        } else if(modo==='calibrar'){
          if(faseCalibracion!=='plano'||puntosCalibracion.length>=4) return;
          puntosCalibracion.push(p);
          redraw();
          if(puntosCalibracion.length<4){
            $('#instrucciones').textContent=`Paso B — marcá el punto ${puntosCalibracion.length+1} de 4 en el plano (el mismo que marcaste en el video).`;
          } else {
            $('#instrucciones').textContent='Los 4 pares están listos. Revisá y confirmá, o rehacé.';
            $('#controles-calibracion').style.display='block';
          }
        } else if(modo==='zona'){
          if(event.detail===2){
            if(poligonoZona.length>=3) $('#instrucciones').textContent='Zona lista — ponele nombre y guardala.';
            return;
          }
          poligonoZona.push(p); redraw();
        } else if(modo==='contorno'){
          if(event.detail===2){
            if(poligonoContorno.length>=3) $('#instrucciones').textContent='Forma lista — guardala.';
            return;
          }
          poligonoContorno.push(p); redraw();
        }
      });

      async function guardarPosicion(punto){
        const r=await fetch(`/api/camaras/${camaraSel}/posicion`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({posicion:punto})});
        const data=await r.json();
        if(!r.ok){alert(data.detail||'Error al ubicar la cámara');return;}
        modo=null;
        $('#instrucciones').textContent='Ubicación guardada. Seguí con el paso 3 para calibrar lo que ve.';
        await cargarCamaras();
      }

      async function guardarContorno(){
        if(poligonoContorno.length<3) return alert('Dibujá al menos 3 puntos.');
        const r=await fetch('/api/plano/contorno',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({polygon:poligonoContorno})});
        const data=await r.json();
        if(!r.ok){alert(data.detail||'Error al guardar la forma');return;}
        modo=null; poligonoContorno=[];
        ocultarControles();
        $('#instrucciones').textContent='Forma del local guardada.';
        await cargarPlano();
      }

      async function borrarContorno(){
        if(!confirm('¿Borrar la forma del local?')) return;
        await fetch('/api/plano/contorno',{method:'DELETE'});
        poligonoContorno=[]; modo=null; ocultarControles();
        await cargarPlano();
      }

      async function confirmarCalibracion(){
        const r=await fetch(`/api/camaras/${camaraSel}/calibracion`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({puntos_plano:puntosCalibracion,puntos_camara:puntosVideo})});
        const data=await r.json();
        if(!r.ok){alert(data.detail||'Error al calibrar');return;}
        modo=null; puntosCalibracion=[]; puntosVideo=[];
        ocultarControles();
        $('#instrucciones').textContent='Calibración guardada. Lo que ve esta cámara ya se proyecta al plano.';
        await cargarCamaras(); redrawVideo();
      }

      async function guardarZona(){
        const nombre=$('#z_nombre').value.trim();
        if(!nombre) return alert('Ponele un nombre a la zona.');
        if(poligonoZona.length<3) return alert('Dibujá al menos 3 puntos.');
        const r=await fetch('/api/zonas',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:nombre,color:$('#z_color').value,polygon:poligonoZona})});
        const data=await r.json();
        if(!r.ok){alert(data.detail||'Error al guardar la zona');return;}
        poligonoZona=[]; $('#z_nombre').value='';
        $('#instrucciones').textContent='Zona guardada.';
        await cargarZonas();
      }

      async function registrarCamara(){
        const camara_id=$('#c_id').value.trim(), video_url=$('#c_url').value.trim();
        if(!camara_id||!video_url) return alert('ID y URL de video son obligatorios.');
        await fetch('/api/camaras',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({camara_id,nombre:$('#c_nombre').value.trim(),video_url})});
        $('#c_id').value='';$('#c_nombre').value='';$('#c_url').value='';
        await cargarCamaras();
      }
      function seleccionarCamara(id){
        camaraSel=id;
        const cam=camaras.find(c=>c.camara_id===id);
        $('#video-previa').src=cam?cam.video_url:'';
        $('#paso-video').textContent=cam?`Viendo "${cam.nombre||id}".`:'';
        if(modo==='calibrar'||modo==='posicion'){modo=null; ocultarControles(); $('#instrucciones').textContent='Cámara cambiada: elegí de nuevo el paso 2 o 3.';}
        puntosVideo=[]; puntosCalibracion=[];
        cargarCamaras(); redrawVideo();
      }
      async function borrarCalibracion(id){
        await fetch(`/api/camaras/${id}/calibracion`,{method:'DELETE'});
        await cargarCamaras();
      }
      async function quitarCamara(id){
        if(!confirm(`¿Quitar la cámara "${id}"?`)) return;
        await fetch(`/api/camaras/${id}`,{method:'DELETE'});
        if(camaraSel===id) camaraSel=null;
        await cargarCamaras();
      }
      async function borrarZona(id){
        await fetch(`/api/zonas/${id}`,{method:'DELETE'});
        await cargarZonas();
      }

      async function cargarCamaras(){
        camaras=(await (await fetch('/api/camaras')).json()).camaras;
        $('#lista-camaras').innerHTML=camaras.map(c=>`
          <div class="item" style="border-left:4px solid ${colorCamara(c.camara_id)}${camaraSel===c.camara_id?';outline:2px solid #7dd3fc':''}">
            <b>${c.nombre||c.camara_id}
              <span class="badge ${c.posicion?'pos':'no'}">${c.posicion?'ubicada':'sin ubicar'}</span>
              <span class="badge ${c.calibrada?'si':'no'}">${c.calibrada?'calibrada':'sin calibrar'}</span>
            </b>
            ${c.camara_id}
            <div class="acciones">
              <button onclick="seleccionarCamara('${c.camara_id}')">Seleccionar</button>
              ${c.calibrada?`<button class="gris" onclick="borrarCalibracion('${c.camara_id}')">Borrar calibración</button>`:''}
              <button class="rojo" onclick="quitarCamara('${c.camara_id}')">Quitar</button>
            </div>
          </div>`).join('')||'<p>Sin cámaras registradas.</p>';
        redraw();
      }
      async function cargarZonas(){
        zonas=(await (await fetch('/api/zonas')).json()).zones;
        $('#lista-zonas').innerHTML=zonas.map(z=>`
          <div class="item" style="border-left:4px solid ${z.color}">
            <b>${z.name}</b>
            <div class="acciones"><button class="rojo" onclick="borrarZona('${z.id}')">Borrar</button></div>
          </div>`).join('')||'<p>Sin zonas todavía.</p>';
        redraw();
      }
      function aplicarFondoPlano(tieneImagen){
        const el=$('#map');
        if(tieneImagen){
          el.style.backgroundImage=`url(/api/plano/imagen?t=${Date.now()})`;
          el.style.backgroundSize='cover';
          el.style.backgroundPosition='center';
          $('#btn-quitar-imagen').style.display='inline-block';
        } else {
          el.style.backgroundImage='';
          el.style.backgroundSize='';
          el.style.backgroundPosition='';
          $('#btn-quitar-imagen').style.display='none';
        }
      }

      async function cargarPlano(){
        const p=await (await fetch('/api/plano')).json();
        contorno=p.contorno||[];
        aplicarFondoPlano(p.tiene_imagen);
        redraw();
      }

      async function subirImagenPlano(){
        const input=$('#imagen_input');
        if(!input.files.length) return alert('Elegí un archivo de imagen primero.');
        const form=new FormData();
        form.append('file',input.files[0]);
        const r=await fetch('/api/plano/imagen',{method:'POST',body:form});
        const data=await r.json();
        if(!r.ok){alert(data.detail||'Error al subir la imagen');return;}
        input.value='';
        await cargarPlano();
      }

      async function quitarImagenPlano(){
        if(!confirm('¿Quitar la imagen del plano? Vuelve a la grilla.')) return;
        await fetch('/api/plano/imagen',{method:'DELETE'});
        await cargarPlano();
      }

      cargarPlano(); cargarCamaras(); cargarZonas();
    </script></body></html>
    """


@app.get("/", response_class=HTMLResponse)
def panel_web() -> str:
    return """
    <!doctype html><html lang="es"><head><meta charset="utf-8">
    <title>LeanVision Cerebro</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/heatmap.js/2.0.2/heatmap.min.js"></script>
    <style>
      body{margin:0;padding:24px;background:#0f172a;color:#e2e8f0;font:15px system-ui,sans-serif}
      header,.layout{max-width:1280px;margin:auto}.layout{display:flex;gap:28px;align-items:start;flex-wrap:wrap}
      .panel{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px}.clients{flex:1;min-width:300px}.map{width:640px;max-width:100%;box-sizing:border-box}
      #map{position:relative;width:640px;max-width:100%;aspect-ratio:640/480;overflow:hidden;background:#111827;background-image:linear-gradient(#33415555 1px,transparent 1px),linear-gradient(90deg,#33415555 1px,transparent 1px);background-size:40px 40px}
      #heatmap,#zones{position:absolute;inset:0;width:100%;height:100%}#zones{pointer-events:none}.grid{display:flex;gap:12px;flex-wrap:wrap}.card{width:170px;background:#334155;border-radius:8px;padding:10px}.card img{width:100%;height:130px;object-fit:cover;border-radius:6px}button{padding:9px 12px;background:#ef4444;color:white;border:0;border-radius:6px;cursor:pointer}
    </style></head><body><header><h1>LeanVision Cerebro</h1><p id="health">Cargando métricas…</p><a href="/calibrar" style="color:#7dd3fc;margin-right:12px">Calibrar cámaras / zonas</a><button onclick="resetear()">Resetear memoria</button></header>
    <main class="layout"><section class="panel clients"><h2>Personas activas</h2><div id="clientes" class="grid"></div></section><section class="panel map"><h2>Mapa de calor — plano de tienda</h2><div id="map"><div id="heatmap"></div><canvas id="zones" width="640" height="480"></canvas></div></section></main>
    <script>
      const heatmap=h337.create({container:document.querySelector('#heatmap'),radius:38,maxOpacity:.72,blur:.82});
      const $=s=>document.querySelector(s);
      let fondoImagenAplicado=null;
      function aplicarFondoPlano(tieneImagen){
        if(tieneImagen===fondoImagenAplicado) return;
        fondoImagenAplicado=tieneImagen;
        const el=$('#map');
        if(tieneImagen){el.style.backgroundImage=`url(/api/plano/imagen?t=${Date.now()})`;el.style.backgroundSize='cover';el.style.backgroundPosition='center';}
        else{el.style.backgroundImage='';el.style.backgroundSize='';el.style.backgroundPosition='';}
      }
      async function actualizar(){try{const [c,h,m,z,p,cam]=await Promise.all(['/api/clientes','/api/heatmap','/health','/api/zonas','/api/plano','/api/camaras'].map(u=>fetch(u).then(r=>r.json())));$('#clientes').innerHTML=c.clientes.map(x=>`<article class="card"><img src="data:image/jpeg;base64,${x.foto}"><b>${x.id}</b><br>${x.zona}<br><small>${x.genero} · ${x.edad}</small><br><small>${x.similitud} · ${x.ultima_vista}</small></article>`).join('')||'Sin personas activas';heatmap.setData({max:Math.max(3,h.max||0),data:h.puntos});$('#health').textContent=`Cola: ${m.queue_size}/${m.queue_capacity} · Procesados: ${m.processed} · Rechazados: ${m.rejected_full} · Último Re-ID: ${m.last_processing_ms} ms`;aplicarFondoPlano(p.tiene_imagen);dibujarPlano(z.zones,p.contorno||[],cam.camaras||[])}catch(e){console.warn(e)}}
      function dibujarPlano(zonas,contorno,camaras){
        const c=$('#zones'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);
        if(contorno.length>=3){x.beginPath();x.moveTo(...contorno[0]);contorno.slice(1).forEach(p=>x.lineTo(...p));x.closePath();x.strokeStyle='#94a3b8';x.lineWidth=3;x.stroke()}
        zonas.forEach(z=>{if(!z.polygon||z.polygon.length<3)return;x.beginPath();x.moveTo(...z.polygon[0]);z.polygon.slice(1).forEach(p=>x.lineTo(...p));x.closePath();x.strokeStyle=z.color||'#00ff88';x.lineWidth=2;x.stroke();x.fillStyle=(z.color||'#00ff88')+'33';x.fill();x.fillStyle='white';x.fillText(z.name,z.polygon[0][0]+5,z.polygon[0][1]-5)});
        camaras.forEach(cm=>{if(!cm.posicion)return;const[px,py]=cm.posicion;x.beginPath();x.arc(px,py,10,0,7);x.fillStyle='#38bdf8';x.fill();x.strokeStyle='#0f172a';x.lineWidth=2;x.stroke();x.fillStyle='#0f172a';x.font='bold 11px sans-serif';x.fillText('C',px-3,py+4);x.fillStyle='#7dd3fc';x.font='11px sans-serif';x.fillText(cm.nombre||cm.camara_id,px+14,py+4)});
      }
      async function resetear(){if(confirm('¿Borrar memoria y mapa?')){await fetch('/api/reset',{method:'POST'});actualizar()}} actualizar();setInterval(actualizar,1500);
    </script></body></html>
    """


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("LEANVISION_PORT", "8081")))
