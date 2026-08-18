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
import math
import os
import unicodedata
import uuid
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
#
# La acumulación se hace siempre en celdas base finas y fijas; la vista agrupa
# esas celdas al tamaño que corresponda (1 m² si hay escala real definida,
# HEATMAP_CELDA_PX si no). Separar acumulación de presentación permite definir
# o corregir la escala en cualquier momento sin perder lo ya acumulado.
HEATMAP_CELDA_BASE_PX = 10
HEATMAP_CELDA_PX = 20  # celda visible cuando no hay escala real definida.
HEATMAP_CELDA_MIN_PX = 10  # nunca menor que la celda base.
HEATMAP_CELDA_MAX_PX = 160

# Las zonas de negocio se pintan sobre una grilla, no se dibujan vértice por
# vértice. Media hora de metro cuadrado: entran 4 celdas de zona en cada celda
# de 1 m² del heatmap, así que las dos grillas quedan alineadas y una zona
# siempre cubre un número entero de celdas de heatmap.
ZONA_CELDA_METROS = 0.5
ZONA_CELDA_PX_SIN_ESCALA = 20  # fallback mientras no se definió la escala real.
ZONA_CELDA_MIN_PX = 8
ZONA_CELDA_MAX_PX = 120
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
INSTANCIA_ID = str(int(time.time()))
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
# Escala real del plano: {"puntos": [[x1,y1],[x2,y2]], "metros": float} — dos
# puntos de referencia del plano y la distancia real entre ellos. None = sin
# definir (el heatmap cae a celdas de HEATMAP_CELDA_PX, como siempre).
escala_plano: dict | None = None
# Catálogo de etiquetas de zona: [{"id","nombre","color"}]. Define UNA vez los
# tipos de zona del negocio ("Ropa de mujer", "Caja", "Probadores") y cada
# sucursal sólo pinta dónde caen. Que el id sea estable entre sucursales es lo
# que permite comparar la misma zona en varios locales sin depender de que
# alguien escriba el nombre igual en cada uno.
catalogo_zonas: list[dict] = []
_homografias_cache: dict[str, np.ndarray] = {}
# zona_id -> set de (col, fila). Se arma al cargar/guardar zonas para que la
# búsqueda de zona por punto sea O(1) en el camino caliente de detecciones.
_celdas_zona_cache: dict[str, set[tuple[int, int]]] = {}
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
    # Formato nuevo: celdas [[col, fila], ...] pintadas sobre la grilla.
    # Formato viejo: polygon [[x, y], ...]. Se acepta cualquiera de los dos
    # (las zonas dibujadas antes de la grilla siguen funcionando igual).
    celdas: list[list[int]] | None = None
    polygon: list[list[float]] | None = None
    # Etiqueta del catálogo a la que pertenece, si se eligió una.
    catalogo_id: str | None = None


class EtiquetaIn(BaseModel):
    nombre: str
    color: str = "#00ff88"


class CatalogoIn(BaseModel):
    # Para importar un catálogo completo desde otra sucursal.
    etiquetas: list[dict]


class ContornoIn(BaseModel):
    polygon: list[list[float]]


class EscalaIn(BaseModel):
    # Dos puntos del plano y la distancia real en metros entre ellos.
    puntos: list[list[float]]
    metros: float


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


def _escala_valida(escala) -> bool:
    """Valida la forma COMPLETA de una escala: 2 puntos [x, y] numéricos
    finitos y metros finito > 0. Se usa tanto al cargar el JSON del disco como
    en el endpoint: cualquier cosa que no cumpla se descarta, porque una
    escala malformada en memoria rompe /api/plano y /api/heatmap en cada
    request (round(inf/NaN) revienta, y el desempaquetado de puntos también).
    """
    if not isinstance(escala, dict):
        return False
    puntos = escala.get("puntos")
    if not isinstance(puntos, list) or len(puntos) != 2:
        return False
    for punto in puntos:
        if not isinstance(punto, (list, tuple)) or len(punto) != 2:
            return False
        if not all(isinstance(c, (int, float)) and math.isfinite(c) for c in punto):
            return False
    metros = escala.get("metros")
    return isinstance(metros, (int, float)) and math.isfinite(metros) and metros > 0


def _pixeles_por_metro() -> float | None:
    """Cuántos píxeles del plano representan un metro real, o None si la
    escala todavía no fue definida en /calibrar."""
    if not escala_plano:
        return None
    (x1, y1), (x2, y2) = escala_plano["puntos"]
    distancia_px = math.hypot(x2 - x1, y2 - y1)
    metros = escala_plano["metros"]
    if metros <= 0 or distancia_px <= 0:
        return None
    return distancia_px / metros


def _celda_heatmap_px() -> int:
    """Lado de la celda visible del heatmap: 1 metro real si hay escala,
    HEATMAP_CELDA_PX si no. Acotado para que un plano con escala extrema no
    genere celdas más finas que la grilla base ni cuadrados absurdos."""
    ppm = _pixeles_por_metro()
    if ppm is None:
        return HEATMAP_CELDA_PX
    return max(HEATMAP_CELDA_MIN_PX, min(HEATMAP_CELDA_MAX_PX, round(ppm)))


def _celda_zona_px() -> float:
    """Lado en píxeles de la celda de zona (ZONA_CELDA_METROS reales).

    Devuelve float a propósito: con escalas reales el valor casi nunca es
    entero (0,5 m a 53,83 px/m son 26,9 px), y redondearlo acumularía error
    de alineación a lo ancho del plano.
    """
    ppm = _pixeles_por_metro()
    if ppm is None:
        return float(ZONA_CELDA_PX_SIN_ESCALA)
    return max(ZONA_CELDA_MIN_PX, min(ZONA_CELDA_MAX_PX, ppm * ZONA_CELDA_METROS))


def _slug_etiqueta(nombre: str) -> str:
    """Id estable derivado del nombre, para que la misma etiqueta tenga el
    MISMO id en todas las sucursales aunque cada una la haya cargado por su
    cuenta — que es lo único que permite comparar la zona entre locales.

    Los acentos se normalizan a propósito: "Café" y "Cafe" son la misma zona
    de negocio, y si generaran ids distintos el catálogo no cumpliría su
    función justo en el caso que quiere resolver. Devuelve "" si el nombre no
    tiene ningún carácter alfanumérico (el llamador lo rechaza en vez de caer
    a un id con timestamp, que no sería reproducible entre sucursales).
    """
    sin_acentos = "".join(
        c for c in unicodedata.normalize("NFD", nombre) if unicodedata.category(c) != "Mn"
    )
    base = "".join(c if c.isalnum() else "_" for c in sin_acentos.lower())
    # Colapsa corridas de "_" para que "Caja 1" y "Caja  1" no difieran.
    while "__" in base:
        base = base.replace("__", "_")
    base = base.strip("_")
    return f"cat_{base}" if base else ""


def _zona_valida_en_disco(zona) -> bool:
    """Descarta zonas malformadas al cargar plano_config.json. Sin esto, una
    zona sin 'id' propagaba un KeyError desde el evento de startup (uvicorn
    aborta y systemd entra en loop de reinicios) y además rompía
    _zona_en_punto en cada detección."""
    if not isinstance(zona, dict) or not isinstance(zona.get("id"), str) or not zona["id"]:
        return False
    if not isinstance(zona.get("name"), str) or not zona["name"]:
        return False
    return bool(zona.get("celdas")) or bool(zona.get("polygon"))


def _etiqueta_valida_en_disco(etiqueta) -> bool:
    """Igual que las zonas: una entrada sin 'nombre' o sin 'id' dejaba TODOS
    los endpoints del catálogo devolviendo 500 apenas alguien los tocaba."""
    return (
        isinstance(etiqueta, dict)
        and isinstance(etiqueta.get("id"), str)
        and etiqueta["id"]
        and isinstance(etiqueta.get("nombre"), str)
        and etiqueta["nombre"]
    )


def _reindexar_celdas_zonas() -> None:
    """Rearma el índice (zona_id -> set de celdas) que usa _zona_en_punto.

    Se llama al cargar del disco y en cada alta/baja/edición de zonas: son
    operaciones administrativas y poco frecuentes, así que conviene pagar acá
    y dejar la búsqueda por detección en O(1).
    """
    _celdas_zona_cache.clear()
    for zona in zonas_negocio:
        celdas = zona.get("celdas")
        if not celdas:
            continue
        _celdas_zona_cache[zona["id"]] = {
            (int(celda[0]), int(celda[1]))
            for celda in celdas
            if isinstance(celda, (list, tuple)) and len(celda) == 2
        }


def _registrar_heatmap(pos_x: float, pos_y: float) -> None:
    """Acumula presencia en la celda base de la grilla, sin decaimiento."""
    if not (0 <= pos_x < HEATMAP_WIDTH and 0 <= pos_y < HEATMAP_HEIGHT):
        return
    celda_id = (int(pos_x // HEATMAP_CELDA_BASE_PX), int(pos_y // HEATMAP_CELDA_BASE_PX))
    celda = heatmap_celdas.setdefault(celda_id, {"valor": 0.0})
    celda["valor"] += 1.0


def _snapshot_heatmap() -> tuple[list[dict], float, int]:
    """Agrupa las celdas base en celdas visibles (1 m² si hay escala real) y
    devuelve (celdas con su esquina superior izquierda, máximo, lado en px)."""
    celda_px = _celda_heatmap_px()
    agregado: dict[tuple[int, int], float] = {}
    for (col, fila), celda in heatmap_celdas.items():
        centro_x = col * HEATMAP_CELDA_BASE_PX + HEATMAP_CELDA_BASE_PX / 2
        centro_y = fila * HEATMAP_CELDA_BASE_PX + HEATMAP_CELDA_BASE_PX / 2
        clave = (int(centro_x // celda_px), int(centro_y // celda_px))
        agregado[clave] = agregado.get(clave, 0.0) + celda["valor"]
    celdas = [
        {"x": col * celda_px, "y": fila * celda_px, "value": round(valor, 3)}
        for (col, fila), valor in agregado.items()
    ]
    maximo = max((c["value"] for c in celdas), default=0.0)
    return celdas, maximo, celda_px


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
    global escala_plano
    if not os.path.exists(ruta):
        return
    try:
        with open(ruta, "r", encoding="utf-8-sig") as archivo:
            datos = json.load(archivo)
        if not isinstance(datos, dict):
            raise ValueError(f"el contenido no es un objeto JSON sino {type(datos).__name__}")
        calibraciones_camaras.update(datos.get("camaras", {}))
        zonas = [z for z in datos.get("zonas", []) if _zona_valida_en_disco(z)]
        descartadas = len(datos.get("zonas", [])) - len(zonas)
        if descartadas:
            logger.warning("Se descartaron %d zona(s) malformada(s) de %s.", descartadas, ruta)
        zonas_negocio.extend(zonas)
        contorno_local[:] = datos.get("contorno", [])
        catalogo_zonas.extend(e for e in datos.get("catalogo", []) if _etiqueta_valida_en_disco(e))
        escala = datos.get("escala")
        if escala is not None:
            if _escala_valida(escala):
                escala_plano = {"puntos": escala["puntos"], "metros": float(escala["metros"])}
            else:
                logger.warning("Escala inválida en %s, se ignora (%r).", ruta, escala)
        for camara_id in calibraciones_camaras:
            _recalcular_homografia(camara_id)
    except (OSError, AttributeError, KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
        logger.warning("No se pudo cargar %s, se ignora (%r).", ruta, error)
    finally:
        # Va en finally: si algo falla DESPUÉS de cargar las zonas (escala,
        # homografías), sin esto quedaba zonas_negocio poblado y el cache
        # vacío — y ese desajuste es indistinguible de "no hay zonas": las
        # detecciones dejan de atribuirse a ninguna zona, sin error ni log.
        _reindexar_celdas_zonas()


def _guardar_plano_config(ruta: str = PLANO_CONFIG_PATH) -> None:
    """Escritura atómica (mismo patrón que app_limpia.py: .tmp + replace)."""
    payload = {
        "camaras": calibraciones_camaras,
        "zonas": zonas_negocio,
        "contorno": contorno_local,
        "escala": escala_plano,
        "catalogo": catalogo_zonas,
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
    """Nombre de la primera zona de negocio que contiene el punto, o None.

    Soporta los dos formatos: zonas nuevas pintadas sobre la grilla (celdas,
    búsqueda O(1) contra un set) y zonas viejas dibujadas vértice por vértice
    (polígono, ray-casting). Corre por cada detección, así que las de grilla
    se resuelven sin recorrer geometría.
    """
    for zona in zonas_negocio:
        celdas = _celdas_zona_cache.get(zona["id"])
        if celdas is not None:
            # Cada zona guarda el tamaño de celda con el que fue pintada: si
            # después se corrige la escala del plano, la zona conserva la
            # superficie real que se dibujó en su momento.
            celda_px = zona.get("celda_px") or _celda_zona_px()
            if celda_px > 0 and (int(x // celda_px), int(y // celda_px)) in celdas:
                return zona["name"]
            continue
        poligono = zona.get("polygon")
        if poligono and _punto_en_poligono(x, y, poligono):
            return zona["name"]
    return None


def _area_zona_m2(zona: dict) -> float | None:
    """Superficie real de una zona de grilla, o None si no se puede calcular
    (zona vieja de polígono, o plano sin escala definida)."""
    celdas = zona.get("celdas")
    celda_px = zona.get("celda_px")
    ppm = _pixeles_por_metro()
    if not celdas or not celda_px or ppm is None:
        return None
    lado_metros = celda_px / ppm
    return round(len(celdas) * lado_metros * lado_metros, 2)


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
            # Cambia en cada arranque. El dashboard lo usa para recargarse solo
            # tras un deploy: su HTML/JS viaja embebido en este archivo, así que
            # una pestaña abierta desde antes seguiría hablando el contrato viejo.
            "instancia": INSTANCIA_ID,
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
        celdas, maximo, celda_px = _snapshot_heatmap()
        ppm = _pixeles_por_metro()
    return {
        "width": HEATMAP_WIDTH,
        "height": HEATMAP_HEIGHT,
        "celda_px": celda_px,
        # Cuántos metros reales representa el lado de cada celda (None sin
        # escala). Con escala es ~1.0 salvo que el clamp de tamaño actúe.
        "celda_metros": round(celda_px / ppm, 2) if ppm else None,
        "celdas": celdas,
        "max": round(maximo, 3),
    }


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
        ppm = _pixeles_por_metro()
        celda_px = _celda_heatmap_px()
        return {
            "width": HEATMAP_WIDTH,
            "height": HEATMAP_HEIGHT,
            "contorno": list(contorno_local),
            "tiene_imagen": os.path.exists(PLANO_IMAGEN_PATH),
            "escala": escala_plano,
            "pixeles_por_metro": round(ppm, 2) if ppm else None,
            "celda_px": celda_px,
            "celda_metros": round(celda_px / ppm, 2) if ppm else None,
            # Grilla de zonas (más fina que la del heatmap, ver ZONA_CELDA_METROS).
            # Se deriva del px real y no se devuelve la constante: cuando el
            # clamp de _celda_zona_px() actúa, la celda deja de medir
            # ZONA_CELDA_METROS y la UI mostraría un área que no es la que
            # calcula el servidor.
            "zona_celda_px": round(_celda_zona_px(), 3),
            "zona_celda_metros": round(_celda_zona_px() / ppm, 3) if ppm else None,
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


@app.post("/api/plano/escala")
def guardar_escala(escala: EscalaIn) -> dict:
    """Define la escala real: dos puntos del plano + los metros entre ellos.
    Con eso las celdas del heatmap pasan a representar 1 m² real."""
    global escala_plano
    candidata = {"puntos": escala.puntos, "metros": escala.metros}
    # Rechaza NaN/Infinity antes de tocar nada: json.loads los acepta, y una
    # escala no finita persistida deja /api/plano y /api/heatmap en 500 hasta
    # que alguien edite el archivo a mano.
    if not _escala_valida(candidata):
        raise HTTPException(status_code=422, detail="La escala necesita exactamente 2 puntos [x, y] con números válidos.")
    (x1, y1), (x2, y2) = escala.puntos
    distancia_px = math.hypot(x2 - x1, y2 - y1)
    if distancia_px < 10:
        raise HTTPException(status_code=422, detail="Los 2 puntos están demasiado cerca; marcá referencias bien separadas.")
    if not (0.1 <= escala.metros <= 500):
        raise HTTPException(status_code=422, detail="La distancia real debe estar entre 0.1 y 500 metros.")
    with state_lock:
        escala_plano = candidata
        ppm = _pixeles_por_metro()
        celda_px = _celda_heatmap_px()
        _guardar_plano_config()
    return {
        "ok": True,
        "pixeles_por_metro": round(ppm, 2) if ppm else None,
        "celda_px": celda_px,
    }


@app.delete("/api/plano/escala")
def borrar_escala() -> dict:
    global escala_plano
    with state_lock:
        escala_plano = None
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


def _validar_celdas(celdas: list, celda_px: float) -> list[list[int]]:
    """Normaliza y deduplica las celdas pintadas. Las celdas fuera del plano
    se descartan en vez de rechazar todo: pintar arrastrando el mouse hasta
    el borde es normal y no debería fallar.

    El tamaño de celda llega por parámetro (no se toma el actual del plano):
    al editar una zona vieja hay que acotar con la grilla CON LA QUE FUE
    PINTADA, o las celdas de los bordes se descartarían por caer fuera de una
    grilla que no es la suya.
    """
    if celda_px <= 0:
        raise HTTPException(status_code=422, detail="Tamaño de celda inválido.")
    max_col = int(HEATMAP_WIDTH // celda_px) + 1
    max_fila = int(HEATMAP_HEIGHT // celda_px) + 1
    limpias: set[tuple[int, int]] = set()
    for celda in celdas:
        if not isinstance(celda, (list, tuple)) or len(celda) != 2:
            raise HTTPException(status_code=422, detail="Cada celda debe ser un par [columna, fila].")
        try:
            col, fila = int(celda[0]), int(celda[1])
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail="Las celdas deben ser números enteros.") from error
        if 0 <= col <= max_col and 0 <= fila <= max_fila:
            limpias.add((col, fila))
    if not limpias:
        raise HTTPException(status_code=422, detail="La zona no tiene ninguna celda pintada dentro del plano.")
    return [[col, fila] for col, fila in sorted(limpias)]


def _validar_poligono(polygon: list) -> list[list[float]]:
    """Valida que el polígono sea una lista de pares [x, y] finitos.

    pydantic acepta list[list[float]] con sublistas de largo 1 y con
    NaN/Infinity, y un polígono así se persiste con 200 OK pero después hace
    reventar _punto_en_poligono en el camino caliente — lo que deja de
    procesar TODAS las detecciones mientras el servidor sigue respondiendo
    como si estuviera sano.
    """
    if len(polygon) < 3:
        raise HTTPException(status_code=422, detail="Una zona por polígono necesita al menos 3 puntos.")
    limpio: list[list[float]] = []
    for punto in polygon:
        if not isinstance(punto, (list, tuple)) or len(punto) != 2:
            raise HTTPException(status_code=422, detail="Cada punto del polígono debe ser un par [x, y].")
        if not all(isinstance(c, (int, float)) and math.isfinite(c) for c in punto):
            raise HTTPException(status_code=422, detail="Las coordenadas del polígono deben ser números finitos.")
        limpio.append([float(punto[0]), float(punto[1])])
    return limpio


def _armar_zona(zona: ZonaIn, zona_id: str, celda_px_existente: float | None = None) -> dict:
    """Construye el dict de una zona desde el payload, aceptando tanto el
    formato de grilla (celdas) como el viejo de polígono.

    `celda_px_existente` se pasa al EDITAR: las celdas son índices sin
    unidad, sólo significan algo junto al tamaño de celda con el que se
    pintaron. Re-estampar el tamaño actual movería y redimensionaría la zona
    sobre el piso si alguien corrigió la escala del plano en el medio.
    """
    if not zona.name.strip():
        raise HTTPException(status_code=422, detail="La zona necesita un nombre.")
    nueva: dict = {
        "id": zona_id,
        "name": zona.name.strip(),
        "color": zona.color,
    }
    if zona.catalogo_id:
        nueva["catalogo_id"] = zona.catalogo_id
    if zona.celdas:
        celda_px = celda_px_existente if celda_px_existente else _celda_zona_px()
        nueva["celdas"] = _validar_celdas(zona.celdas, celda_px)
        nueva["celda_px"] = round(celda_px, 3)
    elif zona.polygon:
        nueva["polygon"] = _validar_poligono(zona.polygon)
    else:
        raise HTTPException(status_code=422, detail="Pintá al menos una celda para definir la zona.")
    return nueva


@app.get("/api/zonas")
def listar_zonas() -> dict:
    with state_lock:
        ppm = _pixeles_por_metro()
        celda_px = _celda_zona_px()
        zonas = [{**zona, "area_m2": _area_zona_m2(zona)} for zona in zonas_negocio]
        return {
            "zones": zonas,
            "celda_px": round(celda_px, 3),
            # None sin escala: ahí las celdas son píxeles sin significado
            # físico y devolver 0.5 haría que la UI muestre metros inventados.
            "celda_metros": round(celda_px / ppm, 3) if ppm else None,
        }


@app.post("/api/zonas")
def crear_zona(zona: ZonaIn) -> dict:
    with state_lock:
        # uuid en vez de contador+timestamp: el contador se recicla al borrar
        # zonas y el timestamp no distingue dos altas en el mismo segundo. El
        # id ahora es la clave del cache de celdas, así que una colisión hace
        # que una zona reporte las detecciones de otra.
        nueva = _armar_zona(zona, f"zona_{uuid.uuid4().hex[:12]}")
        zonas_negocio.append(nueva)
        _reindexar_celdas_zonas()
        _guardar_plano_config()
    return nueva


@app.put("/api/zonas/{zona_id}")
def editar_zona(zona_id: str, zona: ZonaIn) -> dict:
    with state_lock:
        for indice, existente in enumerate(zonas_negocio):
            if existente["id"] == zona_id:
                zonas_negocio[indice] = _armar_zona(zona, zona_id, existente.get("celda_px"))
                _reindexar_celdas_zonas()
                _guardar_plano_config()
                return zonas_negocio[indice]
    raise HTTPException(status_code=404, detail="Zona no encontrada.")


@app.delete("/api/zonas/{zona_id}")
def borrar_zona(zona_id: str) -> dict:
    with state_lock:
        restantes = [z for z in zonas_negocio if z["id"] != zona_id]
        if len(restantes) == len(zonas_negocio):
            raise HTTPException(status_code=404, detail="Zona no encontrada.")
        zonas_negocio[:] = restantes
        _reindexar_celdas_zonas()
        _guardar_plano_config()
    return {"ok": True}


# --- Catálogo de etiquetas de zona: se define una vez y se reutiliza en
# todas las sucursales, para poder comparar la misma zona entre locales. ---


@app.get("/api/catalogo")
def listar_catalogo() -> dict:
    with state_lock:
        return {"etiquetas": list(catalogo_zonas)}


@app.post("/api/catalogo")
def crear_etiqueta(etiqueta: EtiquetaIn) -> dict:
    nombre = etiqueta.nombre.strip()
    if not nombre:
        raise HTTPException(status_code=422, detail="La etiqueta necesita un nombre.")
    id_nuevo = _slug_etiqueta(nombre)
    if not id_nuevo:
        raise HTTPException(
            status_code=422,
            detail="El nombre necesita al menos una letra o número (no puede ser sólo símbolos o emojis).",
        )
    with state_lock:
        if any(e.get("nombre", "").lower() == nombre.lower() for e in catalogo_zonas):
            raise HTTPException(status_code=422, detail=f"Ya existe una etiqueta llamada '{nombre}'.")
        # También se valida el id, no sólo el nombre: "Caja" y "Caja!" tienen
        # nombres distintos pero producen el mismo id, y con ids repetidos el
        # desplegable de la UI resuelve siempre a la primera y borrar una
        # elimina las dos.
        existente = next((e for e in catalogo_zonas if e.get("id") == id_nuevo), None)
        if existente:
            raise HTTPException(
                status_code=422,
                detail=f"'{nombre}' generaría el mismo identificador que '{existente.get('nombre')}'. Usá un nombre más distinto.",
            )
        nueva = {"id": id_nuevo, "nombre": nombre, "color": etiqueta.color}
        catalogo_zonas.append(nueva)
        _guardar_plano_config()
    return nueva


@app.delete("/api/catalogo/{etiqueta_id}")
def borrar_etiqueta(etiqueta_id: str) -> dict:
    with state_lock:
        en_uso = [z["name"] for z in zonas_negocio if z.get("catalogo_id") == etiqueta_id]
        if en_uso:
            raise HTTPException(
                status_code=422,
                detail=f"La etiqueta está en uso por {len(en_uso)} zona(s) de este plano. Borrá esas zonas primero.",
            )
        restantes = [e for e in catalogo_zonas if e["id"] != etiqueta_id]
        if len(restantes) == len(catalogo_zonas):
            raise HTTPException(status_code=404, detail="Etiqueta no encontrada.")
        catalogo_zonas[:] = restantes
        _guardar_plano_config()
    return {"ok": True}


@app.post("/api/catalogo/importar")
def importar_catalogo(payload: CatalogoIn) -> dict:
    """Agrega las etiquetas de otra sucursal sin pisar las locales: las que
    ya existen (mismo id) se saltean, para poder replicar el catálogo entre
    locales corriendo esto sin miedo más de una vez."""
    agregadas = 0
    with state_lock:
        existentes = {e["id"] for e in catalogo_zonas}
        for etiqueta in payload.etiquetas:
            if not isinstance(etiqueta, dict):
                continue
            id_etiqueta, nombre = etiqueta.get("id"), etiqueta.get("nombre")
            if not id_etiqueta or not nombre or id_etiqueta in existentes:
                continue
            catalogo_zonas.append({
                "id": str(id_etiqueta),
                "nombre": str(nombre),
                "color": str(etiqueta.get("color", "#00ff88")),
            })
            existentes.add(id_etiqueta)
            agregadas += 1
        if agregadas:
            _guardar_plano_config()
    return {"ok": True, "agregadas": agregadas, "total": len(catalogo_zonas)}


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
      #map{position:relative;width:640px;max-width:100%;aspect-ratio:640/480;overflow:hidden;background:#111827}
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
          <button onclick="modoEscala()">2. Escala (metros)</button>
          <button onclick="modoPosicion()">3. Ubicar cámara</button>
          <button onclick="modoCalibrar()">4. Calibrar cámara</button>
          <button onclick="modoZona()">5. Dibujar zona</button>
        </div>
        <p id="instrucciones">Empezá dibujando la forma del local con el paso 1.</p>
        <p class="paso" id="escala-info"></p>
        <div id="map"><canvas id="plano" width="640" height="480"></canvas></div>
        <div id="controles-contorno" style="display:none">
          <button onclick="guardarContorno()">Guardar forma del local</button>
          <button class="gris" onclick="limpiarDibujo()">Limpiar</button>
          <button class="rojo" onclick="borrarContorno()">Borrar forma guardada</button>
        </div>
        <div id="controles-escala" style="display:none">
          <div class="fila">
            <input id="escala_metros" type="number" min="0.1" step="0.1" placeholder="Distancia real entre los 2 puntos, en metros">
            <button onclick="guardarEscala()">Guardar escala</button>
          </div>
          <button class="gris" onclick="limpiarDibujo()">Rehacer puntos</button>
          <button class="rojo" onclick="borrarEscala()">Borrar escala guardada</button>
        </div>
        <div id="controles-calibracion" style="display:none">
          <button onclick="confirmarCalibracion()">Confirmar calibración</button>
          <button class="gris" onclick="limpiarDibujo()">Rehacer</button>
        </div>
        <div id="controles-zona" style="display:none">
          <div class="fila">
            <select id="z_etiqueta" onchange="alSeleccionarEtiqueta()"></select>
            <input id="z_color" type="color" value="#00ff88" style="width:44px;padding:2px">
          </div>
          <p class="paso" id="zona-resumen">Sin celdas pintadas.</p>
          <button onclick="guardarZona()">Guardar zona</button>
          <button class="gris" onclick="limpiarDibujo()">Limpiar</button>
        </div>
      </section>

      <section class="panel lado">
        <h2>Etiquetas</h2>
        <p class="paso">Se definen una vez y se reutilizan en todas las sucursales. Comparar la misma zona entre locales depende de que usen la misma etiqueta.</p>
        <div class="fila">
          <input id="e_nombre" placeholder="Ej: Ropa de mujer">
          <input id="e_color" type="color" value="#38bdf8" style="width:44px;padding:2px">
        </div>
        <button onclick="crearEtiqueta()">Agregar etiqueta</button>
        <div id="lista-etiquetas" style="margin-top:10px"></div>
        <div style="margin-top:12px;border-top:1px solid #334155;padding-top:10px">
          <button class="gris" onclick="exportarCatalogo()">Exportar catálogo</button>
          <button class="gris" onclick="$('#importar_input').click()">Importar</button>
          <input id="importar_input" type="file" accept="application/json" style="display:none" onchange="importarCatalogo(this)">
          <p class="paso">Para replicar las mismas etiquetas en otra sucursal.</p>
        </div>

        <h2 style="margin-top:18px">Zonas de este local</h2>
        <div id="lista-zonas"></div>
      </section>

    </main>
    <script>
      const $=s=>document.querySelector(s);
      const canvas=$('#plano'), ctx=canvas.getContext('2d');
      const vcanvas=$('#video-canvas'), vctx=vcanvas.getContext('2d');
      const PALETA=['#38bdf8','#f472b6','#a78bfa','#fb923c','#4ade80'];
      let camaras=[], zonas=[], contorno=[], camaraSel=null, modo=null;
      let puntosCalibracion=[], poligonoContorno=[], puntosVideo=[], previewContorno=null, contornoCerrado=false;
      let faseCalibracion='video';
      let escala=null, puntosEscala=[];
      // Zonas por grilla: se pinta arrastrando, como seleccionar celdas en Excel.
      // zonaCeldaPxPlano es la grilla actual del plano; zonaCeldaPx es la que
      // se está usando para pintar, que al EDITAR pasa a ser la de la zona
      // (las celdas son índices: reinterpretarlas con otra grilla movería y
      // redimensionaría la zona sobre el piso).
      let etiquetas=[], celdasZona=new Set(), zonaCeldaPx=20, zonaCeldaPxPlano=20, zonaCeldaMetros=null;
      let pintando=false, borrandoCeldas=false, zonaEditandoId=null;

      const claveCelda=(c,f)=>`${c},${f}`;
      // Los nombres de etiqueta y de zona se muestran con innerHTML, y el
      // catálogo se puede IMPORTAR de un archivo que no escribió quien lo
      // carga: sin escapar, ese archivo puede inyectar HTML/JS en un panel
      // que tiene acceso completo a la API de calibración.
      const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
      // El detail de un 422 de pydantic es una LISTA de errores, no un
      // string: sin esto el alert mostraba "[object Object]".
      const msgError=(d,porDefecto)=>typeof d==='string'?d:(Array.isArray(d)?d.map(e=>e.msg||JSON.stringify(e)).join(' · '):porDefecto);
      function celdaDesdeEvento(event){
        const rect=canvas.getBoundingClientRect();
        const x=(event.clientX-rect.left)*(canvas.width/rect.width);
        const y=(event.clientY-rect.top)*(canvas.height/rect.height);
        return [Math.floor(x/zonaCeldaPx), Math.floor(y/zonaCeldaPx)];
      }

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

      function dibujarRegla(puntos,metros,provisoria){
        if(puntos.length<1) return;
        ctx.strokeStyle='#f59e0b';ctx.fillStyle='#f59e0b';
        puntos.forEach(p=>{ctx.beginPath();ctx.arc(p[0],p[1],5,0,7);ctx.fill();});
        if(puntos.length<2) return;
        ctx.setLineDash(provisoria?[6,4]:[]);
        ctx.beginPath();ctx.moveTo(...puntos[0]);ctx.lineTo(...puntos[1]);
        ctx.lineWidth=2;ctx.stroke();ctx.setLineDash([]);
        if(metros){
          const mx=(puntos[0][0]+puntos[1][0])/2, my=(puntos[0][1]+puntos[1][1])/2;
          ctx.font='bold 13px sans-serif';
          ctx.fillText(`${metros} m`,mx+8,my-8);
        }
      }

      function dibujarGrilla(resaltada){
        // Se dibuja en el canvas y no con CSS a propósito: el canvas se
        // escala solo si el plano no entra a 640px, y una grilla de CSS
        // (que mide en píxeles de pantalla) quedaría desalineada de las
        // celdas reales justo cuando más importa.
        const g=zonaCeldaPx;
        if(g<4) return;
        ctx.strokeStyle=resaltada?'#64748b66':'#64748b2e';ctx.lineWidth=1;
        ctx.beginPath();
        for(let x=0;x<=canvas.width;x+=g){ctx.moveTo(Math.round(x)+0.5,0);ctx.lineTo(Math.round(x)+0.5,canvas.height);}
        for(let y=0;y<=canvas.height;y+=g){ctx.moveTo(0,Math.round(y)+0.5);ctx.lineTo(canvas.width,Math.round(y)+0.5);}
        ctx.stroke();
      }

      function pintarCeldas(celdas,color,celdaPx,resaltar){
        const lado=celdaPx||zonaCeldaPx;
        ctx.fillStyle=color+(resaltar?'99':'55');
        celdas.forEach(([c,f])=>ctx.fillRect(c*lado,f*lado,lado,lado));
        if(resaltar){
          ctx.strokeStyle=color;ctx.lineWidth=1;
          celdas.forEach(([c,f])=>ctx.strokeRect(c*lado+0.5,f*lado+0.5,lado-1,lado-1));
        }
      }

      function etiquetaZona(z){
        if(!z.celdas||!z.celdas.length) return;
        // Etiqueta en la celda de más arriba a la izquierda, para que no quede
        // flotando en un punto vacío si la zona tiene forma irregular.
        const lado=z.celda_px||zonaCeldaPx;
        const [c,f]=z.celdas.reduce((a,b)=>(b[1]<a[1]||(b[1]===a[1]&&b[0]<a[0]))?b:a);
        ctx.fillStyle='white';ctx.font='12px sans-serif';
        ctx.fillText(z.name,c*lado+3,f*lado+13);
      }

      function redraw(){
        ctx.clearRect(0,0,canvas.width,canvas.height);
        if(contorno.length>=3){
          ctx.beginPath();ctx.moveTo(...contorno[0]);contorno.slice(1).forEach(p=>ctx.lineTo(...p));ctx.closePath();
          ctx.fillStyle='#1e293b88';ctx.fill();
          ctx.strokeStyle='#94a3b8';ctx.lineWidth=3;ctx.stroke();
        }
        dibujarGrilla(modo==='zona'||modo==='contorno');
        zonas.forEach(z=>{
          // La zona en edición no se dibuja acá: se dibuja abajo desde
          // celdasZona. Si no, se verían las dos copias superpuestas.
          if(z.id===zonaEditandoId) return;
          if(z.celdas&&z.celdas.length){
            pintarCeldas(z.celdas,z.color||'#00ff88',z.celda_px,false);
            etiquetaZona(z);
          } else if(z.polygon){
            dibujarPoligono(z.polygon,z.color||'#00ff88',true,z.name,false);
          }
        });
        camaras.forEach(c=>{if(c.puntos_plano) dibujarPoligono(c.puntos_plano,colorCamara(c.camara_id),false,null,true);});
        camaras.forEach(dibujarCamara);
        if(modo==='escala'){
          dibujarRegla(puntosEscala,null,true);
        } else if(escala&&escala.puntos){
          dibujarRegla(escala.puntos,escala.metros,false);
        }
        if(modo==='calibrar'&&faseCalibracion==='plano'){
          dibujarPoligono(puntosCalibracion,'#facc15',false,null,false);
          marcarPuntos(ctx,puntosCalibracion,'#facc15',true);
        } else if(modo==='zona'){
          const celdas=[...celdasZona].map(k=>k.split(',').map(Number));
          pintarCeldas(celdas,$('#z_color').value||'#facc15',zonaCeldaPx,true);
        } else if(modo==='contorno'){
          dibujarPoligono(poligonoContorno,'#38bdf8',poligonoContorno.length>=3,null,false);
          if(previewContorno&&poligonoContorno.length){
            const ultimo=poligonoContorno[poligonoContorno.length-1];
            ctx.setLineDash([5,4]); ctx.strokeStyle='#38bdf8aa'; ctx.lineWidth=2;
            ctx.beginPath(); ctx.moveTo(...ultimo); ctx.lineTo(...previewContorno); ctx.stroke();
            ctx.setLineDash([]);
          }
          marcarPuntos(ctx,poligonoContorno,'#38bdf8',false);
          if(poligonoContorno.length){
            // Resalta la primera esquina: es la referencia para saber dónde
            // va a cerrar la forma.
            ctx.strokeStyle='#38bdf8'; ctx.lineWidth=2;
            ctx.beginPath(); ctx.arc(poligonoContorno[0][0],poligonoContorno[0][1],10,0,7); ctx.stroke();
          }
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
        $('#controles-escala').style.display='none';
        vcanvas.classList.remove('activo');
      }

      function modoEscala(){
        modo='escala'; puntosEscala=[];
        ocultarControles();
        $('#controles-escala').style.display='block';
        $('#instrucciones').textContent='Marcá 2 puntos del plano cuya distancia real conozcas (ej: los extremos de una pared), y escribí cuántos metros hay entre ellos.';
        redraw(); redrawVideo();
      }

      function modoContorno(){
        modo='contorno'; poligonoContorno=[]; previewContorno=null; contornoCerrado=false;
        zonaCeldaPx=zonaCeldaPxPlano; zonaEditandoId=null;
        ocultarControles();
        $('#controles-contorno').style.display='block';
        $('#instrucciones').textContent='Clic en cada esquina del local. Las paredes salen rectas solas y se pegan a la grilla; doble clic para cerrar.';
        redraw(); redrawVideo();
      }

      // --- Dibujo del local con paredes en ángulo recto ---
      function ajustarAGrilla(p){
        const g=zonaCeldaPxPlano;
        // Se acota al plano: redondear al múltiplo más cercano puede empujar
        // la esquina hasta media celda AFUERA del canvas, y dibujar el
        // contorno pegado al borde es el caso normal, no el raro.
        return [
          Math.min(canvas.width, Math.max(0, Math.round(p[0]/g)*g)),
          Math.min(canvas.height, Math.max(0, Math.round(p[1]/g)*g)),
        ];
      }
      function forzarOrtogonal(anterior,p){
        // Cada pared es horizontal o vertical: se conserva el eje en el que
        // más se movió el mouse y se alinea el otro con la esquina anterior.
        if(!anterior) return p;
        return Math.abs(p[0]-anterior[0])>=Math.abs(p[1]-anterior[1])
          ? [p[0],anterior[1]]
          : [anterior[0],p[1]];
      }
      function esquinaContorno(evento){
        const p=ajustarAGrilla(pointFromEvent(evento));
        return forzarOrtogonal(poligonoContorno[poligonoContorno.length-1],p);
      }
      function cerrarContorno(){
        if(poligonoContorno.length<3){
          $('#instrucciones').textContent='Marcá al menos 3 esquinas antes de cerrar.';
          return;
        }
        const primero=poligonoContorno[0], ultimo=poligonoContorno[poligonoContorno.length-1];
        // La pared de cierre también tiene que ser recta: si la última
        // esquina no comparte eje con la primera, se agrega la esquina que
        // falta en vez de dejar una diagonal.
        if(ultimo[0]!==primero[0]&&ultimo[1]!==primero[1]){
          // CUÁL de las dos esquinas posibles depende de hacia dónde iba la
          // última pared: salir en la misma dirección sería retroceder sobre
          // ella, dejando una espiga de ancho cero que recorta parte del
          // local (y que igual pasa la validación de área del servidor).
          const anteultimo=poligonoContorno[poligonoContorno.length-2];
          const ultimaHorizontal=anteultimo[1]===ultimo[1];
          poligonoContorno.push(ultimaHorizontal?[ultimo[0],primero[1]]:[primero[0],ultimo[1]]);
        }
        contornoCerrado=true;
        previewContorno=null;
        $('#instrucciones').textContent=`Local cerrado (${poligonoContorno.length} esquinas). Guardalo, o Limpiar para rehacerlo.`;
        redraw();
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
        modo='zona'; celdasZona=new Set(); zonaEditandoId=null;
        zonaCeldaPx=zonaCeldaPxPlano;  // vuelve a la grilla actual del plano
        ocultarControles();
        $('#controles-zona').style.display='block';
        $('#instrucciones').textContent='Pintá las celdas arrastrando el mouse. Shift+arrastrar borra. Elegí la etiqueta y guardá.';
        actualizarResumenZona();
        redraw(); redrawVideo();
      }

      function actualizarResumenZona(){
        const n=celdasZona.size;
        if(!n){ $('#zona-resumen').textContent='Sin celdas pintadas.'; return; }
        const area=zonaCeldaMetros?` · ${(n*zonaCeldaMetros*zonaCeldaMetros).toFixed(2)} m²`:'';
        $('#zona-resumen').textContent=`${n} celda${n===1?'':'s'}${area}`;
      }

      function alSeleccionarEtiqueta(){
        const et=etiquetas.find(e=>e.id===$('#z_etiqueta').value);
        if(et) $('#z_color').value=et.color;
        redraw();
      }

      function limpiarDibujo(){
        puntosCalibracion=[]; poligonoContorno=[]; puntosVideo=[]; puntosEscala=[]; previewContorno=null; contornoCerrado=false;
        if(modo==='calibrar') modoCalibrar();
        else if(modo==='contorno') modoContorno();
        else if(modo==='escala') modoEscala();
        else if(modo==='zona'){
          // "Limpiar" borra lo pintado pero NO cancela la edición en curso:
          // si la cancelara, el guardado siguiente crearía una zona nueva y
          // quedarían dos superpuestas con el mismo nombre.
          const editando=zonaEditandoId, px=zonaCeldaPx;
          modoZona();
          if(editando){
            zonaEditandoId=editando; zonaCeldaPx=px;
            const z=zonas.find(x=>x.id===editando);
            $('#instrucciones').textContent=`Editando "${z?z.name:''}" — repintá las celdas y guardá.`;
          }
        }
        else {redraw(); redrawVideo();}
      }

      // --- Pintado de celdas (arrastrar como en una planilla) ---
      function aplicarCelda(event){
        const [c,f]=celdaDesdeEvento(event);
        if(c<0||f<0) return;
        const clave=claveCelda(c,f);
        if(borrandoCeldas) celdasZona.delete(clave); else celdasZona.add(clave);
        actualizarResumenZona();
        redraw();
      }
      canvas.addEventListener('mousedown',(event)=>{
        if(modo!=='zona') return;
        event.preventDefault();
        pintando=true; borrandoCeldas=event.shiftKey;
        aplicarCelda(event);
      });
      canvas.addEventListener('mousemove',(event)=>{
        if(modo==='zona'&&pintando){ aplicarCelda(event); return; }
        if(modo==='contorno'&&poligonoContorno.length&&!contornoCerrado){
          // Previsualiza la pared antes de fijarla, así se ve para qué lado
          // va a salir sin tener que adivinar.
          previewContorno=esquinaContorno(event);
          redraw();
        }
      });
      canvas.addEventListener('mouseleave',()=>{
        if(previewContorno){ previewContorno=null; redraw(); }
      });
      // El mouseup va en window y no en el canvas: si se suelta el botón
      // fuera del plano (muy común al pintar hasta el borde), sin esto el
      // pintado quedaría pegado al mouse.
      window.addEventListener('mouseup',()=>{pintando=false;});

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
        } else if(modo==='contorno'){
          if(event.detail>=2){ cerrarContorno(); return; }
          if(contornoCerrado){
            $('#instrucciones').textContent='El local ya está cerrado. Guardalo, o tocá Limpiar para rehacerlo.';
            return;
          }
          const esquina=esquinaContorno(event);
          const anterior=poligonoContorno[poligonoContorno.length-1];
          // Dos clics en la misma celda dejarían un vértice de arista cero.
          if(anterior&&anterior[0]===esquina[0]&&anterior[1]===esquina[1]) return;
          poligonoContorno.push(esquina);
          previewContorno=null;
          $('#instrucciones').textContent=`${poligonoContorno.length} esquina${poligonoContorno.length===1?'':'s'}. Seguí marcando; doble clic para cerrar.`;
          redraw();
        } else if(modo==='escala'){
          // El paso 1 cierra el contorno con doble clic: acá ese mismo gesto
          // marcaría los 2 puntos superpuestos, así que se ignora el 2do click.
          if(event.detail===2||puntosEscala.length>=2) return;
          puntosEscala.push(p); redraw();
          if(puntosEscala.length===1){
            $('#instrucciones').textContent='Ahora marcá el segundo punto de referencia.';
          } else {
            $('#instrucciones').textContent='Escribí la distancia real en metros entre esos 2 puntos y guardá.';
            $('#escala_metros').focus();
          }
        }
      });

      async function guardarPosicion(punto){
        const r=await fetch(`/api/camaras/${camaraSel}/posicion`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({posicion:punto})});
        const data=await r.json();
        if(!r.ok){alert(msgError(data.detail,'Error al ubicar la cámara'));return;}
        modo=null;
        $('#instrucciones').textContent='Ubicación guardada. Seguí con el paso 4 para calibrar lo que ve.';
        await cargarCamaras();
      }

      async function guardarContorno(){
        if(poligonoContorno.length<3) return alert('Dibujá al menos 3 puntos.');
        const r=await fetch('/api/plano/contorno',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({polygon:poligonoContorno})});
        const data=await r.json();
        if(!r.ok){alert(msgError(data.detail,'Error al guardar la forma'));return;}
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

      async function guardarEscala(){
        if(puntosEscala.length!==2) return alert('Marcá primero los 2 puntos de referencia en el plano.');
        const metros=parseFloat($('#escala_metros').value);
        if(!metros||metros<=0) return alert('Escribí la distancia real en metros (mayor a 0).');
        const r=await fetch('/api/plano/escala',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({puntos:puntosEscala,metros})});
        const data=await r.json();
        if(!r.ok){alert(msgError(data.detail,'Error al guardar la escala'));return;}
        modo=null; puntosEscala=[]; $('#escala_metros').value='';
        ocultarControles();
        const lado=(data.celda_px/data.pixeles_por_metro).toFixed(2);
        $('#instrucciones').textContent=`Escala guardada: 1 m ≈ ${data.pixeles_por_metro} px. El heatmap ahora usa celdas de ${lado}×${lado} m.`;
        await cargarPlano();
      }

      async function borrarEscala(){
        if(!confirm('¿Borrar la escala? El heatmap vuelve a celdas de 20 px sin significado real.')) return;
        await fetch('/api/plano/escala',{method:'DELETE'});
        puntosEscala=[]; modo=null; ocultarControles();
        $('#instrucciones').textContent='Escala borrada. El heatmap vuelve a celdas de 20 px.';
        await cargarPlano();
      }

      async function confirmarCalibracion(){
        const r=await fetch(`/api/camaras/${camaraSel}/calibracion`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({puntos_plano:puntosCalibracion,puntos_camara:puntosVideo})});
        const data=await r.json();
        if(!r.ok){alert(msgError(data.detail,'Error al calibrar'));return;}
        modo=null; puntosCalibracion=[]; puntosVideo=[];
        ocultarControles();
        $('#instrucciones').textContent='Calibración guardada. Lo que ve esta cámara ya se proyecta al plano.';
        await cargarCamaras(); redrawVideo();
      }

      async function guardarZona(){
        if(!celdasZona.size) return alert('Pintá al menos una celda arrastrando el mouse sobre el plano.');
        const etiquetaId=$('#z_etiqueta').value;
        const etiqueta=etiquetas.find(e=>e.id===etiquetaId);
        if(!etiqueta) return alert('Elegí una etiqueta. Si no hay ninguna, creala en el panel de la derecha.');
        const celdas=[...celdasZona].map(k=>k.split(',').map(Number));
        const cuerpo={name:etiqueta.nombre,color:$('#z_color').value,celdas,catalogo_id:etiqueta.id};
        const url=zonaEditandoId?`/api/zonas/${zonaEditandoId}`:'/api/zonas';
        const r=await fetch(url,{method:zonaEditandoId?'PUT':'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cuerpo)});
        const data=await r.json();
        if(!r.ok){alert(msgError(data.detail,'Error al guardar la zona'));return;}
        celdasZona=new Set(); zonaEditandoId=null;
        $('#instrucciones').textContent='Zona guardada. Podés pintar la siguiente.';
        actualizarResumenZona();
        await cargarZonas(); await cargarCatalogo();
      }

      function editarZona(id){
        const z=zonas.find(x=>x.id===id);
        if(!z) return;
        if(!z.celdas||!z.celdas.length) return alert('Esta zona es de las viejas (por polígono). Borrala y pintala de nuevo con la grilla.');
        modoZona();
        zonaEditandoId=id;
        // Se pinta sobre la grilla CON LA QUE fue creada, no sobre la actual:
        // si alguien corrigió la escala del plano después, reinterpretar los
        // índices con la grilla nueva movería la zona de lugar.
        zonaCeldaPx=z.celda_px||zonaCeldaPxPlano;
        celdasZona=new Set(z.celdas.map(([c,f])=>claveCelda(c,f)));
        if(z.catalogo_id) $('#z_etiqueta').value=z.catalogo_id;
        $('#z_color').value=z.color||'#00ff88';
        const aviso=Math.abs(zonaCeldaPx-zonaCeldaPxPlano)>0.01
          ? ' (se pintó con otra escala: se mantiene su grilla original)' : '';
        $('#instrucciones').textContent=`Editando "${z.name}". Pintá o borrá celdas y guardá.${aviso}`;
        actualizarResumenZona();
        redraw();
      }

      // --- Catálogo de etiquetas ---
      async function crearEtiqueta(){
        const nombre=$('#e_nombre').value.trim();
        if(!nombre) return alert('Escribí un nombre para la etiqueta.');
        const r=await fetch('/api/catalogo',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({nombre,color:$('#e_color').value})});
        const data=await r.json();
        if(!r.ok){alert(msgError(data.detail,'Error al crear la etiqueta'));return;}
        $('#e_nombre').value='';
        await cargarCatalogo();
      }
      async function borrarEtiqueta(id){
        const r=await fetch(`/api/catalogo/${id}`,{method:'DELETE'});
        const data=await r.json();
        // Se recarga incluso al fallar: el 422 'esta en uso' suele
        // significar que la vista tenia el badge desactualizado.
        if(!r.ok){alert(msgError(data.detail,'Error al borrar la etiqueta'));await cargarCatalogo();return;}
        await cargarCatalogo();
      }
      function exportarCatalogo(){
        if(!etiquetas.length) return alert('No hay etiquetas para exportar.');
        const blob=new Blob([JSON.stringify({etiquetas},null,2)],{type:'application/json'});
        const a=document.createElement('a');
        a.href=URL.createObjectURL(blob);
        a.download='catalogo-zonas.json';
        a.click();
        URL.revokeObjectURL(a.href);
      }
      async function importarCatalogo(input){
        if(!input.files.length) return;
        try{
          const texto=await input.files[0].text();
          const datos=JSON.parse(texto);
          const r=await fetch('/api/catalogo/importar',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({etiquetas:datos.etiquetas||datos})});
          const res=await r.json();
          if(!r.ok){alert(msgError(res.detail,'Error al importar'));return;}
          alert(`Importadas ${res.agregadas} etiqueta(s) nueva(s). Total: ${res.total}.`);
          await cargarCatalogo();
        }catch(e){ alert('El archivo no es un catálogo válido.'); }
        finally{ input.value=''; }
      }
      async function cargarCatalogo(){
        etiquetas=(await (await fetch('/api/catalogo')).json()).etiquetas;
        const usadas=new Set(zonas.map(z=>z.catalogo_id).filter(Boolean));
        $('#lista-etiquetas').innerHTML=etiquetas.map(e=>`
          <div class="item" style="border-left:4px solid ${esc(e.color)}">
            <b>${esc(e.nombre)}${usadas.has(e.id)?' <span class="badge si">en uso</span>':''}</b>
            <div class="acciones"><button class="rojo" onclick="borrarEtiqueta('${esc(e.id)}')">Borrar</button></div>
          </div>`).join('')||'<p>Sin etiquetas. Creá la primera arriba.</p>';
        const sel=$('#z_etiqueta'), previo=sel.value;
        sel.innerHTML=etiquetas.map(e=>`<option value="${esc(e.id)}">${esc(e.nombre)}</option>`).join('')||'<option value="">(sin etiquetas)</option>';
        if(previo&&etiquetas.some(e=>e.id===previo)) sel.value=previo;
        alSeleccionarEtiqueta();
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
        if(modo==='calibrar'||modo==='posicion'){modo=null; ocultarControles(); $('#instrucciones').textContent='Cámara cambiada: elegí de nuevo el paso 3 o 4.';}
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
        // cargarCatalogo recalcula el badge 'en uso' a partir de zonas.
        await cargarZonas(); await cargarCatalogo();
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
        const datos=await (await fetch('/api/zonas')).json();
        zonas=datos.zones;
        $('#lista-zonas').innerHTML=zonas.map(z=>{
          const detalle=z.area_m2!=null?`${z.area_m2} m²`:(z.celdas?`${z.celdas.length} celdas`:'polígono (formato viejo)');
          return `
          <div class="item" style="border-left:4px solid ${esc(z.color)}">
            <b>${esc(z.name)}</b>
            <small>${esc(detalle)}</small>
            <div class="acciones">
              ${z.celdas?`<button onclick="editarZona('${esc(z.id)}')">Editar</button>`:''}
              <button class="rojo" onclick="borrarZona('${esc(z.id)}')">Borrar</button>
            </div>
          </div>`;
        }).join('')||'<p>Sin zonas todavía.</p>';
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
        escala=p.escala||null;
        zonaCeldaPxPlano=p.zona_celda_px||20;
        if(!zonaEditandoId) zonaCeldaPx=zonaCeldaPxPlano;
        zonaCeldaMetros=p.zona_celda_metros;
        $('#escala-info').textContent=p.pixeles_por_metro
          ?`Escala definida: 1 m ≈ ${p.pixeles_por_metro} px · heatmap en celdas de ${p.celda_metros}×${p.celda_metros} m · grilla de zonas de ${p.zona_celda_metros}×${p.zona_celda_metros} m.`
          :'Sin escala real: la grilla de zonas usa celdas de 20 px sin significado físico. Definila con el paso 2.';
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
        if(!r.ok){alert(msgError(data.detail,'Error al subir la imagen'));return;}
        input.value='';
        await cargarPlano();
      }

      async function quitarImagenPlano(){
        if(!confirm('¿Quitar la imagen del plano? Vuelve a la grilla.')) return;
        await fetch('/api/plano/imagen',{method:'DELETE'});
        await cargarPlano();
      }

      // cargarZonas antes que cargarCatalogo: el catálogo marca qué etiquetas
      // están en uso, y eso lo saca de las zonas ya cargadas.
      (async()=>{ await cargarPlano(); await cargarCamaras(); await cargarZonas(); await cargarCatalogo(); })();
    </script></body></html>
    """


@app.get("/", response_class=HTMLResponse)
def panel_web() -> str:
    return """
    <!doctype html><html lang="es"><head><meta charset="utf-8">
    <title>LeanVision Cerebro</title>
    <style>
      body{margin:0;padding:24px;background:#0f172a;color:#e2e8f0;font:15px system-ui,sans-serif}
      header,.layout{max-width:1280px;margin:auto}.layout{display:flex;gap:28px;align-items:start;flex-wrap:wrap}
      .panel{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px}.clients{flex:1;min-width:300px}.map{width:640px;max-width:100%;box-sizing:border-box}
      #map{position:relative;width:640px;max-width:100%;aspect-ratio:640/480;overflow:hidden;background:#111827}
      #heatmap,#zones{position:absolute;inset:0;width:100%;height:100%}#zones{pointer-events:none}.grid{display:flex;gap:12px;flex-wrap:wrap}.card{width:170px;background:#334155;border-radius:8px;padding:10px}.card img{width:100%;height:130px;object-fit:cover;border-radius:6px}button{padding:9px 12px;background:#ef4444;color:white;border:0;border-radius:6px;cursor:pointer}
    </style></head><body><header><h1>LeanVision Cerebro</h1><p id="health">Cargando métricas…</p><a href="/calibrar" style="color:#7dd3fc;margin-right:12px">Calibrar cámaras / zonas</a><button onclick="resetear()">Resetear memoria</button></header>
    <main class="layout"><section class="panel clients"><h2>Personas activas</h2><div id="clientes" class="grid"></div></section><section class="panel map"><h2>Mapa de calor — plano de tienda</h2><div id="map"><canvas id="heatmap" width="640" height="480"></canvas><canvas id="zones" width="640" height="480"></canvas></div><p id="escala-legenda" style="margin:8px 0 0;font-size:12px;color:#94a3b8"></p></section></main>
    <script>
      const $=s=>document.querySelector(s);
      // Tras un deploy el servidor reinicia y su JS embebido cambia; esta
      // pestaña seguiría con el viejo. Al cambiar la instancia, se recarga.
      let instanciaServidor=null;
      function verificarInstancia(id){
        if(instanciaServidor===null){instanciaServidor=id;return;}
        if(id&&id!==instanciaServidor) location.reload();
      }
      // Rampa de calor por cuadrantes: azul (poco) -> verde -> amarillo -> naranja -> rojo (mucho).
      const COLORES_CALOR=[[59,130,246],[34,197,94],[234,179,8],[249,115,22],[239,68,68]];
      function colorCalor(t){
        const tramo=Math.min(COLORES_CALOR.length-2,Math.floor(t*(COLORES_CALOR.length-1)));
        const f=t*(COLORES_CALOR.length-1)-tramo;
        const a=COLORES_CALOR[tramo],b=COLORES_CALOR[tramo+1];
        return [Math.round(a[0]+(b[0]-a[0])*f),Math.round(a[1]+(b[1]-a[1])*f),Math.round(a[2]+(b[2]-a[2])*f)];
      }
      function dibujarHeatmap(h){
        const c=$('#heatmap'),x=c.getContext('2d');
        x.clearRect(0,0,c.width,c.height);
        const max=Math.max(h.max||0,1);
        (h.celdas||[]).forEach(cel=>{
          const t=Math.min(1,cel.value/max);
          const [r,g,b]=colorCalor(t);
          x.fillStyle=`rgba(${r},${g},${b},${(0.25+0.5*t).toFixed(2)})`;
          x.fillRect(cel.x,cel.y,h.celda_px,h.celda_px);
        });
        $('#escala-legenda').textContent=h.celda_metros
          ?`Cuadrantes de ${h.celda_metros}×${h.celda_metros} m — azul: poco tráfico · rojo: mucho tráfico`
          :'Cuadrantes de '+h.celda_px+' px — definí la escala real en Calibrar para verlos en metros';
      }
      let fondoImagenAplicado=null;
      function aplicarFondoPlano(tieneImagen){
        if(tieneImagen===fondoImagenAplicado) return;
        fondoImagenAplicado=tieneImagen;
        const el=$('#map');
        if(tieneImagen){el.style.backgroundImage=`url(/api/plano/imagen?t=${Date.now()})`;el.style.backgroundSize='cover';el.style.backgroundPosition='center';}
        else{el.style.backgroundImage='';el.style.backgroundSize='';el.style.backgroundPosition='';}
      }
      async function actualizar(){try{const [c,h,m,z,p,cam]=await Promise.all(['/api/clientes','/api/heatmap','/health','/api/zonas','/api/plano','/api/camaras'].map(u=>fetch(u).then(r=>r.json())));$('#clientes').innerHTML=c.clientes.map(x=>`<article class="card"><img src="data:image/jpeg;base64,${x.foto}"><b>${x.id}</b><br>${x.zona}<br><small>${x.genero} · ${x.edad}</small><br><small>${x.similitud} · ${x.ultima_vista}</small></article>`).join('')||'Sin personas activas';dibujarHeatmap(h);$('#health').textContent=`Cola: ${m.queue_size}/${m.queue_capacity} · Procesados: ${m.processed} · Rechazados: ${m.rejected_full} · Último Re-ID: ${m.last_processing_ms} ms`;verificarInstancia(m.instancia);aplicarFondoPlano(p.tiene_imagen);dibujarPlano(z.zones,p.contorno||[],cam.camaras||[],h.celda_px)}catch(e){console.warn(e)}}
      function dibujarPlano(zonas,contorno,camaras,celdaPx){
        const c=$('#zones'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);
        // Grilla de los cuadrantes del heatmap. Va en el canvas y no en CSS
        // para que quede alineada con las celdas aunque el plano se muestre
        // a menos de 640px de ancho.
        if(celdaPx>=4){
          x.strokeStyle='#64748b2e';x.lineWidth=1;x.beginPath();
          for(let gx=0;gx<=c.width;gx+=celdaPx){x.moveTo(Math.round(gx)+0.5,0);x.lineTo(Math.round(gx)+0.5,c.height);}
          for(let gy=0;gy<=c.height;gy+=celdaPx){x.moveTo(0,Math.round(gy)+0.5);x.lineTo(c.width,Math.round(gy)+0.5);}
          x.stroke();
        }
        if(contorno.length>=3){x.beginPath();x.moveTo(...contorno[0]);contorno.slice(1).forEach(p=>x.lineTo(...p));x.closePath();x.strokeStyle='#94a3b8';x.lineWidth=3;x.stroke()}
        zonas.forEach(z=>{
          const color=z.color||'#00ff88';
          if(z.celdas&&z.celdas.length){
            // Zona pintada sobre la grilla: se rellenan las celdas y se
            // recuadra el contorno exterior (los bordes que no lindan con
            // otra celda de la misma zona), para que se lea como una figura
            // sola y no como un damero.
            const lado=z.celda_px||20, ocupadas=new Set(z.celdas.map(([c,f])=>c+','+f));
            x.fillStyle=color+'44';
            z.celdas.forEach(([c,f])=>x.fillRect(c*lado,f*lado,lado,lado));
            x.strokeStyle=color;x.lineWidth=2;x.beginPath();
            z.celdas.forEach(([c,f])=>{
              const px=c*lado, py=f*lado;
              if(!ocupadas.has((c)+','+(f-1))){x.moveTo(px,py);x.lineTo(px+lado,py);}
              if(!ocupadas.has((c)+','+(f+1))){x.moveTo(px,py+lado);x.lineTo(px+lado,py+lado);}
              if(!ocupadas.has((c-1)+','+f)){x.moveTo(px,py);x.lineTo(px,py+lado);}
              if(!ocupadas.has((c+1)+','+f)){x.moveTo(px+lado,py);x.lineTo(px+lado,py+lado);}
            });
            x.stroke();
            const [ec,ef]=z.celdas.reduce((a,b)=>(b[1]<a[1]||(b[1]===a[1]&&b[0]<a[0]))?b:a);
            x.fillStyle='white';x.font='12px sans-serif';x.fillText(z.name,ec*lado+3,ef*lado+13);
            return;
          }
          if(!z.polygon||z.polygon.length<3)return;
          x.beginPath();x.moveTo(...z.polygon[0]);z.polygon.slice(1).forEach(p=>x.lineTo(...p));x.closePath();
          x.strokeStyle=color;x.lineWidth=2;x.stroke();x.fillStyle=color+'33';x.fill();
          x.fillStyle='white';x.fillText(z.name,z.polygon[0][0]+5,z.polygon[0][1]-5);
        });
        camaras.forEach(cm=>{if(!cm.posicion)return;const[px,py]=cm.posicion;x.beginPath();x.arc(px,py,10,0,7);x.fillStyle='#38bdf8';x.fill();x.strokeStyle='#0f172a';x.lineWidth=2;x.stroke();x.fillStyle='#0f172a';x.font='bold 11px sans-serif';x.fillText('C',px-3,py+4);x.fillStyle='#7dd3fc';x.font='11px sans-serif';x.fillText(cm.nombre||cm.camara_id,px+14,py+4)});
      }
      async function resetear(){if(confirm('¿Borrar memoria y mapa?')){await fetch('/api/reset',{method:'POST'});actualizar()}} actualizar();setInterval(actualizar,1500);
    </script></body></html>
    """


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("LEANVISION_PORT", "8081")))
