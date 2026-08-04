"""Cerebro LeanVision (CerebroLocal): recepción en cola, Re-ID robusto y heatmap nativo 640x480.

Portado desde mi-proyecto/cerebro_server.py el 2026-08-04, adaptado para
mantener compatibilidad con la configuración de CerebroLocal (Supabase vía
.env local, demografia.onnx y haarcascade_frontalface_default.xml en esta
carpeta).
"""

from __future__ import annotations

import asyncio
import base64
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
from fastapi.responses import HTMLResponse
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
    with open(ruta, "r", encoding="utf-8") as archivo:
        for linea in archivo:
            linea = linea.strip()
            if not linea or linea.startswith("#") or "=" not in linea:
                continue
            clave, _, valor = linea.partition("=")
            os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


_cargar_dotenv_local()

HEATMAP_WIDTH = 640
HEATMAP_HEIGHT = 480
# Heatmap por grilla con decaimiento: refleja actividad reciente (no todo el
# día) y manda un payload chico al dashboard.
HEATMAP_CELDA_PX = 20
HEATMAP_TAU_SEGUNDOS = 45.0  # Constante de enfriamiento: a mayor valor, más "memoria".
HEATMAP_VALOR_MINIMO = 0.05  # Debajo de esto una celda se considera fría y se descarta.
IP_CAMARA = os.getenv("LEANVISION_CAMERA_URL", "http://172.31.99.7:8002")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# La recepción HTTP no ejecuta IA: sólo encola JPEGs. Así el loop de la cámara
# nunca queda esperando al modelo ni se acumulan threads sin límite.
REID_QUEUE_SIZE = 24
REID_WORKERS = 1  # OSNet sobre CPU es más estable con una inferencia a la vez.
SUPABASE_WORKERS = 1

# Re-ID: match flexible para cambios sentado/parado, actualización estricta
# para impedir que un recorte incorrecto contamine el centroide de una persona.
UMBRAL_SIMILITUD = 0.36
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

# Edad/género con InsightFace: se toman varias muestras por sesión y se vota
# (mayoría de género, mediana de edad) para que el resultado sea estable.
DEMO_MAX_VOTOS = 7
DEMO_INTERVALO_SEGUNDOS = 1.5
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
contador_global_ids = 1
cola_reid: asyncio.Queue[DetectionJob] | None = None
reid_executor = ThreadPoolExecutor(max_workers=REID_WORKERS, thread_name_prefix="reid")
supabase_executor = ThreadPoolExecutor(max_workers=SUPABASE_WORKERS, thread_name_prefix="supabase")
demografia_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="demografia")
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
    analizador_rostros.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 => CPU
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


def _decaimiento(valor: float, ultimo: float, ahora: float) -> float:
    """Enfría una celda según el tiempo transcurrido (decaimiento exponencial)."""
    dt = ahora - ultimo
    if dt <= 0:
        return valor
    return valor * float(np.exp(-dt / HEATMAP_TAU_SEGUNDOS))


def _registrar_heatmap(pos_x: float, pos_y: float, ahora: float) -> None:
    """Acumula presencia en la celda de la grilla, enfriando lo previo primero."""
    if not (0 <= pos_x < HEATMAP_WIDTH and 0 <= pos_y < HEATMAP_HEIGHT):
        return
    celda_id = (int(pos_x // HEATMAP_CELDA_PX), int(pos_y // HEATMAP_CELDA_PX))
    celda = heatmap_celdas.get(celda_id)
    valor_previo = _decaimiento(celda["valor"], celda["ultimo"], ahora) if celda else 0.0
    heatmap_celdas[celda_id] = {"valor": valor_previo + 1.0, "ultimo": ahora}


def _snapshot_heatmap(ahora: float) -> tuple[list[dict], float]:
    """Devuelve los puntos vigentes (celdas ya enfriadas) y el valor máximo."""
    puntos: list[dict] = []
    maximo = 0.0
    frias: list[tuple[int, int]] = []
    for celda_id, celda in heatmap_celdas.items():
        valor = _decaimiento(celda["valor"], celda["ultimo"], ahora)
        if valor < HEATMAP_VALOR_MINIMO:
            frias.append(celda_id)
            continue
        celda["valor"], celda["ultimo"] = valor, ahora
        col, fila = celda_id
        puntos.append({
            "x": int(col * HEATMAP_CELDA_PX + HEATMAP_CELDA_PX // 2),
            "y": int(fila * HEATMAP_CELDA_PX + HEATMAP_CELDA_PX // 2),
            "value": round(valor, 3),
        })
        maximo = max(maximo, valor)
    for celda_id in frias:
        heatmap_celdas.pop(celda_id, None)
    return puntos, maximo


def _muestrear_demografia(id_global: str, image_bytes: bytes) -> None:
    """Corre InsightFace sobre un recorte y suma un voto de edad/género.

    Se ejecuta en su propio executor para no frenar el pipeline de Re-ID. Sólo
    cuenta el rostro más confiable del recorte; si no hay cara clara, no vota.
    """
    if not tiene_demografia or analizador_rostros is None:
        return
    try:
        imagen = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if imagen is None:
            return
        rostros = analizador_rostros.get(imagen)
        if not rostros:
            return
        mejor = max(rostros, key=lambda r: float(getattr(r, "det_score", 0.0)))
        if float(getattr(mejor, "det_score", 0.0)) < DEMO_MIN_DET_SCORE:
            return
        genero = "Mujer" if int(mejor.sex == "F") else "Hombre"
        edad = int(mejor.age)
    except Exception as error:
        logger.warning("No se pudo inferir edad/género: %s", error)
        return

    with state_lock:
        datos = clientes_globales.get(id_global)
        if datos is None:
            return
        datos["votos_genero"].append(genero)
        datos["votos_edad"].append(edad)


def _quizas_muestrear_demografia(id_global: str, datos: dict, ahora: float, image_bytes: bytes) -> None:
    """Programa una muestra de demografía si toca (throttle + tope de votos)."""
    if not tiene_demografia:
        return
    if len(datos.get("votos_edad", [])) >= DEMO_MAX_VOTOS:
        return
    if ahora - datos.get("ultimo_voto_demo", 0.0) < DEMO_INTERVALO_SEGUNDOS:
        return
    datos["ultimo_voto_demo"] = ahora
    demografia_executor.submit(_muestrear_demografia, id_global, image_bytes)


def _rango_edad(edad: int) -> str:
    return "-18" if edad < 18 else "18-25" if edad <= 25 else "26-35" if edad <= 35 else "36-45" if edad <= 45 else "46+"


def _resolver_demografia(datos: dict) -> tuple[str, str]:
    """Consolida los votos de la sesión: mayoría de género, mediana de edad."""
    votos_genero = datos.get("votos_genero", [])
    votos_edad = datos.get("votos_edad", [])
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
        _registrar_heatmap(*posicion, ahora)

        # La ruta local es rápida, pero no actualiza el álbum ciegamente: un
        # recorte erróneo de ese tracker no debe alterar la identidad global.
        if id_local and id_local in traductor_camaras:
            id_global = traductor_camaras[id_local]["id_global"]
            if id_global in clientes_globales:
                datos = clientes_globales[id_global]
                _agregar_huella_confiable(datos, huella_nueva, ahora)
                datos["apariciones"] = datos.get("apariciones", 1) + 1
                datos.update({
                    "foto_b64": foto_b64,
                    "zona_actual": job.zona,
                    "timestamp": ahora,
                    "hora_legible": time.strftime("%H:%M:%S"),
                    "camara_id": job.camara_id,
                    "posicion": posicion,
                })
                traductor_camaras[id_local].update(ultimo_update=ahora, posicion=posicion)
                _quizas_muestrear_demografia(id_global, datos, ahora, job.image_bytes)
                _actualizar_metricas_procesado(started_at)
                return id_global

        bloqueados = _ids_bloqueados_en_misma_camara(job.camara_id, id_local, posicion, ahora)
        mejor_id_global: str | None = None
        mejor_puntaje = -1.0
        candidatos_continuidad: list[tuple[str, float]] = []

        for persona_id, datos in clientes_globales.items():
            if datos.get("branch_id") != job.branch_id or persona_id in bloqueados:
                continue
            if job.zona != datos.get("zona_actual") and ahora - datos.get("timestamp", 0.0) < TIEMPO_TELETRANSPORTACION:
                continue

            puntaje = _puntaje_identidad(huella_nueva, datos)
            # Bonus pequeño para recuperar una persona que se mantuvo en el
            # mismo asiento cuando el tracker local cambia de ID.
            if _distancia(datos.get("posicion"), posicion) <= DISTANCIA_REID_LOCAL_PX:
                puntaje += 0.04
            continuidad_postura = (
                datos.get("camara_id") == job.camara_id
                and ahora - datos.get("timestamp", 0.0) <= TIEMPO_CONTINUIDAD_POSTURA
                and _misma_ubicacion_sentado_parado(datos.get("posicion"), posicion)
            )
            if continuidad_postura:
                candidatos_continuidad.append((persona_id, puntaje))
            if puntaje > mejor_puntaje:
                mejor_puntaje, mejor_id_global = puntaje, persona_id

        continuidad_unica = len(candidatos_continuidad) == 1
        puede_recuperar_por_continuidad = (
            continuidad_unica
            and candidatos_continuidad[0][1] >= UMBRAL_SIMILITUD_CONTINUIDAD
        )
        if mejor_id_global is not None and (
            mejor_puntaje >= UMBRAL_SIMILITUD or puede_recuperar_por_continuidad
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
                "zona_entrada": job.zona,
                "branch_id": job.branch_id,
                "apariciones": 1,
                "similitud_ema": 0.0,
                "votos_genero": [],
                "votos_edad": [],
                "ultimo_voto_demo": 0.0,
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
        datos.update({
            "foto_b64": foto_b64,
            "zona_actual": job.zona,
            "similitud": datos["similitud_ema"],
            "timestamp": ahora,
            "hora_legible": time.strftime("%H:%M:%S"),
            "branch_id": job.branch_id,
            "camara_id": job.camara_id,
            "posicion": posicion,
        })
        _quizas_muestrear_demografia(id_global, datos, ahora, job.image_bytes)
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
    genero, rango_edad = _resolver_demografia(datos)
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
        logger.warning("No se pudo guardar la sesión en Supabase: %s", error)


async def reloj_limpiador_background() -> None:
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(20)
        ahora = time.time()
        with state_lock:
            vencidos = [pid for pid, datos in clientes_globales.items() if ahora - datos.get("timestamp", ahora) > TIEMPO_INACTIVIDAD_SEGUNDOS]
            sesiones = [(pid, clientes_globales.pop(pid)) for pid in vencidos]
            for clave in [k for k, v in traductor_camaras.items() if ahora - v.get("ultimo_update", 0.0) > TIEMPO_INACTIVIDAD_SEGUNDOS]:
                traductor_camaras.pop(clave, None)
        for pid, datos in sesiones:
            tiempo_adentro = int(datos["timestamp"] - datos["hora_entrada"])
            if tiempo_adentro > 5:
                await loop.run_in_executor(supabase_executor, procesar_y_enviar_supabase, pid, datos, tiempo_adentro)


@app.on_event("startup")
async def iniciar_servicios() -> None:
    global cola_reid
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
    with state_lock:
        clientes = []
        for persona_id, datos in clientes_globales.items():
            # Los clientes provisionales (una sola aparición) no se muestran
            # todavía: evita tarjetas fantasma por un recorte espurio.
            if datos.get("apariciones", 1) < MIN_APARICIONES_VISIBLE:
                continue
            genero, rango_edad = _resolver_demografia(datos)
            clientes.append({
                "id": persona_id,
                "zona": datos.get("zona_actual", "Desconocida"),
                "similitud": f"{datos.get('similitud', 0.0) * 100:.1f}%",
                "ultima_vista": datos.get("hora_legible", "--:--:--"),
                "genero": genero,
                "edad": rango_edad,
                "foto": datos.get("foto_b64", ""),
            })
    return {"clientes": clientes}


@app.get("/api/heatmap")
def obtener_heatmap() -> dict:
    ahora = time.time()
    with state_lock:
        puntos, maximo = _snapshot_heatmap(ahora)
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
    </style></head><body><header><h1>LeanVision Cerebro</h1><p id="health">Cargando métricas…</p><button onclick="resetear()">Resetear memoria</button></header>
    <main class="layout"><section class="panel clients"><h2>Personas activas</h2><div id="clientes" class="grid"></div></section><section class="panel map"><h2>Mapa de calor nativo — 640×480</h2><div id="map"><div id="heatmap"></div><canvas id="zones" width="640" height="480"></canvas></div></section></main>
    <script>
      const heatmap=h337.create({container:document.querySelector('#heatmap'),radius:38,maxOpacity:.72,blur:.82});
      const $=s=>document.querySelector(s);
      async function actualizar(){try{const [c,h,m,z]=await Promise.all(['/api/clientes','/api/heatmap','/health','/api/zonas_camara'].map(u=>fetch(u).then(r=>r.json())));$('#clientes').innerHTML=c.clientes.map(x=>`<article class="card"><img src="data:image/jpeg;base64,${x.foto}"><b>${x.id}</b><br>${x.zona}<br><small>${x.genero} · ${x.edad}</small><br><small>${x.similitud} · ${x.ultima_vista}</small></article>`).join('')||'Sin personas activas';heatmap.setData({max:Math.max(3,h.max||0),data:h.puntos});$('#health').textContent=`Cola: ${m.queue_size}/${m.queue_capacity} · Procesados: ${m.processed} · Rechazados: ${m.rejected_full} · Último Re-ID: ${m.last_processing_ms} ms`;dibujarZonas(z.zones)}catch(e){console.warn(e)}}
      function dibujarZonas(zonas){const c=$('#zones'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);zonas.forEach(z=>{if(!z.polygon||z.polygon.length<3)return;x.beginPath();x.moveTo(...z.polygon[0]);z.polygon.slice(1).forEach(p=>x.lineTo(...p));x.closePath();x.strokeStyle=z.color||'#00ff88';x.lineWidth=2;x.stroke();x.fillStyle=(z.color||'#00ff88')+'33';x.fill();x.fillStyle='white';x.fillText(z.name,z.polygon[0][0]+5,z.polygon[0][1]-5)})}
      async function resetear(){if(confirm('¿Borrar memoria y mapa?')){await fetch('/api/reset',{method:'POST'});actualizar()}} actualizar();setInterval(actualizar,1500);
    </script></body></html>
    """


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("LEANVISION_PORT", "8081")))
