"""Cola de salida en disco para lo que el Cerebro le manda a la nube.

POR QUÉ EXISTE
Hasta ahora, si el POST a Supabase fallaba, la visita se perdía: no había
reintento ni buffer. Ese modo de falla costó 21 días de datos en agosto de
2026, y desde el PR #17 sólo sabíamos que se perdían, no cómo recuperarlas.

Acá se separan dos cosas que estaban pegadas: **capturar** el dato y
**entregarlo**. Al cerrar la sesión, el payload se escribe en disco antes de
que exista cualquier red. Entregarlo pasa a ser un problema aparte que puede
fallar, esperar y reintentar sin destruir nada.

DÓNDE VIVE EL ARCHIVO
Fuera del checkout de git, a propósito. `cerebro-autopull.sh` hace
`git pull --ff-only` sobre el repo cada cinco minutos; un archivo con estado
adentro de ese directorio es pedir problemas. El default es un hermano del
repo (`../estado/outbox.db`) y se puede mover con LEANVISION_OUTBOX_DB.

Sigue el patrón que ya usa la cámara en `database.py`: una conexión por hilo
con threading.local, WAL y synchronous=NORMAL. Ese par evita un fsync por
escritura, que es lo que realmente desgasta la SD de una Raspberry.
"""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

RUTA_DB = Path(
    os.getenv("LEANVISION_OUTBOX_DB", str(Path(__file__).resolve().parent.parent / "estado" / "outbox.db"))
)

# Espera creciente entre reintentos, en segundos, por número de intento. Se
# queda en 5 minutos: ante una caída larga no tiene sentido golpear más seguido,
# y con el tope el drenaje se recupera rápido cuando la conexión vuelve.
ESPERA_MAXIMA_SEGUNDOS = 300.0
ESPERA_BASE_SEGUNDOS = 5.0

DIAS_DE_RETENCION = float(os.getenv("LEANVISION_OUTBOX_RETENCION_DIAS", "15"))

_local = threading.local()


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conexion() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        RUTA_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(RUTA_DB), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
    return conn


def inicializar() -> None:
    conn = _conexion()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS outbox (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            -- Identifica el hecho, no el envío. Reencolar lo mismo no duplica.
            idempotencia       TEXT NOT NULL UNIQUE,
            tipo               TEXT NOT NULL,
            payload            TEXT NOT NULL,
            creado_en          TEXT NOT NULL,
            sincronizado_en    TEXT,
            intentos           INTEGER NOT NULL DEFAULT 0,
            proximo_intento_en TEXT NOT NULL,
            ultimo_error       TEXT
        )
        """
    )
    # Índice parcial: el drenaje sólo mira lo pendiente, que casi siempre es
    # nada. No paga recorrer quince días de historia ya entregada.
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_outbox_pendiente
            ON outbox(proximo_intento_en) WHERE sincronizado_en IS NULL
        """
    )
    conn.commit()


def encolar(tipo: str, idempotencia: str, payload: dict) -> bool:
    """Deja un envío listo para entregar. Devuelve False si ya estaba.

    El UNIQUE sobre idempotencia hace que reencolar el mismo hecho sea
    inofensivo: si el proceso muere entre el encolado y el commit del estado,
    reintentar no duplica.
    """
    conn = _conexion()
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO outbox
            (idempotencia, tipo, payload, creado_en, proximo_intento_en)
        VALUES (?, ?, ?, ?, ?)
        """,
        (idempotencia, tipo, json.dumps(payload, ensure_ascii=False), _ahora(), _ahora()),
    )
    conn.commit()
    return cursor.rowcount > 0


def pendientes(limite: int = 50) -> list[dict]:
    """Lo que toca entregar ahora: sin sincronizar y con la espera cumplida."""
    conn = _conexion()
    filas = conn.execute(
        """
        SELECT id, tipo, payload, intentos
          FROM outbox
         WHERE sincronizado_en IS NULL
           AND proximo_intento_en <= ?
         ORDER BY id
         LIMIT ?
        """,
        (_ahora(), limite),
    ).fetchall()
    return [
        {"id": f["id"], "tipo": f["tipo"], "payload": json.loads(f["payload"]), "intentos": f["intentos"]}
        for f in filas
    ]


def marcar_enviado(fila_id: int) -> None:
    conn = _conexion()
    conn.execute(
        "UPDATE outbox SET sincronizado_en = ?, ultimo_error = NULL WHERE id = ?",
        (_ahora(), fila_id),
    )
    conn.commit()


def marcar_error(fila_id: int, error: str) -> None:
    """Cuenta el intento y posterga el próximo con espera creciente."""
    conn = _conexion()
    fila = conn.execute("SELECT intentos FROM outbox WHERE id = ?", (fila_id,)).fetchone()
    intentos = (fila["intentos"] if fila else 0) + 1
    espera = min(ESPERA_BASE_SEGUNDOS * (2 ** (intentos - 1)), ESPERA_MAXIMA_SEGUNDOS)
    proximo = (datetime.now(timezone.utc) + timedelta(seconds=espera)).isoformat()
    conn.execute(
        """
        UPDATE outbox
           SET intentos = ?, proximo_intento_en = ?, ultimo_error = ?
         WHERE id = ?
        """,
        (intentos, proximo, error[:200], fila_id),
    )
    conn.commit()


def limpiar(dias: float = DIAS_DE_RETENCION) -> int:
    """Borra lo ya entregado y viejo. Devuelve cuántas filas se fueron.

    La condición es sincronizado Y viejo, NUNCA sólo viejo: borrar por fecha de
    creación descartaría en silencio lo que todavía no se subió, que es
    exactamente el caso que esta cola viene a resolver. Una caída de más de
    `dias` es improbable, pero es donde más duele equivocarse.
    """
    conn = _conexion()
    corte = (datetime.now(timezone.utc) - timedelta(days=dias)).isoformat()
    cursor = conn.execute(
        "DELETE FROM outbox WHERE sincronizado_en IS NOT NULL AND sincronizado_en < ?",
        (corte,),
    )
    conn.commit()
    return cursor.rowcount


def estado() -> dict:
    """Foto para /health.

    Una cola que deja de drenar es tan invisible como lo era la pérdida de
    sesiones antes del PR #17. `segundos_del_mas_viejo` es el número que hay que
    mirar: si crece sin parar, no se está entregando nada.
    """
    conn = _conexion()
    fila = conn.execute(
        """
        SELECT COUNT(*) AS pendientes, MIN(creado_en) AS mas_viejo
          FROM outbox WHERE sincronizado_en IS NULL
        """
    ).fetchone()
    entregados = conn.execute(
        "SELECT COUNT(*) AS n FROM outbox WHERE sincronizado_en IS NOT NULL"
    ).fetchone()

    segundos = None
    if fila["mas_viejo"]:
        nacido = datetime.fromisoformat(fila["mas_viejo"])
        segundos = round((datetime.now(timezone.utc) - nacido).total_seconds())

    return {
        "archivo": str(RUTA_DB),
        "pendientes": fila["pendientes"],
        "entregados_sin_purgar": entregados["n"],
        "segundos_del_mas_viejo": segundos,
    }
