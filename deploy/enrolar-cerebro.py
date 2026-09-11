"""Da de alta al Cerebro como dispositivo de LeanRetail y escribe su credencial en .env.

Uso, en la Pi del Cerebro, parado en la carpeta del repo:

    python3 deploy/enrolar-cerebro.py https://app.leanvision.ai ABCDE-FGHJK

El código de activación se genera en el panel: Dispositivos, agregar uno nuevo en la
sucursal del local. Se muestra una sola vez y vence.

El secreto se genera acá y nunca sale de la Pi: a la app sólo viaja su hash. Si se
filtra, se revoca desvinculando el dispositivo desde el panel.

También borra del .env SUPABASE_URL, SUPABASE_KEY y SUPABASE_HEATMAP_URL: después de
esto la Pi no guarda ninguna clave de la base de datos.
"""
import hashlib
import json
import os
import secrets
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

VIEJAS = {"SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_HEATMAP_URL"}


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    app_url, codigo = sys.argv[1].rstrip("/"), sys.argv[2].strip().upper()

    secreto = secrets.token_urlsafe(32)
    pedido = urllib.request.Request(
        f"{app_url}/api/edge/provision",
        data=json.dumps({
            "activation_code": codigo,
            "device_id": str(uuid.uuid4()),
            "credential_hash": hashlib.sha256(secreto.encode()).hexdigest(),
            "agent_version": "cerebro",
        }).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "LeanVision-Cerebro"},
    )
    try:
        with urllib.request.urlopen(pedido, timeout=20) as respuesta:
            alta = json.load(respuesta)
    except urllib.error.HTTPError as error:
        sys.exit(f"La app rechazó la activación ({error.code}): {error.read().decode()[:200]}")

    nuevas = {
        "LEANRETAIL_INGEST_URL": f"{app_url}/api/edge/ingest",
        "LEANRETAIL_KEY_ID": alta["key_id"],
        "LEANRETAIL_SECRET": secreto,
    }
    env = Path(".env")
    lineas = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    lineas = [l for l in lineas if l.split("=", 1)[0].strip() not in VIEJAS | nuevas.keys()]
    lineas += [f"{clave}={valor}" for clave, valor in nuevas.items()]
    env.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    os.chmod(env, 0o600)

    print(f"Cerebro activado en «{alta['store']['name']}» como {alta['edge_node_id']}.")
    print("Credencial escrita en .env. Reiniciá el servicio para que la tome.")


if __name__ == "__main__":
    main()
