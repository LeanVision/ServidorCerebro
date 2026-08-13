#!/bin/bash
# Chequea si hay commits nuevos en origin/main; si los hay, actualiza el
# código, reinstala dependencias (por si requirements.txt cambió) y reinicia
# el servicio. Pensado para correr cada pocos minutos vía cerebro-autopull.timer.
set -euo pipefail

# Se ubica solo a partir de dónde vive el script (deploy/<esto>.sh), no
# hardcodea el usuario/ruta de ninguna Pi en particular: el mismo archivo,
# tal cual viene del repo, sirve para cualquier Pi sin importar su usuario.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

git fetch origin main

LOCAL_REV="$(git rev-parse HEAD)"
REMOTE_REV="$(git rev-parse origin/main)"

if [ "$LOCAL_REV" = "$REMOTE_REV" ]; then
    echo "cerebro-autopull: sin cambios (${LOCAL_REV:0:8})."
    exit 0
fi

echo "cerebro-autopull: actualizando ${LOCAL_REV:0:8} -> ${REMOTE_REV:0:8}"
git pull --ff-only origin main
"$REPO_DIR/venv/bin/pip" install -q -r requirements.txt
sudo /usr/bin/systemctl restart cerebro.service
echo "cerebro-autopull: listo, servicio reiniciado."
