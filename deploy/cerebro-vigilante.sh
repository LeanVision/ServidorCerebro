#!/bin/bash
# Vigilante del Cerebro: reinicia el servicio si deja de responder.
#
# systemd (Restart=on-failure) sólo reinicia si el proceso MUERE. Un Cerebro
# colgado (deadlock, torch trabado, uvicorn vivo pero sin atender) sigue
# "active" y nadie lo toca. Este script pregunta a /health y, tras
# FALLOS_PARA_REINICIAR chequeos seguidos sin respuesta, reinicia cerebro.service.
# Corre cada minuto vía cerebro-vigilante.timer, como root (puede reiniciar
# sin sudo). Las respuestas lentas no cuentan como falla: el Re-ID carga la CPU.
set -uo pipefail

PUERTO="${LEANVISION_PORT:-8081}"
FALLOS_PARA_REINICIAR="${FALLOS_PARA_REINICIAR:-3}"
TIMEOUT_SEGUNDOS="${TIMEOUT_SEGUNDOS:-15}"
ESTADO="${CEREBRO_VIGILANTE_ESTADO:-/run/cerebro-vigilante.fallos}"

# Si el servicio está parado a propósito o systemd ya lo está reiniciando,
# no interferir.
if ! systemctl is-active --quiet cerebro.service; then
    echo "cerebro-vigilante: cerebro.service no está activo, nada que vigilar."
    rm -f "$ESTADO"
    exit 0
fi

if curl -fsS --max-time "$TIMEOUT_SEGUNDOS" -o /dev/null "http://127.0.0.1:${PUERTO}/health"; then
    rm -f "$ESTADO"
    exit 0
fi

FALLOS=$(( $(cat "$ESTADO" 2>/dev/null || echo 0) + 1 ))
echo "cerebro-vigilante: /health sin respuesta (${FALLOS}/${FALLOS_PARA_REINICIAR})."

if [ "$FALLOS" -ge "$FALLOS_PARA_REINICIAR" ]; then
    echo "cerebro-vigilante: reiniciando cerebro.service."
    rm -f "$ESTADO"
    systemctl restart cerebro.service
else
    echo "$FALLOS" > "$ESTADO"
fi
