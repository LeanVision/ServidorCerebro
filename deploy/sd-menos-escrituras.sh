#!/usr/bin/env bash
# Reduce las escrituras a la tarjeta SD en cualquier Pi de LeanVision
# (Cerebro o cámara). Es idempotente: se puede correr las veces que haga falta.
#
# Lo que NO toca, a propósito: events.db (cámara) y estado/outbox.db (Cerebro).
# Esas dos son la cola de datos pendientes de enviar. Si viven en RAM, un corte
# de luz borra las visitas que todavía no llegaron a Supabase. El desgaste que
# generan es mínimo: un INSERT por cambio de zona, con WAL y synchronous=NORMAL.
#
# Uso:
#   sudo ./sd-menos-escrituras.sh            # aplica los cambios seguros
#   sudo ./sd-menos-escrituras.sh --medir    # sólo muestra cuánto se escribió
#   sudo ./sd-menos-escrituras.sh --sin-swap # además desactiva la swap (ver abajo)
set -euo pipefail

DISCO="${DISCO:-mmcblk0}"

medir() {
    # Campo 10 de /proc/diskstats = sectores escritos desde el último arranque.
    awk -v disco="$DISCO" -v horas="$(awk '{printf "%.1f", $1 / 3600}' /proc/uptime)" '
        $3 == disco {
            printf "%s: %.2f GiB escritos en %s horas de encendido\n",
                disco, $10 * 512 / 1073741824, horas
        }' /proc/diskstats
}

if [[ "${1:-}" == "--medir" ]]; then
    medir
    exit 0
fi

if [[ $EUID -ne 0 ]]; then
    exec sudo -- "$0" "$@"
fi

echo "== Antes =="
medir

# 1. Journal acotado. Sin tope, journald crece hasta el 10% de la partición y
#    reescribe ese volumen entero cada vez que rota. 50M alcanza de sobra para
#    diagnosticar un reinicio del watchdog y mantiene los logs tras un corte,
#    que es justo cuando hacen falta.
install -d /etc/systemd/journald.conf.d
cat > /etc/systemd/journald.conf.d/leanvision.conf <<'EOF'
[Journal]
SystemMaxUse=50M
SystemMaxFileSize=10M
EOF
systemctl restart systemd-journald
echo "journald: tope de 50M aplicado"

# 2. /tmp en RAM. systemd ya trae la unidad tmp.mount, no hace falta tocar
#    /etc/fstab. Se lleva a RAM los temporales de pip, apt y OpenCV.
systemctl enable --now tmp.mount 2>/dev/null || true
echo "tmp.mount: $(systemctl is-enabled tmp.mount 2>/dev/null || echo 'no disponible')"

# 3. Swap. Es el que de verdad castiga la SD cuando falta RAM: escribe y
#    reescribe las mismas celdas sin parar. Opt-in porque desactivarla en una
#    Zero 2W (512MB) cambia el thrashing por un OOM kill, y el proceso muere en
#    vez de ponerse lento. En una Pi 5 de 1GB se puede sacar sin drama.
if [[ "${1:-}" == "--sin-swap" ]]; then
    dphys-swapfile swapoff || true
    dphys-swapfile uninstall || true
    systemctl disable --now dphys-swapfile || true
    echo "swap: desactivada"
else
    echo "swap: intacta (usar --sin-swap para desactivarla)"
fi

echo "== Después =="
medir
echo
echo "Volvé a correr '$0 --medir' dentro de unos días para comparar."
