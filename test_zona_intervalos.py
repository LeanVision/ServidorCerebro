"""Comprueba que el recorrido por zonas se asiente bien.

Corre con: python test_zona_intervalos.py

Importar ServerLocal carga los modelos y tarda ~30s; es el precio de probar las
funciones reales en vez de una copia que se desincroniza.
"""

import ServerLocal as sl

MAX = sl.MAX_INTERVALOS_ZONA_POR_VISITA


def tramos(datos):
    return [(i["zona"], i["segundos"]) for i in datos.get("zona_intervalos", [])]


def ver(datos, zona, ahora):
    """Una detección, como la procesa el servidor.

    El orden importa: _actualizar_zona_y_tiempo cierra el tramo de la zona
    anterior y recién después el llamador sobreescribe zona_actual. Si el test
    no respeta ese orden, no prueba nada de lo que pasa en producción.
    """
    sl._actualizar_zona_y_tiempo(datos, zona, ahora)
    datos["zona_actual"] = zona
    datos["timestamp"] = ahora


def test_cambio_de_zona_cierra_el_tramo_anterior():
    datos = {}
    ver(datos, "Entrada", 1000.0)
    ver(datos, "Cajas", 1010.0)
    assert tramos(datos) == [("Entrada", 10)], tramos(datos)
    # El tramo de Cajas sigue abierto: no se asienta hasta que cambie o cierre.
    assert datos["zona_desde"] == 1010.0


def test_misma_zona_no_parte_el_tramo():
    datos = {}
    ver(datos, "Entrada", 1000.0)
    ver(datos, "Entrada", 1005.0)
    ver(datos, "Cajas", 1010.0)
    assert tramos(datos) == [("Entrada", 10)], tramos(datos)


def test_volver_a_una_zona_abre_un_tramo_nuevo():
    """Lo que FOCA tiene que poder leer: dos pasos por Entrada, no uno."""
    datos = {}
    ver(datos, "Entrada", 1000.0)
    ver(datos, "Cajas", 1010.0)
    ver(datos, "Entrada", 1030.0)
    ver(datos, "Cajas", 1035.0)
    assert tramos(datos) == [("Entrada", 10), ("Cajas", 20), ("Entrada", 5)], tramos(datos)


def test_el_roce_con_el_borde_no_cuenta():
    datos = {}
    ver(datos, "Entrada", 1000.0)
    ver(datos, "Pasillo", 1000.4)  # 0,4s: roce
    ver(datos, "Cajas", 1010.0)
    assert tramos(datos) == [("Pasillo", 9)], tramos(datos)


def test_los_tramos_nunca_suman_mas_que_la_permanencia():
    """La suma truncada de los tramos no puede pasarse del total truncado.

    Es el mismo falso positivo que ya se arregló en zonas_tiempo: si los tramos
    redondearan para arriba, el recorrido mostraría más segundos que la visita.
    """
    datos = {}
    marcas = [0.0, 1.7, 3.4, 5.1, 6.8, 8.5]
    zonas = ["A", "B", "A", "B", "A", "B"]
    for zona, t in zip(zonas, marcas):
        ver(datos, zona, t)
    sl._anotar_intervalo_zona(datos, "B", datos["zona_desde"], 10.0)
    suma = sum(i["segundos"] for i in datos["zona_intervalos"])
    assert suma <= int(10.0 - marcas[0]), (suma, tramos(datos))


def test_el_tope_corta_y_no_crece_sin_fin():
    datos = {}
    for i in range(MAX + 50):
        sl._anotar_intervalo_zona(datos, "Borde", float(i * 2), float(i * 2 + 1))
    assert len(datos["zona_intervalos"]) == MAX, len(datos["zona_intervalos"])
    assert datos["intervalos_truncados"] is True


def test_las_horas_salen_en_utc_iso():
    datos = {}
    sl._anotar_intervalo_zona(datos, "Entrada", 1794528000.0, 1794528030.0)
    tramo = datos["zona_intervalos"][0]
    assert tramo["entrada"] == "2026-11-13T00:00:00+00:00", tramo["entrada"]
    assert tramo["salida"] == "2026-11-13T00:00:30+00:00", tramo["salida"]
    assert tramo["segundos"] == 30


def test_api_clientes_no_asienta_tramos():
    """_cerrar_intervalo_zona la llama el panel en vivo: no debe mutar nada."""
    datos = {}
    ver(datos, "Entrada", 1000.0)
    for _ in range(3):
        sl._cerrar_intervalo_zona(datos, 1050.0)
    assert tramos(datos) == [], tramos(datos)


if __name__ == "__main__":
    for nombre, fn in sorted(globals().items()):
        if nombre.startswith("test_"):
            fn()
            print("ok", nombre)
    print("todo bien")
