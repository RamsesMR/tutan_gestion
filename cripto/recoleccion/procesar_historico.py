from __future__ import annotations

import argparse
from zipfile import BadZipFile

import psycopg
import requests

from cripto.recoleccion.guardar_datos import importar_archivo
from cripto.recoleccion.recolector_historico import descargar_mes
from cripto.recoleccion.validar_datos import validar_periodo


INTERVALOS_PERMITIDOS = {
    "1m",
    "1h",
    "1d",
}


def procesar_historico(
    simbolo: str,
    intervalo: str,
    anio: int,
    mes: int,
) -> None:
    """
    Ejecuta el proceso completo para un mes histórico:

    1. Descarga el archivo oficial.
    2. Verifica su checksum.
    3. Importa las velas en PostgreSQL.
    4. Valida la calidad de los datos.
    """

    simbolo = simbolo.upper().strip()
    intervalo = intervalo.strip()

    print("\n" + "=" * 60)
    print("PROCESAMIENTO HISTÓRICO DE CRIPTOMONEDAS")
    print("=" * 60)
    print(f"Símbolo: {simbolo}")
    print(f"Intervalo: {intervalo}")
    print(f"Periodo: {anio}-{mes:02d}")

    print("\n[1/3] DESCARGA Y VERIFICACIÓN")
    print("-" * 60)

    ruta_zip = descargar_mes(
        simbolo=simbolo,
        intervalo=intervalo,
        anio=anio,
        mes=mes,
    )

    print("\n[2/3] IMPORTACIÓN EN POSTGRESQL")
    print("-" * 60)

    importar_archivo(
        ruta_zip=ruta_zip,
        simbolo=simbolo,
        intervalo=intervalo,
    )

    print("\n[3/3] VALIDACIÓN DE LOS DATOS")
    print("-" * 60)

    validar_periodo(
        simbolo=simbolo,
        intervalo=intervalo,
        anio=anio,
        mes=mes,
    )

    print("\n" + "=" * 60)
    print("PROCESO COMPLETADO CORRECTAMENTE")
    print("=" * 60)


def obtener_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Descarga, importa y valida un mes de velas "
            "históricas de Binance Spot."
        )
    )

    parser.add_argument(
        "--simbolo",
        default="BTCUSDT",
        help="Símbolo del mercado. Ejemplo: BTCUSDT.",
    )

    parser.add_argument(
        "--intervalo",
        default="1m",
        choices=sorted(INTERVALOS_PERMITIDOS),
        help="Intervalo de las velas.",
    )

    parser.add_argument(
        "--anio",
        type=int,
        required=True,
        help="Año que se desea procesar.",
    )

    parser.add_argument(
        "--mes",
        type=int,
        required=True,
        help="Mes que se desea procesar.",
    )

    return parser.parse_args()


def main() -> None:
    argumentos = obtener_argumentos()

    try:
        procesar_historico(
            simbolo=argumentos.simbolo,
            intervalo=argumentos.intervalo,
            anio=argumentos.anio,
            mes=argumentos.mes,
        )

    except (
        requests.RequestException,
        FileNotFoundError,
        BadZipFile,
        psycopg.Error,
        OSError,
        ValueError,
    ) as error:
        print("\n" + "=" * 60)
        print("EL PROCESO NO PUDO COMPLETARSE")
        print("=" * 60)
        print(f"Detalle: {error}")

        raise SystemExit(1)


if __name__ == "__main__":
    main()