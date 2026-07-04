from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from cripto.corto_plazo_v5.configuracion import (
    RUTA_FUTUROS,
)


BASE_URL = "https://fapi.binance.com"

ENDPOINTS_30_DIAS = {
    "interes_abierto": {
        "ruta": "/futures/data/openInterestHist",
        "parametro_simbolo": "symbol",
    },
    "taker": {
        "ruta": "/futures/data/takerlongshortRatio",
        "parametro_simbolo": "symbol",
    },
    "long_short_global": {
        "ruta": "/futures/data/globalLongShortAccountRatio",
        "parametro_simbolo": "symbol",
    },
    "top_posiciones": {
        "ruta": "/futures/data/topLongShortPositionRatio",
        "parametro_simbolo": "symbol",
    },
    "top_cuentas": {
        "ruta": "/futures/data/topLongShortAccountRatio",
        "parametro_simbolo": "symbol",
    },
    "basis": {
        "ruta": "/futures/data/basis",
        "parametro_simbolo": "pair",
    },
}


def solicitar(
    sesion: requests.Session,
    ruta: str,
    parametros: dict[str, Any],
    reintentos: int = 5,
) -> Any:
    """Consulta Binance con reintentos y espera incremental."""

    ultimo_error = None

    for intento in range(
        reintentos
    ):
        try:
            respuesta = sesion.get(
                BASE_URL
                + ruta,
                params=parametros,
                timeout=30,
            )

            respuesta.raise_for_status()

            return respuesta.json()

        except (
            requests.RequestException,
            ValueError,
        ) as error:
            ultimo_error = error

            time.sleep(
                2 ** intento
            )

    raise RuntimeError(
        f"No se pudo consultar {ruta}: {ultimo_error}"
    )


def guardar_parquet(
    datos: list[dict[str, Any]],
    ruta: Path,
    columna_tiempo: str,
) -> None:
    """Agrega, deduplica y guarda un histórico."""

    if not datos:
        return

    nuevos = pd.DataFrame(
        datos
    )

    if columna_tiempo not in nuevos.columns:
        raise ValueError(
            f"No existe la columna temporal {columna_tiempo}."
        )

    nuevos[
        "fecha_utc"
    ] = pd.to_datetime(
        pd.to_numeric(
            nuevos[
                columna_tiempo
            ],
            errors="coerce",
        ),
        unit="ms",
        utc=True,
        errors="coerce",
    )

    nuevos = nuevos.dropna(
        subset=[
            "fecha_utc",
        ]
    )

    for columna in nuevos.columns:
        if columna in (
            columna_tiempo,
            "fecha_utc",
            "symbol",
            "pair",
            "contractType",
        ):
            continue

        convertida = pd.to_numeric(
            nuevos[
                columna
            ],
            errors="coerce",
        )

        if convertida.notna().sum() > 0:
            nuevos[
                columna
            ] = convertida

    if ruta.exists():
        anteriores = pd.read_parquet(
            ruta
        )

        nuevos = pd.concat(
            [
                anteriores,
                nuevos,
            ],
            ignore_index=True,
        )

    nuevos = (
        nuevos
        .drop_duplicates(
            subset=[
                "fecha_utc",
            ],
            keep="last",
        )
        .sort_values(
            "fecha_utc"
        )
        .reset_index(
            drop=True
        )
    )

    ruta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nuevos.to_parquet(
        ruta,
        index=False,
    )


def recolectar_30_dias(
    sesion: requests.Session,
    simbolo: str,
    periodo: str,
    dias: int,
) -> None:
    """Descarga endpoints cuya retención oficial es cercana a 30 días."""

    ahora = datetime.now(
        timezone.utc
    )

    inicio = ahora - timedelta(
        days=dias
    )

    inicio_ms = int(
        inicio.timestamp()
        * 1000
    )

    fin_ms = int(
        ahora.timestamp()
        * 1000
    )

    for nombre, configuracion in ENDPOINTS_30_DIAS.items():
        cursor = inicio_ms
        acumulado = []

        while cursor <= fin_ms:
            parametros = {
                configuracion[
                    "parametro_simbolo"
                ]: simbolo,
                "period": periodo,
                "limit": 500,
                "startTime": cursor,
                "endTime": fin_ms,
            }

            if nombre == "basis":
                parametros[
                    "contractType"
                ] = "PERPETUAL"

            datos = solicitar(
                sesion=sesion,
                ruta=configuracion[
                    "ruta"
                ],
                parametros=parametros,
            )

            if not datos:
                break

            acumulado.extend(
                datos
            )

            ultimo = int(
                datos[
                    -1
                ][
                    "timestamp"
                ]
            )

            siguiente = ultimo + 1

            if siguiente <= cursor:
                break

            cursor = siguiente

            if len(
                datos
            ) < 500:
                break

            time.sleep(
                0.10
            )

        ruta = (
            RUTA_FUTUROS
            / nombre
            / periodo
            / f"{simbolo}.parquet"
        )

        guardar_parquet(
            datos=acumulado,
            ruta=ruta,
            columna_tiempo="timestamp",
        )

        print(
            f"- {nombre}: {len(acumulado)} registros → {ruta}"
        )


def recolectar_funding(
    sesion: requests.Session,
    simbolo: str,
    dias: int,
) -> None:
    """Descarga funding mediante paginación ascendente."""

    ahora = datetime.now(
        timezone.utc
    )

    inicio = ahora - timedelta(
        days=dias
    )

    cursor = int(
        inicio.timestamp()
        * 1000
    )

    fin_ms = int(
        ahora.timestamp()
        * 1000
    )

    acumulado = []

    while cursor <= fin_ms:
        datos = solicitar(
            sesion=sesion,
            ruta="/fapi/v1/fundingRate",
            parametros={
                "symbol": simbolo,
                "startTime": cursor,
                "endTime": fin_ms,
                "limit": 1000,
            },
        )

        if not datos:
            break

        acumulado.extend(
            datos
        )

        ultimo = int(
            datos[
                -1
            ][
                "fundingTime"
            ]
        )

        siguiente = ultimo + 1

        if siguiente <= cursor:
            break

        cursor = siguiente

        if len(
            datos
        ) < 1000:
            break

        time.sleep(
            0.25
        )

    ruta = (
        RUTA_FUTUROS
        / "funding"
        / f"{simbolo}.parquet"
    )

    guardar_parquet(
        datos=acumulado,
        ruta=ruta,
        columna_tiempo="fundingTime",
    )

    print(
        f"- funding: {len(acumulado)} registros → {ruta}"
    )


def main() -> None:
    """Recolecta datos públicos de futuros de Binance."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--simbolos",
        nargs="+",
        default=[
            "BTCUSDT",
            "ETHUSDT",
        ],
    )

    parser.add_argument(
        "--periodo",
        default="5m",
        choices=(
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "12h",
            "1d",
        ),
    )

    parser.add_argument(
        "--dias",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--dias-funding",
        type=int,
        default=3650,
    )

    argumentos = parser.parse_args()

    if argumentos.dias <= 0 or argumentos.dias > 30:
        parser.error(
            "--dias debe estar entre 1 y 30."
        )

    sesion = requests.Session()

    sesion.headers.update(
        {
            "User-Agent": "tutanGestion-v5-research/1.0",
        }
    )

    print(
        "\nRECOLECTOR DE FUTUROS BINANCE"
    )
    print("=" * 72)

    for simbolo in argumentos.simbolos:
        print(
            f"\n{simbolo}"
        )

        recolectar_30_dias(
            sesion=sesion,
            simbolo=simbolo,
            periodo=argumentos.periodo,
            dias=argumentos.dias,
        )

        recolectar_funding(
            sesion=sesion,
            simbolo=simbolo,
            dias=argumentos.dias_funding,
        )


if __name__ == "__main__":
    main()
