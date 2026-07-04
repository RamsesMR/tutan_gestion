from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from cripto.corto_plazo_v5.configuracion import (
    ALIAS_COLUMNAS,
    PERIODOS,
    RUTA_DATOS_V2,
)
from cripto.corto_plazo_v5.utilidades import (
    resolver_columna,
)


def construir_ruta(
    simbolo: str,
    desde: str,
    hasta: str,
) -> Path:
    """Construye el nombre usado por la V2A."""

    return (
        RUTA_DATOS_V2
        / (
            f"{simbolo}_1m_4h_"
            f"{desde}_{hasta}_variables.parquet"
        )
    )


def main() -> None:
    """Muestra si los Parquet contienen OHLC utilizable."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--simbolo",
        default="BTCUSDT",
    )

    argumentos = parser.parse_args()

    desde, hasta = PERIODOS[
        0
    ]

    ruta = construir_ruta(
        simbolo=argumentos.simbolo,
        desde=desde,
        hasta=hasta,
    )

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe: {ruta}"
        )

    archivo = pq.ParquetFile(
        ruta
    )

    columnas_archivo = archivo.schema.names
    filas_archivo = archivo.metadata.num_rows

    print(
        "\nINSPECCIÓN DE FUENTE V5"
    )
    print("=" * 72)
    print(
        f"Archivo: {ruta}"
    )
    print(
        f"Filas: {filas_archivo:,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"Columnas: {len(columnas_archivo)}"
    )

    print(
        "\nColumnas detectadas:"
    )

    faltantes = []

    for canonica, alias in ALIAS_COLUMNAS.items():
        encontrada = resolver_columna(
            columnas_archivo,
            alias,
        )

        print(
            f"- {canonica}: {encontrada}"
        )

        if encontrada is None:
            faltantes.append(
                canonica
            )

    print(
        "\nTodas las columnas:"
    )

    for columna in columnas_archivo:
        print(
            f"- {columna}"
        )

    if faltantes:
        print(
            "\nFALTAN COLUMNAS OHLC EN LA V2A"
        )
        print(
            "El generador necesitará --ruta-precios con los Parquet "
            "originales de velas de 1 minuto."
        )
        print(
            "Faltantes: "
            + ", ".join(
                faltantes
            )
        )
    else:
        print(
            "\nLa V2A ya contiene las columnas necesarias."
        )


if __name__ == "__main__":
    main()
