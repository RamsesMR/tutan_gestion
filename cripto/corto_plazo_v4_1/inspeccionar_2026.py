from __future__ import annotations

import pyarrow.parquet as pq
import pandas as pd

from cripto.corto_plazo_v2.configuracion import (
    RUTA_DATOS_V2,
)
from cripto.corto_plazo_v4.configuracion import (
    COLUMNAS_V4_63,
)


def main() -> None:
    """Localiza archivos BTCUSDT con datos de 2026 y comprueba su esquema."""

    candidatos = sorted(
        ruta
        for ruta in RUTA_DATOS_V2.glob(
            "BTCUSDT*.parquet"
        )
        if "2026" in ruta.name
    )

    print(
        "\nINSPECCIÓN DE DATOS 2026 PARA V4.1"
    )
    print("=" * 72)
    print(
        f"Carpeta: {RUTA_DATOS_V2}"
    )

    if not candidatos:
        print(
            "\nNo se encontraron archivos BTCUSDT con 2026 en el nombre."
        )
        print(
            "Todavía no ejecutes la confirmación final."
        )
        return

    columnas_obligatorias = {
        "fecha_apertura",
        "fecha_objetivo",
        "precio_apertura",
        "precio_maximo",
        "precio_minimo",
        "precio_cierre",
        "rendimiento_objetivo",
        *COLUMNAS_V4_63,
    }

    for ruta in candidatos:
        archivo = pq.ParquetFile(
            ruta
        )

        columnas = set(
            archivo.schema.names
        )

        faltantes = sorted(
            columnas_obligatorias
            - columnas
        )

        fechas = pd.read_parquet(
            ruta,
            columns=[
                "fecha_apertura",
                "fecha_objetivo",
            ],
        )

        fechas[
            "fecha_apertura"
        ] = pd.to_datetime(
            fechas[
                "fecha_apertura"
            ],
            utc=True,
            errors="coerce",
        )

        fechas[
            "fecha_objetivo"
        ] = pd.to_datetime(
            fechas[
                "fecha_objetivo"
            ],
            utc=True,
            errors="coerce",
        )

        print(
            "\n"
            + "-" * 72
        )
        print(
            f"Archivo: {ruta.name}"
        )
        print(
            f"Filas: {archivo.metadata.num_rows:,}".replace(
                ",",
                ".",
            )
        )
        print(
            f"Primera apertura: {fechas['fecha_apertura'].min()}"
        )
        print(
            f"Última apertura: {fechas['fecha_apertura'].max()}"
        )
        print(
            f"Último objetivo: {fechas['fecha_objetivo'].max()}"
        )
        print(
            f"Columnas requeridas completas: {not faltantes}"
        )

        if faltantes:
            print(
                "Faltantes:"
            )

            for columna in faltantes:
                print(
                    f"- {columna}"
                )

    print(
        "\nNo se modificó ningún archivo."
    )


if __name__ == "__main__":
    main()
