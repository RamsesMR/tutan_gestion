from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

from cripto.corto_plazo_v4_4.configuracion import (
    COLUMNAS_PREDICCIONES_REQUERIDAS,
    COSTE,
    HORIZONTE_MAXIMO_MINUTOS,
    MODO_ENTRADA,
    PLIEGUES,
    RUTA_PREDICCIONES_V4_1,
    RUTA_SELECCION_V4_1,
    SIMBOLO,
    UMBRAL_SUBE,
)


def main() -> None:
    """Verifica que V4.4 pueda reutilizar correctamente los artefactos V4.1."""

    print(
        "\nVERIFICACIÓN DEL ENTORNO V4.4"
    )
    print("=" * 72)

    if not RUTA_SELECCION_V4_1.exists():
        raise FileNotFoundError(
            f"No existe la selección de V4.1: {RUTA_SELECCION_V4_1}"
        )

    seleccion = pd.read_csv(
        RUTA_SELECCION_V4_1
    )

    columnas_estrategia = {
        "umbral_sube",
        "modo_entrada",
        "usa_veto_baja",
        "nombre_salida",
        "horizonte_salida",
        "coste",
    }

    faltantes_seleccion = columnas_estrategia.difference(
        seleccion.columns
    )

    if faltantes_seleccion:
        raise ValueError(
            "La selección V4.1 no contiene las columnas esperadas: "
            + ", ".join(
                sorted(faltantes_seleccion)
            )
        )

    usa_veto = (
        seleccion[
            "usa_veto_baja"
        ]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
            ]
        )
    )

    filas_compatibles = seleccion.loc[
        (seleccion["umbral_sube"].astype(float) == UMBRAL_SUBE)
        & (seleccion["modo_entrada"].astype(str) == MODO_ENTRADA)
        & (~usa_veto)
        & (
            seleccion["horizonte_salida"].astype(int)
            == HORIZONTE_MAXIMO_MINUTOS
        )
        & (seleccion["coste"].astype(float) == COSTE)
    ]

    if filas_compatibles.empty:
        raise ValueError(
            "La estrategia guardada de V4.1 no coincide con "
            "cruce 0.44, sin veto, salida 480m y coste 0.001."
        )

    for pliegue in PLIEGUES:
        ruta = (
            RUTA_PREDICCIONES_V4_1
            / f"{SIMBOLO}_{pliegue}.parquet"
        )

        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe: {ruta}"
            )

        columnas = set(
            pq.read_schema(
                ruta
            ).names
        )

        faltantes = set(
            COLUMNAS_PREDICCIONES_REQUERIDAS
        ).difference(
            columnas
        )

        if faltantes:
            raise ValueError(
                f"{ruta.name} no contiene: "
                + ", ".join(
                    sorted(faltantes)
                )
            )

        print(
            f"{pliegue}: archivo y columnas correctos."
        )

    print(
        "\nENTORNO V4.4 CORRECTO"
    )
    print(
        "V4.1 no será reentrenada ni modificada."
    )


if __name__ == "__main__":
    main()
