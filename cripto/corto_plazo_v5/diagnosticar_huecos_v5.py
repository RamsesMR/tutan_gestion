from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cripto.corto_plazo_v5.configuracion import (
    PERIODOS,
    RUTA_DATOS_V2,
)


def construir_ruta(
    simbolo: str,
    desde: str,
    hasta: str,
):
    return (
        RUTA_DATOS_V2
        / (
            f"{simbolo}_1m_4h_"
            f"{desde}_{hasta}_variables.parquet"
        )
    )


def analizar_periodo(
    simbolo: str,
    desde: str,
    hasta: str,
) -> None:
    ruta = construir_ruta(
        simbolo=simbolo,
        desde=desde,
        hasta=hasta,
    )

    if not ruta.exists():
        print(
            f"\n{desde} → {hasta}: NO EXISTE"
        )
        print(
            f"- {ruta}"
        )
        return

    datos = pd.read_parquet(
        ruta,
        columns=[
            "fecha_apertura",
        ],
    )

    fechas = pd.to_datetime(
        datos[
            "fecha_apertura"
        ],
        utc=True,
        errors="coerce",
    ).dropna()

    duplicados = int(
        fechas.duplicated().sum()
    )

    fechas = (
        fechas
        .drop_duplicates()
        .sort_values()
        .reset_index(
            drop=True
        )
    )

    if fechas.empty:
        print(
            f"\n{desde} → {hasta}: SIN FECHAS VÁLIDAS"
        )
        return

    diferencias = fechas.diff()

    cortes = (
        diferencias.isna()
        | (
            diferencias
            != pd.Timedelta(
                minutes=1
            )
        )
    )

    grupos = cortes.cumsum()

    longitudes = (
        fechas
        .groupby(
            grupos
        )
        .size()
        .to_numpy(
            dtype="int64"
        )
    )

    huecos = diferencias.loc[
        diferencias
        > pd.Timedelta(
            minutes=1
        )
    ]

    minutos_faltantes = int(
        (
            huecos
            / pd.Timedelta(
                minutes=1
            )
            - 1
        ).sum()
    )

    esperado_periodo = int(
        (
            pd.Timestamp(
                hasta,
                tz="UTC",
            )
            - pd.Timestamp(
                desde,
                tz="UTC",
            )
        )
        / pd.Timedelta(
            minutes=1
        )
    )

    filas_tras_60 = int(
        np.maximum(
            longitudes - 59,
            0,
        ).sum()
    )

    filas_tras_240 = int(
        np.maximum(
            longitudes - 239,
            0,
        ).sum()
    )

    filas_tras_1440 = int(
        np.maximum(
            longitudes - 1439,
            0,
        ).sum()
    )

    cuantiles = np.quantile(
        longitudes,
        [
            0.25,
            0.50,
            0.75,
            0.90,
            0.99,
        ],
    )

    print(
        "\n"
        + "=" * 78
    )
    print(
        f"{simbolo} | {desde} → {hasta}"
    )
    print(
        "=" * 78
    )
    print(
        f"Archivo: {ruta}"
    )
    print(
        f"Filas originales: {len(datos):,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"Fechas únicas válidas: {len(fechas):,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"Duplicados: {duplicados:,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"Minutos esperados del año: {esperado_periodo:,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"Déficit total frente al año: "
        f"{esperado_periodo - len(fechas):,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"Primera fecha: {fechas.iloc[0]}"
    )
    print(
        f"Última fecha: {fechas.iloc[-1]}"
    )
    print(
        f"Cortes de continuidad: {max(len(longitudes) - 1, 0):,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"Minutos faltantes dentro del rango: {minutos_faltantes:,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"Cantidad de segmentos: {len(longitudes):,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"Segmento máximo: {int(longitudes.max()):,} minutos".replace(
            ",",
            ".",
        )
    )
    print(
        "Longitud de segmentos "
        f"P25/P50/P75/P90/P99: "
        f"{int(cuantiles[0])}/"
        f"{int(cuantiles[1])}/"
        f"{int(cuantiles[2])}/"
        f"{int(cuantiles[3])}/"
        f"{int(cuantiles[4])}"
    )
    print(
        f"Filas posibles tras ventana 60m: {filas_tras_60:,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"Filas posibles tras ventana 240m: {filas_tras_240:,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"Filas posibles tras ventana 1440m: {filas_tras_1440:,}".replace(
            ",",
            ".",
        )
    )

    if not huecos.empty:
        resumen_huecos = (
            (
                huecos
                / pd.Timedelta(
                    minutes=1
                )
                - 1
            )
            .astype(
                "int64"
            )
            .value_counts()
            .sort_index()
            .head(
                15
            )
        )

        print(
            "\nPrimeros tamaños de hueco "
            "(minutos faltantes → cantidad):"
        )

        for minutos, cantidad in resumen_huecos.items():
            print(
                f"- {int(minutos)} → {int(cantidad)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--simbolo",
        default="BTCUSDT",
    )

    argumentos = parser.parse_args()

    print(
        "\nDIAGNÓSTICO DE HUECOS PARA V5"
    )
    print(
        "No modifica ningún archivo."
    )

    for desde, hasta in PERIODOS:
        analizar_periodo(
            simbolo=argumentos.simbolo,
            desde=desde,
            hasta=hasta,
        )


if __name__ == "__main__":
    main()
