from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_2.configuracion import (
    COLUMNAS_META,
    PERIODOS_DESARROLLO,
    RUTA_BASES,
    RUTA_DATOS_V2,
    RUTA_PREDICCIONES_V4_1,
    SIMBOLO,
    UMBRAL_SUBE_BASE,
)


def ruta_v2(desde: str, hasta: str) -> Path:
    return (
        RUTA_DATOS_V2
        / f"{SIMBOLO}_1m_4h_{desde}_{hasta}_variables.parquet"
    )


def ruta_predicciones(nombre: str) -> Path:
    return (
        RUTA_PREDICCIONES_V4_1
        / f"{SIMBOLO}_{nombre}.parquet"
    )


def construir_periodo(
    nombre: str,
    desde: str,
    hasta: str,
) -> pd.DataFrame:
    ruta_prob = ruta_predicciones(nombre)
    ruta_datos = ruta_v2(desde, hasta)

    if not ruta_prob.exists():
        raise FileNotFoundError(f"No existe: {ruta_prob}")

    if not ruta_datos.exists():
        raise FileNotFoundError(f"No existe: {ruta_datos}")

    predicciones = pd.read_parquet(ruta_prob)

    columnas_v2 = [
        "fecha_apertura",
        *[
            columna
            for columna in COLUMNAS_META
            if columna not in {
                "probabilidad_sube",
                "probabilidad_baja",
                "diferencia_sube_baja",
                "pendiente_sube_3m",
                "pendiente_sube_5m",
                "pendiente_sube_10m",
                "distancia_umbral",
            }
        ],
    ]

    datos_v2 = pd.read_parquet(
        ruta_datos,
        columns=columnas_v2,
    )

    predicciones["fecha_apertura"] = pd.to_datetime(
        predicciones["fecha_apertura"],
        utc=True,
        errors="raise",
    )

    datos_v2["fecha_apertura"] = pd.to_datetime(
        datos_v2["fecha_apertura"],
        utc=True,
        errors="raise",
    )

    base = predicciones.merge(
        datos_v2,
        on="fecha_apertura",
        how="inner",
        validate="one_to_one",
    )

    base["diferencia_sube_baja"] = (
        base["probabilidad_sube"]
        - base["probabilidad_baja"]
    )

    for minutos in (3, 5, 10):
        base[f"pendiente_sube_{minutos}m"] = (
            base["probabilidad_sube"]
            - base["probabilidad_sube"].shift(minutos)
        )

    base["distancia_umbral"] = (
        base["probabilidad_sube"]
        - UMBRAL_SUBE_BASE
    )

    columnas_obligatorias = {
        "precio_cierre",
        "objetivo_sube_4h",
        *COLUMNAS_META,
    }

    faltantes = columnas_obligatorias.difference(base.columns)

    if faltantes:
        raise ValueError(
            "Faltan columnas: "
            + ", ".join(sorted(faltantes))
        )

    base = base.dropna(
        subset=list(columnas_obligatorias)
    ).reset_index(drop=True)

    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sobrescribir", action="store_true")
    argumentos = parser.parse_args()

    RUTA_BASES.mkdir(parents=True, exist_ok=True)

    print("\nCONSTRUCCIÓN DE BASES V4.2")
    print("=" * 72)

    for nombre, (desde, hasta) in PERIODOS_DESARROLLO.items():
        ruta_salida = RUTA_BASES / f"{nombre}.parquet"

        if ruta_salida.exists() and not argumentos.sobrescribir:
            raise FileExistsError(
                f"Ya existe: {ruta_salida}. Usa --sobrescribir."
            )

        base = construir_periodo(nombre, desde, hasta)
        base.to_parquet(ruta_salida, index=False)

        print(
            f"{nombre}: {len(base):,} filas".replace(",", ".")
        )
        print(f"- {ruta_salida}")


if __name__ == "__main__":
    main()
