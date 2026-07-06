from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .configuracion import (
    ARCHIVO_INFERENCIA,
    ARCHIVO_MODELO,
    ARCHIVO_VARIABLES_MINUTO,
    CLASES_IMPACTO,
    HORIZONTES_IMPACTO_MINUTOS,
)
from .utilidades import fecha_utc, guardar_tabla, leer_tabla, probabilidades_por_clase


def inferir(df: pd.DataFrame, artefacto: dict) -> pd.DataFrame:
    datos = df.copy()
    datos["fecha_disponible"] = fecha_utc(datos["fecha_disponible"])
    columnas = list(artefacto["columnas_modelo"])
    for columna in columnas:
        if columna not in datos:
            datos[columna] = pd.NA

    salida = datos[["fecha_disponible", *[c for c in columnas if c in datos.columns]]].copy()
    for horizonte in HORIZONTES_IMPACTO_MINUTOS:
        modelos = artefacto["horizontes"][str(horizonte)]
        clf = modelos["clasificador"]
        reg = modelos["regresor"]
        proba = probabilidades_por_clase(clf, datos[columnas], CLASES_IMPACTO)
        for clase in CLASES_IMPACTO:
            salida[f"nb_p_{clase.lower()}_{horizonte}m"] = proba[clase]
        salida[f"nb_retorno_estimado_{horizonte}m"] = reg.predict(datos[columnas])
        calidad = pd.to_numeric(datos.get("nb_raw_calidad_datos", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
        max_proba = pd.concat(
            [salida[f"nb_p_{c.lower()}_{horizonte}m"] for c in CLASES_IMPACTO],
            axis=1,
        ).max(axis=1)
        salida[f"nb_confianza_{horizonte}m"] = max_proba * calidad

    salida["nb_version"] = artefacto["version"]
    salida["nb_modelo_entrenado_hasta"] = artefacto["entrenado_hasta"]
    return salida


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta inferencia de Noticiero Ballena V1.")
    parser.add_argument("--variables", type=Path, default=ARCHIVO_VARIABLES_MINUTO)
    parser.add_argument("--modelo", type=Path, default=ARCHIVO_MODELO)
    parser.add_argument("--salida", type=Path, default=ARCHIVO_INFERENCIA)
    args = parser.parse_args()
    artefacto = joblib.load(args.modelo)
    predicciones = inferir(leer_tabla(args.variables), artefacto)
    guardar_tabla(predicciones, args.salida)
    print(f"Predicciones: {len(predicciones):,}")
    print(f"Salida: {args.salida}")


if __name__ == "__main__":
    main()
