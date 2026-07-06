from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .configuracion import ARCHIVO_EXPORTACION_HISTORICA, HORIZONTES_IMPACTO_MINUTOS, RUTA_AUDITORIAS
from .utilidades import fecha_utc, guardar_json, leer_tabla


def auditar(df: pd.DataFrame) -> dict:
    resultado = {"aprobada": True, "filas": int(len(df)), "errores": [], "avisos": []}
    if "fecha_apertura" not in df:
        resultado["errores"].append("Falta fecha_apertura.")
        resultado["aprobada"] = False
        return resultado
    fechas = fecha_utc(df["fecha_apertura"], permitir_nulos=True)
    resultado["duplicados_fecha"] = int(fechas.duplicated().sum())
    resultado["fechas_nulas"] = int(fechas.isna().sum())
    resultado["ordenada"] = bool(fechas.is_monotonic_increasing)
    if resultado["duplicados_fecha"]:
        resultado["errores"].append("Hay fechas duplicadas.")
    if resultado["fechas_nulas"]:
        resultado["errores"].append("Hay fechas nulas.")
    if not resultado["ordenada"]:
        resultado["errores"].append("La exportación no está ordenada.")

    for horizonte in HORIZONTES_IMPACTO_MINUTOS:
        columnas = [
            f"nb_p_bajista_{horizonte}m",
            f"nb_p_alcista_{horizonte}m",
            f"nb_p_sin_impacto_{horizonte}m",
        ]
        if not all(c in df for c in columnas):
            resultado["avisos"].append(f"No están completas las probabilidades de {horizonte}m.")
            continue
        probs = df[columnas].apply(pd.to_numeric, errors="coerce")
        validas = probs.dropna()
        fuera = int(((validas < 0) | (validas > 1)).any(axis=1).sum())
        suma_incorrecta = int((~np.isclose(validas.sum(axis=1), 1.0, atol=1e-6)).sum())
        resultado[f"probabilidades_fuera_rango_{horizonte}m"] = fuera
        resultado[f"sumas_incorrectas_{horizonte}m"] = suma_incorrecta
        if fuera or suma_incorrecta:
            resultado["errores"].append(f"Probabilidades inválidas en {horizonte}m.")

    resultado["aprobada"] = not resultado["errores"]
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita el contrato exportado por Noticiero Ballena.")
    parser.add_argument("--entrada", type=Path, default=ARCHIVO_EXPORTACION_HISTORICA)
    parser.add_argument("--salida", type=Path, default=RUTA_AUDITORIAS / "auditoria_exportacion.json")
    args = parser.parse_args()
    resultado = auditar(leer_tabla(args.entrada))
    guardar_json(resultado, args.salida)
    print("AUDITORÍA DE EXPORTACIÓN APROBADA" if resultado["aprobada"] else "AUDITORÍA DE EXPORTACIÓN RECHAZADA")
    for error in resultado["errores"]:
        print(f"ERROR: {error}")
    if not resultado["aprobada"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
