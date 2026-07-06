from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .configuracion import ARCHIVO_EVENTOS_NORMALIZADOS, COLUMNAS_EVENTO_REQUERIDAS, RUTA_AUDITORIAS
from .utilidades import fecha_utc, guardar_json, leer_tabla


def auditar(df: pd.DataFrame) -> dict:
    resultado: dict = {
        "filas": int(len(df)),
        "aprobada": True,
        "errores": [],
        "avisos": [],
    }
    faltantes = [c for c in COLUMNAS_EVENTO_REQUERIDAS if c not in df.columns]
    if faltantes:
        resultado["errores"].append(f"Faltan columnas: {faltantes}")
        resultado["aprobada"] = False
        return resultado
    if df.empty:
        resultado["errores"].append("No hay eventos.")
        resultado["aprobada"] = False
        return resultado

    resultado["eventos_duplicados"] = int(df["evento_id"].duplicated().sum())
    resultado["hashes_repetidos"] = int(df["tx_hash"].duplicated().sum())
    fechas = fecha_utc(df["fecha_disponible"], permitir_nulos=True)
    resultado["fechas_nulas"] = int(fechas.isna().sum())
    resultado["fecha_min"] = str(fechas.min())
    resultado["fecha_max"] = str(fechas.max())
    resultado["fuentes"] = df["fuente"].value_counts(dropna=False).to_dict()
    resultado["direcciones_flujo"] = df["direccion_flujo"].value_counts(dropna=False).to_dict()
    resultado["tipos_origen"] = df["tipo_origen"].value_counts(dropna=False).to_dict()
    resultado["tipos_destino"] = df["tipo_destino"].value_counts(dropna=False).to_dict()
    resultado["identidades_verificadas_pct"] = float(pd.Series(df.get("identidad_verificada", False)).fillna(False).mean() * 100.0)
    resultado["confianza_media"] = float(pd.to_numeric(df.get("confianza_etiquetado", 0.0), errors="coerce").fillna(0.0).mean())

    if resultado["eventos_duplicados"]:
        resultado["errores"].append("Hay evento_id duplicados.")
    if resultado["fechas_nulas"]:
        resultado["errores"].append("Hay fechas de disponibilidad nulas.")
    for columna in ("cantidad_activo", "valor_usd"):
        valores = pd.to_numeric(df[columna], errors="coerce")
        nulos = int(valores.isna().sum())
        negativos = int(valores.lt(0).sum())
        resultado[f"{columna}_nulos"] = nulos
        resultado[f"{columna}_negativos"] = negativos
        if columna == "cantidad_activo" and (nulos or int(valores.le(0).sum())):
            resultado["errores"].append("cantidad_activo contiene valores inválidos.")
        if columna == "valor_usd" and nulos:
            resultado["avisos"].append("Hay valor_usd nulo; puede limitar umbrales económicos.")

    resultado["aprobada"] = not resultado["errores"]
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita eventos normalizados de Noticiero Ballena.")
    parser.add_argument("--entrada", type=Path, default=ARCHIVO_EVENTOS_NORMALIZADOS)
    parser.add_argument("--salida", type=Path, default=RUTA_AUDITORIAS / "auditoria_eventos.json")
    args = parser.parse_args()
    resultado = auditar(leer_tabla(args.entrada))
    guardar_json(resultado, args.salida)
    print("AUDITORÍA APROBADA" if resultado["aprobada"] else "AUDITORÍA RECHAZADA")
    for aviso in resultado["avisos"]:
        print(f"AVISO: {aviso}")
    for error in resultado["errores"]:
        print(f"ERROR: {error}")
    if not resultado["aprobada"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
