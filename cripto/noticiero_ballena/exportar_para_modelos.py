from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .configuracion import (
    ARCHIVO_EXPORTACION_HISTORICA,
    ARCHIVO_EXPORTACION_TIEMPO_REAL,
    ARCHIVO_INFERENCIA,
    ARCHIVO_OOF,
    HORIZONTES_IMPACTO_MINUTOS,
    VERSION_MODELO,
)
from .utilidades import fecha_utc, guardar_tabla, leer_tabla


def columnas_contrato(df: pd.DataFrame) -> list[str]:
    columnas = ["fecha_apertura"]
    preferidas = [
        "nb_raw_activo_240m",
        "nb_raw_minutos_desde_evento",
        "nb_raw_calidad_datos",
        "nb_raw_inflow_btc_15m",
        "nb_raw_outflow_btc_15m",
        "nb_raw_netflow_btc_15m",
        "nb_raw_eventos_15m",
        "nb_raw_inflow_btc_60m",
        "nb_raw_outflow_btc_60m",
        "nb_raw_netflow_btc_60m",
        "nb_raw_eventos_60m",
        "nb_raw_inflow_btc_240m",
        "nb_raw_outflow_btc_240m",
        "nb_raw_netflow_btc_240m",
        "nb_raw_eventos_240m",
        "nb_raw_inflow_zscore_robusto_30d",
        "nb_raw_ratio_interno_60m",
        "nb_raw_ratio_verificado_60m",
        "nb_raw_institucional_btc_60m",
        "nb_raw_gobierno_btc_60m",
        "nb_raw_persona_publica_btc_60m",
        "nb_raw_minero_btc_60m",
    ]
    columnas.extend([c for c in preferidas if c in df.columns])
    for horizonte in HORIZONTES_IMPACTO_MINUTOS:
        for columna in (
            f"nb_p_bajista_{horizonte}m",
            f"nb_p_alcista_{horizonte}m",
            f"nb_p_sin_impacto_{horizonte}m",
            f"nb_retorno_estimado_{horizonte}m",
            f"nb_confianza_{horizonte}m",
        ):
            if columna in df.columns:
                columnas.append(columna)
    columnas.extend([c for c in ("nb_version", "nb_modelo_entrenado_hasta") if c in df.columns])
    return list(dict.fromkeys(columnas))


def exportar(df: pd.DataFrame, modo: str) -> pd.DataFrame:
    datos = df.copy()
    datos["fecha_disponible"] = fecha_utc(datos["fecha_disponible"])
    datos = datos.sort_values("fecha_disponible").drop_duplicates("fecha_disponible", keep="last")
    datos["fecha_apertura"] = datos["fecha_disponible"]
    if "nb_version" not in datos:
        datos["nb_version"] = VERSION_MODELO
    datos["nb_modo_salida"] = modo
    columnas = columnas_contrato(datos)
    if "nb_modo_salida" not in columnas:
        columnas.append("nb_modo_salida")
    return datos[columnas].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta contrato estable para modelos consumidores.")
    parser.add_argument("--modo", choices=("historico", "tiempo_real"), required=True)
    parser.add_argument("--entrada", type=Path)
    parser.add_argument("--salida", type=Path)
    args = parser.parse_args()
    entrada = args.entrada or (ARCHIVO_OOF if args.modo == "historico" else ARCHIVO_INFERENCIA)
    salida = args.salida or (
        ARCHIVO_EXPORTACION_HISTORICA if args.modo == "historico" else ARCHIVO_EXPORTACION_TIEMPO_REAL
    )
    exportacion = exportar(leer_tabla(entrada), args.modo)
    guardar_tabla(exportacion, salida)
    print(f"Filas exportadas: {len(exportacion):,}")
    print(f"Salida: {salida}")


if __name__ == "__main__":
    main()
