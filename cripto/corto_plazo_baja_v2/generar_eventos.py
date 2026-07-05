from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cripto.corto_plazo_baja_v2.configuracion import (
    COLUMNAS_BASE_63,
    COLUMNAS_MACRO,
    COSTE_BASE,
    PERIODOS_FUENTE,
    REJILLAS_EVENTOS_MINUTOS,
    RUTA_DATOS_V2,
)
from cripto.corto_plazo_baja_v2.utilidades import (
    asegurar_carpetas,
    columnas_objetivo_fuente,
    construir_interacciones,
    integrar_macro,
    normalizar_resultado,
    ruta_baja_v1,
    ruta_eventos,
    seleccionar_eventos_rejilla,
)


def generar_periodo(
    desde: str,
    hasta: str,
    rejilla: int,
    sobrescribir: bool,
) -> dict:
    destino = ruta_eventos(desde, hasta, rejilla)
    if destino.exists() and not sobrescribir:
        return {"estado": "conservado", "ruta": str(destino)}

    fuente = ruta_baja_v1(desde, hasta)
    if not fuente.exists():
        raise FileNotFoundError(
            f"No existe {fuente}. Primero debe estar generado BAJA V1."
        )

    nombres = columnas_objetivo_fuente()
    datos = pd.read_parquet(fuente)
    datos["fecha_apertura"] = pd.to_datetime(
        datos["fecha_apertura"], utc=True, errors="raise"
    )
    datos[nombres["fecha_fin"]] = pd.to_datetime(
        datos[nombres["fecha_fin"]], utc=True, errors="coerce"
    )

    requeridas = {
        "fecha_apertura",
        *COLUMNAS_BASE_63,
        *nombres.values(),
    }
    faltantes = sorted(requeridas - set(datos.columns))
    if faltantes:
        raise KeyError(f"Faltan columnas en {fuente.name}: {faltantes}")

    datos, origen_macro = integrar_macro(datos, desde, hasta)
    datos = construir_interacciones(datos)

    resultado = datos[nombres["resultado"]].map(normalizar_resultado)
    retorno_bruto = pd.to_numeric(datos[nombres["retorno"]], errors="coerce")
    duracion = pd.to_numeric(datos[nombres["duracion"]], errors="coerce")
    mascara = (
        resultado.notna()
        & retorno_bruto.notna()
        & np.isfinite(retorno_bruto)
        & duracion.notna()
        & (duracion > 0)
        & datos[nombres["fecha_fin"]].notna()
    )
    datos = datos.loc[mascara].copy()
    datos["clase_evento"] = resultado.loc[mascara].astype("int8")
    datos["retorno_bruto"] = retorno_bruto.loc[mascara].astype("float64")
    datos["retorno_neto"] = datos["retorno_bruto"] - COSTE_BASE
    datos["duracion"] = duracion.loc[mascara].astype("int32")
    datos["es_timeout"] = (datos["clase_evento"] == 2).astype("int8")

    eventos = seleccionar_eventos_rejilla(datos, rejilla)
    destino.parent.mkdir(parents=True, exist_ok=True)
    eventos.to_parquet(destino, index=False)

    cobertura = {
        columna: float(eventos[columna].notna().mean())
        for columna in COLUMNAS_MACRO
        if columna in eventos.columns
    }
    return {
        "estado": "generado",
        "ruta": str(destino),
        "filas": int(len(eventos)),
        "origen_macro": origen_macro,
        "cobertura_macro": cobertura,
        "clases": {
            str(k): int(v)
            for k, v in eventos["clase_evento"].value_counts().sort_index().items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rejillas",
        default=",".join(str(v) for v in REJILLAS_EVENTOS_MINUTOS),
        help="Ejemplo: 5,10,15,30",
    )
    parser.add_argument("--sobrescribir", action="store_true")
    args = parser.parse_args()

    rejillas = tuple(int(v.strip()) for v in args.rejillas.split(",") if v.strip())
    asegurar_carpetas(RUTA_DATOS_V2)
    print("=== GENERACIÓN BAJA V2 POR EVENTOS ===")
    for desde, hasta in PERIODOS_FUENTE:
        for rejilla in rejillas:
            info = generar_periodo(desde, hasta, rejilla, args.sobrescribir)
            print(
                f"{desde} -> {hasta} | {rejilla}m | "
                f"{info['estado']} | {info.get('filas', '-')}"
            )
            if info.get("origen_macro") == "macro_no_encontrada":
                print(
                    "  AVISO: no se encontraron Parquet macro integrados. "
                    "El control funcionará, pero las variantes macro no deben entrenarse."
                )
    print("GENERACIÓN BAJA V2 COMPLETADA")


if __name__ == "__main__":
    main()
