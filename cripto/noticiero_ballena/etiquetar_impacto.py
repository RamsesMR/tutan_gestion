from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .configuracion import (
    ARCHIVO_ETIQUETADO,
    ARCHIVO_VARIABLES_MINUTO,
    HORIZONTES_IMPACTO_MINUTOS,
    LATENCIA_EJECUCION_MINUTOS,
    UMBRAL_IMPACTO,
)
from .utilidades import fecha_utc, guardar_tabla, leer_tabla, leer_tablas_desde_ruta, normalizar_precios


def etiquetar_impacto(variables: pd.DataFrame, precios: pd.DataFrame) -> pd.DataFrame:
    datos = variables.copy()
    datos["fecha_disponible"] = fecha_utc(datos["fecha_disponible"])
    mercado = normalizar_precios(precios)

    fechas_precio = mercado["fecha_apertura"].to_numpy(dtype="datetime64[ns]")
    aperturas = mercado["open"].to_numpy(dtype="float64")
    altos = mercado["high"].to_numpy(dtype="float64")
    bajos = mercado["low"].to_numpy(dtype="float64")
    cierres = mercado["close"].to_numpy(dtype="float64")

    fechas_datos = (
        datos["fecha_disponible"] + pd.Timedelta(minutes=LATENCIA_EJECUCION_MINUTOS)
    ).dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
    indices_entrada = np.searchsorted(fechas_precio, fechas_datos, side="left")

    for horizonte in HORIZONTES_IMPACTO_MINUTOS:
        clases: list[str | None] = []
        retornos: list[float] = []
        primera_barrera: list[float] = []
        ambiguas: list[bool] = []
        umbral = float(UMBRAL_IMPACTO[horizonte])

        for indice in indices_entrada:
            if indice >= len(mercado) or indice + horizonte >= len(mercado):
                clases.append(None)
                retornos.append(np.nan)
                primera_barrera.append(np.nan)
                ambiguas.append(False)
                continue

            precio_entrada = aperturas[indice]
            if not np.isfinite(precio_entrada) or precio_entrada <= 0:
                clases.append(None)
                retornos.append(np.nan)
                primera_barrera.append(np.nan)
                ambiguas.append(False)
                continue

            objetivo_alcista = precio_entrada * (1.0 + umbral)
            objetivo_bajista = precio_entrada * (1.0 - umbral)
            segmento_alto = altos[indice : indice + horizonte + 1]
            segmento_bajo = bajos[indice : indice + horizonte + 1]
            hits_alcista = np.flatnonzero(segmento_alto >= objetivo_alcista)
            hits_bajista = np.flatnonzero(segmento_bajo <= objetivo_bajista)
            minuto_alcista = int(hits_alcista[0]) if len(hits_alcista) else None
            minuto_bajista = int(hits_bajista[0]) if len(hits_bajista) else None

            ambigua = minuto_alcista is not None and minuto_bajista is not None and minuto_alcista == minuto_bajista
            ambiguas.append(ambigua)
            if ambigua:
                clases.append("AMBIGUO")
                primera_barrera.append(float(minuto_alcista))
            elif minuto_bajista is not None and (minuto_alcista is None or minuto_bajista < minuto_alcista):
                clases.append("BAJISTA")
                primera_barrera.append(float(minuto_bajista))
            elif minuto_alcista is not None:
                clases.append("ALCISTA")
                primera_barrera.append(float(minuto_alcista))
            else:
                clases.append("SIN_IMPACTO")
                primera_barrera.append(np.nan)

            retorno = cierres[indice + horizonte] / precio_entrada - 1.0
            retornos.append(float(retorno))

        datos[f"nb_clase_impacto_{horizonte}m"] = clases
        datos[f"nb_retorno_futuro_{horizonte}m"] = retornos
        datos[f"nb_minuto_primera_barrera_{horizonte}m"] = primera_barrera
        datos[f"nb_ambigua_{horizonte}m"] = ambiguas

    return datos


def main() -> None:
    parser = argparse.ArgumentParser(description="Etiqueta el impacto futuro de las variables de ballenas.")
    parser.add_argument("--variables", type=Path, default=ARCHIVO_VARIABLES_MINUTO)
    parser.add_argument("--precios", required=True, type=Path)
    parser.add_argument("--salida", type=Path, default=ARCHIVO_ETIQUETADO)
    args = parser.parse_args()
    etiquetados = etiquetar_impacto(leer_tabla(args.variables), leer_tablas_desde_ruta(args.precios))
    guardar_tabla(etiquetados, args.salida)
    print(f"Filas etiquetadas: {len(etiquetados):,}")
    for horizonte in HORIZONTES_IMPACTO_MINUTOS:
        print(f"{horizonte}m: {etiquetados[f'nb_clase_impacto_{horizonte}m'].value_counts(dropna=False).to_dict()}")
    print(f"Salida: {args.salida}")


if __name__ == "__main__":
    main()
