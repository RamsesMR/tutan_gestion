from __future__ import annotations

import math
import numpy as np
import pandas as pd


def convertir_fechas_ns(fechas: pd.Series) -> np.ndarray:
    return (
        pd.to_datetime(fechas, utc=True, errors="raise")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64")
    )


def factor_beneficio(retornos: np.ndarray) -> float:
    ganancias = float(retornos[retornos > 0].sum())
    perdidas = float(-retornos[retornos < 0].sum())

    if perdidas == 0:
        return math.inf if ganancias > 0 else 0.0

    return ganancias / perdidas


def maximo_drawdown(retornos: np.ndarray) -> float:
    if retornos.size == 0:
        return 0.0

    capital = np.cumprod(1.0 + np.clip(retornos, -0.999999, None))
    maximos = np.maximum.accumulate(capital)

    return float((capital / maximos - 1.0).min())


def calcular_retorno_fijo(
    cierres: np.ndarray,
    horizonte: int,
) -> np.ndarray:
    retornos = np.full(len(cierres), np.nan, dtype="float64")
    limite = len(cierres) - horizonte

    if limite > 0:
        retornos[:limite] = (
            cierres[horizonte:]
            / cierres[:limite]
            - 1.0
        )

    return retornos


def seleccionar_no_solapadas(
    fechas_ns: np.ndarray,
    mascara: np.ndarray,
    horizonte: int,
    enfriamiento: int = 0,
) -> np.ndarray:
    candidatos = np.flatnonzero(mascara)
    seleccionados: list[int] = []

    siguiente = np.iinfo(np.int64).min
    minuto_ns = 60 * 1_000_000_000

    for indice in candidatos:
        fecha = int(fechas_ns[indice])

        if fecha < siguiente:
            continue

        seleccionados.append(int(indice))
        siguiente = fecha + (horizonte + enfriamiento) * minuto_ns

    return np.asarray(seleccionados, dtype="int64")


def evaluar_indices(
    indices: np.ndarray,
    retorno_bruto: np.ndarray,
    objetivo_sube: np.ndarray,
    coste: float,
) -> dict[str, float | int]:
    if len(indices) == 0:
        return {
            "operaciones": 0,
            "precision": 0.0,
            "porcentaje_acierto": 0.0,
            "porcentaje_positivas": 0.0,
            "retorno_neto_medio": 0.0,
            "retorno_neto_mediano": 0.0,
            "factor_beneficio": 0.0,
            "maximo_drawdown": 0.0,
        }

    brutos = retorno_bruto[indices]
    netos = brutos - coste
    precision = float(objetivo_sube[indices].mean())

    return {
        "operaciones": int(len(indices)),
        "precision": precision,
        "porcentaje_acierto": precision * 100.0,
        "porcentaje_positivas": float((netos > 0).mean() * 100.0),
        "retorno_neto_medio": float(netos.mean()),
        "retorno_neto_mediano": float(np.median(netos)),
        "factor_beneficio": float(factor_beneficio(netos)),
        "maximo_drawdown": float(maximo_drawdown(netos)),
    }


def cruces_desde_abajo(
    probabilidades: np.ndarray,
    umbral: float,
) -> np.ndarray:
    sobre = probabilidades >= umbral
    anterior = np.empty_like(sobre)
    anterior[0] = False
    anterior[1:] = sobre[:-1]
    return sobre & ~anterior


def aplicar_rearme(
    cruces: np.ndarray,
    probabilidades: np.ndarray,
    umbral_rearme: float | None,
) -> np.ndarray:
    if umbral_rearme is None:
        return cruces.copy()

    salida = np.zeros_like(cruces)
    armado = True

    for indice in range(len(cruces)):
        if probabilidades[indice] <= umbral_rearme:
            armado = True

        if cruces[indice] and armado:
            salida[indice] = True
            armado = False

    return salida
