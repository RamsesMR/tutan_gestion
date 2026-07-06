from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cripto.corto_plazo_baja_v2.configuracion import (
    COLUMNAS_MACRO,
    HORIZONTE_MINUTOS,
    INTERACCIONES_MACRO,
    OBJETIVO_FUENTE,
    RUTA_DATOS_BAJA_V1,
    RUTA_DATOS_MACRO,
    RUTA_DATOS_V2,
    SIMBOLO,
)


def normalizar_fecha_ns_utc(serie: pd.Series) -> pd.Series:
    """
    Convierte cualquier fecha compatible a datetime64[ns, UTC].

    pandas.merge_asof exige que ambas claves temporales tengan exactamente
    el mismo dtype. Algunos Parquet del proyecto cargan como datetime64[us, UTC]
    y otros como datetime64[ns, UTC].
    """
    fechas = pd.to_datetime(serie, utc=True, errors="raise")
    return fechas.astype("datetime64[ns, UTC]")


def asegurar_carpetas(*rutas: Path) -> None:
    for ruta in rutas:
        ruta.mkdir(parents=True, exist_ok=True)


def ruta_baja_v1(desde: str, hasta: str) -> Path:
    return RUTA_DATOS_BAJA_V1 / f"{SIMBOLO}_1m_baja_v1_{desde}_{hasta}.parquet"


def ruta_eventos(desde: str, hasta: str, rejilla: int) -> Path:
    return (
        RUTA_DATOS_V2
        / f"{SIMBOLO}_eventos_baja_v2_{rejilla}m_{desde}_{hasta}.parquet"
    )


def columnas_objetivo_fuente() -> dict[str, str]:
    return {
        "objetivo": f"objetivo_{OBJETIVO_FUENTE}",
        "resultado": f"resultado_{OBJETIVO_FUENTE}",
        "retorno": f"retorno_salida_{OBJETIVO_FUENTE}",
        "duracion": f"minutos_salida_{OBJETIVO_FUENTE}",
        "fecha_fin": f"fecha_fin_horizonte_{OBJETIVO_FUENTE}",
    }


def normalizar_resultado(valor: object) -> float:
    """
    Convierte las etiquetas reales de BAJA V1 a:
    0 = STOP/operación perdedora
    1 = TAKE PROFIT/operación ganadora
    2 = TIMEOUT/sin barrera

    Las filas sin horizonte completo conservan NaN y se filtran después.
    """
    if pd.isna(valor):
        return np.nan

    texto = str(valor).strip().lower()

    if texto in {
        "1",
        "tp",
        "take_profit",
        "take profit",
        "beneficio",
        "ganadora",
    }:
        return 1.0

    if texto in {
        "0",
        "sl",
        "stop",
        "stop_loss",
        "stop loss",
        "perdida",
        "pérdida",
        "perdedora",
    }:
        return 0.0

    if texto in {
        "2",
        "timeout",
        "time",
        "horizonte",
        "cierre_horizonte",
        "fin_horizonte",
        "sin_barrera",
        "sin_resultado",
        "sin resultado",
    }:
        return 2.0

    raise ValueError(f"Resultado short no reconocido: {valor!r}")


def buscar_parquet_macro(desde: str, hasta: str) -> Path | None:
    if not RUTA_DATOS_MACRO.exists():
        return None
    candidatos = sorted(RUTA_DATOS_MACRO.rglob("*.parquet"))
    tokens = (
        desde.replace("-", ""),
        hasta.replace("-", ""),
        desde,
        hasta,
    )
    priorizados = [
        p for p in candidatos
        if any(token in p.name for token in tokens)
    ]
    for ruta in priorizados + candidatos:
        try:
            columnas = set(pd.read_parquet(ruta, columns=None).columns)
        except Exception:
            continue
        if "fecha_apertura" in columnas and any(c in columnas for c in COLUMNAS_MACRO):
            return ruta
    return None


def integrar_macro(
    datos: pd.DataFrame,
    desde: str,
    hasta: str,
) -> tuple[pd.DataFrame, str]:
    presentes = [c for c in COLUMNAS_MACRO if c in datos.columns]
    if len(presentes) == len(COLUMNAS_MACRO):
        return datos, "macro_ya_presente"

    ruta = buscar_parquet_macro(desde, hasta)
    if ruta is None:
        faltantes = [c for c in COLUMNAS_MACRO if c not in datos.columns]
        for columna in faltantes:
            datos[columna] = np.nan
        return datos, "macro_no_encontrada"

    macro = pd.read_parquet(ruta)
    macro["fecha_apertura"] = normalizar_fecha_ns_utc(
        macro["fecha_apertura"]
    )
    columnas = ["fecha_apertura", *[c for c in COLUMNAS_MACRO if c in macro.columns]]
    macro = (
        macro[columnas]
        .sort_values("fecha_apertura")
        .drop_duplicates("fecha_apertura", keep="last")
    )
    base = datos.copy()
    base["fecha_apertura"] = normalizar_fecha_ns_utc(
        base["fecha_apertura"]
    )
    base = base.sort_values("fecha_apertura")
    unidos = pd.merge_asof(
        base,
        macro,
        on="fecha_apertura",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_macro"),
    )
    unidos["fecha_apertura"] = normalizar_fecha_ns_utc(
        unidos["fecha_apertura"]
    )

    for columna in COLUMNAS_MACRO:
        auxiliar = f"{columna}_macro"
        if auxiliar in unidos.columns:
            if columna in unidos.columns:
                unidos[columna] = unidos[columna].combine_first(unidos[auxiliar])
            else:
                unidos[columna] = unidos[auxiliar]
            unidos.drop(columns=[auxiliar], inplace=True)
        elif columna not in unidos.columns:
            unidos[columna] = np.nan
    return unidos, str(ruta)


def construir_interacciones(datos: pd.DataFrame) -> pd.DataFrame:
    salida = datos.copy()
    for nombre, (macro, mercado) in INTERACCIONES_MACRO.items():
        if macro in salida.columns and mercado in salida.columns:
            salida[nombre] = (
                pd.to_numeric(salida[macro], errors="coerce")
                * pd.to_numeric(salida[mercado], errors="coerce")
            )
    return salida


def seleccionar_eventos_rejilla(datos: pd.DataFrame, minutos: int) -> pd.DataFrame:
    fechas = datos["fecha_apertura"]
    mascara = (fechas.dt.minute % minutos == 0) & (fechas.dt.second == 0)
    eventos = datos.loc[mascara].copy()
    eventos["event_id"] = (
        eventos["fecha_apertura"].dt.strftime("%Y%m%dT%H%M%SZ")
        + f"_{minutos}m"
    )
    eventos["rejilla_minutos"] = int(minutos)
    eventos["fecha_fin_evento"] = eventos["fecha_apertura"] + pd.Timedelta(
        minutes=HORIZONTE_MINUTOS
    )
    # Agrupa eventos cuyo horizonte se solapa. Se usa para purga y auditoría.
    segundos = eventos["fecha_apertura"].astype("int64") // 10**9
    eventos["overlap_group"] = (
        (segundos.diff().fillna(HORIZONTE_MINUTOS * 60 + 1)
         > HORIZONTE_MINUTOS * 60)
        .cumsum()
        .astype("int64")
    )
    return eventos.reset_index(drop=True)


def drawdown(retornos: np.ndarray) -> float:
    if len(retornos) == 0:
        return 0.0
    capital = np.cumprod(1.0 + retornos)
    maximo = np.maximum.accumulate(capital)
    return float(np.min(capital / maximo - 1.0))


def metricas_operaciones(operaciones: pd.DataFrame) -> dict[str, float | int]:
    if operaciones.empty:
        return {
            "operaciones": 0,
            "porcentaje_positivas": 0.0,
            "retorno_neto_medio": 0.0,
            "retorno_neto_mediano": 0.0,
            "factor_beneficio": 0.0,
            "drawdown": 0.0,
            "retorno_compuesto": 0.0,
            "concentracion_mensual": 1.0,
            "duracion_media": 0.0,
        }
    r = operaciones["retorno_neto"].to_numpy(dtype="float64")
    ganancias = float(r[r > 0].sum())
    perdidas = abs(float(r[r < 0].sum()))
    pf = ganancias / perdidas if perdidas > 0 else (math.inf if ganancias > 0 else 0.0)
    meses = operaciones["fecha_apertura"].dt.strftime("%Y-%m")
    absolutos = operaciones.assign(_abs=np.abs(r)).groupby(meses)["_abs"].sum()
    concentracion = float(absolutos.max() / absolutos.sum()) if absolutos.sum() > 0 else 1.0
    return {
        "operaciones": int(len(operaciones)),
        "porcentaje_positivas": float((r > 0).mean() * 100.0),
        "retorno_neto_medio": float(np.mean(r)),
        "retorno_neto_mediano": float(np.median(r)),
        "factor_beneficio": float(pf),
        "drawdown": drawdown(r),
        "retorno_compuesto": float(np.prod(1.0 + r) - 1.0),
        "concentracion_mensual": concentracion,
        "duracion_media": float(operaciones["duracion"].mean()),
    }


def seleccionar_no_solapadas(
    datos: pd.DataFrame,
    mascara: np.ndarray,
) -> pd.DataFrame:
    candidatas = datos.loc[mascara].sort_values("fecha_apertura").copy()
    elegidas: list[int] = []
    disponible_desde: pd.Timestamp | None = None
    for indice, fila in candidatas.iterrows():
        fecha = fila["fecha_apertura"]
        if disponible_desde is None or fecha >= disponible_desde:
            elegidas.append(indice)
            duracion = max(1, int(fila["duracion"]))
            disponible_desde = fecha + pd.Timedelta(minutes=duracion)
    return candidatas.loc[elegidas].copy()


def guardar_json(ruta: Path, contenido: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(contenido, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
