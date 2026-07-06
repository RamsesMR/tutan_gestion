from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def asegurar_carpetas(*rutas: Path) -> None:
    for ruta in rutas:
        ruta.mkdir(parents=True, exist_ok=True)


def fecha_utc(serie: pd.Series, *, permitir_nulos: bool = False) -> pd.Series:
    fechas = pd.to_datetime(serie, utc=True, errors="coerce" if permitir_nulos else "raise")
    return fechas.astype("datetime64[ns, UTC]")


def fecha_minuto_disponible(serie: pd.Series) -> pd.Series:
    """Primer minuto en el que el dato ya estaba disponible."""
    fechas = fecha_utc(serie)
    exactas = fechas.dt.second.eq(0) & fechas.dt.microsecond.eq(0)
    return fechas.where(exactas, fechas.dt.ceil("min")).astype("datetime64[ns, UTC]")


def texto_seguro(valor: Any) -> str:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return ""
    return re.sub(r"\s+", " ", str(valor)).strip()


def hash_evento(*partes: Any) -> str:
    contenido = "|".join(texto_seguro(p) for p in partes)
    return hashlib.sha256(contenido.encode("utf-8")).hexdigest()[:32]


def guardar_json(contenido: Any, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(contenido, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def leer_json(ruta: Path) -> Any:
    return json.loads(ruta.read_text(encoding="utf-8"))


def leer_tabla(ruta: Path) -> pd.DataFrame:
    sufijo = ruta.suffix.lower()
    if sufijo == ".parquet":
        return pd.read_parquet(ruta)
    if sufijo == ".csv":
        return pd.read_csv(ruta)
    if sufijo in {".jsonl", ".ndjson"}:
        return pd.read_json(ruta, lines=True)
    if sufijo == ".json":
        contenido = leer_json(ruta)
        if isinstance(contenido, list):
            return pd.DataFrame(contenido)
        if isinstance(contenido, dict):
            for clave in ("data", "result", "transactions", "alerts", "events"):
                valor = contenido.get(clave)
                if isinstance(valor, list):
                    return pd.DataFrame(valor)
            return pd.DataFrame([contenido])
    raise ValueError(f"Formato no soportado: {ruta}")



def leer_tablas_desde_ruta(ruta: Path) -> pd.DataFrame:
    if ruta.is_file():
        return leer_tabla(ruta)
    if not ruta.is_dir():
        raise FileNotFoundError(ruta)
    archivos = sorted([
        *ruta.rglob("*.parquet"),
        *ruta.rglob("*.csv"),
        *ruta.rglob("*.jsonl"),
        *ruta.rglob("*.json"),
    ])
    if not archivos:
        raise FileNotFoundError(f"No hay tablas compatibles en {ruta}")
    partes = []
    for archivo in archivos:
        try:
            partes.append(leer_tabla(archivo))
        except Exception as exc:
            print(f"AVISO: se omite {archivo}: {exc}")
    if not partes:
        raise ValueError(f"Ningún archivo de {ruta} pudo leerse")
    return pd.concat(partes, ignore_index=True, sort=False)

def guardar_tabla(df: pd.DataFrame, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    sufijo = ruta.suffix.lower()
    if sufijo == ".parquet":
        df.to_parquet(ruta, index=False)
    elif sufijo == ".csv":
        df.to_csv(ruta, index=False)
    elif sufijo in {".jsonl", ".ndjson"}:
        df.to_json(ruta, orient="records", lines=True, force_ascii=False, date_format="iso")
    else:
        raise ValueError(f"Formato de salida no soportado: {ruta}")


def primera_columna(columnas: Iterable[str], candidatas: Iterable[str]) -> str | None:
    mapa = {str(c).lower(): str(c) for c in columnas}
    for candidata in candidatas:
        if candidata.lower() in mapa:
            return mapa[candidata.lower()]
    return None


def normalizar_precios(df: pd.DataFrame) -> pd.DataFrame:
    fecha = primera_columna(df.columns, ("fecha_apertura", "fecha", "timestamp", "open_time", "datetime"))
    apertura = primera_columna(df.columns, ("open", "apertura", "precio_apertura"))
    alto = primera_columna(df.columns, ("high", "alto", "maximo"))
    bajo = primera_columna(df.columns, ("low", "bajo", "minimo"))
    cierre = primera_columna(df.columns, ("close", "cierre", "precio_cierre"))
    faltantes = [n for n, v in {"fecha": fecha, "open": apertura, "high": alto, "low": bajo, "close": cierre}.items() if v is None]
    if faltantes:
        raise ValueError(f"Faltan columnas OHLC: {faltantes}. Columnas disponibles: {list(df.columns)}")
    salida = pd.DataFrame({
        "fecha_apertura": fecha_utc(df[fecha]),
        "open": pd.to_numeric(df[apertura], errors="coerce"),
        "high": pd.to_numeric(df[alto], errors="coerce"),
        "low": pd.to_numeric(df[bajo], errors="coerce"),
        "close": pd.to_numeric(df[cierre], errors="coerce"),
    })
    return (
        salida.dropna()
        .sort_values("fecha_apertura")
        .drop_duplicates("fecha_apertura", keep="last")
        .reset_index(drop=True)
    )


def seleccionar_inactivos(df: pd.DataFrame, mascara_activos: pd.Series, ratio: float, semilla: int) -> pd.DataFrame:
    activos = df.loc[mascara_activos]
    inactivos = df.loc[~mascara_activos]
    cantidad = min(len(inactivos), int(max(0, round(len(activos) * ratio))))
    if cantidad == 0:
        return activos.copy()
    muestra = inactivos.sample(n=cantidad, random_state=semilla)
    return pd.concat([activos, muestra]).sort_values("fecha_disponible")


def probabilidades_por_clase(modelo: Any, x: pd.DataFrame, clases_objetivo: Iterable[str]) -> dict[str, np.ndarray]:
    probabilidades = modelo.predict_proba(x)
    clases_modelo = [str(c) for c in modelo.named_steps["modelo"].classes_]
    salida: dict[str, np.ndarray] = {}
    for clase in clases_objetivo:
        if clase in clases_modelo:
            salida[clase] = probabilidades[:, clases_modelo.index(clase)]
        else:
            salida[clase] = np.zeros(len(x), dtype="float64")
    return salida
