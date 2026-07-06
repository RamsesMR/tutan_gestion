from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .configuracion import (
    ARCHIVO_EVENTOS_NORMALIZADOS,
    ARCHIVO_VARIABLES_MINUTO,
    VENTANAS_MINUTOS,
)
from .utilidades import fecha_minuto_disponible, fecha_utc, guardar_tabla, leer_tabla


def _suma_condicional(df: pd.DataFrame, mascara: pd.Series, columna: str) -> pd.Series:
    return pd.to_numeric(df[columna], errors="coerce").fillna(0.0).where(mascara, 0.0)


def _indice_intervalos_activos(
    fechas_evento: pd.Series,
    ventana_maxima: int,
    desde: str | None,
    hasta: str | None,
) -> pd.DatetimeIndex:
    fechas = sorted(pd.DatetimeIndex(fechas_evento.dropna().unique()))
    if not fechas:
        return pd.DatetimeIndex([], tz="UTC")
    intervalos: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    inicio = pd.Timestamp(fechas[0])
    fin = inicio + pd.Timedelta(minutes=ventana_maxima)
    for fecha in fechas[1:]:
        fecha = pd.Timestamp(fecha)
        nuevo_fin = fecha + pd.Timedelta(minutes=ventana_maxima)
        if fecha <= fin + pd.Timedelta(minutes=1):
            fin = max(fin, nuevo_fin)
        else:
            intervalos.append((inicio, fin))
            inicio, fin = fecha, nuevo_fin
    intervalos.append((inicio, fin))

    limite_desde = pd.to_datetime(desde, utc=True) if desde else None
    limite_hasta = pd.to_datetime(hasta, utc=True) if hasta else None
    rangos = []
    for inicio, fin in intervalos:
        if limite_desde is not None:
            inicio = max(inicio, limite_desde)
        if limite_hasta is not None:
            fin = min(fin, limite_hasta)
        if inicio <= fin:
            rangos.append(pd.date_range(inicio.floor("min"), fin.ceil("min"), freq="1min", tz="UTC"))
    if not rangos:
        return pd.DatetimeIndex([], tz="UTC")
    combinado = rangos[0]
    for rango in rangos[1:]:
        combinado = combinado.append(rango)
    return combinado.drop_duplicates().sort_values()


def generar_variables_minuto(
    eventos: pd.DataFrame,
    *,
    desde: str | None = None,
    hasta: str | None = None,
) -> pd.DataFrame:
    if eventos.empty:
        raise ValueError("No hay eventos para generar variables.")
    datos = eventos.copy()
    datos["fecha_disponible"] = fecha_utc(datos["fecha_disponible"])
    datos["fecha_minuto"] = fecha_minuto_disponible(datos["fecha_disponible"])
    datos["cantidad_activo"] = pd.to_numeric(datos["cantidad_activo"], errors="coerce").fillna(0.0)
    datos["valor_usd"] = pd.to_numeric(datos["valor_usd"], errors="coerce").fillna(0.0)
    datos["confianza_etiquetado"] = pd.to_numeric(datos.get("confianza_etiquetado", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
    datos["identidad_verificada"] = pd.Series(datos.get("identidad_verificada", False), index=datos.index).fillna(False).astype(bool)
    datos["es_movimiento_interno_probable"] = pd.Series(datos.get("es_movimiento_interno_probable", False), index=datos.index).fillna(False).astype(bool)

    inflow = datos["direccion_flujo"].eq("HACIA_EXCHANGE")
    outflow = datos["direccion_flujo"].eq("DESDE_EXCHANGE")
    entre = datos["direccion_flujo"].eq("ENTRE_EXCHANGES")
    wallet = datos["direccion_flujo"].eq("WALLET_A_WALLET")
    tipos_institucionales = {"ETF", "FONDO", "EMPRESA", "CUSTODIO"}
    institucional = datos["tipo_origen"].isin(tipos_institucionales) | datos["tipo_destino"].isin(tipos_institucionales)
    gobierno = datos["tipo_origen"].eq("GOBIERNO") | datos["tipo_destino"].eq("GOBIERNO")
    persona_publica = datos["tipo_origen"].eq("PERSONA_PUBLICA") | datos["tipo_destino"].eq("PERSONA_PUBLICA")
    minero = datos["tipo_origen"].eq("MINERO") | datos["tipo_destino"].eq("MINERO")

    instantaneas = pd.DataFrame({
        "fecha_disponible": datos["fecha_minuto"],
        "nb_raw_eventos": 1.0,
        "nb_raw_btc_total": datos["cantidad_activo"],
        "nb_raw_usd_total": datos["valor_usd"],
        "nb_raw_inflow_btc": _suma_condicional(datos, inflow, "cantidad_activo"),
        "nb_raw_outflow_btc": _suma_condicional(datos, outflow, "cantidad_activo"),
        "nb_raw_entre_exchanges_btc": _suma_condicional(datos, entre, "cantidad_activo"),
        "nb_raw_wallet_wallet_btc": _suma_condicional(datos, wallet, "cantidad_activo"),
        "nb_raw_interno_probable_btc": _suma_condicional(datos, datos["es_movimiento_interno_probable"], "cantidad_activo"),
        "nb_raw_verificado_btc": _suma_condicional(datos, datos["identidad_verificada"], "cantidad_activo"),
        "nb_raw_institucional_btc": _suma_condicional(datos, institucional, "cantidad_activo"),
        "nb_raw_gobierno_btc": _suma_condicional(datos, gobierno, "cantidad_activo"),
        "nb_raw_persona_publica_btc": _suma_condicional(datos, persona_publica, "cantidad_activo"),
        "nb_raw_minero_btc": _suma_condicional(datos, minero, "cantidad_activo"),
        "nb_raw_confianza_suma": datos["confianza_etiquetado"],
        "nb_raw_max_evento_usd": datos["valor_usd"],
    })
    agrupadas = instantaneas.groupby("fecha_disponible", as_index=True).agg({
        **{c: "sum" for c in instantaneas.columns if c not in {"fecha_disponible", "nb_raw_max_evento_usd"}},
        "nb_raw_max_evento_usd": "max",
    })

    indice = _indice_intervalos_activos(
        datos["fecha_minuto"],
        max(VENTANAS_MINUTOS),
        desde,
        hasta,
    )
    if len(indice) == 0:
        raise ValueError("No hay intervalos activos dentro del periodo solicitado.")
    base = agrupadas.reindex(indice, fill_value=0.0)
    base.index.name = "fecha_disponible"

    columnas_suma = [c for c in base.columns if c != "nb_raw_max_evento_usd"]
    salida = pd.DataFrame(index=base.index)
    for ventana in VENTANAS_MINUTOS:
        ventana_texto = f"{ventana}min"
        rolling = base[columnas_suma].rolling(ventana_texto, min_periods=1, closed="right").sum()
        for columna in columnas_suma:
            salida[f"{columna}_{ventana}m"] = rolling[columna]
        salida[f"nb_raw_max_evento_usd_{ventana}m"] = base["nb_raw_max_evento_usd"].rolling(
            ventana_texto, min_periods=1, closed="right"
        ).max()
        total = salida[f"nb_raw_btc_total_{ventana}m"]
        inflow_w = salida[f"nb_raw_inflow_btc_{ventana}m"]
        outflow_w = salida[f"nb_raw_outflow_btc_{ventana}m"]
        salida[f"nb_raw_netflow_btc_{ventana}m"] = inflow_w - outflow_w
        salida[f"nb_raw_ratio_inflow_{ventana}m"] = inflow_w / (inflow_w + outflow_w + 1e-9)
        salida[f"nb_raw_presion_neta_{ventana}m"] = (inflow_w - outflow_w) / (inflow_w + outflow_w + 1e-9)
        salida[f"nb_raw_ratio_verificado_{ventana}m"] = salida[f"nb_raw_verificado_btc_{ventana}m"] / (total + 1e-9)
        salida[f"nb_raw_ratio_interno_{ventana}m"] = salida[f"nb_raw_interno_probable_btc_{ventana}m"] / (total + 1e-9)
        salida[f"nb_raw_confianza_media_{ventana}m"] = salida[f"nb_raw_confianza_suma_{ventana}m"] / (salida[f"nb_raw_eventos_{ventana}m"] + 1e-9)
        salida[f"nb_raw_concentracion_top1_{ventana}m"] = salida[f"nb_raw_max_evento_usd_{ventana}m"] / (salida[f"nb_raw_usd_total_{ventana}m"] + 1e-9)

    referencia = salida["nb_raw_inflow_btc_60m"]
    media_pasada = referencia.shift(1).rolling("30D", min_periods=60).median()
    mad_pasada = (referencia.shift(1) - media_pasada).abs().rolling("30D", min_periods=60).median()
    salida["nb_raw_inflow_zscore_robusto_30d"] = (referencia - media_pasada) / (1.4826 * mad_pasada + 1e-9)
    salida["nb_raw_inflow_zscore_robusto_30d"] = salida["nb_raw_inflow_zscore_robusto_30d"].replace([np.inf, -np.inf], np.nan)
    fechas_ultimo_evento = pd.Series(
        pd.DatetimeIndex(base.index).where(base["nb_raw_eventos"].gt(0)),
        index=base.index,
    ).ffill()
    salida["nb_raw_minutos_desde_evento"] = (
        pd.Series(base.index, index=base.index) - fechas_ultimo_evento
    ).dt.total_seconds().div(60.0)
    salida["nb_raw_minutos_desde_evento"] = salida["nb_raw_minutos_desde_evento"].where(
        salida["nb_raw_eventos_240m"].gt(0), np.nan
    )
    salida["nb_raw_activo_240m"] = salida["nb_raw_eventos_240m"].gt(0).astype("int8")
    salida["nb_raw_calidad_datos"] = (
        salida["nb_raw_confianza_media_60m"].clip(0, 1)
        * (1.0 - salida["nb_raw_ratio_interno_60m"].clip(0, 1))
    )
    return salida.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera variables causales por minuto.")
    parser.add_argument("--entrada", type=Path, default=ARCHIVO_EVENTOS_NORMALIZADOS)
    parser.add_argument("--salida", type=Path, default=ARCHIVO_VARIABLES_MINUTO)
    parser.add_argument("--desde")
    parser.add_argument("--hasta")
    args = parser.parse_args()
    variables = generar_variables_minuto(leer_tabla(args.entrada), desde=args.desde, hasta=args.hasta)
    guardar_tabla(variables, args.salida)
    print(f"Minutos generados: {len(variables):,}")
    print(f"Variables: {len(variables.columns) - 1}")
    print(f"Salida: {args.salida}")


if __name__ == "__main__":
    main()
