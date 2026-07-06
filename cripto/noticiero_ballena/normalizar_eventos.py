from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .configuracion import (
    ACTIVO_PRINCIPAL,
    ARCHIVO_EVENTOS_NORMALIZADOS,
    CADENA_PRINCIPAL,
    COLUMNAS_EVENTO_REQUERIDAS,
    TIPOS_ENTIDAD,
)
from .registro_entidades import RegistroEntidades
from .utilidades import fecha_utc, guardar_tabla, hash_evento, leer_tabla, primera_columna, texto_seguro


ALIASES = {
    "tx_hash": ("tx_hash", "hash", "transaction_hash", "transaction", "id"),
    "fecha_blockchain": ("fecha_blockchain", "timestamp", "block_timestamp", "time", "date"),
    "fecha_detectada": ("fecha_detectada", "detected_at", "received_at", "ingested_at"),
    "fecha_disponible": ("fecha_disponible", "available_at", "received_at", "detected_at", "timestamp", "date"),
    "activo": ("activo", "symbol", "asset", "currency"),
    "cantidad_activo": ("cantidad_activo", "amount", "amount_btc", "quantity", "value"),
    "valor_usd": ("valor_usd", "value_usd", "amount_usd", "usd_value"),
    "direccion_origen": ("direccion_origen", "from_address", "source_address"),
    "direccion_destino": ("direccion_destino", "to_address", "destination_address"),
    "entidad_origen": ("entidad_origen", "from_owner", "source_owner", "from_entity"),
    "entidad_destino": ("entidad_destino", "to_owner", "destination_owner", "to_entity"),
    "tipo_origen": ("tipo_origen", "from_type", "source_type"),
    "tipo_destino": ("tipo_destino", "to_type", "destination_type"),
    "exchange_origen": ("exchange_origen", "from_exchange"),
    "exchange_destino": ("exchange_destino", "to_exchange"),
    "confianza_etiquetado": ("confianza_etiquetado", "label_confidence", "confidence"),
    "fuente": ("fuente", "source", "provider"),
    "fuente_id": ("fuente_id", "provider_id", "channel_id"),
}


def _serie(df: pd.DataFrame, nombre: str, predeterminado: Any = None) -> pd.Series:
    columna = primera_columna(df.columns, ALIASES[nombre])
    if columna is None:
        return pd.Series(predeterminado, index=df.index, dtype="object")
    return df[columna]


def _tipo_entidad(valor: Any, entidad: Any, exchange: Any) -> str:
    texto = f"{texto_seguro(valor)} {texto_seguro(entidad)} {texto_seguro(exchange)}".lower()
    mapa = {
        "EXCHANGE": ("exchange", "binance", "coinbase", "kraken", "bitfinex", "okx", "bybit", "gemini"),
        "ETF": ("etf",),
        "FONDO": ("fund", "capital", "ventures", "asset management"),
        "GOBIERNO": ("government", "gobierno", "treasury", "ministry"),
        "PERSONA_PUBLICA": ("public figure", "persona publica", "celebrity"),
        "EMPRESA": ("company", "empresa", "corporate"),
        "MINERO": ("miner", "mining", "pool"),
        "CUSTODIO": ("custody", "custodian"),
        "PROTOCOLO": ("protocol", "bridge", "defi"),
    }
    for tipo, tokens in mapa.items():
        if any(token in texto for token in tokens):
            return tipo
    valor_normalizado = texto_seguro(valor).upper()
    if valor_normalizado in TIPOS_ENTIDAD:
        return valor_normalizado
    return "WALLET" if texto.strip() else "DESCONOCIDO"


def _direccion_flujo(tipo_origen: str, tipo_destino: str) -> str:
    if tipo_origen == "EXCHANGE" and tipo_destino == "EXCHANGE":
        return "ENTRE_EXCHANGES"
    if tipo_destino == "EXCHANGE":
        return "HACIA_EXCHANGE"
    if tipo_origen == "EXCHANGE":
        return "DESDE_EXCHANGE"
    if tipo_origen != "DESCONOCIDO" or tipo_destino != "DESCONOCIDO":
        return "WALLET_A_WALLET"
    return "DESCONOCIDO"


def normalizar_dataframe(
    df: pd.DataFrame,
    *,
    fuente_predeterminada: str = "archivo",
    registro: RegistroEntidades | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=COLUMNAS_EVENTO_REQUERIDAS)

    salida = pd.DataFrame(index=df.index)
    salida["tx_hash"] = _serie(df, "tx_hash", "").map(texto_seguro)
    salida["fecha_blockchain"] = fecha_utc(_serie(df, "fecha_blockchain"), permitir_nulos=True)
    salida["fecha_detectada"] = fecha_utc(_serie(df, "fecha_detectada"), permitir_nulos=True)
    salida["fecha_disponible"] = fecha_utc(_serie(df, "fecha_disponible"), permitir_nulos=True)
    salida["fecha_disponible"] = salida["fecha_disponible"].combine_first(salida["fecha_detectada"]).combine_first(salida["fecha_blockchain"])
    salida["activo"] = _serie(df, "activo", ACTIVO_PRINCIPAL).map(texto_seguro).str.upper().replace("", ACTIVO_PRINCIPAL)
    salida["cantidad_activo"] = pd.to_numeric(_serie(df, "cantidad_activo", np.nan), errors="coerce")
    salida["valor_usd"] = pd.to_numeric(_serie(df, "valor_usd", np.nan), errors="coerce")
    salida["cadena"] = CADENA_PRINCIPAL

    for columna in (
        "direccion_origen",
        "direccion_destino",
        "entidad_origen",
        "entidad_destino",
        "exchange_origen",
        "exchange_destino",
        "fuente_id",
    ):
        salida[columna] = _serie(df, columna, None).map(lambda x: texto_seguro(x) or None)

    tipo_origen_crudo = _serie(df, "tipo_origen", None)
    tipo_destino_crudo = _serie(df, "tipo_destino", None)
    salida["tipo_origen"] = [
        _tipo_entidad(t, e, x)
        for t, e, x in zip(tipo_origen_crudo, salida["entidad_origen"], salida["exchange_origen"])
    ]
    salida["tipo_destino"] = [
        _tipo_entidad(t, e, x)
        for t, e, x in zip(tipo_destino_crudo, salida["entidad_destino"], salida["exchange_destino"])
    ]
    salida["direccion_flujo"] = [
        _direccion_flujo(o, d) for o, d in zip(salida["tipo_origen"], salida["tipo_destino"])
    ]

    salida["confianza_etiquetado"] = pd.to_numeric(_serie(df, "confianza_etiquetado", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
    salida["fuente"] = _serie(df, "fuente", fuente_predeterminada).map(texto_seguro).replace("", fuente_predeterminada)
    salida["identidad_verificada"] = salida["confianza_etiquetado"].ge(0.8)
    salida["es_movimiento_interno_probable"] = (
        salida["entidad_origen"].notna()
        & salida["entidad_destino"].notna()
        & salida["entidad_origen"].str.lower().eq(salida["entidad_destino"].str.lower())
    )
    salida["identificador_entidad_origen"] = None
    salida["identificador_entidad_destino"] = None

    if registro is not None:
        salida = registro.enriquecer(salida)
        salida["direccion_flujo"] = [
            _direccion_flujo(o, d) for o, d in zip(salida["tipo_origen"], salida["tipo_destino"])
        ]

    salida["evento_id"] = [
        hash_evento(fuente, tx, origen, destino, cantidad, fecha)
        for fuente, tx, origen, destino, cantidad, fecha in zip(
            salida["fuente"],
            salida["tx_hash"],
            salida["direccion_origen"],
            salida["direccion_destino"],
            salida["cantidad_activo"],
            salida["fecha_blockchain"],
        )
    ]
    salida["tx_hash"] = salida["tx_hash"].where(salida["tx_hash"].ne(""), salida["evento_id"])
    salida["metadata_json"] = [json.dumps(fila, ensure_ascii=False, default=str) for fila in df.to_dict("records")]

    salida = salida.loc[salida["activo"].eq(ACTIVO_PRINCIPAL)].copy()
    salida = salida.dropna(subset=["fecha_disponible", "cantidad_activo"])
    salida = salida.loc[salida["cantidad_activo"].gt(0)]
    salida["valor_usd"] = salida["valor_usd"].fillna(0.0)
    salida = (
        salida.sort_values(["fecha_disponible", "evento_id"])
        .drop_duplicates("evento_id", keep="last")
        .reset_index(drop=True)
    )
    return salida


def main() -> None:
    parser = argparse.ArgumentParser(description="Normaliza eventos de ballenas de cualquier proveedor.")
    parser.add_argument("--entrada", required=True, type=Path)
    parser.add_argument("--salida", type=Path, default=ARCHIVO_EVENTOS_NORMALIZADOS)
    parser.add_argument("--fuente", default="archivo")
    parser.add_argument("--registro-entidades", type=Path)
    args = parser.parse_args()

    registro = RegistroEntidades.desde_json(args.registro_entidades)
    normalizados = normalizar_dataframe(
        leer_tabla(args.entrada),
        fuente_predeterminada=args.fuente,
        registro=registro,
    )
    guardar_tabla(normalizados, args.salida)
    print(f"Eventos normalizados: {len(normalizados):,}")
    print(f"Salida: {args.salida}")


if __name__ == "__main__":
    main()
