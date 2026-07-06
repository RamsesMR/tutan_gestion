from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from .configuracion import RUTA_CRUDOS, RUTA_PROYECTO, UMBRAL_BALLENA_USD
from .proveedores.bigquery_bitcoin import ClienteBigQueryBitcoin, SATOSHIS_POR_BTC
from .utilidades import (
    guardar_json,
    guardar_tabla,
    hash_evento,
    leer_tablas_desde_ruta,
    normalizar_precios,
)

FUENTE = "bigquery_public_bitcoin_outputs"
RUTA_HISTORICO = RUTA_CRUDOS / "historico_bigquery"
RUTA_PARTES = RUTA_HISTORICO / "partes"
ARCHIVO_HISTORICO = RUTA_HISTORICO / "blockchain_bigquery_historico.parquet"
ARCHIVO_MANIFIESTO = RUTA_HISTORICO / "manifiesto_historico.json"
RUTA_PRECIOS_PREDETERMINADA = (
    RUTA_PROYECTO / "datos" / "cripto" / "preparados" / "corto_plazo_baja_v1"
)


@dataclass(slots=True)
class ResumenMes:
    mes: str
    inicio: str
    fin: str
    max_precio_btc: float
    min_btc_consulta: float
    min_satoshis_consulta: int
    gib_estimados: float
    estado: str
    eventos_guardados: int = 0
    archivo: str | None = None


def _fecha(valor: str) -> pd.Timestamp:
    fecha = pd.Timestamp(valor)
    if fecha.tzinfo is None:
        fecha = fecha.tz_localize("UTC")
    else:
        fecha = fecha.tz_convert("UTC")
    return fecha


def iterar_meses(desde: pd.Timestamp, hasta: pd.Timestamp) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    cursor = pd.Timestamp(year=desde.year, month=desde.month, day=1, tz="UTC")
    while cursor < hasta:
        siguiente = (cursor.tz_localize(None) + pd.offsets.MonthBegin(1)).tz_localize("UTC")
        inicio = max(cursor, desde)
        fin = min(siguiente, hasta)
        if inicio < fin:
            yield inicio, fin
        cursor = siguiente


def cargar_precios(ruta: Path, desde: pd.Timestamp, hasta: pd.Timestamp) -> pd.DataFrame:
    precios = normalizar_precios(leer_tablas_desde_ruta(ruta))
    precios = precios.loc[
        precios["fecha_apertura"].between(
            desde - pd.Timedelta(minutes=30),
            hasta,
            inclusive="left",
        ),
        ["fecha_apertura", "close"],
    ].copy()
    if precios.empty:
        raise ValueError(f"No hay precios BTC 1m en {ruta} para el período solicitado")
    precios = precios.rename(columns={"close": "precio_btc_usd"})
    precios["precio_btc_usd"] = pd.to_numeric(precios["precio_btc_usd"], errors="coerce")
    precios = precios.dropna().sort_values("fecha_apertura")
    return precios


def minimo_satoshis_mes(
    precios: pd.DataFrame,
    inicio: pd.Timestamp,
    fin: pd.Timestamp,
    min_usd: float,
    margen_seguridad: float = 0.98,
) -> tuple[int, float, float]:
    bloque = precios.loc[
        precios["fecha_apertura"].between(inicio, fin, inclusive="left"),
        "precio_btc_usd",
    ]
    if bloque.empty:
        raise ValueError(f"No hay precios para {inicio:%Y-%m}")
    max_precio = float(bloque.max())
    min_btc = (float(min_usd) / max_precio) * float(margen_seguridad)
    min_satoshis = max(1, math.floor(min_btc * SATOSHIS_POR_BTC))
    return min_satoshis, min_btc, max_precio


def _normalizar_datetime_ns_utc(serie: pd.Series) -> pd.Series:
    """Normaliza timestamps a nanosegundos UTC para uniones temporales estables."""
    fechas = pd.to_datetime(serie, utc=True, errors="coerce")
    return fechas.astype("datetime64[ns, UTC]")


def procesar_salidas(
    crudo: pd.DataFrame,
    precios: pd.DataFrame,
    *,
    min_usd: float,
    latencia_minutos: int,
    tolerancia_precio_minutos: int,
) -> pd.DataFrame:
    if crudo.empty:
        return pd.DataFrame()

    datos = crudo.copy()
    datos["block_timestamp"] = _normalizar_datetime_ns_utc(datos["block_timestamp"])
    datos["valor_satoshis"] = pd.to_numeric(datos["valor_satoshis"], errors="coerce")
    datos["cantidad_activo"] = datos["valor_satoshis"] / SATOSHIS_POR_BTC
    datos = datos.dropna(subset=["block_timestamp", "cantidad_activo"])
    datos = datos.loc[datos["cantidad_activo"].gt(0)].sort_values("block_timestamp")

    precios_merge = precios.copy()
    precios_merge["fecha_apertura"] = _normalizar_datetime_ns_utc(
        precios_merge["fecha_apertura"]
    )
    precios_merge = precios_merge.dropna(subset=["fecha_apertura"]).sort_values(
        "fecha_apertura"
    )

    valorados = pd.merge_asof(
        datos,
        precios_merge,
        left_on="block_timestamp",
        right_on="fecha_apertura",
        direction="backward",
        tolerance=pd.Timedelta(minutes=int(tolerancia_precio_minutos)),
    )
    valorados["valor_usd"] = valorados["cantidad_activo"] * valorados["precio_btc_usd"]
    valorados = valorados.dropna(subset=["precio_btc_usd", "valor_usd"])
    valorados = valorados.loc[valorados["valor_usd"].ge(float(min_usd))].copy()
    if valorados.empty:
        return pd.DataFrame()

    fecha_disponible = valorados["block_timestamp"] + pd.Timedelta(minutes=int(latencia_minutos))
    direcciones = valorados["direcciones_destino"].fillna("").astype(str)
    direccion_unica = direcciones.where(~direcciones.str.contains(r"\|", regex=True), None)
    direccion_unica = direccion_unica.replace("", None)

    salida = pd.DataFrame(index=valorados.index)
    salida["evento_id"] = [
        hash_evento("bitcoin_output", tx, indice, direcciones_tx, cantidad)
        for tx, indice, direcciones_tx, cantidad in zip(
            valorados["tx_hash"],
            valorados["output_index"],
            direcciones,
            valorados["cantidad_activo"],
        )
    ]
    salida["tx_hash"] = valorados["tx_hash"].astype(str)
    salida["fecha_blockchain"] = valorados["block_timestamp"]
    salida["fecha_detectada"] = fecha_disponible
    salida["fecha_disponible"] = fecha_disponible
    salida["activo"] = "BTC"
    salida["cantidad_activo"] = valorados["cantidad_activo"].astype(float)
    salida["valor_usd"] = valorados["valor_usd"].astype(float)
    salida["cadena"] = "bitcoin"
    salida["direccion_origen"] = None
    salida["direccion_destino"] = direccion_unica
    salida["entidad_origen"] = None
    salida["entidad_destino"] = None
    salida["tipo_origen"] = "DESCONOCIDO"
    salida["tipo_destino"] = "DESCONOCIDO"
    salida["direccion_flujo"] = "DESCONOCIDO"
    salida["exchange_origen"] = None
    salida["exchange_destino"] = None
    salida["identificador_entidad_origen"] = None
    salida["identificador_entidad_destino"] = None
    salida["confianza_etiquetado"] = 0.0
    salida["identidad_verificada"] = False
    salida["es_movimiento_interno_probable"] = False
    salida["fuente"] = FUENTE
    salida["fuente_id"] = [
        f"{tx}:{int(indice)}"
        for tx, indice in zip(valorados["tx_hash"], valorados["output_index"])
    ]
    salida["metadata_json"] = [
        json.dumps(
            {
                "proveedor": FUENTE,
                "historico_confirmado": True,
                "latencia_conservadora_minutos": int(latencia_minutos),
                "block_hash": fila.block_hash,
                "block_number": int(fila.block_number),
                "output_index": int(fila.output_index),
                "tipo_salida": fila.tipo_salida,
                "direcciones_destino": fila.direcciones_destino,
                "precio_btc_usd": float(fila.precio_btc_usd),
            },
            ensure_ascii=False,
            default=str,
        )
        for fila in valorados.itertuples(index=False)
    ]

    return (
        salida.sort_values(["fecha_disponible", "evento_id"])
        .drop_duplicates("evento_id", keep="last")
        .reset_index(drop=True)
    )




def calcular_max_bytes_consulta(
    bytes_estimados: int,
    limite_mes: int,
    *,
    margen_porcentual: float = 0.10,
    margen_minimo_bytes: int = 512 * 1024 ** 2,
) -> int:
    """Devuelve un límite facturable seguro para la consulta real.

    BigQuery puede exigir algunos bytes más que el *dry run* por diferencias
    entre la estimación y la ejecución. El margen no aumenta el consumo real:
    solo evita rechazos por un límite demasiado ajustado. Nunca supera el
    límite duro mensual configurado por el usuario.
    """
    estimados = max(0, int(bytes_estimados))
    duro = max(0, int(limite_mes))
    extra = max(
        int(estimados * float(margen_porcentual)),
        int(margen_minimo_bytes),
    )
    return min(estimados + extra, duro)

def consolidar(partes: list[Path], salida: Path) -> pd.DataFrame:
    disponibles = [ruta for ruta in partes if ruta.exists()]
    if not disponibles:
        raise FileNotFoundError("No existen partes históricas para consolidar")
    tablas = [pd.read_parquet(ruta) for ruta in disponibles]
    total = pd.concat(tablas, ignore_index=True, sort=False)
    total["fecha_disponible"] = pd.to_datetime(total["fecha_disponible"], utc=True)
    total = (
        total.sort_values(["fecha_disponible", "evento_id"])
        .drop_duplicates("evento_id", keep="last")
        .reset_index(drop=True)
    )
    guardar_tabla(total, salida)
    return total


def _datetime_bigquery(fecha: pd.Timestamp) -> datetime:
    return fecha.to_pydatetime().astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construye el histórico gratuito de salidas grandes de Bitcoin usando "
            "el dataset público de BigQuery y los precios BTC 1m del proyecto."
        )
    )
    parser.add_argument("--proyecto-gcp", default=os.getenv("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--desde", default="2021-01-01")
    parser.add_argument("--hasta", default="2025-01-01")
    parser.add_argument("--precios", type=Path, default=RUTA_PRECIOS_PREDETERMINADA)
    parser.add_argument("--min-usd", type=float, default=1_000_000.0)
    parser.add_argument("--latencia-minutos", type=int, default=15)
    parser.add_argument("--tolerancia-precio-minutos", type=int, default=10)
    parser.add_argument("--max-gb-mes", type=float, default=60.0)
    parser.add_argument("--max-gb-total", type=float, default=900.0)
    parser.add_argument("--ejecutar", action="store_true")
    parser.add_argument("--sobrescribir", action="store_true")
    args = parser.parse_args()

    desde = _fecha(args.desde)
    hasta = _fecha(args.hasta)
    if desde >= hasta:
        raise ValueError("--desde debe ser anterior a --hasta")
    if args.min_usd <= 0:
        raise ValueError("--min-usd debe ser positivo")

    precios = cargar_precios(args.precios, desde, hasta)
    cliente = ClienteBigQueryBitcoin(args.proyecto_gcp)
    RUTA_PARTES.mkdir(parents=True, exist_ok=True)

    resumenes: list[ResumenMes] = []
    planes: list[tuple[pd.Timestamp, pd.Timestamp, int, Path, int]] = []
    total_bytes = 0
    limite_mes = int(args.max_gb_mes * (1024 ** 3))
    limite_total = int(args.max_gb_total * (1024 ** 3))

    print("HISTÓRICO NOTICIERO BALLENA — BIGQUERY PÚBLICO")
    print(f"Período: {desde.isoformat()} → {hasta.isoformat()}")
    print(f"Umbral final: USD {args.min_usd:,.0f}")
    print(f"Precios locales: {args.precios}")
    print("Primero se realiza una estimación sin ejecutar descargas.\n")

    for inicio, fin in iterar_meses(desde, hasta):
        etiqueta = inicio.strftime("%Y-%m")
        archivo_mes = RUTA_PARTES / f"{etiqueta}.parquet"
        min_sats, min_btc, max_precio = minimo_satoshis_mes(
            precios,
            inicio,
            fin,
            args.min_usd,
        )

        if archivo_mes.exists() and not args.sobrescribir:
            cantidad = len(pd.read_parquet(archivo_mes, columns=["evento_id"]))
            resumen = ResumenMes(
                mes=etiqueta,
                inicio=inicio.isoformat(),
                fin=fin.isoformat(),
                max_precio_btc=max_precio,
                min_btc_consulta=min_btc,
                min_satoshis_consulta=min_sats,
                gib_estimados=0.0,
                estado="YA_EXISTE",
                eventos_guardados=cantidad,
                archivo=str(archivo_mes.relative_to(RUTA_PROYECTO)),
            )
            resumenes.append(resumen)
            print(f"{etiqueta}: ya existe ({cantidad:,} eventos)")
            continue

        estimacion = cliente.estimar(
            _datetime_bigquery(inicio),
            _datetime_bigquery(fin),
            min_sats,
        )
        total_bytes += estimacion.bytes_procesados
        estado = "ESTIMADO"
        if estimacion.bytes_procesados > limite_mes:
            estado = "BLOQUEADO_LIMITE_MENSUAL"
        resumenes.append(ResumenMes(
            mes=etiqueta,
            inicio=inicio.isoformat(),
            fin=fin.isoformat(),
            max_precio_btc=max_precio,
            min_btc_consulta=min_btc,
            min_satoshis_consulta=min_sats,
            gib_estimados=estimacion.gib_procesados,
            estado=estado,
        ))
        planes.append((inicio, fin, min_sats, archivo_mes, estimacion.bytes_procesados))
        print(
            f"{etiqueta}: {estimacion.gib_procesados:,.2f} GiB estimados; "
            f"prefiltro >= {min_btc:,.4f} BTC"
        )

    print(f"\nTotal pendiente estimado: {total_bytes / (1024 ** 3):,.2f} GiB")
    if any(r.estado == "BLOQUEADO_LIMITE_MENSUAL" for r in resumenes):
        guardar_json({"meses": [asdict(r) for r in resumenes]}, ARCHIVO_MANIFIESTO)
        raise RuntimeError(
            "Algún mes supera --max-gb-mes. No se ejecutó ninguna descarga."
        )
    if total_bytes > limite_total:
        guardar_json({"meses": [asdict(r) for r in resumenes]}, ARCHIVO_MANIFIESTO)
        raise RuntimeError(
            "La estimación supera --max-gb-total. Reduce el período y ejecútalo por etapas."
        )

    if not args.ejecutar:
        guardar_json(
            {
                "modo": "solo_estimacion",
                "min_usd": args.min_usd,
                "bytes_pendientes_estimados": total_bytes,
                "gib_pendientes_estimados": total_bytes / (1024 ** 3),
                "meses": [asdict(r) for r in resumenes],
            },
            ARCHIVO_MANIFIESTO,
        )
        print("\nEstimación terminada. No se descargó nada.")
        print("Repite el comando añadiendo --ejecutar para construir el histórico.")
        return

    resumen_por_mes = {r.mes: r for r in resumenes}
    for inicio, fin, min_sats, archivo_mes, bytes_mes in planes:
        etiqueta = inicio.strftime("%Y-%m")
        print(f"\nDescargando {etiqueta}...")
        max_bytes_consulta = calcular_max_bytes_consulta(bytes_mes, limite_mes)
        crudo = cliente.descargar(
            _datetime_bigquery(inicio),
            _datetime_bigquery(fin),
            min_sats,
            max_bytes=max_bytes_consulta,
        )
        procesado = procesar_salidas(
            crudo,
            precios,
            min_usd=args.min_usd,
            latencia_minutos=args.latencia_minutos,
            tolerancia_precio_minutos=args.tolerancia_precio_minutos,
        )
        guardar_tabla(procesado, archivo_mes)
        resumen = resumen_por_mes[etiqueta]
        resumen.estado = "DESCARGADO"
        resumen.eventos_guardados = len(procesado)
        resumen.archivo = str(archivo_mes.relative_to(RUTA_PROYECTO))
        print(f"{etiqueta}: {len(procesado):,} eventos >= USD {args.min_usd:,.0f}")

        guardar_json(
            {
                "modo": "descarga_en_progreso",
                "min_usd": args.min_usd,
                "latencia_minutos": args.latencia_minutos,
                "meses": [asdict(r) for r in resumenes],
            },
            ARCHIVO_MANIFIESTO,
        )

    partes = [RUTA_PARTES / f"{inicio:%Y-%m}.parquet" for inicio, _ in iterar_meses(desde, hasta)]
    historico = consolidar(partes, ARCHIVO_HISTORICO)
    manifiesto = {
        "modo": "completo",
        "fuente": FUENTE,
        "desde": desde.isoformat(),
        "hasta": hasta.isoformat(),
        "min_usd": args.min_usd,
        "latencia_minutos": args.latencia_minutos,
        "precios": str(args.precios),
        "eventos": len(historico),
        "transacciones_unicas": int(historico["tx_hash"].nunique()),
        "fecha_minima": historico["fecha_disponible"].min() if len(historico) else None,
        "fecha_maxima": historico["fecha_disponible"].max() if len(historico) else None,
        "archivo_historico": str(ARCHIVO_HISTORICO.relative_to(RUTA_PROYECTO)),
        "meses": [asdict(r) for r in resumenes],
    }
    guardar_json(manifiesto, ARCHIVO_MANIFIESTO)

    print("\nHISTÓRICO CONSTRUIDO")
    print(f"Eventos: {len(historico):,}")
    print(f"Transacciones únicas: {historico['tx_hash'].nunique():,}")
    print(f"Salida: {ARCHIVO_HISTORICO}")
    print(f"Manifiesto: {ARCHIVO_MANIFIESTO}")


if __name__ == "__main__":
    main()
