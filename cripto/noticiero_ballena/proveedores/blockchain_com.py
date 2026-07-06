from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from ..esquemas import EventoBallena
from ..utilidades import hash_evento, texto_seguro

URL_WEBSOCKET = "wss://ws.blockchain.info/inv"
URL_PRECIO_BTC_USD = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
SATOSHIS_POR_BTC = 100_000_000


class PrecioBTC:
    """Obtiene un precio público de BTC sin API key y lo cachea brevemente."""

    def __init__(self, ttl_segundos: int = 60) -> None:
        self.ttl_segundos = max(10, int(ttl_segundos))
        self._precio: float | None = None
        self._actualizado_monotonic = 0.0

    def obtener(self) -> float | None:
        ahora = time.monotonic()
        if self._precio and ahora - self._actualizado_monotonic < self.ttl_segundos:
            return self._precio
        try:
            respuesta = requests.get(URL_PRECIO_BTC_USD, timeout=10)
            respuesta.raise_for_status()
            precio = float(respuesta.json()["data"]["amount"])
            if precio > 0:
                self._precio = precio
                self._actualizado_monotonic = ahora
        except (requests.RequestException, KeyError, TypeError, ValueError):
            pass
        return self._precio


def _direcciones_entrada(transaccion: dict[str, Any]) -> list[str]:
    direcciones: list[str] = []
    for entrada in transaccion.get("inputs") or []:
        if not isinstance(entrada, dict):
            continue
        prev_out = entrada.get("prev_out")
        if not isinstance(prev_out, dict):
            continue
        direccion = texto_seguro(prev_out.get("addr"))
        if direccion:
            direcciones.append(direccion)
    return sorted(set(direcciones))


def convertir_transaccion(
    payload: dict[str, Any],
    *,
    recibido_utc: datetime | None = None,
    precio_btc_usd: float | None = None,
    min_value_usd: float = 5_000_000.0,
    min_btc_respaldo: float = 50.0,
) -> list[EventoBallena]:
    """Convierte una transacción pública en eventos por salida grande.

    Bitcoin usa UTXO y una salida grande puede ser pago, cambio, batching o
    reorganización. Por eso no se atribuye intención ni identidad aquí.
    """

    if texto_seguro(payload.get("op")).lower() != "utx":
        return []
    transaccion = payload.get("x")
    if not isinstance(transaccion, dict):
        return []

    recibido = recibido_utc or datetime.now(timezone.utc)
    marca_tiempo = transaccion.get("time")
    try:
        fecha_tx = datetime.fromtimestamp(float(marca_tiempo), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        fecha_tx = recibido

    tx_hash = texto_seguro(transaccion.get("hash") or transaccion.get("tx_index"))
    entradas = _direcciones_entrada(transaccion)
    direccion_origen = entradas[0] if len(entradas) == 1 else None
    total_entrada_sats = 0
    for entrada in transaccion.get("inputs") or []:
        if isinstance(entrada, dict) and isinstance(entrada.get("prev_out"), dict):
            try:
                total_entrada_sats += int(entrada["prev_out"].get("value") or 0)
            except (TypeError, ValueError):
                continue

    salidas = transaccion.get("out") or []
    eventos: list[EventoBallena] = []
    for indice, salida in enumerate(salidas):
        if not isinstance(salida, dict):
            continue
        try:
            sats = int(salida.get("value") or 0)
        except (TypeError, ValueError):
            continue
        cantidad_btc = sats / SATOSHIS_POR_BTC
        valor_usd = cantidad_btc * precio_btc_usd if precio_btc_usd else 0.0
        supera_umbral = (
            valor_usd >= float(min_value_usd)
            if precio_btc_usd
            else cantidad_btc >= float(min_btc_respaldo)
        )
        if not supera_umbral:
            continue

        direccion_destino = texto_seguro(salida.get("addr")) or None
        evento_id = hash_evento(
            "blockchain_com_public",
            tx_hash,
            indice,
            direccion_destino,
            cantidad_btc,
        )
        metadata = {
            "proveedor": "blockchain.com_websocket_publico",
            "indice_salida": indice,
            "numero_entradas": len(transaccion.get("inputs") or []),
            "numero_salidas": len(salidas),
            "direcciones_entrada_unicas": entradas,
            "total_entrada_btc": total_entrada_sats / SATOSHIS_POR_BTC,
            "precio_btc_usd": precio_btc_usd,
            "tamanio_bytes": transaccion.get("size"),
            "tx_index": transaccion.get("tx_index"),
        }
        eventos.append(EventoBallena(
            evento_id=evento_id,
            tx_hash=tx_hash or evento_id,
            fecha_disponible=recibido.isoformat(),
            fecha_detectada=recibido.isoformat(),
            fecha_blockchain=fecha_tx.isoformat(),
            activo="BTC",
            cantidad_activo=cantidad_btc,
            valor_usd=valor_usd,
            tipo_origen="DESCONOCIDO",
            tipo_destino="DESCONOCIDO",
            direccion_flujo="DESCONOCIDO",
            fuente="blockchain_com_public",
            direccion_origen=direccion_origen,
            direccion_destino=direccion_destino,
            confianza_etiquetado=0.0,
            identidad_verificada=False,
            es_movimiento_interno_probable=False,
            fuente_id=f"{tx_hash}:{indice}",
            metadata_json=json.dumps(metadata, ensure_ascii=False),
        ))
    return eventos


async def escuchar_transacciones(
    salida_jsonl: Path,
    *,
    min_value_usd: float = 5_000_000.0,
    min_btc_respaldo: float = 50.0,
) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("Instala websockets para usar Blockchain.com en tiempo real") from exc

    salida_jsonl.parent.mkdir(parents=True, exist_ok=True)
    precio = PrecioBTC()
    suscripcion = json.dumps({"op": "unconfirmed_sub"})
    capturados = 0

    print("Fuente gratuita: Blockchain.com WebSocket público")
    print("No requiere cuenta ni API key.")
    print(f"Umbral: USD {min_value_usd:,.0f}; respaldo sin precio: {min_btc_respaldo:g} BTC")
    print(f"Salida: {salida_jsonl}")

    while True:
        try:
            async with websockets.connect(
                URL_WEBSOCKET,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
                max_size=8 * 1024 * 1024,
            ) as ws:
                await ws.send(suscripcion)
                print("Conectado. Escuchando transacciones Bitcoin sin confirmar...")
                async for mensaje in ws:
                    payload = json.loads(mensaje)
                    eventos = convertir_transaccion(
                        payload,
                        precio_btc_usd=precio.obtener(),
                        min_value_usd=min_value_usd,
                        min_btc_respaldo=min_btc_respaldo,
                    )
                    if not eventos:
                        continue
                    with salida_jsonl.open("a", encoding="utf-8") as archivo:
                        for evento in eventos:
                            archivo.write(json.dumps(evento.a_dict(), ensure_ascii=False) + "\n")
                            capturados += 1
                            print(
                                f"BALLENA #{capturados}: {evento.cantidad_activo:,.4f} BTC "
                                f"(~USD {evento.valor_usd:,.0f}) tx={evento.tx_hash[:12]}..."
                            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"Blockchain.com desconectado: {exc}. Reintento en 5 segundos.")
            await asyncio.sleep(5)
