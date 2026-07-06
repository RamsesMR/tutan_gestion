from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..esquemas import EventoBallena
from ..utilidades import hash_evento, texto_seguro


def _campo_propietario(valor: Any) -> tuple[str | None, str | None, str | None]:
    if isinstance(valor, dict):
        propietario = texto_seguro(valor.get("owner") or valor.get("name") or valor.get("entity")) or None
        direccion = texto_seguro(valor.get("address") or valor.get("hash")) or None
        tipo = texto_seguro(valor.get("owner_type") or valor.get("type")).lower() or None
        return propietario, direccion, tipo
    texto = texto_seguro(valor)
    return (texto or None), None, None


def _tipo_entidad(nombre: str | None, tipo_proveedor: str | None) -> str:
    texto = f"{nombre or ''} {tipo_proveedor or ''}".lower()
    if not texto.strip() or "unknown" in texto:
        return "DESCONOCIDO"
    if any(x in texto for x in ("exchange", "binance", "coinbase", "kraken", "bitfinex", "okx", "bybit", "gemini")):
        return "EXCHANGE"
    if any(x in texto for x in ("miner", "mining", "pool")):
        return "MINERO"
    if any(x in texto for x in ("custody", "custodian")):
        return "CUSTODIO"
    if any(x in texto for x in ("etf",)):
        return "ETF"
    if any(x in texto for x in ("fund", "capital", "ventures")):
        return "FONDO"
    if any(x in texto for x in ("government", "govt", "treasury")):
        return "GOBIERNO"
    return "WALLET"


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


def convertir_alerta(payload: dict[str, Any], recibido_utc: datetime | None = None) -> list[EventoBallena]:
    tipo_mensaje = texto_seguro(payload.get("type")).lower()
    if tipo_mensaje and tipo_mensaje not in {"alert", "transaction"}:
        return []

    recibido = recibido_utc or datetime.now(timezone.utc)
    marca_tiempo = payload.get("timestamp") or payload.get("time")
    try:
        fecha_tx = datetime.fromtimestamp(float(marca_tiempo), tz=timezone.utc) if marca_tiempo else recibido
    except (TypeError, ValueError, OSError):
        fecha_tx = recibido

    transaccion = payload.get("transaction") if isinstance(payload.get("transaction"), dict) else {}
    tx_hash = texto_seguro(transaccion.get("hash") or payload.get("hash") or payload.get("channel_id"))
    propietario_origen, direccion_origen, tipo_origen_proveedor = _campo_propietario(payload.get("from") or transaccion.get("from"))
    propietario_destino, direccion_destino, tipo_destino_proveedor = _campo_propietario(payload.get("to") or transaccion.get("to"))
    tipo_origen = _tipo_entidad(propietario_origen, tipo_origen_proveedor)
    tipo_destino = _tipo_entidad(propietario_destino, tipo_destino_proveedor)

    montos = payload.get("amounts")
    if not isinstance(montos, list):
        montos = [{
            "symbol": payload.get("symbol") or transaccion.get("symbol"),
            "amount": payload.get("amount") or transaccion.get("amount"),
            "value_usd": payload.get("amount_usd") or payload.get("value_usd") or transaccion.get("value_usd"),
        }]

    eventos: list[EventoBallena] = []
    for indice, monto in enumerate(montos):
        if not isinstance(monto, dict):
            continue
        simbolo = texto_seguro(monto.get("symbol")).upper()
        if simbolo != "BTC":
            continue
        cantidad = float(monto.get("amount") or 0.0)
        valor_usd = float(monto.get("value_usd") or monto.get("amount_usd") or 0.0)
        evento_id = hash_evento("whale_alert", tx_hash, indice, simbolo, cantidad, propietario_origen, propietario_destino)
        interno = bool(propietario_origen and propietario_destino and propietario_origen.lower() == propietario_destino.lower())
        direccion = _direccion_flujo(tipo_origen, tipo_destino)
        eventos.append(EventoBallena(
            evento_id=evento_id,
            tx_hash=tx_hash or evento_id,
            fecha_disponible=recibido.isoformat(),
            fecha_detectada=recibido.isoformat(),
            fecha_blockchain=fecha_tx.isoformat(),
            activo="BTC",
            cantidad_activo=cantidad,
            valor_usd=valor_usd,
            tipo_origen=tipo_origen,
            tipo_destino=tipo_destino,
            direccion_flujo=direccion,
            fuente="whale_alert",
            direccion_origen=direccion_origen,
            direccion_destino=direccion_destino,
            entidad_origen=propietario_origen,
            entidad_destino=propietario_destino,
            exchange_origen=propietario_origen if tipo_origen == "EXCHANGE" else None,
            exchange_destino=propietario_destino if tipo_destino == "EXCHANGE" else None,
            confianza_etiquetado=0.85 if tipo_origen != "DESCONOCIDO" or tipo_destino != "DESCONOCIDO" else 0.35,
            identidad_verificada=tipo_origen != "DESCONOCIDO" or tipo_destino != "DESCONOCIDO",
            es_movimiento_interno_probable=interno,
            fuente_id=texto_seguro(payload.get("channel_id")) or None,
            metadata_json=json.dumps(payload, ensure_ascii=False),
        ))
    return eventos


async def escuchar_alertas(salida_jsonl: Path, min_value_usd: float = 1_000_000.0) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("Instala websockets para usar Whale Alert en tiempo real") from exc

    api_key = os.getenv("WHALE_ALERT_API_KEY")
    if not api_key:
        raise RuntimeError("Falta WHALE_ALERT_API_KEY")

    salida_jsonl.parent.mkdir(parents=True, exist_ok=True)
    url = f"wss://leviathan.whale-alert.io/ws?api_key={api_key}"
    suscripcion = {
        "type": "subscribe_alerts",
        "id": "tutan-noticiero-ballena-btc-v1",
        "blockchains": ["bitcoin"],
        "symbols": ["btc"],
        "tx_types": ["transfer"],
        "min_value_usd": float(min_value_usd),
    }

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps(suscripcion))
                async for mensaje in ws:
                    payload = json.loads(mensaje)
                    eventos = convertir_alerta(payload)
                    if not eventos:
                        continue
                    with salida_jsonl.open("a", encoding="utf-8") as archivo:
                        for evento in eventos:
                            archivo.write(json.dumps(evento.a_dict(), ensure_ascii=False) + "\n")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"Whale Alert desconectado: {exc}. Reintento en 5 segundos.")
            await asyncio.sleep(5)
