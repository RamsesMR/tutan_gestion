from __future__ import annotations

from datetime import datetime, timezone

from cripto.noticiero_ballena.proveedores.whale_alert import convertir_alerta


def test_convertir_alerta_whale_alert() -> None:
    payload = {
        "type": "alert",
        "timestamp": 1704067200,
        "transaction": {"hash": "abc"},
        "from": {"owner": "unknown", "address": "from1"},
        "to": {"owner": "Binance", "address": "to1", "owner_type": "exchange"},
        "amounts": [{"symbol": "BTC", "amount": 500, "value_usd": 20_000_000}],
    }
    eventos = convertir_alerta(payload, datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert len(eventos) == 1
    evento = eventos[0]
    assert evento.direccion_flujo == "HACIA_EXCHANGE"
    assert evento.cantidad_activo == 500
    assert evento.exchange_destino == "Binance"
