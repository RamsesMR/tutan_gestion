from __future__ import annotations

from datetime import datetime, timezone

from cripto.noticiero_ballena.proveedores.blockchain_com import convertir_transaccion


def test_convertir_transaccion_grande_publica() -> None:
    payload = {
        "op": "utx",
        "x": {
            "hash": "abc123",
            "time": 1704067200,
            "inputs": [
                {"prev_out": {"addr": "origen1", "value": 12_000_000_000}},
            ],
            "out": [
                {"addr": "destino1", "value": 10_000_000_000},
                {"addr": "cambio1", "value": 1_999_000_000},
            ],
            "size": 250,
        },
    }
    eventos = convertir_transaccion(
        payload,
        recibido_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        precio_btc_usd=60_000.0,
        min_value_usd=5_000_000.0,
        min_btc_respaldo=100.0,
    )
    assert len(eventos) == 1
    evento = eventos[0]
    assert evento.cantidad_activo == 100.0
    assert evento.valor_usd == 6_000_000.0
    assert evento.direccion_destino == "destino1"
    assert evento.fuente == "blockchain_com_public"
    assert evento.identidad_verificada is False


def test_filtra_transaccion_pequena() -> None:
    payload = {
        "op": "utx",
        "x": {
            "hash": "pequena",
            "time": 1704067200,
            "inputs": [{"prev_out": {"value": 100_000_000}}],
            "out": [{"value": 90_000_000}],
        },
    }
    eventos = convertir_transaccion(
        payload,
        precio_btc_usd=60_000.0,
        min_value_usd=5_000_000.0,
    )
    assert eventos == []
