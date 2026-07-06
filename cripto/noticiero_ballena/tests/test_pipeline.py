from __future__ import annotations

import pandas as pd

from cripto.noticiero_ballena.etiquetar_impacto import etiquetar_impacto
from cripto.noticiero_ballena.generar_variables import generar_variables_minuto
from cripto.noticiero_ballena.normalizar_eventos import normalizar_dataframe


def test_normalizacion_y_variables() -> None:
    crudos = pd.DataFrame([
        {
            "hash": "tx1",
            "timestamp": "2024-01-01T00:00:10Z",
            "symbol": "BTC",
            "amount": 250,
            "value_usd": 10_000_000,
            "from_owner": "unknown",
            "to_owner": "Binance",
            "to_type": "exchange",
            "source": "test",
            "confidence": 0.9,
        },
        {
            "hash": "tx2",
            "timestamp": "2024-01-01T00:04:00Z",
            "symbol": "BTC",
            "amount": 100,
            "value_usd": 4_000_000,
            "from_owner": "Coinbase",
            "from_type": "exchange",
            "to_owner": "unknown",
            "source": "test",
            "confidence": 0.8,
        },
    ])
    normalizados = normalizar_dataframe(crudos)
    assert len(normalizados) == 2
    assert set(normalizados["direccion_flujo"]) == {"HACIA_EXCHANGE", "DESDE_EXCHANGE"}

    variables = generar_variables_minuto(normalizados, desde="2024-01-01", hasta="2024-01-01T00:10:00Z")
    fila = variables.loc[variables["fecha_disponible"] == pd.Timestamp("2024-01-01T00:05:00Z")].iloc[0]
    assert fila["nb_raw_inflow_btc_15m"] == 250
    assert fila["nb_raw_outflow_btc_15m"] == 100
    assert fila["nb_raw_netflow_btc_15m"] == 150


def test_etiquetado_bajista() -> None:
    fechas = pd.date_range("2024-01-01", periods=20, freq="1min", tz="UTC")
    precios = pd.DataFrame({
        "fecha_apertura": fechas,
        "open": [100.0] * 20,
        "high": [100.1] * 20,
        "low": [100.0, 100.0, 99.0] + [100.0] * 17,
        "close": [100.0] * 20,
    })
    variables = pd.DataFrame({
        "fecha_disponible": [fechas[0]],
        "nb_raw_activo_240m": [1],
    })
    salida = etiquetar_impacto(variables, precios)
    assert salida.loc[0, "nb_clase_impacto_15m"] == "BAJISTA"
