from __future__ import annotations

import pandas as pd

from cripto.noticiero_ballena.construir_historico import (
    calcular_max_bytes_consulta,
    iterar_meses,
    minimo_satoshis_mes,
    procesar_salidas,
)
from cripto.noticiero_ballena.proveedores.bigquery_bitcoin import CONSULTA_OUTPUTS_GRANDES


def test_consulta_usa_tabla_particionada_y_filtra_satoshis() -> None:
    assert "crypto_bitcoin.transactions" in CONSULTA_OUTPUTS_GRANDES
    assert "CROSS JOIN UNNEST(t.outputs)" in CONSULTA_OUTPUTS_GRANDES
    assert "t.block_timestamp_month = @mes_particion" in CONSULTA_OUTPUTS_GRANDES
    assert "t.block_timestamp >= @inicio" in CONSULTA_OUTPUTS_GRANDES
    assert "t.block_timestamp < @fin" in CONSULTA_OUTPUTS_GRANDES
    assert "o.value >= @min_satoshis" in CONSULTA_OUTPUTS_GRANDES


def test_iterar_meses_no_solapa() -> None:
    meses = list(iterar_meses(
        pd.Timestamp("2023-01-15", tz="UTC"),
        pd.Timestamp("2023-03-10", tz="UTC"),
    ))
    assert meses == [
        (pd.Timestamp("2023-01-15", tz="UTC"), pd.Timestamp("2023-02-01", tz="UTC")),
        (pd.Timestamp("2023-02-01", tz="UTC"), pd.Timestamp("2023-03-01", tz="UTC")),
        (pd.Timestamp("2023-03-01", tz="UTC"), pd.Timestamp("2023-03-10", tz="UTC")),
    ]


def test_minimo_satoshis_usa_maximo_mensual() -> None:
    precios = pd.DataFrame({
        "fecha_apertura": pd.to_datetime(["2023-01-01", "2023-01-02"], utc=True),
        "precio_btc_usd": [20_000.0, 25_000.0],
    })
    sats, min_btc, max_precio = minimo_satoshis_mes(
        precios,
        pd.Timestamp("2023-01-01", tz="UTC"),
        pd.Timestamp("2023-02-01", tz="UTC"),
        1_000_000.0,
        margen_seguridad=0.98,
    )
    assert max_precio == 25_000.0
    assert round(min_btc, 2) == 39.20
    assert sats == 3_920_000_000


def test_procesar_salidas_filtra_por_usd_y_aplica_latencia() -> None:
    crudo = pd.DataFrame({
        "tx_hash": ["tx_grande", "tx_pequena"],
        "block_hash": ["b1", "b1"],
        "block_number": [100, 100],
        "block_timestamp": pd.to_datetime(
            ["2023-01-01T00:00:30Z", "2023-01-01T00:00:40Z"], utc=True
        ),
        "output_index": [0, 1],
        "direcciones_destino": ["addr1", "addr2|addr3"],
        "tipo_salida": ["pubkeyhash", "scripthash"],
        "valor_satoshis": [5_100_000_000, 4_000_000_000],
    })
    precios = pd.DataFrame({
        "fecha_apertura": pd.to_datetime(["2023-01-01T00:00:00Z"], utc=True),
        "precio_btc_usd": [20_000.0],
    })
    salida = procesar_salidas(
        crudo,
        precios,
        min_usd=1_000_000.0,
        latencia_minutos=15,
        tolerancia_precio_minutos=10,
    )
    assert len(salida) == 1
    assert salida.iloc[0]["tx_hash"] == "tx_grande"
    assert salida.iloc[0]["direccion_destino"] == "addr1"
    assert salida.iloc[0]["valor_usd"] == 1_020_000.0
    assert salida.iloc[0]["fecha_disponible"] == pd.Timestamp("2023-01-01T00:15:30Z")


def test_procesar_salidas_acepta_precision_us_y_ns() -> None:
    crudo = pd.DataFrame({
        "tx_hash": ["tx_precision"],
        "block_hash": ["b1"],
        "block_number": [100],
        "block_timestamp": pd.Series(
            pd.array(["2023-01-01T00:00:30Z"], dtype="datetime64[us, UTC]")
        ),
        "output_index": [0],
        "direcciones_destino": ["addr1"],
        "tipo_salida": ["pubkeyhash"],
        "valor_satoshis": [5_100_000_000],
    })
    precios = pd.DataFrame({
        "fecha_apertura": pd.Series(
            pd.array(["2023-01-01T00:00:00Z"], dtype="datetime64[ns, UTC]")
        ),
        "precio_btc_usd": [20_000.0],
    })

    salida = procesar_salidas(
        crudo,
        precios,
        min_usd=1_000_000.0,
        latencia_minutos=15,
        tolerancia_precio_minutos=10,
    )

    assert len(salida) == 1
    assert salida.iloc[0]["tx_hash"] == "tx_precision"
    assert str(salida["fecha_blockchain"].dtype) == "datetime64[ns, UTC]"


def test_calcular_max_bytes_consulta_aplica_margen_y_respeta_limite() -> None:
    gib = 1024 ** 3
    estimados = int(6.35 * gib)
    limite_mes = 60 * gib

    limite = calcular_max_bytes_consulta(estimados, limite_mes)

    assert limite >= estimados + 512 * 1024 ** 2
    assert limite <= limite_mes


def test_calcular_max_bytes_consulta_no_supera_limite_duro() -> None:
    gib = 1024 ** 3
    assert calcular_max_bytes_consulta(59 * gib, 60 * gib) == 60 * gib
