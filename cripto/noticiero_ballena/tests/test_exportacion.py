from __future__ import annotations

import pandas as pd

from cripto.noticiero_ballena.auditar_exportacion import auditar
from cripto.noticiero_ballena.exportar_para_modelos import exportar


def test_contrato_exportacion() -> None:
    df = pd.DataFrame({
        "fecha_disponible": pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC"),
        "nb_raw_activo_240m": [1, 1],
        "nb_raw_calidad_datos": [0.8, 0.9],
        "nb_p_bajista_15m": [0.6, 0.2],
        "nb_p_alcista_15m": [0.2, 0.5],
        "nb_p_sin_impacto_15m": [0.2, 0.3],
        "nb_retorno_estimado_15m": [-0.003, 0.001],
        "nb_confianza_15m": [0.48, 0.45],
    })
    salida = exportar(df, "historico")
    assert "fecha_apertura" in salida
    resultado = auditar(salida)
    assert resultado["aprobada"]
