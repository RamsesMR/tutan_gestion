from __future__ import annotations

import gc

import pandas as pd

from cripto.corto_plazo_v2.configuracion import (
    FILAS_HISTORIAL,
)
from cripto.corto_plazo_v2.generar_variables_cruzadas import (
    calcular_variables_cruzadas,
    cargar_periodo,
    construir_base_relaciones,
    guardar_periodo,
    validar_alineacion,
)


DESDE_HISTORIAL = "2025-01-01"
HASTA_HISTORIAL = "2026-01-01"

DESDE_ACTUAL = "2026-01-01"
HASTA_ACTUAL = "2026-06-01"


def main() -> None:
    """Genera únicamente las variables cruzadas V2A de enero-mayo de 2026."""

    print(
        "\nGENERACIÓN V2A PARA 2026"
    )
    print("=" * 70)
    print(
        f"Historial causal: {DESDE_HISTORIAL} a {HASTA_HISTORIAL}"
    )
    print(
        f"Periodo nuevo: {DESDE_ACTUAL} a {HASTA_ACTUAL}"
    )
    print(
        "No se regenerarán 2021-2025."
    )

    btc_historial, ruta_btc_historial = cargar_periodo(
        simbolo="BTCUSDT",
        desde=DESDE_HISTORIAL,
        hasta=HASTA_HISTORIAL,
    )

    eth_historial, ruta_eth_historial = cargar_periodo(
        simbolo="ETHUSDT",
        desde=DESDE_HISTORIAL,
        hasta=HASTA_HISTORIAL,
    )

    validar_alineacion(
        btc=btc_historial,
        eth=eth_historial,
        desde=DESDE_HISTORIAL,
        hasta=HASTA_HISTORIAL,
    )

    btc_actual, ruta_btc_actual = cargar_periodo(
        simbolo="BTCUSDT",
        desde=DESDE_ACTUAL,
        hasta=HASTA_ACTUAL,
    )

    eth_actual, ruta_eth_actual = cargar_periodo(
        simbolo="ETHUSDT",
        desde=DESDE_ACTUAL,
        hasta=HASTA_ACTUAL,
    )

    validar_alineacion(
        btc=btc_actual,
        eth=eth_actual,
        desde=DESDE_ACTUAL,
        hasta=HASTA_ACTUAL,
    )

    print(
        f"\nEntrada BTC historial: {ruta_btc_historial.name}"
    )
    print(
        f"Entrada ETH historial: {ruta_eth_historial.name}"
    )
    print(
        f"Entrada BTC actual: {ruta_btc_actual.name}"
    )
    print(
        f"Entrada ETH actual: {ruta_eth_actual.name}"
    )

    base_historial = construir_base_relaciones(
        btc=btc_historial,
        eth=eth_historial,
    ).tail(
        FILAS_HISTORIAL
    ).reset_index(
        drop=True
    )

    base_historial[
        "es_periodo_actual"
    ] = False

    base_actual = construir_base_relaciones(
        btc=btc_actual,
        eth=eth_actual,
    )

    base_actual[
        "es_periodo_actual"
    ] = True

    base_completa = pd.concat(
        [
            base_historial,
            base_actual,
        ],
        ignore_index=True,
    )

    variables_cruzadas = calcular_variables_cruzadas(
        base=base_completa.drop(
            columns=[
                "es_periodo_actual",
            ]
        )
    )

    mascara_actual = base_completa[
        "es_periodo_actual"
    ]

    resumen_btc = guardar_periodo(
        datos_originales=btc_actual,
        variables_cruzadas=variables_cruzadas,
        mascara_periodo_actual=mascara_actual,
        simbolo="BTCUSDT",
        desde=DESDE_ACTUAL,
        hasta=HASTA_ACTUAL,
        sobrescribir=False,
    )

    resumen_eth = guardar_periodo(
        datos_originales=eth_actual,
        variables_cruzadas=variables_cruzadas,
        mascara_periodo_actual=mascara_actual,
        simbolo="ETHUSDT",
        desde=DESDE_ACTUAL,
        hasta=HASTA_ACTUAL,
        sobrescribir=False,
    )

    print(
        "\nVARIABLES V2A 2026 GENERADAS CORRECTAMENTE"
    )
    print("=" * 70)
    print(
        f"BTC filas: {resumen_btc['archivo_salida_filas']:,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"ETH filas: {resumen_eth['archivo_salida_filas']:,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"BTC salida: {resumen_btc['archivo_salida']}"
    )
    print(
        f"ETH salida: {resumen_eth['archivo_salida']}"
    )

    del btc_historial
    del eth_historial
    del btc_actual
    del eth_actual
    del base_historial
    del base_actual
    del base_completa
    del variables_cruzadas
    gc.collect()


if __name__ == "__main__":
    main()
