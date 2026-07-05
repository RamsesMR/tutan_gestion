from __future__ import annotations

from cripto.corto_plazo_v4.configuracion import (
    COLUMNAS_V4_63,
    RUTA_PROYECTO,
    VARIANTES,
)


VERSION_MODELO = "v4_3_historico_ampliado"
SIMBOLO = "BTCUSDT"
VARIANTE_SUBE = "v4_63_sin_pesos"
COLUMNAS_SUBE = tuple(
    VARIANTES[
        VARIANTE_SUBE
    ][
        "columnas"
    ]
)

# Estrategia congelada de V4.1. No se seleccionará otra estrategia.
UMBRAL_SUBE = 0.44
MODO_ENTRADA = "cruce"
USA_VETO_BAJA = False
UMBRAL_VETO_BAJA = None
TIPO_SALIDA = "fija"
MINUTOS_SALIDA = 480
NOMBRE_SALIDA = "fija_480m"
COSTE = 0.001
UMBRAL_CLASE_SUBE = 0.005

ARCHIVOS_HISTORICOS = (
    (
        "2017-08-17",
        "2018-01-01",
    ),
    (
        "2018-01-01",
        "2019-01-01",
    ),
    (
        "2019-01-01",
        "2020-01-01",
    ),
    (
        "2020-01-01",
        "2021-01-01",
    ),
)

ARCHIVOS_2021_2022 = (
    (
        "2021-01-01",
        "2022-01-01",
    ),
    (
        "2022-01-01",
        "2023-01-01",
    ),
)

PLIEGUES_TEMPORALES = {
    "validacion_2023": {
        "entrenamiento_desde": "2017-08-17",
        "entrenamiento_hasta": "2023-01-01",
        "archivos_entrenamiento": (
            *ARCHIVOS_HISTORICOS,
            *ARCHIVOS_2021_2022,
        ),
        "validacion_desde": "2023-01-01",
        "validacion_hasta": "2024-01-01",
        "archivo_validacion": (
            "2023-01-01",
            "2024-01-01",
        ),
    },
    "validacion_2024": {
        "entrenamiento_desde": "2017-08-17",
        "entrenamiento_hasta": "2024-01-01",
        "archivos_entrenamiento": (
            *ARCHIVOS_HISTORICOS,
            *ARCHIVOS_2021_2022,
            (
                "2023-01-01",
                "2024-01-01",
            ),
        ),
        "validacion_desde": "2024-01-01",
        "validacion_hasta": "2025-01-01",
        "archivo_validacion": (
            "2024-01-01",
            "2025-01-01",
        ),
    },
}

PERIODO_2025_CONOCIDO = {
    "entrenamiento_desde": "2017-08-17",
    "entrenamiento_hasta": "2025-01-01",
    "archivos_entrenamiento": (
        *ARCHIVOS_HISTORICOS,
        *ARCHIVOS_2021_2022,
        (
            "2023-01-01",
            "2024-01-01",
        ),
        (
            "2024-01-01",
            "2025-01-01",
        ),
    ),
    "validacion_desde": "2025-01-01",
    "validacion_hasta": "2026-01-01",
    "archivo_validacion": (
        "2025-01-01",
        "2026-01-01",
    ),
}

PERIODO_2026_CONOCIDO = {
    "entrenamiento_desde": "2017-08-17",
    "entrenamiento_hasta": "2026-01-01",
    "archivos_entrenamiento": (
        *ARCHIVOS_HISTORICOS,
        *ARCHIVOS_2021_2022,
        (
            "2023-01-01",
            "2024-01-01",
        ),
        (
            "2024-01-01",
            "2025-01-01",
        ),
        (
            "2025-01-01",
            "2026-01-01",
        ),
    ),
    "validacion_desde": "2026-01-01",
    "validacion_hasta": "2026-06-01",
    "archivo_validacion": (
        "2026-01-01",
        "2026-06-01",
    ),
}

RUTA_MODELOS_V4_3 = (
    RUTA_PROYECTO
    / "modelos_entrenados"
    / "cripto"
    / "corto_plazo_v4_3"
)

RUTA_PREDICCIONES_DESARROLLO = (
    RUTA_MODELOS_V4_3
    / "predicciones_desarrollo"
)

RUTA_RESULTADOS_DESARROLLO = (
    RUTA_MODELOS_V4_3
    / "resultados_desarrollo"
)

RUTA_RESULTADO_2025 = (
    RUTA_MODELOS_V4_3
    / "evaluacion_conocida_2025"
)

RUTA_RESULTADO_2026 = (
    RUTA_MODELOS_V4_3
    / "evaluacion_conocida_2026"
)

RUTA_COMPARACION = (
    RUTA_MODELOS_V4_3
    / "comparacion_v4_1"
)

ESTRATEGIA_CONGELADA = {
    "umbral_sube": UMBRAL_SUBE,
    "modo_entrada": MODO_ENTRADA,
    "usa_veto_baja": USA_VETO_BAJA,
    "tipo_salida": TIPO_SALIDA,
    "nombre_salida": NOMBRE_SALIDA,
    "horizonte_salida": MINUTOS_SALIDA,
    "coste": COSTE,
}

if len(COLUMNAS_SUBE) != 63:
    raise RuntimeError(
        "V4.3 debe conservar exactamente las 63 variables de V4.1."
    )

if tuple(COLUMNAS_SUBE) != tuple(COLUMNAS_V4_63):
    raise RuntimeError(
        "Las variables de V4.3 no coinciden con la configuración completa de V4."
    )

if VARIANTE_SUBE != "v4_63_sin_pesos":
    raise RuntimeError(
        "V4.3 debe conservar la misma variante SUBE de V4.1."
    )
