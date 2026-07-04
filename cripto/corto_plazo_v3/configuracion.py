from __future__ import annotations

from pathlib import Path

from cripto.corto_plazo_v2.configuracion import (
    CLASES,
    COLUMNAS_MODELO_V2A,
    NOMBRE_HORIZONTE,
    RUTA_DATOS_V2,
    RUTA_PROYECTO,
    SIMBOLOS,
    UMBRAL_CLASE,
)


VERSION_MODELO = "v3a_ablaciones"

RUTA_MODELOS_V3 = (
    RUTA_PROYECTO
    / "modelos_entrenados"
    / "cripto"
    / "corto_plazo_v3"
)

RUTA_RESULTADOS_ABLACIONES = (
    RUTA_MODELOS_V3
    / "ablaciones"
)

RUTA_HISTORIAL_ABLACIONES = (
    RUTA_RESULTADOS_ABLACIONES
    / "historial_ablaciones.csv"
)


# ============================================================
# VALIDACIONES TEMPORALES INTERNAS
# ============================================================

# 2025 no se utiliza para escoger variables de la V3.
PLIEGUES_TEMPORALES = {
    "validacion_2023": {
        "entrenamiento_desde": "2021-01-01",
        "entrenamiento_hasta": "2023-01-01",
        "archivos_entrenamiento": (
            ("2021-01-01", "2022-01-01"),
            ("2022-01-01", "2023-01-01"),
        ),
        "validacion_desde": "2023-01-01",
        "validacion_hasta": "2024-01-01",
        "archivo_validacion": (
            "2023-01-01",
            "2024-01-01",
        ),
    },
    "validacion_2024": {
        "entrenamiento_desde": "2021-01-01",
        "entrenamiento_hasta": "2024-01-01",
        "archivos_entrenamiento": (
            ("2021-01-01", "2022-01-01"),
            ("2022-01-01", "2023-01-01"),
            ("2023-01-01", "2024-01-01"),
        ),
        "validacion_desde": "2024-01-01",
        "validacion_hasta": "2025-01-01",
        "archivo_validacion": (
            "2024-01-01",
            "2025-01-01",
        ),
    },
}


# ============================================================
# REDUCCIÓN DE VARIABLES
# ============================================================

VARIABLES_RETIRADAS_V3A_52 = {
    "proporcion_compradora_base",
    "proporcion_compradora_cotizacion",
    "otro_rendimiento_5m",
    "otro_rendimiento_15m",
    "otro_rendimiento_30m",
    "otro_rendimiento_60m",
    "divergencia_rendimiento_5m",
    "divergencia_rendimiento_15m",
    "divergencia_rendimiento_30m",
    "zscore_divergencia_240m",
    "zscore_divergencia_1440m",
}

VARIABLES_RETIRADAS_ADICIONALES_V3A_46 = {
    "rendimiento_5m",
    "rendimiento_15m",
    "rendimiento_30m",
    "volumen_relativo_5m",
    "volumen_relativo_30m",
    "desviacion_media_cierre_5m",
}


COLUMNAS_V3A_63 = tuple(
    COLUMNAS_MODELO_V2A
)

COLUMNAS_V3A_52 = tuple(
    columna
    for columna in COLUMNAS_V3A_63
    if columna not in VARIABLES_RETIRADAS_V3A_52
)

COLUMNAS_V3A_46 = tuple(
    columna
    for columna in COLUMNAS_V3A_52
    if columna not in VARIABLES_RETIRADAS_ADICIONALES_V3A_46
)


# ============================================================
# VARIANTES
# ============================================================

VARIANTES = {
    "v3a_63_l2": {
        "columnas": COLUMNAS_V3A_63,
        "penalty": "l2",
        "alpha": 0.0001,
        "l1_ratio": 0.15,
        "grupo": "reduccion",
    },
    "v3a_52_l2": {
        "columnas": COLUMNAS_V3A_52,
        "penalty": "l2",
        "alpha": 0.0001,
        "l1_ratio": 0.15,
        "grupo": "reduccion",
    },
    "v3a_46_l2": {
        "columnas": COLUMNAS_V3A_46,
        "penalty": "l2",
        "alpha": 0.0001,
        "l1_ratio": 0.15,
        "grupo": "reduccion",
    },
    "v3a_52_elasticnet_015": {
        "columnas": COLUMNAS_V3A_52,
        "penalty": "elasticnet",
        "alpha": 0.0001,
        "l1_ratio": 0.15,
        "grupo": "elasticnet",
    },
    "v3a_52_elasticnet_050": {
        "columnas": COLUMNAS_V3A_52,
        "penalty": "elasticnet",
        "alpha": 0.0001,
        "l1_ratio": 0.50,
        "grupo": "elasticnet",
    },
    "v3a_52_elasticnet_085": {
        "columnas": COLUMNAS_V3A_52,
        "penalty": "elasticnet",
        "alpha": 0.0001,
        "l1_ratio": 0.85,
        "grupo": "elasticnet",
    },
}


# ============================================================
# VALIDACIONES DE CONFIGURACIÓN
# ============================================================

if len(COLUMNAS_V3A_63) != 63:
    raise RuntimeError(
        "La variante completa debe contener 63 variables."
    )

if len(COLUMNAS_V3A_52) != 52:
    raise RuntimeError(
        "La variante reducida debe contener 52 variables."
    )

if len(COLUMNAS_V3A_46) != 46:
    raise RuntimeError(
        "La variante agresiva debe contener 46 variables."
    )

if not VARIABLES_RETIRADAS_V3A_52.issubset(
    set(COLUMNAS_V3A_63)
):
    raise RuntimeError(
        "Alguna variable del corte V3A-52 no existe en la V2A."
    )

if not VARIABLES_RETIRADAS_ADICIONALES_V3A_46.issubset(
    set(COLUMNAS_V3A_52)
):
    raise RuntimeError(
        "Alguna variable del corte V3A-46 no existe en V3A-52."
    )
