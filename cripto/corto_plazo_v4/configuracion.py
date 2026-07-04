from __future__ import annotations

from cripto.corto_plazo_v2.configuracion import (
    COLUMNAS_MODELO_V2A,
    HORIZONTE_MINUTOS,
    NOMBRE_HORIZONTE,
    RUTA_DATOS_V2,
    RUTA_PROYECTO,
    SIMBOLOS,
    UMBRAL_CLASE,
)


VERSION_MODELO = "v4_detectores_binarios"

DETECTORES = (
    "BAJA",
    "SUBE",
)

RUTA_MODELOS_V4 = (
    RUTA_PROYECTO
    / "modelos_entrenados"
    / "cripto"
    / "corto_plazo_v4"
)

RUTA_RESULTADOS_DESARROLLO = (
    RUTA_MODELOS_V4
    / "desarrollo"
)

RUTA_HISTORIAL_DESARROLLO = (
    RUTA_RESULTADOS_DESARROLLO
    / "historial_detectores.csv"
)

RUTA_RESUMEN_DESARROLLO = (
    RUTA_RESULTADOS_DESARROLLO
    / "resumen"
)

RUTA_SELECCION_UMBRALES = (
    RUTA_RESUMEN_DESARROLLO
    / "seleccion_umbrales.csv"
)

RUTA_CONFIRMACION_2025 = (
    RUTA_MODELOS_V4
    / "confirmacion_2025"
)

RUTA_HISTORIAL_CONFIRMACION_2025 = (
    RUTA_CONFIRMACION_2025
    / "historial_confirmacion_2025.csv"
)

RUTA_COMBINACION_2025 = (
    RUTA_MODELOS_V4
    / "combinacion_2025"
)


# ============================================================
# VALIDACIONES TEMPORALES INTERNAS
# ============================================================

# Se usan para elegir variables, ponderación y umbrales.
# 2025 y 2026 quedan fuera de esta selección.
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
# CONFIRMACIÓN SECUNDARIA EN 2025
# ============================================================

# 2025 ya fue observado durante versiones anteriores.
# Solo se usa para confirmar una configuración elegida con 2023/2024.
PERIODO_CONFIRMACION_2025 = {
    "entrenamiento_desde": "2021-01-01",
    "entrenamiento_hasta": "2025-01-01",
    "archivos_entrenamiento": (
        ("2021-01-01", "2022-01-01"),
        ("2022-01-01", "2023-01-01"),
        ("2023-01-01", "2024-01-01"),
        ("2024-01-01", "2025-01-01"),
    ),
    "validacion_desde": "2025-01-01",
    "validacion_hasta": "2026-01-01",
    "archivo_validacion": (
        "2025-01-01",
        "2026-01-01",
    ),
}


# ============================================================
# CONJUNTOS DE VARIABLES
# ============================================================

VARIABLES_RETIRADAS_V4_52 = {
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

COLUMNAS_V4_63 = tuple(
    COLUMNAS_MODELO_V2A
)

COLUMNAS_V4_52 = tuple(
    columna
    for columna in COLUMNAS_V4_63
    if columna not in VARIABLES_RETIRADAS_V4_52
)


# ============================================================
# VARIANTES
# ============================================================

# potencia_peso_positivo:
# 1.0 = balanceo completo.
# 0.5 = balanceo moderado.
# 0.0 = sin ponderación.
#
# Los pesos se normalizan para que el peso medio sea 1.
VARIANTES = {
    "v4_63_balanceado": {
        "columnas": COLUMNAS_V4_63,
        "potencia_peso_positivo": 1.0,
        "penalty": "l2",
        "alpha": 0.0001,
        "grupo": "base",
    },
    "v4_52_balanceado": {
        "columnas": COLUMNAS_V4_52,
        "potencia_peso_positivo": 1.0,
        "penalty": "l2",
        "alpha": 0.0001,
        "grupo": "base",
    },
    "v4_63_moderado": {
        "columnas": COLUMNAS_V4_63,
        "potencia_peso_positivo": 0.5,
        "penalty": "l2",
        "alpha": 0.0001,
        "grupo": "pesos",
    },
    "v4_63_sin_pesos": {
        "columnas": COLUMNAS_V4_63,
        "potencia_peso_positivo": 0.0,
        "penalty": "l2",
        "alpha": 0.0001,
        "grupo": "pesos",
    },
}


# Umbrales comunes para poder comparar 2023 y 2024 exactamente.
UMBRALES_PROBABILIDAD = tuple(
    round(
        valor / 100,
        2,
    )
    for valor in range(
        5,
        96,
    )
)

PERFILES_UMBRALES = (
    "equilibrado",
    "precision",
    "operativo",
)


# ============================================================
# VALIDACIONES
# ============================================================

if len(COLUMNAS_V4_63) != 63:
    raise RuntimeError(
        "La variante completa de la V4 debe contener 63 variables."
    )

if len(COLUMNAS_V4_52) != 52:
    raise RuntimeError(
        "La variante reducida de la V4 debe contener 52 variables."
    )

if not VARIABLES_RETIRADAS_V4_52.issubset(
    set(COLUMNAS_V4_63)
):
    raise RuntimeError(
        "Alguna variable retirada no existe en las 63 variables."
    )
