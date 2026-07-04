from __future__ import annotations

from pathlib import Path


# ============================================================
# IDENTIDAD DE LA VERSIÓN
# ============================================================

VERSION_MODELO = "v2a_cruce_btc_eth"

SIMBOLOS = (
    "BTCUSDT",
    "ETHUSDT",
)

OTRO_SIMBOLO = {
    "BTCUSDT": "ETHUSDT",
    "ETHUSDT": "BTCUSDT",
}

INTERVALO = "1m"
HORIZONTE_MINUTOS = 240
NOMBRE_HORIZONTE = "4h"


# ============================================================
# RUTAS
# ============================================================

# Raíz del proyecto: tutan_gestion/
RUTA_PROYECTO = Path(__file__).resolve().parents[2]

# Entrada: datos definitivos de la V1.
# Esta carpeta no se modifica.
RUTA_DATOS_V1 = (
    RUTA_PROYECTO
    / "datos"
    / "cripto"
    / "preparados"
    / "corto_plazo"
)

# Salida independiente para la V2A.
RUTA_DATOS_V2 = (
    RUTA_PROYECTO
    / "datos"
    / "cripto"
    / "preparados"
    / "corto_plazo_v2"
)

RUTA_MODELOS_V2 = (
    RUTA_PROYECTO
    / "modelos_entrenados"
    / "cripto"
    / "corto_plazo_v2"
)

RUTA_RESUMEN_GENERACION = (
    RUTA_DATOS_V2
    / "resumen_generacion_variables_cruzadas.csv"
)


# ============================================================
# PERIODOS PERMITIDOS EN EL DESARROLLO DE LA V2A
# ============================================================

# Se generan únicamente entrenamiento 2021-2024 y validación 2025.
# Enero-mayo de 2026 ya fue utilizado en la evaluación final de la V1
# y no se usará para seleccionar variables de la V2.
PERIODOS_DESARROLLO = (
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-01-01"),
)


# ============================================================
# VARIABLES DE LA V1
# ============================================================

VENTANAS_MINUTOS_V1 = (
    5,
    15,
    30,
    60,
    240,
)


def construir_columnas_modelo_v1() -> list[str]:
    """Reproduce exactamente las 36 variables del modelo V1."""

    columnas = [
        "rango_relativo",
        "cuerpo_relativo",
        "mecha_superior_relativa",
        "mecha_inferior_relativa",
        "proporcion_compradora_base",
        "proporcion_compradora_cotizacion",
        "rendimiento_1m",
    ]

    for ventana in VENTANAS_MINUTOS_V1:
        columnas.extend(
            [
                f"rendimiento_{ventana}m",
                f"volatilidad_{ventana}m",
                f"desviacion_media_cierre_{ventana}m",
                f"volumen_relativo_{ventana}m",
                f"operaciones_relativas_{ventana}m",
            ]
        )

    columnas.extend(
        [
            "minuto_dia_seno",
            "minuto_dia_coseno",
            "dia_semana_seno",
            "dia_semana_coseno",
        ]
    )

    return columnas


COLUMNAS_MODELO_V1 = tuple(
    construir_columnas_modelo_v1()
)


# ============================================================
# VARIABLES CRUZADAS BTC-ETH
# ============================================================

VENTANAS_RENDIMIENTO_CRUZADO = (
    1,
    5,
    15,
    30,
    60,
    240,
)

VENTANAS_RELACION = (
    60,
    240,
    1440,
)

VENTANAS_VOLATILIDAD_RELATIVA = (
    60,
    240,
)

VENTANAS_ZSCORE_DIVERGENCIA = (
    240,
    1440,
)

# Historial suficiente para calcular la mayor ventana sin mezclar futuro.
FILAS_HISTORIAL = max(
    VENTANAS_RELACION
)


def construir_columnas_cruzadas() -> list[str]:
    """Devuelve las 27 variables nuevas de la V2A."""

    columnas: list[str] = []

    for ventana in VENTANAS_RENDIMIENTO_CRUZADO:
        columnas.extend(
            [
                f"otro_rendimiento_{ventana}m",
                f"divergencia_rendimiento_{ventana}m",
            ]
        )

    for ventana in VENTANAS_VOLATILIDAD_RELATIVA:
        columnas.extend(
            [
                f"otro_volatilidad_{ventana}m",
                f"ratio_volatilidad_{ventana}m",
            ]
        )

    for ventana in VENTANAS_RELACION:
        columnas.extend(
            [
                f"correlacion_btc_eth_{ventana}m",
                f"beta_propio_otro_{ventana}m",
                f"residual_propio_otro_{ventana}m",
            ]
        )

    for ventana in VENTANAS_ZSCORE_DIVERGENCIA:
        columnas.append(
            f"zscore_divergencia_{ventana}m"
        )

    return columnas


COLUMNAS_CRUZADAS = tuple(
    construir_columnas_cruzadas()
)

COLUMNAS_MODELO_V2A = (
    *COLUMNAS_MODELO_V1,
    *COLUMNAS_CRUZADAS,
)

if len(COLUMNAS_MODELO_V1) != 36:
    raise RuntimeError(
        "La configuración V2A no reproduce las 36 variables de la V1."
    )

if len(COLUMNAS_CRUZADAS) != 27:
    raise RuntimeError(
        "La configuración V2A debe contener 27 variables cruzadas."
    )

if len(COLUMNAS_MODELO_V2A) != 63:
    raise RuntimeError(
        "La V2A debe contener exactamente 63 variables."
    )
