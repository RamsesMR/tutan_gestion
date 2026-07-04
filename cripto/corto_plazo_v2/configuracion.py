from __future__ import annotations

from pathlib import Path


# ============================================================
# IDENTIDAD DEL MODELO
# ============================================================

VERSION_MODELO = "v2a_variables_cruzadas"

SIMBOLOS = (
    "BTCUSDT",
    "ETHUSDT",
)

CLASES = (
    "BAJA",
    "NEUTRAL",
    "SUBE",
)

INTERVALO = "1m"
HORIZONTE_MINUTOS = 240
NOMBRE_HORIZONTE = "4h"
UMBRAL_CLASE = 0.005


# ============================================================
# RUTAS
# ============================================================

# Raíz del proyecto: tutan_gestion/
RUTA_PROYECTO = Path(__file__).resolve().parents[2]

# Datos definitivos de la V1. Solo se leen.
RUTA_DATOS_V1 = (
    RUTA_PROYECTO
    / "datos"
    / "cripto"
    / "preparados"
    / "corto_plazo"
)

# Datos independientes de la V2A.
RUTA_DATOS_V2 = (
    RUTA_PROYECTO
    / "datos"
    / "cripto"
    / "preparados"
    / "corto_plazo_v2"
)

# Modelos y resultados independientes de la V2A.
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

RUTA_MANIFIESTO_V2 = (
    RUTA_DATOS_V2
    / "manifiesto_divisiones_4h.csv"
)

RUTA_RESUMEN_DIVISIONES_V2 = (
    RUTA_DATOS_V2
    / "resumen_divisiones_4h.csv"
)


# ============================================================
# PERIODOS DE DESARROLLO
# ============================================================

# La V2A se selecciona únicamente con:
# - entrenamiento: 2021-2024
# - validación: 2025
#
# Enero-mayo de 2026 ya fue observado con la V1. No se incluye
# en el desarrollo ni en la selección de la V2A.
DIVISIONES_DESARROLLO = {
    "entrenamiento": {
        "desde": "2021-01-01",
        "hasta": "2025-01-01",
        "periodos": (
            ("2021-01-01", "2022-01-01"),
            ("2022-01-01", "2023-01-01"),
            ("2023-01-01", "2024-01-01"),
            ("2024-01-01", "2025-01-01"),
        ),
    },
    "validacion": {
        "desde": "2025-01-01",
        "hasta": "2026-01-01",
        "periodos": (
            ("2025-01-01", "2026-01-01"),
        ),
    },
}


# Lista plana conservada para que generar_variables_cruzadas.py
# continúe funcionando sin duplicar la definición de periodos.
PERIODOS_DESARROLLO = (
    *DIVISIONES_DESARROLLO["entrenamiento"]["periodos"],
    *DIVISIONES_DESARROLLO["validacion"]["periodos"],
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
    """Reproduce exactamente las 36 variables de la V1."""

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


# ============================================================
# VALIDACIONES DE CONFIGURACIÓN
# ============================================================

if len(COLUMNAS_MODELO_V1) != 36:
    raise RuntimeError(
        "La configuración no reproduce las 36 variables de la V1."
    )

if len(COLUMNAS_CRUZADAS) != 27:
    raise RuntimeError(
        "La configuración debe contener 27 variables cruzadas."
    )

if len(COLUMNAS_MODELO_V2A) != 63:
    raise RuntimeError(
        "La V2A debe contener exactamente 63 variables."
    )

if set(COLUMNAS_MODELO_V1).intersection(COLUMNAS_CRUZADAS):
    raise RuntimeError(
        "Existen nombres repetidos entre las variables V1 y V2A."
    )
