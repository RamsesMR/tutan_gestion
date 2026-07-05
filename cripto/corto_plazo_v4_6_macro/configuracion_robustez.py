from __future__ import annotations

from cripto.corto_plazo_v4_6_macro.configuracion import (
    RUTA_MODELOS_V4_6,
    SERIES_MACRO,
)


SEMILLAS_ROBUSTEZ = (
    11,
    23,
    42,
    77,
    101,
)

VARIANTES_ROBUSTEZ = (
    "v41_control_macro",
    "v41_reservas_fed",
    "v41_vix",
    "v45_control_macro",
    "v45_reservas_fed",
)

COMPARACIONES_ROBUSTEZ = {
    "v41_reservas_044": {
        "control": "v41_control_macro",
        "candidata": "v41_reservas_fed",
        "umbral": 0.44,
        "variable_macro": SERIES_MACRO[
            "reservas_fed"
        ]["columna"],
        "descripcion": (
            "V4.1 control frente a V4.1 con reservas Fed, "
            "ambas con umbral 0,44."
        ),
    },
    "v41_vix_042": {
        "control": "v41_control_macro",
        "candidata": "v41_vix",
        "umbral": 0.42,
        "variable_macro": SERIES_MACRO[
            "vix"
        ]["columna"],
        "descripcion": (
            "V4.1 control frente a V4.1 con VIX, "
            "ambas con umbral 0,42."
        ),
    },
    "v45_reservas_044": {
        "control": "v45_control_macro",
        "candidata": "v45_reservas_fed",
        "umbral": 0.44,
        "variable_macro": SERIES_MACRO[
            "reservas_fed"
        ]["columna"],
        "descripcion": (
            "V4.5-Spot control frente a V4.5-Spot con "
            "reservas Fed, ambas con umbral 0,44."
        ),
    },
}

RUTA_ROBUSTEZ = (
    RUTA_MODELOS_V4_6
    / "robustez_semillas"
)

RUTA_MODELOS_ROBUSTEZ = (
    RUTA_ROBUSTEZ
    / "modelos"
)

RUTA_ESCALADORES_ROBUSTEZ = (
    RUTA_ROBUSTEZ
    / "escaladores_comunes"
)

RUTA_PREDICCIONES_ROBUSTEZ = (
    RUTA_ROBUSTEZ
    / "predicciones"
)

RUTA_RESULTADOS_ROBUSTEZ = (
    RUTA_ROBUSTEZ
    / "resultados"
)

RUTA_SELECCION_ROBUSTEZ = (
    RUTA_ROBUSTEZ
    / "seleccion"
)

RETENCION_MINIMA_RETORNO_ANUAL = 0.95
RETENCION_MINIMA_FACTOR_ANUAL = 0.98
EMPEORAMIENTO_MAXIMO_DRAWDOWN = -0.01
OPERACIONES_MINIMAS = 50
SEMILLAS_MINIMAS_APROBADAS = 4
CONCENTRACION_MENSUAL_MAXIMA = 0.35
