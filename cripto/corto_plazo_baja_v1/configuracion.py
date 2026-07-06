from __future__ import annotations

from cripto.corto_plazo_v2.configuracion import (
    COLUMNAS_MODELO_V2A,
    HORIZONTE_MINUTOS,
    RUTA_PROYECTO,
    UMBRAL_CLASE,
)
from cripto.corto_plazo_v5.configuracion import (
    COLUMNAS_TECNICAS_V5,
    RUTA_DATOS_V5,
)


VERSION_MODELO = "baja_v1_short"

SIMBOLO = "BTCUSDT"

PERIODOS_FUENTE = (
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    # Se utiliza solamente como contexto futuro para las últimas
    # operaciones de 2024. No participa en la selección.
    ("2025-01-01", "2026-01-01"),
)

PERIODOS_DESARROLLO = PERIODOS_FUENTE[:4]

PLIEGUES_TEMPORALES = {
    "validacion_2023": {
        "entrenamiento": (
            ("2021-01-01", "2022-01-01"),
            ("2022-01-01", "2023-01-01"),
        ),
        "validacion": ("2023-01-01", "2024-01-01"),
        "hasta_entrenamiento": "2023-01-01",
        "hasta_validacion": "2024-01-01",
    },
    "validacion_2024": {
        "entrenamiento": (
            ("2021-01-01", "2022-01-01"),
            ("2022-01-01", "2023-01-01"),
            ("2023-01-01", "2024-01-01"),
        ),
        "validacion": ("2024-01-01", "2025-01-01"),
        "hasta_entrenamiento": "2024-01-01",
        "hasta_validacion": "2025-01-01",
    },
}

# Control histórico: misma definición de clase BAJA que V4.
# La clasificación mira HORIZONTE_MINUTOS (4 horas), mientras la
# estrategia de control mantiene la salida fija de V4.1: 480 minutos.
OBJETIVOS = {
    "baja_cierre_4h": {
        "tipo": "cierre",
        "umbral_baja": float(UMBRAL_CLASE),
        "horizonte_objetivo_minutos": int(HORIZONTE_MINUTOS),
        "horizonte_salida_minutos": 480,
        "grupo": "control_v4",
    },
    "corto_tp050_sl030_120m": {
        "tipo": "barreras_short",
        "take_profit": 0.0050,
        "stop_loss": 0.0030,
        "horizonte_minutos": 120,
        "grupo": "short",
    },
    "corto_tp070_sl035_240m": {
        "tipo": "barreras_short",
        "take_profit": 0.0070,
        "stop_loss": 0.0035,
        "horizonte_minutos": 240,
        "grupo": "short",
    },
    "corto_tp100_sl050_480m": {
        "tipo": "barreras_short",
        "take_profit": 0.0100,
        "stop_loss": 0.0050,
        "horizonte_minutos": 480,
        "grupo": "short",
    },
}

OBJETIVO_BASE = "corto_tp070_sl035_240m"

COLUMNAS_BASE_63 = tuple(COLUMNAS_MODELO_V2A)
COLUMNAS_TECNICAS_38 = tuple(COLUMNAS_TECNICAS_V5)
COLUMNAS_COMPLETAS_101 = (
    *COLUMNAS_BASE_63,
    *COLUMNAS_TECNICAS_38,
)

VARIANTES = {
    # Reproduce el planteamiento del detector BAJA V4:
    # 63 variables y ponderación moderada.
    "control_v4_63_moderado": {
        "tipo": "sgd",
        "columnas": COLUMNAS_BASE_63,
        "potencia_peso_positivo": 0.5,
        "alpha": 0.0001,
        "grupo": "control",
    },
    "sgd_63_sin_pesos": {
        "tipo": "sgd",
        "columnas": COLUMNAS_BASE_63,
        "potencia_peso_positivo": 0.0,
        "alpha": 0.0001,
        "grupo": "base",
    },
    "histgb_63_moderado": {
        "tipo": "histgb",
        "columnas": COLUMNAS_BASE_63,
        "potencia_peso_positivo": 0.5,
        "max_iter": 160,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 100,
        "l2_regularization": 0.20,
        "max_filas_entrenamiento": 600_000,
        "grupo": "algoritmo",
    },
    "sgd_101_moderado": {
        "tipo": "sgd",
        "columnas": COLUMNAS_COMPLETAS_101,
        "potencia_peso_positivo": 0.5,
        "alpha": 0.0001,
        "grupo": "estructura_velas",
    },
    "histgb_101_moderado": {
        "tipo": "histgb",
        "columnas": COLUMNAS_COMPLETAS_101,
        "potencia_peso_positivo": 0.5,
        "max_iter": 160,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 100,
        "l2_regularization": 0.20,
        "max_filas_entrenamiento": 600_000,
        "grupo": "estructura_velas",
    },
}

UMBRALES = tuple(
    round(valor / 100, 2)
    for valor in range(5, 96)
)

COSTE_BASE = 0.0010
COSTES_AUDITORIA = (
    0.0005,
    0.0010,
    0.0020,
)

OPERACIONES_MINIMAS_POR_PLIEGUE = 50
FACTOR_BENEFICIO_MINIMO = 1.10
DRAWDOWN_MINIMO_ADMITIDO = -0.20
PORCENTAJE_POSITIVAS_MINIMO = 50.0

EPOCAS_SGD = 3
TAMANO_LOTE = 100_000
SEMILLA = 42

MAXIMO_CONTEXTO_FUTURO = max(
    max(
        int(configuracion.get("horizonte_minutos", 0)),
        int(configuracion.get("horizonte_salida_minutos", 0)),
        int(configuracion.get("horizonte_objetivo_minutos", 0)),
    )
    for configuracion in OBJETIVOS.values()
)

RUTA_DATOS_BAJA = (
    RUTA_PROYECTO
    / "datos"
    / "cripto"
    / "preparados"
    / "corto_plazo_baja_v1"
)

RUTA_MODELOS_BAJA = (
    RUTA_PROYECTO
    / "modelos_entrenados"
    / "cripto"
    / "corto_plazo_baja_v1"
)

RUTA_DESARROLLO = (
    RUTA_MODELOS_BAJA
    / "desarrollo"
)

RUTA_MODELOS_DESARROLLO = (
    RUTA_DESARROLLO
    / "modelos"
)

RUTA_RESULTADOS = (
    RUTA_DESARROLLO
    / "resultados"
)

RUTA_HISTORIAL = (
    RUTA_RESULTADOS
    / "historial_desarrollo.csv"
)

RUTA_SELECCION = (
    RUTA_DESARROLLO
    / "seleccion"
)

if len(COLUMNAS_BASE_63) != 63:
    raise RuntimeError(
        "La base V4 debe contener exactamente 63 variables."
    )

if len(COLUMNAS_TECNICAS_38) != 38:
    raise RuntimeError(
        "La familia técnica V5 debe contener exactamente 38 variables."
    )

if len(COLUMNAS_COMPLETAS_101) != 101:
    raise RuntimeError(
        "La familia completa debe contener exactamente 101 variables."
    )

if len(set(COLUMNAS_COMPLETAS_101)) != len(COLUMNAS_COMPLETAS_101):
    raise RuntimeError(
        "Hay variables duplicadas entre la base y la familia técnica."
    )

if OBJETIVO_BASE not in OBJETIVOS:
    raise RuntimeError(
        "El objetivo base no existe en OBJETIVOS."
    )
