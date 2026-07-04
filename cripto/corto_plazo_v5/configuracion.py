from __future__ import annotations

from cripto.corto_plazo_v2.configuracion import (
    COLUMNAS_MODELO_V2A,
    RUTA_DATOS_V2,
    RUTA_PROYECTO,
)


VERSION_MODELO = "v5_oportunidades_financieras"

SIMBOLO_INICIAL = "BTCUSDT"

RUTA_DATOS_V5 = (
    RUTA_PROYECTO
    / "datos"
    / "cripto"
    / "preparados"
    / "corto_plazo_v5"
)

RUTA_MODELOS_V5 = (
    RUTA_PROYECTO
    / "modelos_entrenados"
    / "cripto"
    / "corto_plazo_v5"
)

RUTA_DESARROLLO_V5 = (
    RUTA_MODELOS_V5
    / "desarrollo"
)

RUTA_HISTORIAL_V5 = (
    RUTA_DESARROLLO_V5
    / "historial_modelos.csv"
)

RUTA_RESUMEN_V5 = (
    RUTA_DESARROLLO_V5
    / "resumen"
)

RUTA_SELECCION_V5 = (
    RUTA_RESUMEN_V5
    / "seleccion_v5.csv"
)

RUTA_CONFIRMACION_2025 = (
    RUTA_MODELOS_V5
    / "confirmacion_2025"
)

RUTA_FUTUROS = (
    RUTA_PROYECTO
    / "datos"
    / "cripto"
    / "futuros"
)


PERIODOS = (
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-01-01"),
)


PLIEGUES_TEMPORALES = {
    "validacion_2023": {
        "entrenamiento": (
            ("2021-01-01", "2022-01-01"),
            ("2022-01-01", "2023-01-01"),
        ),
        "validacion": (
            "2023-01-01",
            "2024-01-01",
        ),
        "hasta_entrenamiento": "2023-01-01",
        "hasta_validacion": "2024-01-01",
    },
    "validacion_2024": {
        "entrenamiento": (
            ("2021-01-01", "2022-01-01"),
            ("2022-01-01", "2023-01-01"),
            ("2023-01-01", "2024-01-01"),
        ),
        "validacion": (
            "2024-01-01",
            "2025-01-01",
        ),
        "hasta_entrenamiento": "2024-01-01",
        "hasta_validacion": "2025-01-01",
    },
}


PERIODO_CONFIRMACION_2025 = {
    "entrenamiento": (
        ("2021-01-01", "2022-01-01"),
        ("2022-01-01", "2023-01-01"),
        ("2023-01-01", "2024-01-01"),
        ("2024-01-01", "2025-01-01"),
    ),
    "validacion": (
        "2025-01-01",
        "2026-01-01",
    ),
    "hasta_entrenamiento": "2025-01-01",
    "hasta_validacion": "2026-01-01",
}


OBJETIVOS_BARRERAS = {
    "largo_tp050_sl030_120m": {
        "take_profit": 0.0050,
        "stop_loss": 0.0030,
        "horizonte_minutos": 120,
    },
    "largo_tp070_sl035_240m": {
        "take_profit": 0.0070,
        "stop_loss": 0.0035,
        "horizonte_minutos": 240,
    },
    "largo_tp100_sl050_480m": {
        "take_profit": 0.0100,
        "stop_loss": 0.0050,
        "horizonte_minutos": 480,
    },
}

OBJETIVO_BASE = "largo_tp070_sl035_240m"


COLUMNAS_TECNICAS_V5 = (
    "pendiente_log_15m",
    "r2_tendencia_15m",
    "pendiente_log_60m",
    "r2_tendencia_60m",
    "pendiente_log_240m",
    "r2_tendencia_240m",
    "aceleracion_5m_15m",
    "aceleracion_15m_60m",
    "aceleracion_60m_240m",
    "distancia_maximo_60m",
    "distancia_minimo_60m",
    "posicion_rango_60m",
    "ruptura_maximo_60m",
    "ruptura_minimo_60m",
    "distancia_maximo_240m",
    "distancia_minimo_240m",
    "posicion_rango_240m",
    "ruptura_maximo_240m",
    "ruptura_minimo_240m",
    "distancia_maximo_1440m",
    "distancia_minimo_1440m",
    "posicion_rango_1440m",
    "semivolatilidad_positiva_60m",
    "semivolatilidad_negativa_60m",
    "ratio_semivolatilidad_60m",
    "semivolatilidad_positiva_240m",
    "semivolatilidad_negativa_240m",
    "ratio_semivolatilidad_240m",
    "porcentaje_velas_positivas_15m",
    "porcentaje_velas_positivas_60m",
    "cuerpos_positivos_acumulados_15m",
    "cuerpos_negativos_acumulados_15m",
    "cuerpos_positivos_acumulados_60m",
    "cuerpos_negativos_acumulados_60m",
    "velas_positivas_consecutivas",
    "velas_negativas_consecutivas",
    "expansion_volatilidad_15m_60m",
    "expansion_volatilidad_60m_240m",
)

COLUMNAS_MODELO_V5 = tuple(
    COLUMNAS_MODELO_V2A
) + COLUMNAS_TECNICAS_V5


VARIANTES_MODELO = {
    "sgd_sin_pesos": {
        "tipo": "sgd",
        "potencia_peso_positivo": 0.0,
        "alpha": 0.0001,
    },
    "sgd_moderado": {
        "tipo": "sgd",
        "potencia_peso_positivo": 0.5,
        "alpha": 0.0001,
    },
    "histgb_moderado": {
        "tipo": "histgb",
        "potencia_peso_positivo": 0.5,
        "max_iter": 160,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 100,
        "l2_regularization": 0.20,
        "max_filas_entrenamiento": 600_000,
    },
}


UMBRALES = tuple(
    round(
        valor / 100,
        2,
    )
    for valor in range(
        5,
        96,
    )
)

COSTES_OPERACION = (
    0.0005,
    0.0010,
    0.0020,
)

COSTE_BASE = 0.0010

OPERACIONES_MINIMAS_POR_PLIEGUE = 50


ALIAS_COLUMNAS = {
    "fecha_apertura": (
        "fecha_apertura",
        "open_time",
        "timestamp",
        "fecha",
        "datetime",
    ),
    "apertura": (
        "apertura",
        "open",
        "precio_apertura",
    ),
    "maximo": (
        "maximo",
        "high",
        "precio_maximo",
    ),
    "minimo": (
        "minimo",
        "low",
        "precio_minimo",
    ),
    "cierre": (
        "cierre",
        "close",
        "precio_cierre",
    ),
}


if len(COLUMNAS_MODELO_V5) != (
    len(COLUMNAS_MODELO_V2A)
    + len(COLUMNAS_TECNICAS_V5)
):
    raise RuntimeError(
        "Hay columnas duplicadas entre la V2A y las variables técnicas V5."
    )
