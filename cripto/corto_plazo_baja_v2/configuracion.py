from __future__ import annotations

from cripto.corto_plazo_v2.configuracion import COLUMNAS_MODELO_V2A, RUTA_PROYECTO

VERSION_MODELO = "baja_v2_eventos_macro"
SIMBOLO = "BTCUSDT"

# BAJA V2 parte del hallazgo más cercano a equilibrio de BAJA V1.
OBJETIVO_FUENTE = "corto_tp100_sl050_480m"
TAKE_PROFIT = 0.0100
STOP_LOSS = 0.0050
HORIZONTE_MINUTOS = 480
COSTE_BASE = 0.0010
COSTES_AUDITORIA = (0.0005, 0.0010, 0.0020)

# Desarrollo. 2025 y 2026 quedan bloqueados.
PERIODOS_FUENTE = (
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
)
PLIEGUES_TEMPORALES = {
    "validacion_2023": {
        "entrenamiento": PERIODOS_FUENTE[:2],
        "validacion": PERIODOS_FUENTE[2],
        "hasta_entrenamiento": "2023-01-01",
        "hasta_validacion": "2024-01-01",
    },
    "validacion_2024": {
        "entrenamiento": PERIODOS_FUENTE[:3],
        "validacion": PERIODOS_FUENTE[3],
        "hasta_entrenamiento": "2024-01-01",
        "hasta_validacion": "2025-01-01",
    },
}

REJILLAS_EVENTOS_MINUTOS = (5, 10, 15, 30)
MARGENES_EV = (0.0, 0.00025, 0.00050, 0.00100)
SEMILLAS = (11, 23, 42, 77, 101)

COLUMNAS_BASE_63 = tuple(COLUMNAS_MODELO_V2A)
COLUMNAS_MACRO = (
    "impulso_dolar_amplio_20s",
    "impulso_rendimiento_real_10y_20s",
    "impulso_spread_baa_20s",
    "impulso_vix_20s",
    "retorno_nasdaq100_20s",
    "impulso_reservas_fed_4s",
)

# Interacciones con hipótesis económica explícita. Se construyen después del merge causal.
INTERACCIONES_MACRO = {
    "estres_dolar_eth": ("impulso_dolar_amplio_20s", "otro_retorno_60m"),
    "estres_vix_btc": ("impulso_vix_20s", "retorno_60m"),
    "riesgo_nasdaq_btc": ("retorno_nasdaq100_20s", "retorno_60m"),
    "tipos_reales_momentum": ("impulso_rendimiento_real_10y_20s", "retorno_240m"),
    "credito_volatilidad": ("impulso_spread_baa_20s", "volatilidad_60m"),
}

VARIANTES = {
    "control_63_eventos": {
        "columnas_macro": (),
        "usar_interacciones": False,
    },
    "macro_dolar": {
        "columnas_macro": ("impulso_dolar_amplio_20s",),
        "usar_interacciones": False,
    },
    "macro_rendimiento_real": {
        "columnas_macro": ("impulso_rendimiento_real_10y_20s",),
        "usar_interacciones": False,
    },
    "macro_spread_baa": {
        "columnas_macro": ("impulso_spread_baa_20s",),
        "usar_interacciones": False,
    },
    "macro_vix": {
        "columnas_macro": ("impulso_vix_20s",),
        "usar_interacciones": False,
    },
    "macro_nasdaq100": {
        "columnas_macro": ("retorno_nasdaq100_20s",),
        "usar_interacciones": False,
    },
    "macro_reservas_fed": {
        "columnas_macro": ("impulso_reservas_fed_4s",),
        "usar_interacciones": False,
    },
    "macro_completo": {
        "columnas_macro": COLUMNAS_MACRO,
        "usar_interacciones": False,
    },
    "macro_completo_interacciones": {
        "columnas_macro": COLUMNAS_MACRO,
        "usar_interacciones": True,
    },
}

PARAMETROS_HISTGB = {
    "learning_rate": 0.05,
    "max_iter": 180,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 40,
    "l2_regularization": 0.20,
}

OPERACIONES_MINIMAS_POR_PLIEGUE = 50
FACTOR_BENEFICIO_MINIMO = 1.10
DRAWDOWN_MINIMO_ADMITIDO = -0.20
CONCENTRACION_MENSUAL_MAXIMA = 0.35
PROMOCION_AUTOMATICA = False

RUTA_DATOS_BAJA_V1 = (
    RUTA_PROYECTO / "datos" / "cripto" / "preparados" / "corto_plazo_baja_v1"
)
RUTA_DATOS_MACRO = (
    RUTA_PROYECTO / "datos" / "cripto" / "preparados" / "corto_plazo_v4_6_macro"
)
RUTA_DATOS_V2 = (
    RUTA_PROYECTO / "datos" / "cripto" / "preparados" / "corto_plazo_baja_v2"
)
RUTA_MODELOS_V2 = (
    RUTA_PROYECTO / "modelos_entrenados" / "cripto" / "corto_plazo_baja_v2"
)
RUTA_DESARROLLO = RUTA_MODELOS_V2 / "desarrollo"
RUTA_MODELOS = RUTA_DESARROLLO / "modelos"
RUTA_RESULTADOS = RUTA_DESARROLLO / "resultados"
RUTA_HISTORIAL = RUTA_RESULTADOS / "historial_desarrollo.csv"
RUTA_SELECCION = RUTA_DESARROLLO / "seleccion"
RUTA_AUDITORIAS = RUTA_DESARROLLO / "auditorias"

if len(COLUMNAS_BASE_63) != 63:
    raise RuntimeError("BAJA V2 requiere exactamente las 63 variables V2A/V4.")
