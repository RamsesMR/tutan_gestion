from __future__ import annotations

from cripto.corto_plazo_v2.configuracion import RUTA_PROYECTO

VERSION_MODELO = "v4_2_mejora_ejecucion"

SIMBOLO = "BTCUSDT"

RUTA_MODELOS_V4_2 = (
    RUTA_PROYECTO
    / "modelos_entrenados"
    / "cripto"
    / "corto_plazo_v4_2"
)

RUTA_BASES = RUTA_MODELOS_V4_2 / "bases"
RUTA_RESULTADOS = RUTA_MODELOS_V4_2 / "resultados"
RUTA_MODELOS = RUTA_MODELOS_V4_2 / "modelos"
RUTA_CONFIRMACIONES = RUTA_MODELOS_V4_2 / "confirmaciones"

RUTA_PREDICCIONES_V4_1 = (
    RUTA_PROYECTO
    / "modelos_entrenados"
    / "cripto"
    / "corto_plazo_v4_1"
    / "predicciones_desarrollo"
)

RUTA_DATOS_V2 = (
    RUTA_PROYECTO
    / "datos"
    / "cripto"
    / "preparados"
    / "corto_plazo_v2"
)

PERIODOS_DESARROLLO = {
    "validacion_2023": ("2023-01-01", "2024-01-01"),
    "validacion_2024": ("2024-01-01", "2025-01-01"),
}

PERIODOS_CONFIRMACION = {
    "confirmacion_2025": ("2025-01-01", "2026-01-01"),
    "confirmacion_2026": ("2026-01-01", "2026-06-01"),
}

UMBRAL_SUBE_BASE = 0.44
COSTE_TOTAL = 0.001

HORIZONTES_MINUTOS = (
    300,
    360,
    420,
    480,
    540,
    600,
    720,
)

UMBRAL_REARME = (
    None,
    0.42,
    0.40,
    0.38,
)

ENFRIAMIENTOS_MINUTOS = (
    0,
    30,
    60,
    120,
)

PENDIENTES_MINUTOS = (
    3,
    5,
    10,
)

DISTANCIAS_CRUCE = (
    0.000,
    0.005,
    0.010,
    0.020,
)

UMBRALES_DIFERENCIA_SUBE_BAJA = (
    None,
    0.00,
    0.05,
    0.10,
    0.15,
)

CUANTILES_FILTRO = (
    0.20,
    0.40,
    0.60,
    0.80,
)

OPERACIONES_MINIMAS_ANIO = 40

COLUMNAS_META = (
    "probabilidad_sube",
    "probabilidad_baja",
    "diferencia_sube_baja",
    "pendiente_sube_3m",
    "pendiente_sube_5m",
    "pendiente_sube_10m",
    "distancia_umbral",
    "rendimiento_60m",
    "rendimiento_240m",
    "volatilidad_60m",
    "volatilidad_240m",
    "volumen_relativo_60m",
    "volumen_relativo_240m",
    "correlacion_btc_eth_240m",
    "beta_propio_otro_240m",
    "residual_propio_otro_240m",
    "minuto_dia_seno",
    "minuto_dia_coseno",
    "dia_semana_seno",
    "dia_semana_coseno",
)

RUTA_RESULTADOS_SALIDAS = RUTA_RESULTADOS / "salidas.csv"
RUTA_RESULTADOS_FILTROS = RUTA_RESULTADOS / "filtros.csv"
RUTA_RESULTADOS_META = RUTA_RESULTADOS / "meta_modelo.csv"
RUTA_POLITICA = RUTA_RESULTADOS / "politica_v4_2.json"
