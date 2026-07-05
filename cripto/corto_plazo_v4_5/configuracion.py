from __future__ import annotations

from cripto.corto_plazo_v2.configuracion import RUTA_PROYECTO
from cripto.corto_plazo_v4.configuracion import COLUMNAS_V4_63

VERSION_MODELO = "v4_5_microestructura_historica"

SIMBOLOS = (
    "BTCUSDT",
    "ETHUSDT",
)

SIMBOLO_OPERATIVO = "BTCUSDT"
INTERVALO = "1m"
HORIZONTE_MINUTOS = 240
UMBRAL_CLASE = 0.005

# Se mantienen los mismos periodos de desarrollo de V4.1.
PERIODOS_ARCHIVOS = (
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
)

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
        "archivo_validacion": ("2023-01-01", "2024-01-01"),
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
        "archivo_validacion": ("2024-01-01", "2025-01-01"),
    },
}

VENTANAS_FLUJO_SPOT = (
    5,
    15,
    30,
    60,
    240,
)

VENTANAS_FUTUROS_RENDIMIENTO = (
    1,
    5,
    15,
    30,
    60,
    240,
)

VENTANAS_FLUJO_FUTUROS = (
    5,
    15,
    30,
    60,
    240,
)

# Cubre variables con doble ventana y la media de funding de 24 horas.
FILAS_CONTEXTO = 1440

COLUMNAS_FLUJO_SPOT: list[str] = []

for ventana in VENTANAS_FLUJO_SPOT:
    COLUMNAS_FLUJO_SPOT.extend(
        [
            f"desequilibrio_spot_base_{ventana}m",
            f"desequilibrio_spot_cotizacion_{ventana}m",
            f"log_ratio_compra_venta_spot_base_{ventana}m",
            f"aceleracion_desequilibrio_spot_base_{ventana}m",
            f"confirmacion_precio_flujo_spot_{ventana}m",
        ]
    )

COLUMNAS_FLUJO_SPOT.extend(
    [
        "tamano_medio_operacion_spot_15m_relativo_240m",
        "tamano_medio_operacion_spot_60m_relativo_240m",
    ]
)

COLUMNAS_FUTUROS: list[str] = [
    "basis_futuros_spot",
    "funding_ultimo",
    "funding_absoluto",
    "cambio_funding_8h",
    "funding_media_24h",
]

for ventana in VENTANAS_FUTUROS_RENDIMIENTO:
    COLUMNAS_FUTUROS.extend(
        [
            f"rendimiento_futuros_{ventana}m",
            f"diferencia_rendimiento_futuros_spot_{ventana}m",
            f"cambio_basis_{ventana}m",
        ]
    )

for ventana in VENTANAS_FLUJO_FUTUROS:
    COLUMNAS_FUTUROS.extend(
        [
            f"desequilibrio_futuros_base_{ventana}m",
            f"desequilibrio_futuros_cotizacion_{ventana}m",
            f"divergencia_flujo_spot_futuros_{ventana}m",
            f"confirmacion_precio_flujo_futuros_{ventana}m",
        ]
    )

COLUMNAS_FLUJO_SPOT = tuple(COLUMNAS_FLUJO_SPOT)
COLUMNAS_FUTUROS = tuple(COLUMNAS_FUTUROS)
COLUMNAS_CONTROL = tuple(COLUMNAS_V4_63)
COLUMNAS_SPOT = (*COLUMNAS_CONTROL, *COLUMNAS_FLUJO_SPOT)
COLUMNAS_COMPLETAS = (*COLUMNAS_SPOT, *COLUMNAS_FUTUROS)

VARIANTES = {
    "control_63": {
        "columnas": COLUMNAS_CONTROL,
        "etapa_datos": "spot",
        "descripcion": "Control exacto de 63 variables.",
    },
    "flujo_spot": {
        "columnas": COLUMNAS_SPOT,
        "etapa_datos": "spot",
        "descripcion": "63 variables y flujo agresor Spot.",
    },
    "flujo_spot_futuros": {
        "columnas": COLUMNAS_COMPLETAS,
        "etapa_datos": "completo",
        "descripcion": (
            "63 variables, flujo Spot, flujo de futuros, basis y funding."
        ),
    },
}

RUTA_DATOS_BASE = (
    RUTA_PROYECTO
    / "datos"
    / "cripto"
    / "preparados"
    / "corto_plazo_v2"
)

RUTA_DATOS_V4_5 = (
    RUTA_PROYECTO
    / "datos"
    / "cripto"
    / "preparados"
    / "corto_plazo_v4_5"
)

RUTA_DATOS_SPOT = RUTA_DATOS_V4_5 / "spot"
RUTA_DATOS_COMPLETOS = RUTA_DATOS_V4_5 / "completo"
RUTA_AUDITORIA = RUTA_DATOS_V4_5 / "auditoria"

RUTA_FUTUROS_BRUTOS = (
    RUTA_PROYECTO
    / "datos"
    / "cripto"
    / "bruto"
    / "binance"
    / "futuros_um"
)

RUTA_KLINES_FUTUROS = RUTA_FUTUROS_BRUTOS / "klines"
RUTA_FUNDING = RUTA_FUTUROS_BRUTOS / "funding"

RUTA_MODELOS_V4_5 = (
    RUTA_PROYECTO
    / "modelos_entrenados"
    / "cripto"
    / "corto_plazo_v4_5"
)

RUTA_MODELOS_DESARROLLO = RUTA_MODELOS_V4_5 / "modelos_desarrollo"
RUTA_PREDICCIONES = RUTA_MODELOS_V4_5 / "predicciones_desarrollo"
RUTA_RESULTADOS = RUTA_MODELOS_V4_5 / "resultados_desarrollo"
RUTA_SELECCION = RUTA_MODELOS_V4_5 / "seleccion"
RUTA_COMPARACION = RUTA_MODELOS_V4_5 / "comparacion"

UMBRAL_SUBE_CONGELADO = 0.44
UMBRALES_LABORATORIO = (
    0.40,
    0.42,
    0.44,
    0.46,
    0.48,
    0.50,
)

HORIZONTE_SALIDA_MINUTOS = 480
COSTE = 0.001

COLUMNAS_META = (
    "fecha_apertura",
    "fecha_objetivo",
    "precio_apertura",
    "precio_maximo",
    "precio_minimo",
    "precio_cierre",
    "rendimiento_objetivo",
)

URL_BASE_ARCHIVOS_BINANCE = "https://data.binance.vision/data"
URL_FUNDING_BINANCE = "https://fapi.binance.com/fapi/v1/fundingRate"

if len(COLUMNAS_CONTROL) != 63:
    raise RuntimeError("El control de V4.5 debe conservar exactamente 63 variables.")

if set(COLUMNAS_CONTROL).intersection(COLUMNAS_FLUJO_SPOT):
    raise RuntimeError("Hay variables Spot repetidas respecto al control.")

if set(COLUMNAS_SPOT).intersection(COLUMNAS_FUTUROS):
    raise RuntimeError("Hay variables de futuros repetidas.")
