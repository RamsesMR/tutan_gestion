from __future__ import annotations

from cripto.corto_plazo_v2.configuracion import (
    RUTA_PROYECTO,
)
from cripto.corto_plazo_v4.configuracion import (
    COLUMNAS_V4_63,
    PLIEGUES_TEMPORALES,
    PERIODO_CONFIRMACION_2025,
    VARIANTES,
)


VERSION_MODELO = "v4_1_laboratorio_ejecucion"

SIMBOLO = "BTCUSDT"

VARIANTE_SUBE = "v4_63_sin_pesos"
VARIANTE_BAJA = "v4_63_moderado"

COLUMNAS_SUBE = tuple(
    VARIANTES[
        VARIANTE_SUBE
    ][
        "columnas"
    ]
)

COLUMNAS_BAJA = tuple(
    VARIANTES[
        VARIANTE_BAJA
    ][
        "columnas"
    ]
)

RUTA_MODELOS_V4_1 = (
    RUTA_PROYECTO
    / "modelos_entrenados"
    / "cripto"
    / "corto_plazo_v4_1"
)

RUTA_PREDICCIONES_DESARROLLO = (
    RUTA_MODELOS_V4_1
    / "predicciones_desarrollo"
)

RUTA_LABORATORIO = (
    RUTA_MODELOS_V4_1
    / "laboratorio"
)

RUTA_RESULTADOS_LABORATORIO = (
    RUTA_LABORATORIO
    / "resultados_laboratorio.csv"
)

RUTA_RESUMEN_LABORATORIO = (
    RUTA_LABORATORIO
    / "resumen_estrategias.csv"
)

RUTA_SELECCION = (
    RUTA_LABORATORIO
    / "estrategia_seleccionada.csv"
)

RUTA_CONFIRMACION_2025 = (
    RUTA_MODELOS_V4_1
    / "confirmacion_2025"
)


UMBRALES_SUBE = tuple(
    round(
        valor / 100,
        2,
    )
    for valor in range(
        44,
        61,
    )
)

UMBRALES_VETO_BAJA = (
    None,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
)

MODOS_ENTRADA = (
    "nivel",
    "cruce",
    "confirmacion_3m",
    "confirmacion_5m",
    "confirmacion_10m",
    "creciente_3m",
    "creciente_5m",
    "creciente_10m",
)

SALIDAS_FIJAS_MINUTOS = (
    60,
    120,
    240,
    480,
)

BARRERAS = (
    {
        "nombre": "tp030_sl020_120m",
        "take_profit": 0.0030,
        "stop_loss": 0.0020,
        "horizonte": 120,
    },
    {
        "nombre": "tp050_sl030_120m",
        "take_profit": 0.0050,
        "stop_loss": 0.0030,
        "horizonte": 120,
    },
    {
        "nombre": "tp050_sl030_240m",
        "take_profit": 0.0050,
        "stop_loss": 0.0030,
        "horizonte": 240,
    },
    {
        "nombre": "tp070_sl035_240m",
        "take_profit": 0.0070,
        "stop_loss": 0.0035,
        "horizonte": 240,
    },
    {
        "nombre": "tp070_sl040_480m",
        "take_profit": 0.0070,
        "stop_loss": 0.0040,
        "horizonte": 480,
    },
    {
        "nombre": "tp100_sl050_480m",
        "take_profit": 0.0100,
        "stop_loss": 0.0050,
        "horizonte": 480,
    },
)

COSTES = (
    0.0005,
    0.0010,
    0.0020,
)

COSTE_SELECCION = 0.0010

OPERACIONES_MINIMAS_POR_PLIEGUE = 50

UMBRAL_CLASE_SUBE = 0.0050

if len(COLUMNAS_SUBE) != 63:
    raise RuntimeError(
        "El detector SUBE debe usar 63 variables."
    )

if len(COLUMNAS_BAJA) != 63:
    raise RuntimeError(
        "El detector BAJA debe usar 63 variables."
    )
