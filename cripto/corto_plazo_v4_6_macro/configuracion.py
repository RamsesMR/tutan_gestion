from __future__ import annotations

from cripto.corto_plazo_v2.configuracion import RUTA_PROYECTO
from cripto.corto_plazo_v4.configuracion import COLUMNAS_V4_63
from cripto.corto_plazo_v4_5.configuracion import (
    COLUMNAS_META,
    COLUMNAS_SPOT,
    COSTE,
    HORIZONTE_SALIDA_MINUTOS,
    PLIEGUES_TEMPORALES,
    RUTA_DATOS_BASE,
    RUTA_DATOS_SPOT,
    UMBRAL_CLASE,
)


VERSION_MODELO = "v4_6_macro_regimen"
SIMBOLOS = (
    "BTCUSDT",
    "ETHUSDT",
)
SIMBOLO_OPERATIVO = "BTCUSDT"

DESCARGA_DESDE = "2019-01-01"
DESCARGA_HASTA = "2024-12-31"

URL_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

SERIES_MACRO = {
    "dolar": {
        "serie": "DTWEXBGS",
        "columna": "impulso_dolar_amplio_20s",
        "transformacion": "log_ratio",
        "periodos": 20,
        "disponibilidad": "h10_semanal",
        "descripcion": (
            "Cambio logarítmico del índice amplio del dólar "
            "durante 20 observaciones."
        ),
    },
    "tipo_real": {
        "serie": "DFII10",
        "columna": "impulso_rendimiento_real_10y_20s",
        "transformacion": "diferencia",
        "periodos": 20,
        "disponibilidad": "h15_siguiente_habil",
        "descripcion": (
            "Cambio en puntos porcentuales del rendimiento real "
            "estadounidense a diez años durante 20 observaciones."
        ),
    },
    "credito": {
        "serie": "BAA10Y",
        "columna": "impulso_spread_baa_20s",
        "transformacion": "diferencia",
        "periodos": 20,
        "disponibilidad": "siguiente_habil_conservador",
        "descripcion": (
            "Cambio del spread de bonos corporativos Baa frente "
            "al Treasury a diez años durante 20 observaciones."
        ),
    },
    "vix": {
        "serie": "VIXCLS",
        "columna": "impulso_vix_20s",
        "transformacion": "log_ratio",
        "periodos": 20,
        "disponibilidad": "cierre_mercado",
        "descripcion": (
            "Cambio logarítmico del VIX durante 20 sesiones."
        ),
    },
    "nasdaq": {
        "serie": "NASDAQ100",
        "columna": "retorno_nasdaq100_20s",
        "transformacion": "log_ratio",
        "periodos": 20,
        "disponibilidad": "cierre_mercado",
        "descripcion": (
            "Retorno logarítmico del Nasdaq 100 durante 20 sesiones."
        ),
    },
    "reservas_fed": {
        "serie": "WRESBAL",
        "columna": "impulso_reservas_fed_4s",
        "transformacion": "log_ratio",
        "periodos": 4,
        "disponibilidad": "h41_jueves",
        "descripcion": (
            "Cambio logarítmico de las reservas bancarias en la Fed "
            "durante cuatro publicaciones semanales."
        ),
    },
}

COLUMNAS_MACRO = tuple(
    detalle["columna"]
    for detalle in SERIES_MACRO.values()
)

BASES = {
    "v41": {
        "columnas": tuple(COLUMNAS_V4_63),
        "ruta_referencia": RUTA_DATOS_BASE,
        "umbral": 0.44,
        "descripcion": "Base V4.1 de 63 variables.",
    },
    "v45": {
        "columnas": tuple(COLUMNAS_SPOT),
        "ruta_referencia": RUTA_DATOS_SPOT,
        "umbral": 0.42,
        "descripcion": "Base V4.5-Spot de 90 variables.",
    },
}

VARIANTES: dict[str, dict[str, object]] = {}

for nombre_base, configuracion_base in BASES.items():
    columnas_base = tuple(
        configuracion_base["columnas"]
    )

    VARIANTES[
        f"{nombre_base}_control_macro"
    ] = {
        "base": nombre_base,
        "columnas": columnas_base,
        "variable_macro": None,
        "umbral_base": float(
            configuracion_base["umbral"]
        ),
        "descripcion": (
            f"Control {nombre_base} sobre la misma muestra macro."
        ),
    }

    for clave_macro, detalle_macro in SERIES_MACRO.items():
        columna_macro = str(
            detalle_macro["columna"]
        )

        VARIANTES[
            f"{nombre_base}_{clave_macro}"
        ] = {
            "base": nombre_base,
            "columnas": (
                *columnas_base,
                columna_macro,
            ),
            "variable_macro": columna_macro,
            "umbral_base": float(
                configuracion_base["umbral"]
            ),
            "descripcion": (
                f"{nombre_base} más {columna_macro}."
            ),
        }

VARIANTES_FASE_INDIVIDUAL = tuple(VARIANTES)

RUTA_MACRO_BRUTO = (
    RUTA_PROYECTO
    / "datos"
    / "cripto"
    / "bruto"
    / "macro"
    / "fred"
)

RUTA_DATOS_V4_6 = (
    RUTA_PROYECTO
    / "datos"
    / "cripto"
    / "preparados"
    / "corto_plazo_v4_6_macro"
)

RUTA_VARIABLES_MACRO = (
    RUTA_DATOS_V4_6
    / "variables_macro"
)

RUTA_DATOS_INTEGRADOS = (
    RUTA_DATOS_V4_6
    / "integrados"
)

RUTA_AUDITORIA = (
    RUTA_DATOS_V4_6
    / "auditoria"
)

RUTA_MODELOS_V4_6 = (
    RUTA_PROYECTO
    / "modelos_entrenados"
    / "cripto"
    / "corto_plazo_v4_6_macro"
)

RUTA_MODELOS_DESARROLLO = (
    RUTA_MODELOS_V4_6
    / "modelos_desarrollo"
)

RUTA_PREDICCIONES = (
    RUTA_MODELOS_V4_6
    / "predicciones_desarrollo"
)

RUTA_RESULTADOS = (
    RUTA_MODELOS_V4_6
    / "resultados_desarrollo"
)

RUTA_SELECCION = (
    RUTA_MODELOS_V4_6
    / "seleccion"
)

UMBRALES_LABORATORIO = (
    0.40,
    0.42,
    0.44,
    0.46,
    0.48,
    0.50,
)

if len(COLUMNAS_V4_63) != 63:
    raise RuntimeError(
        "La base V4.1 debe conservar exactamente 63 variables."
    )

if len(COLUMNAS_SPOT) != 90:
    raise RuntimeError(
        "La base V4.5-Spot debe conservar exactamente 90 variables."
    )

if len(COLUMNAS_MACRO) != 6:
    raise RuntimeError(
        "V4.6 debe estudiar exactamente seis variables macro."
    )

if len(set(COLUMNAS_MACRO)) != len(COLUMNAS_MACRO):
    raise RuntimeError(
        "Hay nombres de variables macro repetidos."
    )
