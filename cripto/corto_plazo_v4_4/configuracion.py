from __future__ import annotations

from cripto.corto_plazo_v2.configuracion import RUTA_PROYECTO
from cripto.corto_plazo_v4_1.configuracion import (
    RUTA_PREDICCIONES_DESARROLLO as RUTA_PREDICCIONES_V4_1,
    RUTA_SELECCION as RUTA_SELECCION_V4_1,
    SIMBOLO,
)

VERSION_MODELO = "v4_4_gestion_riesgo_baja"

PLIEGUES = (
    "validacion_2023",
    "validacion_2024",
)

# Estrategia campeona V4.1. Estos parámetros quedan congelados.
UMBRAL_SUBE = 0.44
MODO_ENTRADA = "cruce"
HORIZONTE_MAXIMO_MINUTOS = 480
COSTE = 0.001

# Umbrales BAJA obtenidos anteriormente sin usar 2025 ni 2026:
# 0.32: detector equilibrado.
# 0.47: detector de mayor precisión.
UMBRALES_BAJA = (
    0.32,
    0.47,
)

# Evita reaccionar inmediatamente a ruido tras abrir una posición.
MINUTOS_MINIMOS_EN_POSICION = (
    0,
    30,
    60,
)

CONDICIONES_SALIDA = (
    "siempre",
    "solo_beneficio_neto",
)

MODOS_EVALUACION = (
    "cohorte_v4_1",
    "reinversion",
)

RUTA_MODELOS_V4_4 = (
    RUTA_PROYECTO
    / "modelos_entrenados"
    / "cripto"
    / "corto_plazo_v4_4"
)

RUTA_DIAGNOSTICO = (
    RUTA_MODELOS_V4_4
    / "diagnostico"
)

RUTA_LABORATORIO = (
    RUTA_MODELOS_V4_4
    / "laboratorio"
)

RUTA_SELECCION = (
    RUTA_MODELOS_V4_4
    / "seleccion"
)

RUTA_COMPARACION = (
    RUTA_MODELOS_V4_4
    / "comparacion"
)

COLUMNAS_PREDICCIONES_REQUERIDAS = (
    "fecha_apertura",
    "precio_apertura",
    "precio_maximo",
    "precio_minimo",
    "precio_cierre",
    "probabilidad_sube",
    "probabilidad_baja",
    "objetivo_sube_4h",
)

# Filtros duros para promover V4.4.
RETENCION_MINIMA_RETORNO = 0.90
RETENCION_MINIMA_FACTOR_BENEFICIO = 0.95
MEJORA_MINIMA_DRAWDOWN_ABSOLUTA = 0.01
