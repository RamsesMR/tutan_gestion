from __future__ import annotations

from pathlib import Path

try:
    from cripto.corto_plazo_v2.configuracion import RUTA_PROYECTO
except Exception:
    RUTA_PROYECTO = Path(__file__).resolve().parents[2]

VERSION_MODELO = "noticiero_ballena_v1"
ACTIVO_PRINCIPAL = "BTC"
CADENA_PRINCIPAL = "bitcoin"
ZONA_HORARIA = "UTC"

# Umbrales iniciales. Se combinan con medidas relativas; no son la definición
# única de ballena.
UMBRAL_BALLENA_BTC = 100.0
UMBRAL_BALLENA_USD = 5_000_000.0

VENTANAS_MINUTOS = (5, 15, 60, 240)
HORIZONTES_IMPACTO_MINUTOS = (15, 60, 240)
VENTANA_ACTIVIDAD_MINUTOS = 240
LATENCIA_EJECUCION_MINUTOS = 1
UMBRAL_IMPACTO = {
    15: 0.0025,
    60: 0.0040,
    240: 0.0075,
}

ANIOS_DESARROLLO = (2021, 2022, 2023, 2024)
ANIOS_BLOQUEADOS = (2025, 2026)
MIN_DIAS_ENTRENAMIENTO_OOF = 180
DIAS_BLOQUE_VALIDACION_OOF = 90
RATIO_INACTIVOS_ENTRENAMIENTO = 0.25
SEMILLA = 42

PARAMETROS_CLASIFICADOR = {
    "learning_rate": 0.05,
    "max_iter": 180,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 30,
    "l2_regularization": 0.20,
    "random_state": SEMILLA,
}
PARAMETROS_REGRESOR = {
    "learning_rate": 0.05,
    "max_iter": 180,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 30,
    "l2_regularization": 0.20,
    "random_state": SEMILLA,
}

# Estructura alineada con el repositorio: crudos y preparados separados.
RUTA_CRUDOS = RUTA_PROYECTO / "datos" / "cripto" / "crudos" / "noticiero_ballena"
RUTA_PREPARADOS = RUTA_PROYECTO / "datos" / "cripto" / "preparados" / "noticiero_ballena"
RUTA_NORMALIZADOS = RUTA_PREPARADOS / "normalizados"
RUTA_VARIABLES = RUTA_PREPARADOS / "variables"
RUTA_ETIQUETADOS = RUTA_PREPARADOS / "etiquetados"
RUTA_EXPORTACIONES = RUTA_PREPARADOS / "exportaciones"

RUTA_MODELOS = RUTA_PROYECTO / "modelos_entrenados" / "cripto" / "noticiero_ballena"
RUTA_ARTEFACTOS = RUTA_MODELOS / "artefactos"
RUTA_RESULTADOS = RUTA_MODELOS / "resultados"
RUTA_AUDITORIAS = RUTA_MODELOS / "auditorias"
RUTA_PREDICCIONES = RUTA_MODELOS / "predicciones"

ARCHIVO_EVENTOS_NORMALIZADOS = RUTA_NORMALIZADOS / "eventos_ballena.parquet"
ARCHIVO_VARIABLES_MINUTO = RUTA_VARIABLES / "variables_ballena_1m.parquet"
ARCHIVO_ETIQUETADO = RUTA_ETIQUETADOS / "variables_ballena_etiquetadas.parquet"
ARCHIVO_MODELO = RUTA_ARTEFACTOS / "modelo_noticiero_ballena_v1.joblib"
ARCHIVO_MANIFIESTO_MODELO = RUTA_ARTEFACTOS / "manifiesto_modelo.json"
ARCHIVO_METRICAS = RUTA_RESULTADOS / "metricas_desarrollo.json"
ARCHIVO_OOF = RUTA_PREDICCIONES / "predicciones_oof_1m.parquet"
ARCHIVO_INFERENCIA = RUTA_PREDICCIONES / "predicciones_tiempo_real_1m.parquet"
ARCHIVO_EXPORTACION_HISTORICA = RUTA_EXPORTACIONES / "noticiero_ballena_oof_1m.parquet"
ARCHIVO_EXPORTACION_TIEMPO_REAL = RUTA_EXPORTACIONES / "noticiero_ballena_tiempo_real_1m.parquet"

COLUMNAS_EVENTO_REQUERIDAS = (
    "evento_id",
    "tx_hash",
    "fecha_disponible",
    "activo",
    "cantidad_activo",
    "valor_usd",
    "tipo_origen",
    "tipo_destino",
    "direccion_flujo",
    "fuente",
)

TIPOS_ENTIDAD = (
    "EXCHANGE",
    "ETF",
    "FONDO",
    "GOBIERNO",
    "PERSONA_PUBLICA",
    "EMPRESA",
    "MINERO",
    "CUSTODIO",
    "PROTOCOLO",
    "WALLET",
    "DESCONOCIDO",
)

CLASES_IMPACTO = ("ALCISTA", "BAJISTA", "SIN_IMPACTO")
