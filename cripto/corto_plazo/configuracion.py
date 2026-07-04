from pathlib import Path


# Mercados que utilizará el modelo de corto plazo.
SIMBOLOS = (
    "BTCUSDT",
    "ETHUSDT",
)

# Frecuencia de las velas almacenadas en PostgreSQL.
INTERVALO = "1m"

# El modelo intentará predecir el comportamiento
# del precio cuatro horas después.
HORIZONTE_MINUTOS = 240

# Nombre utilizado para identificar este horizonte.
NOMBRE_HORIZONTE = "4h"

# Raíz del proyecto: tutan_gestion/
RUTA_PROYECTO = Path(__file__).resolve().parents[2]

# Carpeta existente donde se guardarán los datos
# preparados para entrenar el modelo.
RUTA_DATOS_PREPARADOS = (
    RUTA_PROYECTO
    / "datos"
    / "cripto"
    / "preparados"
    / "corto_plazo"
)

# Columnas procedentes directamente de la tabla velas.
COLUMNAS_BASE = (
    "fecha_apertura",
    "precio_apertura",
    "precio_maximo",
    "precio_minimo",
    "precio_cierre",
    "volumen",
    "volumen_activo_cotizacion",
    "numero_operaciones",
    "volumen_comprador_base",
    "volumen_comprador_cotizacion",
)