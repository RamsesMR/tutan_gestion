from __future__ import annotations

from cripto.corto_plazo_v4_5.configuracion import (
    COSTE,
    HORIZONTE_SALIDA_MINUTOS,
    RUTA_MODELOS_V4_5,
    SIMBOLO_OPERATIVO,
    UMBRAL_CLASE,
    VARIANTES,
)


VERSION_CONFIRMACION = "v4_5_spot_confirmacion_2025_2026"

PERIODOS_DATOS_CONFIRMACION = (
    (
        "2025-01-01",
        "2026-01-01",
    ),
    (
        "2026-01-01",
        "2026-06-01",
    ),
)

PERIODOS_CONFIRMACION = {
    "confirmacion_2025": {
        "entrenamiento_desde": "2021-01-01",
        "entrenamiento_hasta": "2025-01-01",
        "archivos_entrenamiento": (
            (
                "2021-01-01",
                "2022-01-01",
            ),
            (
                "2022-01-01",
                "2023-01-01",
            ),
            (
                "2023-01-01",
                "2024-01-01",
            ),
            (
                "2024-01-01",
                "2025-01-01",
            ),
        ),
        "validacion_desde": "2025-01-01",
        "validacion_hasta": "2026-01-01",
        "archivo_validacion": (
            "2025-01-01",
            "2026-01-01",
        ),
        "operaciones_minimas": 50,
        "periodo_conocido_en_el_proyecto": True,
        "usado_para_seleccionar_v4_5": False,
    },
    "confirmacion_2026": {
        "entrenamiento_desde": "2021-01-01",
        "entrenamiento_hasta": "2026-01-01",
        "archivos_entrenamiento": (
            (
                "2021-01-01",
                "2022-01-01",
            ),
            (
                "2022-01-01",
                "2023-01-01",
            ),
            (
                "2023-01-01",
                "2024-01-01",
            ),
            (
                "2024-01-01",
                "2025-01-01",
            ),
            (
                "2025-01-01",
                "2026-01-01",
            ),
        ),
        "validacion_desde": "2026-01-01",
        "validacion_hasta": "2026-06-01",
        "archivo_validacion": (
            "2026-01-01",
            "2026-06-01",
        ),
        "operaciones_minimas": 30,
        "periodo_conocido_en_el_proyecto": True,
        "usado_para_seleccionar_v4_5": False,
    },
}

VARIANTES_CONFIRMACION = {
    "v41_control_63": {
        "variante_v4_5": "control_63",
        "columnas": tuple(
            VARIANTES[
                "control_63"
            ]["columnas"]
        ),
        "etapa_datos": "spot",
        "umbral": 0.44,
        "descripcion": (
            "Reproducción de V4.1 sobre exactamente la misma muestra "
            "Spot utilizada por V4.5."
        ),
    },
    "v45_spot_90": {
        "variante_v4_5": "flujo_spot",
        "columnas": tuple(
            VARIANTES[
                "flujo_spot"
            ]["columnas"]
        ),
        "etapa_datos": "spot",
        "umbral": 0.42,
        "descripcion": (
            "V4.5-Spot congelada con 90 variables."
        ),
    },
}

ESTRATEGIA_CONGELADA = {
    "modo_entrada": "cruce",
    "horizonte_salida_minutos": HORIZONTE_SALIDA_MINUTOS,
    "coste": COSTE,
    "epocas": 3,
    "tamano_lote": 100000,
    "semilla": 42,
    "sin_pesos": True,
    "sin_veto_baja": True,
}

RUTA_CONFIRMACION = (
    RUTA_MODELOS_V4_5
    / "confirmacion_2025_2026"
)

RUTA_ARTEFACTOS = (
    RUTA_CONFIRMACION
    / "artefactos"
)

RUTA_RESULTADOS_CONFIRMACION = (
    RUTA_CONFIRMACION
    / "resultados"
)

RUTA_AUDITORIA_CONFIRMACION = (
    RUTA_CONFIRMACION
    / "auditoria"
)

if len(
    VARIANTES_CONFIRMACION[
        "v41_control_63"
    ]["columnas"]
) != 63:
    raise RuntimeError(
        "El control V4.1 debe conservar 63 variables."
    )

if len(
    VARIANTES_CONFIRMACION[
        "v45_spot_90"
    ]["columnas"]
) != 90:
    raise RuntimeError(
        "V4.5-Spot debe conservar 90 variables."
    )

if not (
    ESTRATEGIA_CONGELADA[
        "horizonte_salida_minutos"
    ]
    == 480
    and abs(
        ESTRATEGIA_CONGELADA[
            "coste"
        ]
        - 0.001
    )
    < 1e-12
    and abs(
        VARIANTES_CONFIRMACION[
            "v41_control_63"
        ]["umbral"]
        - 0.44
    )
    < 1e-12
    and abs(
        VARIANTES_CONFIRMACION[
            "v45_spot_90"
        ]["umbral"]
        - 0.42
    )
    < 1e-12
):
    raise RuntimeError(
        "La confirmación ya no coincide con la configuración congelada."
    )

if abs(
    UMBRAL_CLASE
    - 0.005
) > 1e-12:
    raise RuntimeError(
        "El objetivo SUBE debe conservar el umbral de 0,5 %."
    )

if SIMBOLO_OPERATIVO != "BTCUSDT":
    raise RuntimeError(
        "La confirmación congelada debe ejecutarse sobre BTCUSDT."
    )
