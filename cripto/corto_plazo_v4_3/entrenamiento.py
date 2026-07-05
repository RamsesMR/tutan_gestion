from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4.configuracion import (
    VARIANTES,
)
from cripto.corto_plazo_v4.utilidades import (
    ajustar_escalador_y_contar,
    calcular_pesos_binarios,
    construir_ruta_datos,
    entrenar_modelo_binario,
    predecir_periodo,
)
from cripto.corto_plazo_v4_3.configuracion import (
    COLUMNAS_SUBE,
    SIMBOLO,
    VARIANTE_SUBE,
)


def entrenar_y_predecir_sube(
    configuracion_periodo: dict[str, Any],
    epocas: int,
    tamano_lote: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Entrena el mismo detector SUBE de V4.1 y devuelve probabilidades."""

    configuracion_variante = VARIANTES[
        VARIANTE_SUBE
    ]

    escalador, conteos = ajustar_escalador_y_contar(
        simbolo=SIMBOLO,
        detector="SUBE",
        configuracion_periodo=configuracion_periodo,
        columnas_modelo=COLUMNAS_SUBE,
        tamano_lote=tamano_lote,
    )

    pesos = calcular_pesos_binarios(
        conteos=conteos,
        potencia_peso_positivo=float(
            configuracion_variante[
                "potencia_peso_positivo"
            ]
        ),
    )

    modelo = entrenar_modelo_binario(
        simbolo=SIMBOLO,
        detector="SUBE",
        configuracion_periodo=configuracion_periodo,
        configuracion_variante=configuracion_variante,
        escalador=escalador,
        pesos=pesos,
        epocas=epocas,
        tamano_lote=tamano_lote,
    )

    (
        _,
        probabilidades,
        _,
        _,
    ) = predecir_periodo(
        simbolo=SIMBOLO,
        detector="SUBE",
        configuracion_periodo=configuracion_periodo,
        columnas_modelo=COLUMNAS_SUBE,
        modelo=modelo,
        escalador=escalador,
        tamano_lote=tamano_lote,
    )

    detalle = {
        "conteos_entrenamiento": {
            str(clase): int(cantidad)
            for clase, cantidad in conteos.items()
        },
        "pesos": {
            str(clase): float(peso)
            for clase, peso in pesos.items()
        },
    }

    return probabilidades, detalle


def cargar_validacion_completa(
    configuracion_periodo: dict[str, Any],
) -> pd.DataFrame:
    """Carga precios con la misma purga temporal utilizada por V4.1."""

    desde_archivo, hasta_archivo = configuracion_periodo[
        "archivo_validacion"
    ]

    ruta = construir_ruta_datos(
        simbolo=SIMBOLO,
        desde=desde_archivo,
        hasta=hasta_archivo,
    )

    columnas = [
        "fecha_apertura",
        "fecha_objetivo",
        "precio_apertura",
        "precio_maximo",
        "precio_minimo",
        "precio_cierre",
        "rendimiento_objetivo",
    ]

    datos = pd.read_parquet(
        ruta,
        columns=columnas,
    )

    datos[
        "fecha_apertura"
    ] = pd.to_datetime(
        datos[
            "fecha_apertura"
        ],
        utc=True,
        errors="raise",
    )

    datos[
        "fecha_objetivo"
    ] = pd.to_datetime(
        datos[
            "fecha_objetivo"
        ],
        utc=True,
        errors="raise",
    )

    desde = pd.Timestamp(
        configuracion_periodo[
            "validacion_desde"
        ],
        tz="UTC",
    )

    hasta = pd.Timestamp(
        configuracion_periodo[
            "validacion_hasta"
        ],
        tz="UTC",
    )

    mascara = (
        (
            datos[
                "fecha_apertura"
            ]
            >= desde
        )
        & (
            datos[
                "fecha_apertura"
            ]
            < hasta
        )
        & (
            datos[
                "fecha_objetivo"
            ]
            > datos[
                "fecha_apertura"
            ]
        )
        & (
            datos[
                "fecha_objetivo"
            ]
            < hasta
        )
    )

    return (
        datos
        .loc[
            mascara
        ]
        .reset_index(
            drop=True
        )
    )
