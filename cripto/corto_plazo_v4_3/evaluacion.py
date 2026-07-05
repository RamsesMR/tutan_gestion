from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_3.configuracion import (
    COSTE,
    MINUTOS_SALIDA,
    MODO_ENTRADA,
    UMBRAL_CLASE_SUBE,
    UMBRAL_SUBE,
)


def convertir_fechas_ns(
    fechas: pd.Series,
) -> np.ndarray:
    """Convierte fechas UTC a nanosegundos de forma explícita."""

    return (
        pd.to_datetime(
            fechas,
            utc=True,
            errors="raise",
        )
        .dt.tz_convert(
            "UTC"
        )
        .dt.tz_localize(
            None
        )
        .to_numpy(
            dtype="datetime64[ns]"
        )
        .astype(
            "int64"
        )
    )


def construir_mascara_entrada(
    probabilidades_sube: np.ndarray,
) -> np.ndarray:
    """Aplica el cruce de 0,44 congelado en V4.1."""

    sobre_umbral = (
        probabilidades_sube
        >= UMBRAL_SUBE
    )

    if MODO_ENTRADA != "cruce":
        raise RuntimeError(
            "V4.3 debe conservar el modo de entrada por cruce."
        )

    anterior = np.empty_like(
        sobre_umbral
    )

    anterior[0] = False
    anterior[1:] = sobre_umbral[:-1]

    return (
        sobre_umbral
        & ~anterior
    )


def calcular_salida_fija(
    cierres: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula la misma salida fija de 480 minutos de V4.1."""

    total = len(
        cierres
    )

    retornos = np.full(
        total,
        np.nan,
        dtype="float64",
    )

    duraciones = np.full(
        total,
        MINUTOS_SALIDA,
        dtype="int16",
    )

    limite = (
        total
        - MINUTOS_SALIDA
    )

    if limite > 0:
        retornos[:limite] = (
            cierres[
                MINUTOS_SALIDA:
            ]
            / cierres[
                :limite
            ]
            - 1.0
        )

    return retornos, duraciones


def seleccionar_operaciones(
    fechas_ns: np.ndarray,
    mascara: np.ndarray,
    minutos_salida: np.ndarray,
) -> np.ndarray:
    """Selecciona señales cronológicas sin posiciones simultáneas."""

    candidatos = np.flatnonzero(
        mascara
    )

    seleccionados: list[int] = []

    siguiente_fecha_permitida = np.iinfo(
        np.int64
    ).min

    minuto_ns = (
        60
        * 1_000_000_000
    )

    for indice in candidatos:
        fecha = int(
            fechas_ns[
                indice
            ]
        )

        if fecha < siguiente_fecha_permitida:
            continue

        seleccionados.append(
            int(indice)
        )

        duracion = max(
            1,
            int(
                minutos_salida[
                    indice
                ]
            ),
        )

        siguiente_fecha_permitida = (
            fecha
            + duracion
            * minuto_ns
        )

    return np.asarray(
        seleccionados,
        dtype="int64",
    )


def calcular_factor_beneficio(
    retornos: np.ndarray,
) -> float:
    """Calcula ganancias brutas divididas entre pérdidas brutas."""

    ganancias = float(
        retornos[
            retornos > 0
        ].sum()
    )

    perdidas = float(
        -retornos[
            retornos < 0
        ].sum()
    )

    if perdidas == 0:
        return (
            math.inf
            if ganancias > 0
            else 0.0
        )

    return (
        ganancias
        / perdidas
    )


def calcular_maximo_drawdown(
    retornos: np.ndarray,
) -> float:
    """Calcula el drawdown compuesto de las operaciones."""

    if retornos.size == 0:
        return 0.0

    capital = np.cumprod(
        1.0
        + np.clip(
            retornos,
            -0.999999,
            None,
        )
    )

    maximos = np.maximum.accumulate(
        capital
    )

    return float(
        (
            capital
            / maximos
            - 1.0
        ).min()
    )


def evaluar_datos(
    datos: pd.DataFrame,
) -> dict[str, float | int]:
    """Evalúa la estrategia V4.1 sin cambiar ningún parámetro."""

    probabilidades = datos[
        "probabilidad_sube"
    ].to_numpy(
        dtype="float64"
    )

    cierres = datos[
        "precio_cierre"
    ].to_numpy(
        dtype="float64"
    )

    rendimientos_objetivo = datos[
        "rendimiento_objetivo"
    ].to_numpy(
        dtype="float64"
    )

    objetivo_sube = (
        rendimientos_objetivo
        >= UMBRAL_CLASE_SUBE
    ).astype(
        "int8"
    )

    fechas_ns = convertir_fechas_ns(
        datos[
            "fecha_apertura"
        ]
    )

    mascara_entrada = construir_mascara_entrada(
        probabilidades_sube=probabilidades
    )

    (
        retornos_brutos,
        minutos_salida,
    ) = calcular_salida_fija(
        cierres=cierres
    )

    mascara_valida = (
        mascara_entrada
        & np.isfinite(
            retornos_brutos
        )
    )

    indices = seleccionar_operaciones(
        fechas_ns=fechas_ns,
        mascara=mascara_valida,
        minutos_salida=minutos_salida,
    )

    operaciones = int(
        len(indices)
    )

    if operaciones == 0:
        return {
            "operaciones": 0,
            "precision_clasificacion": 0.0,
            "porcentaje_acierto_clasificacion": 0.0,
            "porcentaje_operaciones_positivas": 0.0,
            "retorno_bruto_medio": 0.0,
            "retorno_neto_medio": 0.0,
            "retorno_neto_mediano": 0.0,
            "factor_beneficio": 0.0,
            "maximo_drawdown": 0.0,
        }

    retornos_brutos_seleccionados = retornos_brutos[
        indices
    ]

    retornos_netos = (
        retornos_brutos_seleccionados
        - COSTE
    )

    precision = float(
        objetivo_sube[
            indices
        ].mean()
    )

    return {
        "operaciones": operaciones,
        "precision_clasificacion": precision,
        "porcentaje_acierto_clasificacion": (
            precision
            * 100.0
        ),
        "porcentaje_operaciones_positivas": float(
            np.mean(
                retornos_netos > 0
            )
            * 100.0
        ),
        "retorno_bruto_medio": float(
            retornos_brutos_seleccionados.mean()
        ),
        "retorno_neto_medio": float(
            retornos_netos.mean()
        ),
        "retorno_neto_mediano": float(
            np.median(
                retornos_netos
            )
        ),
        "factor_beneficio": float(
            calcular_factor_beneficio(
                retornos_netos
            )
        ),
        "maximo_drawdown": float(
            calcular_maximo_drawdown(
                retornos_netos
            )
        ),
    }
