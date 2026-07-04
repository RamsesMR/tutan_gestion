from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


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


def factor_beneficio(
    retornos: np.ndarray,
) -> float:
    """Ganancias brutas divididas entre pérdidas brutas."""

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
        return math.inf if ganancias > 0 else 0.0

    return ganancias / perdidas


def maximo_drawdown(
    retornos: np.ndarray,
) -> float:
    """Drawdown compuesto de una secuencia de operaciones."""

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
            capital / maximos
            - 1.0
        ).min()
    )


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
            int(
                indice
            )
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


def racha_confirmada(
    condicion: np.ndarray,
    minutos: int,
) -> np.ndarray:
    """Verdadero cuando la condición se mantuvo durante N minutos."""

    serie = pd.Series(
        condicion.astype(
            "int8"
        )
    )

    return (
        serie.rolling(
            minutos,
            min_periods=minutos,
        ).sum()
        == minutos
    ).to_numpy(
        dtype=bool
    )


def construir_mascara_entrada(
    probabilidades_sube: np.ndarray,
    probabilidades_baja: np.ndarray,
    umbral_sube: float,
    umbral_veto_baja: float | None,
    modo: str,
) -> np.ndarray:
    """Construye una regla de entrada causal."""

    sobre_umbral = (
        probabilidades_sube
        >= umbral_sube
    )

    if modo == "nivel":
        entrada = sobre_umbral

    elif modo == "cruce":
        anterior = np.empty_like(
            sobre_umbral
        )

        anterior[
            0
        ] = False

        anterior[
            1:
        ] = sobre_umbral[
            :-1
        ]

        entrada = (
            sobre_umbral
            & ~anterior
        )

    elif modo.startswith(
        "confirmacion_"
    ):
        minutos = int(
            modo.split(
                "_"
            )[
                1
            ].replace(
                "m",
                "",
            )
        )

        confirmada = racha_confirmada(
            sobre_umbral,
            minutos,
        )

        anterior = np.empty_like(
            confirmada
        )

        anterior[
            0
        ] = False

        anterior[
            1:
        ] = confirmada[
            :-1
        ]

        entrada = (
            confirmada
            & ~anterior
        )

    elif modo.startswith(
        "creciente_"
    ):
        minutos = int(
            modo.split(
                "_"
            )[
                1
            ].replace(
                "m",
                "",
            )
        )

        diferencias = np.diff(
            probabilidades_sube,
            prepend=np.nan,
        )

        creciente = (
            diferencias > 0
        )

        entrada = (
            sobre_umbral
            & racha_confirmada(
                creciente,
                minutos,
            )
        )

    else:
        raise ValueError(
            f"Modo desconocido: {modo}"
        )

    if umbral_veto_baja is not None:
        entrada = (
            entrada
            & (
                probabilidades_baja
                < umbral_veto_baja
            )
        )

    return entrada


def calcular_salida_fija(
    cierres: np.ndarray,
    minutos: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Retorno al cierre de un horizonte fijo."""

    n = len(
        cierres
    )

    retornos = np.full(
        n,
        np.nan,
        dtype="float64",
    )

    duraciones = np.full(
        n,
        minutos,
        dtype="int16",
    )

    limite = n - minutos

    if limite > 0:
        retornos[
            :limite
        ] = (
            cierres[
                minutos:
            ]
            / cierres[
                :limite
            ]
            - 1.0
        )

    return retornos, duraciones


def calcular_salida_barrera(
    cierres: np.ndarray,
    maximos: np.ndarray,
    minimos: np.ndarray,
    take_profit: float,
    stop_loss: float,
    horizonte: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Salida por TP/SL. En empate intravela se aplica el stop."""

    n = len(
        cierres
    )

    retornos = np.full(
        n,
        np.nan,
        dtype="float64",
    )

    duraciones = np.full(
        n,
        horizonte,
        dtype="int16",
    )

    estado = np.zeros(
        n,
        dtype="int8",
    )

    for paso in range(
        1,
        horizonte + 1,
    ):
        limite = n - paso

        if limite <= 0:
            break

        activos = (
            estado[
                :limite
            ]
            == 0
        )

        toca_stop = (
            minimos[
                paso:
            ]
            <= cierres[
                :limite
            ]
            * (
                1.0
                - stop_loss
            )
        )

        toca_take = (
            maximos[
                paso:
            ]
            >= cierres[
                :limite
            ]
            * (
                1.0
                + take_profit
            )
        )

        perdedoras = np.flatnonzero(
            activos
            & toca_stop
        )

        ganadoras = np.flatnonzero(
            activos
            & ~toca_stop
            & toca_take
        )

        estado[
            perdedoras
        ] = -1

        retornos[
            perdedoras
        ] = -stop_loss

        duraciones[
            perdedoras
        ] = paso

        estado[
            ganadoras
        ] = 1

        retornos[
            ganadoras
        ] = take_profit

        duraciones[
            ganadoras
        ] = paso

    limite = n - horizonte

    if limite > 0:
        sin_barrera = np.flatnonzero(
            (
                estado[
                    :limite
                ]
                == 0
            )
        )

        retornos[
            sin_barrera
        ] = (
            cierres[
                sin_barrera
                + horizonte
            ]
            / cierres[
                sin_barrera
            ]
            - 1.0
        )

    return retornos, duraciones


def evaluar_estrategia(
    fechas_ns: np.ndarray,
    mascara_entrada: np.ndarray,
    retornos_brutos: np.ndarray,
    minutos_salida: np.ndarray,
    objetivo_sube_4h: np.ndarray,
    coste: float,
) -> dict[str, float | int]:
    """Evalúa entradas no solapadas y muestra precisión claramente."""

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
        len(
            indices
        )
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
        - coste
    )

    precision = float(
        objetivo_sube_4h[
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
            factor_beneficio(
                retornos_netos
            )
        ),
        "maximo_drawdown": float(
            maximo_drawdown(
                retornos_netos
            )
        ),
    }
