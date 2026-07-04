from __future__ import annotations

import gc
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def resolver_columna(
    columnas: Iterable[str],
    alias: tuple[str, ...],
) -> str | None:
    """Encuentra una columna ignorando mayúsculas y minúsculas."""

    mapa = {
        str(columna).lower(): str(columna)
        for columna in columnas
    }

    for candidata in alias:
        encontrada = mapa.get(
            candidata.lower()
        )

        if encontrada is not None:
            return encontrada

    return None


def recorrer_lotes(
    total: int,
    tamano_lote: int,
):
    """Genera límites de lotes."""

    for inicio in range(
        0,
        total,
        tamano_lote,
    ):
        yield (
            inicio,
            min(
                inicio + tamano_lote,
                total,
            ),
        )


def calcular_pesos_binarios(
    objetivo: np.ndarray,
    potencia_positivo: float,
) -> dict[int, float]:
    """Calcula pesos normalizados para una clase positiva minoritaria."""

    conteos = Counter(
        objetivo.tolist()
    )

    negativos = int(
        conteos[0]
    )

    positivos = int(
        conteos[1]
    )

    if negativos == 0 or positivos == 0:
        raise ValueError(
            "El objetivo necesita positivos y negativos."
        )

    razon = negativos / positivos

    peso_negativo_bruto = 1.0
    peso_positivo_bruto = razon ** potencia_positivo

    total = negativos + positivos

    normalizador = total / (
        negativos * peso_negativo_bruto
        + positivos * peso_positivo_bruto
    )

    return {
        0: peso_negativo_bruto * normalizador,
        1: peso_positivo_bruto * normalizador,
    }


def calcular_metricas_probabilidad(
    objetivo: np.ndarray,
    probabilidades: np.ndarray,
) -> dict[str, float]:
    """Calcula métricas que no dependen del umbral."""

    prevalencia = float(
        objetivo.mean()
    )

    pr_auc = float(
        average_precision_score(
            objetivo,
            probabilidades,
        )
    )

    return {
        "muestras": int(
            len(
                objetivo
            )
        ),
        "positivos": int(
            objetivo.sum()
        ),
        "prevalencia": prevalencia,
        "pr_auc": pr_auc,
        "pr_auc_lift": (
            pr_auc / prevalencia
            if prevalencia > 0
            else math.nan
        ),
        "roc_auc": float(
            roc_auc_score(
                objetivo,
                probabilidades,
            )
        ),
        "brier": float(
            brier_score_loss(
                objetivo,
                probabilidades,
            )
        ),
        "log_loss": float(
            log_loss(
                objetivo,
                probabilidades,
                labels=[
                    0,
                    1,
                ],
            )
        ),
    }


def seleccionar_indices_no_solapados(
    fechas_ns: np.ndarray,
    mascara: np.ndarray,
    minutos_bloqueo: int,
) -> np.ndarray:
    """Conserva la primera señal y bloquea nuevas entradas hasta la salida."""

    candidatos = np.flatnonzero(
        mascara
    )

    if candidatos.size == 0:
        return np.array(
            [],
            dtype="int64",
        )

    tiempos = fechas_ns[
        candidatos
    ]

    separacion = int(
        minutos_bloqueo
        * 60
        * 1_000_000_000
    )

    seleccionados: list[int] = []
    posicion = 0

    while posicion < len(
        candidatos
    ):
        indice = int(
            candidatos[
                posicion
            ]
        )

        seleccionados.append(
            indice
        )

        siguiente = (
            fechas_ns[
                indice
            ]
            + separacion
        )

        posicion = int(
            np.searchsorted(
                tiempos,
                siguiente,
                side="left",
            )
        )

    return np.asarray(
        seleccionados,
        dtype="int64",
    )


def maximo_drawdown(
    retornos: np.ndarray,
) -> float:
    """Calcula drawdown compuesto."""

    if retornos.size == 0:
        return 0.0

    retornos = np.clip(
        retornos,
        -0.999999,
        None,
    )

    capital = np.cumprod(
        1.0
        + retornos
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


def factor_beneficio(
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
        return math.inf if ganancias > 0 else 0.0

    return ganancias / perdidas


def evaluar_umbral(
    objetivo: np.ndarray,
    probabilidades: np.ndarray,
    fechas_ns: np.ndarray,
    retornos_salida: np.ndarray,
    umbral: float,
    horizonte_minutos: int,
    coste: float,
) -> dict[str, float | int]:
    """Evalúa clasificación y operaciones no solapadas."""

    senal = probabilidades >= umbral

    precision = float(
        precision_score(
            objetivo,
            senal.astype(
                "int8"
            ),
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            objetivo,
            senal.astype(
                "int8"
            ),
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            objetivo,
            senal.astype(
                "int8"
            ),
            zero_division=0,
        )
    )

    indices = seleccionar_indices_no_solapados(
        fechas_ns=fechas_ns,
        mascara=senal,
        minutos_bloqueo=horizonte_minutos,
    )

    retornos = (
        retornos_salida[
            indices
        ]
        - coste
    )

    if retornos.size:
        retorno_medio = float(
            retornos.mean()
        )

        retorno_mediano = float(
            np.median(
                retornos
            )
        )

        tasa_positiva = float(
            np.mean(
                retornos > 0
            )
        )

    else:
        retorno_medio = 0.0
        retorno_mediano = 0.0
        tasa_positiva = 0.0

    if len(
        fechas_ns
    ) > 1:
        dias = max(
            (
                fechas_ns.max()
                - fechas_ns.min()
            )
            / (
                86_400
                * 1_000_000_000
            ),
            1.0,
        )
    else:
        dias = 1.0

    prevalencia = float(
        objetivo.mean()
    )

    return {
        "umbral": float(
            umbral
        ),
        "coste": float(
            coste
        ),
        "senales": int(
            senal.sum()
        ),
        "cobertura": float(
            senal.mean()
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_lift": (
            precision / prevalencia
            if prevalencia > 0
            else math.nan
        ),
        "operaciones_no_solapadas": int(
            len(
                indices
            )
        ),
        "operaciones_por_dia": float(
            len(
                indices
            )
            / dias
        ),
        "tasa_retorno_positivo": tasa_positiva,
        "retorno_neto_medio": retorno_medio,
        "retorno_neto_mediano": retorno_mediano,
        "factor_beneficio": float(
            factor_beneficio(
                retornos
            )
        ),
        "maximo_drawdown": maximo_drawdown(
            retornos
        ),
    }


def limpiar_memoria(*objetos: Any) -> None:
    """Elimina referencias y fuerza recolección."""

    for objeto in objetos:
        del objeto

    gc.collect()
