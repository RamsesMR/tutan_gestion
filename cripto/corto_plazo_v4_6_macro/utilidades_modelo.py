from __future__ import annotations

import gc
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from cripto.corto_plazo_v4_6_macro.configuracion import (
    COLUMNAS_META,
    RUTA_DATOS_INTEGRADOS,
    UMBRAL_CLASE,
)


CLASES_BINARIAS = np.asarray(
    [
        0,
        1,
    ],
    dtype="int8",
)


def ruta_datos(
    base: str,
    simbolo: str,
    desde: str,
    hasta: str,
) -> Path:
    return (
        RUTA_DATOS_INTEGRADOS
        / base
        / f"{simbolo}_1m_4h_{desde}_{hasta}_variables.parquet"
    )


def cargar_datos(
    ruta: Path,
    columnas_modelo: tuple[str, ...],
    desde: pd.Timestamp,
    hasta: pd.Timestamp,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    columnas = list(
        dict.fromkeys(
            [
                *COLUMNAS_META,
                *columnas_modelo,
            ]
        )
    )

    datos = pd.read_parquet(
        ruta,
        columns=columnas,
    )

    datos["fecha_apertura"] = pd.to_datetime(
        datos["fecha_apertura"],
        utc=True,
        errors="raise",
    )

    datos["fecha_objetivo"] = pd.to_datetime(
        datos["fecha_objetivo"],
        utc=True,
        errors="raise",
    )

    mascara = (
        (
            datos["fecha_apertura"]
            >= desde
        )
        & (
            datos["fecha_apertura"]
            < hasta
        )
        & (
            datos["fecha_objetivo"]
            > datos["fecha_apertura"]
        )
        & (
            datos["fecha_objetivo"]
            < hasta
        )
    )

    datos = (
        datos.loc[
            mascara
        ]
        .reset_index(drop=True)
    )

    variables = datos[
        list(columnas_modelo)
    ].to_numpy(dtype="float64")

    if not np.isfinite(
        variables
    ).all():
        raise ValueError(
            f"Hay variables no finitas en {ruta.name}."
        )

    objetivo = (
        datos[
            "rendimiento_objetivo"
        ]
        .to_numpy(dtype="float64")
        >= UMBRAL_CLASE
    ).astype("int8")

    return variables, objetivo, datos


def recorrer_lotes(
    total: int,
    tamano_lote: int,
):
    for inicio in range(
        0,
        total,
        tamano_lote,
    ):
        yield inicio, min(
            inicio + tamano_lote,
            total,
        )


def ajustar_escalador(
    base: str,
    simbolo: str,
    configuracion_periodo: dict[str, Any],
    columnas_modelo: tuple[str, ...],
    tamano_lote: int,
) -> tuple[
    StandardScaler,
    Counter,
]:
    escalador = StandardScaler()
    conteos: Counter = Counter()

    desde = pd.Timestamp(
        configuracion_periodo[
            "entrenamiento_desde"
        ],
        tz="UTC",
    )

    hasta = pd.Timestamp(
        configuracion_periodo[
            "entrenamiento_hasta"
        ],
        tz="UTC",
    )

    for desde_archivo, hasta_archivo in (
        configuracion_periodo[
            "archivos_entrenamiento"
        ]
    ):
        ruta = ruta_datos(
            base,
            simbolo,
            desde_archivo,
            hasta_archivo,
        )

        print(
            f"  Escalador: {ruta.name}"
        )

        variables, objetivo, _ = cargar_datos(
            ruta,
            columnas_modelo,
            desde,
            hasta,
        )

        conteos.update(
            objetivo.tolist()
        )

        for inicio, final in recorrer_lotes(
            len(variables),
            tamano_lote,
        ):
            escalador.partial_fit(
                variables[
                    inicio:final
                ]
            )

        del variables
        del objetivo
        gc.collect()

    return escalador, conteos


def entrenar_modelo(
    base: str,
    simbolo: str,
    configuracion_periodo: dict[str, Any],
    columnas_modelo: tuple[str, ...],
    escalador: StandardScaler,
    epocas: int,
    tamano_lote: int,
    semilla: int = 42,
) -> SGDClassifier:
    modelo = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        learning_rate="optimal",
        class_weight={
            0: 1.0,
            1: 1.0,
        },
        average=True,
        random_state=semilla,
    )

    desde = pd.Timestamp(
        configuracion_periodo[
            "entrenamiento_desde"
        ],
        tz="UTC",
    )

    hasta = pd.Timestamp(
        configuracion_periodo[
            "entrenamiento_hasta"
        ],
        tz="UTC",
    )

    primera_actualizacion = True

    for epoca in range(
        1,
        epocas + 1,
    ):
        print(
            f"  Época {epoca}/{epocas}"
        )

        for indice_archivo, (
            desde_archivo,
            hasta_archivo,
        ) in enumerate(
            configuracion_periodo[
                "archivos_entrenamiento"
            ]
        ):
            ruta = ruta_datos(
                base,
                simbolo,
                desde_archivo,
                hasta_archivo,
            )

            print(
                f"    Entrenando: {ruta.name}"
            )

            variables, objetivo, _ = cargar_datos(
                ruta,
                columnas_modelo,
                desde,
                hasta,
            )

            generador = np.random.default_rng(
                semilla
                + epoca * 100
                + indice_archivo
            )

            indices = generador.permutation(
                len(variables)
            )

            for inicio, final in recorrer_lotes(
                len(indices),
                tamano_lote,
            ):
                indices_lote = indices[
                    inicio:final
                ]

                variables_lote = (
                    escalador.transform(
                        variables[
                            indices_lote
                        ]
                    )
                )

                objetivo_lote = objetivo[
                    indices_lote
                ]

                if primera_actualizacion:
                    modelo.partial_fit(
                        variables_lote,
                        objetivo_lote,
                        classes=CLASES_BINARIAS,
                    )
                    primera_actualizacion = False
                else:
                    modelo.partial_fit(
                        variables_lote,
                        objetivo_lote,
                    )

            del variables
            del objetivo
            del indices
            gc.collect()

    return modelo


def predecir_validacion(
    base: str,
    simbolo: str,
    configuracion_periodo: dict[str, Any],
    columnas_modelo: tuple[str, ...],
    modelo: SGDClassifier,
    escalador: StandardScaler,
    tamano_lote: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    desde_archivo, hasta_archivo = (
        configuracion_periodo[
            "archivo_validacion"
        ]
    )

    ruta = ruta_datos(
        base,
        simbolo,
        desde_archivo,
        hasta_archivo,
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

    variables, objetivo, datos = cargar_datos(
        ruta,
        columnas_modelo,
        desde,
        hasta,
    )

    indice_positivo = int(
        np.flatnonzero(
            modelo.classes_ == 1
        )[0]
    )

    probabilidades: list[
        np.ndarray
    ] = []

    for inicio, final in recorrer_lotes(
        len(variables),
        tamano_lote,
    ):
        lote = escalador.transform(
            variables[
                inicio:final
            ]
        )

        probabilidades.append(
            modelo.predict_proba(
                lote
            )[
                :,
                indice_positivo,
            ]
        )

    return (
        np.concatenate(
            probabilidades
        ),
        objetivo,
        datos,
    )
