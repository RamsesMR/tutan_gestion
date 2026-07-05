from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from cripto.corto_plazo_baja_v1.configuracion import (
    COSTE_BASE,
    RUTA_DATOS_BAJA,
    SEMILLA,
    SIMBOLO,
)


CLASES = np.asarray(
    [0, 1],
    dtype="int8",
)


def ruta_datos(
    desde: str,
    hasta: str,
) -> Path:
    return (
        RUTA_DATOS_BAJA
        / f"{SIMBOLO}_1m_baja_v1_{desde}_{hasta}.parquet"
    )


def nombres_columnas_objetivo(
    objetivo: str,
) -> dict[str, str]:
    return {
        "objetivo": f"objetivo_{objetivo}",
        "resultado": f"resultado_{objetivo}",
        "retorno": f"retorno_salida_{objetivo}",
        "duracion": f"minutos_salida_{objetivo}",
        "fecha_fin": f"fecha_fin_horizonte_{objetivo}",
    }


def cargar_periodo(
    periodo: tuple[str, str],
    columnas_modelo: tuple[str, ...],
    objetivo: str,
    hasta_exclusivo: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    desde, hasta = periodo
    ruta = ruta_datos(
        desde,
        hasta,
    )

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe: {ruta}"
        )

    nombres = nombres_columnas_objetivo(
        objetivo
    )

    columnas = list(
        dict.fromkeys(
            [
                "fecha_apertura",
                "cierre",
                *columnas_modelo,
                *nombres.values(),
            ]
        )
    )

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
        nombres["fecha_fin"]
    ] = pd.to_datetime(
        datos[
            nombres["fecha_fin"]
        ],
        utc=True,
        errors="raise",
    )

    objetivo_serie = pd.to_numeric(
        datos[
            nombres["objetivo"]
        ],
        errors="coerce",
    )

    retorno_serie = pd.to_numeric(
        datos[
            nombres["retorno"]
        ],
        errors="coerce",
    )

    duracion_serie = pd.to_numeric(
        datos[
            nombres["duracion"]
        ],
        errors="coerce",
    )

    limite = pd.Timestamp(
        hasta_exclusivo,
        tz="UTC",
    )

    mascara = (
        objetivo_serie.notna()
        & retorno_serie.notna()
        & np.isfinite(
            retorno_serie
        )
        & (
            duracion_serie > 0
        )
        & (
            datos[
                nombres["fecha_fin"]
            ]
            < limite
        )
    )

    datos = (
        datos.loc[
            mascara
        ]
        .reset_index(drop=True)
    )

    x = datos[
        list(
            columnas_modelo
        )
    ].to_numpy(
        dtype="float64"
    )

    if not np.isfinite(
        x
    ).all():
        raise ValueError(
            f"Hay variables no finitas en {ruta.name}."
        )

    y = (
        pd.to_numeric(
            datos[
                nombres["objetivo"]
            ],
            errors="raise",
        )
        .to_numpy(
            dtype="int8"
        )
    )

    return x, y, datos


def recorrer_lotes(
    total: int,
    tamano_lote: int,
):
    for inicio in range(
        0,
        total,
        tamano_lote,
    ):
        yield (
            inicio,
            min(
                inicio
                + tamano_lote,
                total,
            ),
        )


def calcular_pesos_clase(
    y: np.ndarray,
    potencia_positivo: float,
) -> dict[int, float]:
    conteos = Counter(
        y.tolist()
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

    razon = (
        negativos
        / positivos
    )

    peso_positivo_bruto = (
        razon
        ** potencia_positivo
    )

    denominador = (
        negativos
        + positivos
        * peso_positivo_bruto
    )

    factor = (
        len(y)
        / denominador
    )

    return {
        0: float(factor),
        1: float(
            factor
            * peso_positivo_bruto
        ),
    }


def pesos_muestra(
    y: np.ndarray,
    pesos: dict[int, float],
) -> np.ndarray:
    return np.where(
        y == 1,
        pesos[1],
        pesos[0],
    ).astype(
        "float64"
    )


def contar_entrenamiento(
    periodos: tuple[
        tuple[str, str],
        ...
    ],
    columnas_modelo: tuple[str, ...],
    objetivo: str,
    hasta_entrenamiento: str,
) -> Counter:
    conteos: Counter = Counter()

    for periodo in periodos:
        _, y, _ = cargar_periodo(
            periodo=periodo,
            columnas_modelo=columnas_modelo,
            objetivo=objetivo,
            hasta_exclusivo=(
                hasta_entrenamiento
            ),
        )

        conteos.update(
            y.tolist()
        )

    return conteos


def pesos_desde_conteos(
    conteos: Counter,
    potencia_positivo: float,
) -> dict[int, float]:
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

    razon = (
        negativos
        / positivos
    )

    peso_positivo_bruto = (
        razon
        ** potencia_positivo
    )

    total = (
        negativos
        + positivos
    )

    denominador = (
        negativos
        + positivos
        * peso_positivo_bruto
    )

    factor = (
        total
        / denominador
    )

    return {
        0: float(factor),
        1: float(
            factor
            * peso_positivo_bruto
        ),
    }


def ajustar_escalador(
    periodos: tuple[
        tuple[str, str],
        ...
    ],
    columnas_modelo: tuple[str, ...],
    objetivo: str,
    hasta_entrenamiento: str,
    tamano_lote: int,
) -> StandardScaler:
    escalador = StandardScaler()

    for periodo in periodos:
        x, _, _ = cargar_periodo(
            periodo=periodo,
            columnas_modelo=columnas_modelo,
            objetivo=objetivo,
            hasta_exclusivo=(
                hasta_entrenamiento
            ),
        )

        for inicio, final in recorrer_lotes(
            len(x),
            tamano_lote,
        ):
            escalador.partial_fit(
                x[
                    inicio:final
                ]
            )

    return escalador


def entrenar_sgd(
    periodos: tuple[
        tuple[str, str],
        ...
    ],
    columnas_modelo: tuple[str, ...],
    objetivo: str,
    hasta_entrenamiento: str,
    configuracion: dict[str, Any],
    epocas: int,
    tamano_lote: int,
) -> tuple[
    SGDClassifier,
    StandardScaler,
    Counter,
]:
    conteos = contar_entrenamiento(
        periodos=periodos,
        columnas_modelo=columnas_modelo,
        objetivo=objetivo,
        hasta_entrenamiento=(
            hasta_entrenamiento
        ),
    )

    pesos = pesos_desde_conteos(
        conteos=conteos,
        potencia_positivo=float(
            configuracion[
                "potencia_peso_positivo"
            ]
        ),
    )

    escalador = ajustar_escalador(
        periodos=periodos,
        columnas_modelo=columnas_modelo,
        objetivo=objetivo,
        hasta_entrenamiento=(
            hasta_entrenamiento
        ),
        tamano_lote=tamano_lote,
    )

    modelo = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(
            configuracion["alpha"]
        ),
        learning_rate="optimal",
        average=True,
        random_state=SEMILLA,
    )

    primera = True

    for epoca in range(
        1,
        epocas + 1,
    ):
        print(
            f"    Época {epoca}/{epocas}"
        )

        for indice_archivo, periodo in enumerate(
            periodos
        ):
            x, y, _ = cargar_periodo(
                periodo=periodo,
                columnas_modelo=columnas_modelo,
                objetivo=objetivo,
                hasta_exclusivo=(
                    hasta_entrenamiento
                ),
            )

            generador = np.random.default_rng(
                SEMILLA
                + epoca * 100
                + indice_archivo
            )

            indices = generador.permutation(
                len(x)
            )

            for inicio, final in recorrer_lotes(
                len(indices),
                tamano_lote,
            ):
                indices_lote = indices[
                    inicio:final
                ]

                x_lote = escalador.transform(
                    x[
                        indices_lote
                    ]
                )

                y_lote = y[
                    indices_lote
                ]

                peso_lote = pesos_muestra(
                    y_lote,
                    pesos,
                )

                if primera:
                    modelo.partial_fit(
                        x_lote,
                        y_lote,
                        classes=CLASES,
                        sample_weight=(
                            peso_lote
                        ),
                    )

                    primera = False
                else:
                    modelo.partial_fit(
                        x_lote,
                        y_lote,
                        sample_weight=(
                            peso_lote
                        ),
                    )

    return (
        modelo,
        escalador,
        conteos,
    )


def muestrear_histgb(
    periodos: tuple[
        tuple[str, str],
        ...
    ],
    columnas_modelo: tuple[str, ...],
    objetivo: str,
    hasta_entrenamiento: str,
    maximo_filas: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    partes_x = []
    partes_y = []

    cuota = max(
        1,
        maximo_filas
        // len(periodos),
    )

    for indice, periodo in enumerate(
        periodos
    ):
        x, y, _ = cargar_periodo(
            periodo=periodo,
            columnas_modelo=columnas_modelo,
            objetivo=objetivo,
            hasta_exclusivo=(
                hasta_entrenamiento
            ),
        )

        if len(x) > cuota:
            generador = np.random.default_rng(
                SEMILLA
                + indice
            )

            indices = generador.choice(
                len(x),
                size=cuota,
                replace=False,
            )

            indices.sort()

            x = x[indices]
            y = y[indices]

        partes_x.append(
            x
        )

        partes_y.append(
            y
        )

    x_total = np.concatenate(
        partes_x,
        axis=0,
    )

    y_total = np.concatenate(
        partes_y,
        axis=0,
    )

    if len(x_total) > maximo_filas:
        generador = np.random.default_rng(
            SEMILLA
        )

        indices = generador.choice(
            len(x_total),
            size=maximo_filas,
            replace=False,
        )

        indices.sort()

        x_total = x_total[
            indices
        ]

        y_total = y_total[
            indices
        ]

    return (
        x_total,
        y_total,
    )


def entrenar_histgb(
    periodos: tuple[
        tuple[str, str],
        ...
    ],
    columnas_modelo: tuple[str, ...],
    objetivo: str,
    hasta_entrenamiento: str,
    configuracion: dict[str, Any],
) -> tuple[
    HistGradientBoostingClassifier,
    None,
    Counter,
]:
    x, y = muestrear_histgb(
        periodos=periodos,
        columnas_modelo=columnas_modelo,
        objetivo=objetivo,
        hasta_entrenamiento=(
            hasta_entrenamiento
        ),
        maximo_filas=int(
            configuracion[
                "max_filas_entrenamiento"
            ]
        ),
    )

    conteos = Counter(
        y.tolist()
    )

    pesos = calcular_pesos_clase(
        y=y,
        potencia_positivo=float(
            configuracion[
                "potencia_peso_positivo"
            ]
        ),
    )

    modelo = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=float(
            configuracion[
                "learning_rate"
            ]
        ),
        max_iter=int(
            configuracion[
                "max_iter"
            ]
        ),
        max_leaf_nodes=int(
            configuracion[
                "max_leaf_nodes"
            ]
        ),
        min_samples_leaf=int(
            configuracion[
                "min_samples_leaf"
            ]
        ),
        l2_regularization=float(
            configuracion[
                "l2_regularization"
            ]
        ),
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=15,
        random_state=SEMILLA,
    )

    modelo.fit(
        x,
        y,
        sample_weight=pesos_muestra(
            y,
            pesos,
        ),
    )

    return (
        modelo,
        None,
        conteos,
    )


def predecir(
    modelo: Any,
    escalador: StandardScaler | None,
    x: np.ndarray,
    tamano_lote: int,
) -> np.ndarray:
    probabilidades = []

    indice_positivo = int(
        np.flatnonzero(
            modelo.classes_ == 1
        )[0]
    )

    for inicio, final in recorrer_lotes(
        len(x),
        tamano_lote,
    ):
        lote = x[
            inicio:final
        ]

        if escalador is not None:
            lote = escalador.transform(
                lote
            )

        probabilidades.append(
            modelo.predict_proba(
                lote
            )[
                :,
                indice_positivo,
            ]
        )

    return np.concatenate(
        probabilidades
    )


def metricas_discriminacion(
    y: np.ndarray,
    probabilidades: np.ndarray,
) -> dict[str, float]:
    prevalencia = float(
        y.mean()
    )

    pr_auc = float(
        average_precision_score(
            y,
            probabilidades,
        )
    )

    roc_auc = float(
        roc_auc_score(
            y,
            probabilidades,
        )
    )

    return {
        "prevalencia": prevalencia,
        "pr_auc": pr_auc,
        "pr_auc_lift": (
            pr_auc / prevalencia
            if prevalencia > 0
            else 0.0
        ),
        "roc_auc": roc_auc,
    }


def metricas_clasificacion(
    y: np.ndarray,
    probabilidades: np.ndarray,
    umbral: float,
) -> dict[str, float | int]:
    predicho = (
        probabilidades
        >= umbral
    ).astype(
        "int8"
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        predicho,
        labels=[0, 1],
    ).ravel()

    precision = float(
        precision_score(
            y,
            predicho,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y,
            predicho,
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            y,
            predicho,
            zero_division=0,
        )
    )

    return {
        "precision": precision,
        "porcentaje_acierto": (
            precision * 100.0
        ),
        "recall": recall,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

def factor_beneficio(
    retornos: np.ndarray,
) -> float:
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
            float("inf")
            if ganancias > 0
            else 0.0
        )

    return (
        ganancias
        / perdidas
    )


def maximo_drawdown(
    retornos: np.ndarray,
) -> float:
    if retornos.size == 0:
        return 0.0

    curva = np.cumprod(
        1.0
        + np.clip(
            retornos,
            -0.999999,
            None,
        )
    )

    maximos = np.maximum.accumulate(
        curva
    )

    drawdowns = (
        curva
        / maximos
        - 1.0
    )

    return float(
        drawdowns.min()
    )


def simular_short(
    datos: pd.DataFrame,
    probabilidades: np.ndarray,
    objetivo: str,
    umbral: float,
    coste: float = COSTE_BASE,
) -> tuple[
    dict[str, float | int],
    pd.DataFrame,
]:
    nombres = nombres_columnas_objetivo(
        objetivo
    )

    fechas = pd.to_datetime(
        datos[
            "fecha_apertura"
        ],
        utc=True,
        errors="raise",
    )

    y = pd.to_numeric(
        datos[
            nombres["objetivo"]
        ],
        errors="raise",
    ).to_numpy(
        dtype="int8"
    )

    retornos_brutos = pd.to_numeric(
        datos[
            nombres["retorno"]
        ],
        errors="raise",
    ).to_numpy(
        dtype="float64"
    )

    duraciones = pd.to_numeric(
        datos[
            nombres["duracion"]
        ],
        errors="raise",
    ).to_numpy(
        dtype="int32"
    )

    sobre = (
        probabilidades
        >= umbral
    )

    anterior = np.empty_like(
        sobre
    )

    anterior[0] = False
    anterior[1:] = sobre[:-1]

    cruces = (
        sobre
        & ~anterior
    )

    operaciones = []

    siguiente_fecha_permitida = (
        pd.Timestamp.min.tz_localize(
            "UTC"
        )
    )

    for indice in np.flatnonzero(
        cruces
    ):
        fecha_entrada = fechas.iloc[
            indice
        ]

        if (
            fecha_entrada
            < siguiente_fecha_permitida
        ):
            continue

        duracion = int(
            duraciones[indice]
        )

        retorno_bruto = float(
            retornos_brutos[indice]
        )

        if (
            duracion <= 0
            or not np.isfinite(
                retorno_bruto
            )
        ):
            continue

        fecha_salida = (
            fecha_entrada
            + pd.Timedelta(
                minutes=duracion
            )
        )

        retorno_neto = (
            retorno_bruto
            - coste
        )

        operaciones.append(
            {
                "fecha_entrada": fecha_entrada,
                "fecha_salida": fecha_salida,
                "duracion_minutos": duracion,
                "probabilidad": float(
                    probabilidades[indice]
                ),
                "objetivo_positivo": int(
                    y[indice]
                ),
                "retorno_bruto": retorno_bruto,
                "retorno_neto": retorno_neto,
            }
        )

        siguiente_fecha_permitida = (
            fecha_salida
        )

    if not operaciones:
        return (
            {
                "operaciones": 0,
                "precision_operaciones": 0.0,
                "porcentaje_acierto_operaciones": 0.0,
                "porcentaje_operaciones_positivas": 0.0,
                "retorno_neto_medio": 0.0,
                "retorno_neto_mediano": 0.0,
                "factor_beneficio": 0.0,
                "maximo_drawdown": 0.0,
                "retorno_compuesto": 0.0,
                "duracion_media_minutos": 0.0,
                "concentracion_mensual_maxima": 0.0,
            },
            pd.DataFrame(),
        )

    tabla = pd.DataFrame(
        operaciones
    )

    retornos = tabla[
        "retorno_neto"
    ].to_numpy(
        dtype="float64"
    )

    precision = float(
        tabla[
            "objetivo_positivo"
        ].mean()
    )

    tabla[
        "mes"
    ] = tabla[
        "fecha_entrada"
    ].dt.to_period(
        "M"
    ).astype(str)

    mensual = (
        tabla.groupby(
            "mes",
            as_index=False,
        )[
            "retorno_neto"
        ]
        .sum()
    )

    total_absoluto = float(
        mensual[
            "retorno_neto"
        ].abs().sum()
    )

    concentracion = (
        float(
            mensual[
                "retorno_neto"
            ].abs().max()
            / total_absoluto
        )
        if total_absoluto > 0
        else 0.0
    )

    metricas = {
        "operaciones": len(
            tabla
        ),
        "precision_operaciones": precision,
        "porcentaje_acierto_operaciones": (
            precision * 100.0
        ),
        "porcentaje_operaciones_positivas": float(
            (
                retornos > 0
            ).mean()
            * 100.0
        ),
        "retorno_neto_medio": float(
            retornos.mean()
        ),
        "retorno_neto_mediano": float(
            np.median(
                retornos
            )
        ),
        "factor_beneficio": factor_beneficio(
            retornos
        ),
        "maximo_drawdown": maximo_drawdown(
            retornos
        ),
        "retorno_compuesto": float(
            np.prod(
                1.0
                + np.clip(
                    retornos,
                    -0.999999,
                    None,
                )
            )
            - 1.0
        ),
        "duracion_media_minutos": float(
            tabla[
                "duracion_minutos"
            ].mean()
        ),
        "concentracion_mensual_maxima": concentracion,
    }

    return (
        metricas,
        tabla,
    )
