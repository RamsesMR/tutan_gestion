from __future__ import annotations

import gc
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss as perdida_logaritmica,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from cripto.corto_plazo_v4.configuracion import (
    HORIZONTE_MINUTOS,
    NOMBRE_HORIZONTE,
    RUTA_DATOS_V2,
    UMBRAL_CLASE,
)


CLASES_BINARIAS = np.array(
    [
        0,
        1,
    ],
    dtype="int8",
)


def construir_ruta_datos(
    simbolo: str,
    desde: str,
    hasta: str,
) -> Path:
    """Construye la ruta de un Parquet con variables V2A."""

    nombre = (
        f"{simbolo}_1m_{NOMBRE_HORIZONTE}_"
        f"{desde}_{hasta}_variables.parquet"
    )

    return RUTA_DATOS_V2 / nombre


def construir_objetivo_binario(
    rendimientos: np.ndarray,
    detector: str,
) -> np.ndarray:
    """Convierte el rendimiento futuro en BAJA/NO BAJA o SUBE/NO SUBE."""

    if detector == "BAJA":
        objetivo = rendimientos <= -UMBRAL_CLASE

    elif detector == "SUBE":
        objetivo = rendimientos >= UMBRAL_CLASE

    else:
        raise ValueError(
            f"Detector desconocido: {detector}"
        )

    return objetivo.astype(
        "int8"
    )


def ajustar_retorno_a_direccion(
    rendimientos: np.ndarray,
    detector: str,
) -> np.ndarray:
    """Convierte el retorno futuro a retorno favorable para el detector."""

    if detector == "BAJA":
        return -rendimientos

    if detector == "SUBE":
        return rendimientos

    raise ValueError(
        f"Detector desconocido: {detector}"
    )


def cargar_datos(
    ruta: Path,
    columnas_modelo: tuple[str, ...],
    detector: str,
    desde: pd.Timestamp,
    hasta: pd.Timestamp,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Carga un periodo y purga objetivos que crucen el corte temporal."""

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {ruta}"
        )

    columnas = [
        *columnas_modelo,
        "fecha_apertura",
        "fecha_objetivo",
        "rendimiento_objetivo",
    ]

    datos = pd.read_parquet(
        ruta,
        columns=columnas,
    )

    datos["fecha_apertura"] = pd.to_datetime(
        datos["fecha_apertura"],
        utc=True,
        errors="coerce",
    )

    datos["fecha_objetivo"] = pd.to_datetime(
        datos["fecha_objetivo"],
        utc=True,
        errors="coerce",
    )

    if datos[
        [
            "fecha_apertura",
            "fecha_objetivo",
        ]
    ].isna().any().any():
        raise ValueError(
            f"{ruta.name}: contiene fechas inválidas."
        )

    mascara = (
        (datos["fecha_apertura"] >= desde)
        & (datos["fecha_apertura"] < hasta)
        & (datos["fecha_objetivo"] > datos["fecha_apertura"])
        & (datos["fecha_objetivo"] < hasta)
    )

    datos = datos.loc[
        mascara
    ].reset_index(
        drop=True
    )

    if datos.empty:
        raise ValueError(
            f"{ruta.name}: no quedaron muestras después de la purga."
        )

    variables = datos[
        list(
            columnas_modelo
        )
    ].to_numpy(
        dtype="float32"
    )

    if not np.isfinite(
        variables
    ).all():
        raise ValueError(
            f"{ruta.name}: contiene variables nulas o infinitas."
        )

    rendimientos = pd.to_numeric(
        datos["rendimiento_objetivo"],
        errors="coerce",
    ).to_numpy(
        dtype="float64"
    )

    if not np.isfinite(
        rendimientos
    ).all():
        raise ValueError(
            f"{ruta.name}: contiene objetivos nulos o infinitos."
        )

    objetivo = construir_objetivo_binario(
        rendimientos=rendimientos,
        detector=detector,
    )

    # PyArrow puede devolver datetime64[us, UTC].
    # Convertir directamente con astype("int64") conservaría microsegundos,
    # mientras que el filtro de operaciones trabaja en nanosegundos.
    # Forzamos explícitamente datetime64[ns] para evitar un cooldown
    # mil veces mayor de lo previsto.
    fechas_sin_zona = (
        datos["fecha_apertura"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )

    fechas_ns = (
        fechas_sin_zona
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64")
    )

    if fechas_ns.size > 0 and fechas_ns.max() < 100_000_000_000_000_000:
        raise RuntimeError(
            f"{ruta.name}: las fechas no quedaron expresadas en nanosegundos."
        )

    del datos
    gc.collect()

    return (
        variables,
        objetivo,
        fechas_ns,
        rendimientos,
    )


def recorrer_lotes(
    total: int,
    tamano_lote: int,
) -> Iterable[tuple[int, int]]:
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
    conteos: Counter,
    potencia_peso_positivo: float,
) -> dict[int, float]:
    """Calcula pesos normalizados para la clase positiva."""

    negativos = int(
        conteos[0]
    )

    positivos = int(
        conteos[1]
    )

    if negativos <= 0 or positivos <= 0:
        raise ValueError(
            "El entrenamiento necesita ejemplos positivos y negativos."
        )

    proporcion = negativos / positivos

    peso_negativo_bruto = 1.0
    peso_positivo_bruto = proporcion ** potencia_peso_positivo

    total = negativos + positivos

    normalizador = total / (
        negativos * peso_negativo_bruto
        + positivos * peso_positivo_bruto
    )

    return {
        0: peso_negativo_bruto * normalizador,
        1: peso_positivo_bruto * normalizador,
    }


def ajustar_escalador_y_contar(
    simbolo: str,
    detector: str,
    configuracion_periodo: dict[str, Any],
    columnas_modelo: tuple[str, ...],
    tamano_lote: int,
) -> tuple[StandardScaler, Counter]:
    """Ajusta el escalador solo con datos de entrenamiento."""

    escalador = StandardScaler()
    conteos: Counter = Counter()

    desde = pd.Timestamp(
        configuracion_periodo["entrenamiento_desde"],
        tz="UTC",
    )

    hasta = pd.Timestamp(
        configuracion_periodo["entrenamiento_hasta"],
        tz="UTC",
    )

    for desde_archivo, hasta_archivo in configuracion_periodo[
        "archivos_entrenamiento"
    ]:
        ruta = construir_ruta_datos(
            simbolo=simbolo,
            desde=desde_archivo,
            hasta=hasta_archivo,
        )

        print(
            f"  Escalador: {ruta.name}"
        )

        variables, objetivo, _, _ = cargar_datos(
            ruta=ruta,
            columnas_modelo=columnas_modelo,
            detector=detector,
            desde=desde,
            hasta=hasta,
        )

        conteos.update(
            objetivo.tolist()
        )

        for inicio, final in recorrer_lotes(
            len(
                variables
            ),
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


def entrenar_modelo_binario(
    simbolo: str,
    detector: str,
    configuracion_periodo: dict[str, Any],
    configuracion_variante: dict[str, Any],
    escalador: StandardScaler,
    pesos: dict[int, float],
    epocas: int,
    tamano_lote: int,
) -> SGDClassifier:
    """Entrena un detector binario incremental."""

    columnas_modelo = tuple(
        configuracion_variante["columnas"]
    )

    modelo = SGDClassifier(
        loss="log_loss",
        penalty=configuracion_variante["penalty"],
        alpha=float(
            configuracion_variante["alpha"]
        ),
        learning_rate="optimal",
        class_weight=pesos,
        average=True,
        random_state=42,
    )

    desde = pd.Timestamp(
        configuracion_periodo["entrenamiento_desde"],
        tz="UTC",
    )

    hasta = pd.Timestamp(
        configuracion_periodo["entrenamiento_hasta"],
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
            ruta = construir_ruta_datos(
                simbolo=simbolo,
                desde=desde_archivo,
                hasta=hasta_archivo,
            )

            print(
                f"    Entrenando: {ruta.name}"
            )

            variables, objetivo, _, _ = cargar_datos(
                ruta=ruta,
                columnas_modelo=columnas_modelo,
                detector=detector,
                desde=desde,
                hasta=hasta,
            )

            generador = np.random.default_rng(
                42
                + epoca * 100
                + indice_archivo
            )

            indices = generador.permutation(
                len(
                    variables
                )
            )

            for inicio, final in recorrer_lotes(
                len(
                    indices
                ),
                tamano_lote,
            ):
                indices_lote = indices[
                    inicio:final
                ]

                variables_lote = escalador.transform(
                    variables[
                        indices_lote
                    ]
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


def predecir_periodo(
    simbolo: str,
    detector: str,
    configuracion_periodo: dict[str, Any],
    columnas_modelo: tuple[str, ...],
    modelo: SGDClassifier,
    escalador: StandardScaler,
    tamano_lote: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Predice el periodo de validación."""

    desde_archivo, hasta_archivo = configuracion_periodo[
        "archivo_validacion"
    ]

    ruta = construir_ruta_datos(
        simbolo=simbolo,
        desde=desde_archivo,
        hasta=hasta_archivo,
    )

    desde = pd.Timestamp(
        configuracion_periodo["validacion_desde"],
        tz="UTC",
    )

    hasta = pd.Timestamp(
        configuracion_periodo["validacion_hasta"],
        tz="UTC",
    )

    variables, objetivo, fechas_ns, rendimientos = cargar_datos(
        ruta=ruta,
        columnas_modelo=columnas_modelo,
        detector=detector,
        desde=desde,
        hasta=hasta,
    )

    probabilidades: list[np.ndarray] = []

    indice_positivo = int(
        np.flatnonzero(
            modelo.classes_ == 1
        )[0]
    )

    for inicio, final in recorrer_lotes(
        len(
            variables
        ),
        tamano_lote,
    ):
        variables_lote = escalador.transform(
            variables[
                inicio:final
            ]
        )

        probabilidades.append(
            modelo.predict_proba(
                variables_lote
            )[
                :,
                indice_positivo
            ]
        )

    del variables
    gc.collect()

    return (
        objetivo,
        np.concatenate(
            probabilidades
        ),
        fechas_ns,
        rendimientos,
    )


def calcular_metricas_generales(
    objetivo: np.ndarray,
    probabilidades: np.ndarray,
) -> dict[str, float]:
    """Calcula métricas independientes del umbral y métricas a 0.50."""

    prevalencia = float(
        np.mean(
            objetivo
        )
    )

    pr_auc = float(
        average_precision_score(
            objetivo,
            probabilidades,
        )
    )

    roc_auc = float(
        roc_auc_score(
            objetivo,
            probabilidades,
        )
    )

    prediccion = (
        probabilidades >= 0.50
    ).astype(
        "int8"
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
        "roc_auc": roc_auc,
        "brier": float(
            brier_score_loss(
                objetivo,
                probabilidades,
            )
        ),
        "log_loss": float(
            perdida_logaritmica(
                objetivo,
                probabilidades,
                labels=[
                    0,
                    1,
                ],
            )
        ),
        "precision_050": float(
            precision_score(
                objetivo,
                prediccion,
                zero_division=0,
            )
        ),
        "recall_050": float(
            recall_score(
                objetivo,
                prediccion,
                zero_division=0,
            )
        ),
        "f1_050": float(
            f1_score(
                objetivo,
                prediccion,
                zero_division=0,
            )
        ),
        "balanced_accuracy_050": float(
            balanced_accuracy_score(
                objetivo,
                prediccion,
            )
        ),
        "tasa_senales_050": float(
            np.mean(
                prediccion
            )
        ),
    }


def seleccionar_indices_no_solapados(
    fechas_ns: np.ndarray,
    mascara_senal: np.ndarray,
    horizonte_minutos: int = HORIZONTE_MINUTOS,
) -> np.ndarray:
    """Selecciona señales cronológicas sin posiciones de 4h solapadas."""

    candidatos = np.flatnonzero(
        mascara_senal
    )

    if candidatos.size == 0:
        return np.array(
            [],
            dtype="int64",
        )

    tiempos_candidatos = fechas_ns[
        candidatos
    ]

    separacion_ns = int(
        horizonte_minutos
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

        siguiente_tiempo = (
            fechas_ns[
                indice
            ]
            + separacion_ns
        )

        posicion = int(
            np.searchsorted(
                tiempos_candidatos,
                siguiente_tiempo,
                side="left",
            )
        )

    return np.asarray(
        seleccionados,
        dtype="int64",
    )


def calcular_maximo_drawdown(
    retornos: np.ndarray,
) -> float:
    """Calcula drawdown sobre una secuencia no solapada de retornos."""

    if retornos.size == 0:
        return 0.0

    retornos_seguros = np.clip(
        retornos,
        -0.999999,
        None,
    )

    capital = np.cumprod(
        1.0
        + retornos_seguros
    )

    maximos = np.maximum.accumulate(
        capital
    )

    drawdowns = (
        capital / maximos
        - 1.0
    )

    return float(
        drawdowns.min()
    )


def calcular_factor_beneficio(
    retornos: np.ndarray,
) -> float:
    """Calcula suma de ganancias entre suma absoluta de pérdidas."""

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
    rendimientos: np.ndarray,
    detector: str,
    umbral: float,
    coste_operacion: float,
) -> dict[str, float | int]:
    """Evalúa un umbral sobre todas las filas y señales no solapadas."""

    prediccion = probabilidades >= umbral

    verdaderos_positivos = int(
        np.sum(
            prediccion
            & (
                objetivo == 1
            )
        )
    )

    falsos_positivos = int(
        np.sum(
            prediccion
            & (
                objetivo == 0
            )
        )
    )

    falsos_negativos = int(
        np.sum(
            ~prediccion
            & (
                objetivo == 1
            )
        )
    )

    verdaderos_negativos = int(
        np.sum(
            ~prediccion
            & (
                objetivo == 0
            )
        )
    )

    precision = (
        verdaderos_positivos
        / (
            verdaderos_positivos
            + falsos_positivos
        )
        if (
            verdaderos_positivos
            + falsos_positivos
        ) > 0
        else 0.0
    )

    recall = (
        verdaderos_positivos
        / (
            verdaderos_positivos
            + falsos_negativos
        )
        if (
            verdaderos_positivos
            + falsos_negativos
        ) > 0
        else 0.0
    )

    especificidad = (
        verdaderos_negativos
        / (
            verdaderos_negativos
            + falsos_positivos
        )
        if (
            verdaderos_negativos
            + falsos_positivos
        ) > 0
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        ) > 0
        else 0.0
    )

    prevalencia = float(
        np.mean(
            objetivo
        )
    )

    senales = int(
        prediccion.sum()
    )

    retornos_ajustados = ajustar_retorno_a_direccion(
        rendimientos=rendimientos,
        detector=detector,
    )

    if senales > 0:
        retorno_bruto_medio = float(
            retornos_ajustados[
                prediccion
            ].mean()
        )

        retorno_bruto_mediano = float(
            np.median(
                retornos_ajustados[
                    prediccion
                ]
            )
        )

    else:
        retorno_bruto_medio = 0.0
        retorno_bruto_mediano = 0.0

    indices_no_solapados = seleccionar_indices_no_solapados(
        fechas_ns=fechas_ns,
        mascara_senal=prediccion,
    )

    operaciones = int(
        len(
            indices_no_solapados
        )
    )

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

    if operaciones > 0:
        objetivo_operaciones = objetivo[
            indices_no_solapados
        ]

        retornos_brutos_operaciones = retornos_ajustados[
            indices_no_solapados
        ]

        retornos_netos_operaciones = (
            retornos_brutos_operaciones
            - coste_operacion
        )

        precision_objetivo_no_solapada = float(
            objetivo_operaciones.mean()
        )

        tasa_retorno_positivo = float(
            np.mean(
                retornos_netos_operaciones > 0
            )
        )

        retorno_bruto_medio_no_solapado = float(
            retornos_brutos_operaciones.mean()
        )

        retorno_neto_medio_no_solapado = float(
            retornos_netos_operaciones.mean()
        )

        retorno_neto_mediano_no_solapado = float(
            np.median(
                retornos_netos_operaciones
            )
        )

        factor_beneficio = float(
            calcular_factor_beneficio(
                retornos_netos_operaciones
            )
        )

        maximo_drawdown = calcular_maximo_drawdown(
            retornos_netos_operaciones
        )

    else:
        precision_objetivo_no_solapada = 0.0
        tasa_retorno_positivo = 0.0
        retorno_bruto_medio_no_solapado = 0.0
        retorno_neto_medio_no_solapado = 0.0
        retorno_neto_mediano_no_solapado = 0.0
        factor_beneficio = 0.0
        maximo_drawdown = 0.0

    return {
        "umbral": float(
            umbral
        ),
        "prevalencia": prevalencia,
        "verdaderos_positivos": verdaderos_positivos,
        "falsos_positivos": falsos_positivos,
        "falsos_negativos": falsos_negativos,
        "verdaderos_negativos": verdaderos_negativos,
        "senales": senales,
        "cobertura": (
            senales / len(
                objetivo
            )
        ),
        "precision": precision,
        "recall": recall,
        "especificidad": especificidad,
        "balanced_accuracy": (
            recall
            + especificidad
        ) / 2,
        "f1": f1,
        "precision_lift": (
            precision / prevalencia
            if prevalencia > 0
            else math.nan
        ),
        "retorno_bruto_medio_todas_senales": retorno_bruto_medio,
        "retorno_bruto_mediano_todas_senales": retorno_bruto_mediano,
        "operaciones_no_solapadas": operaciones,
        "operaciones_no_solapadas_por_dia": operaciones / dias,
        "precision_objetivo_no_solapada": precision_objetivo_no_solapada,
        "tasa_retorno_positivo_no_solapada": tasa_retorno_positivo,
        "retorno_bruto_medio_no_solapado": retorno_bruto_medio_no_solapado,
        "retorno_neto_medio_no_solapado": retorno_neto_medio_no_solapado,
        "retorno_neto_mediano_no_solapado": retorno_neto_mediano_no_solapado,
        "factor_beneficio_no_solapado": factor_beneficio,
        "maximo_drawdown_no_solapado": maximo_drawdown,
        "coste_operacion": float(
            coste_operacion
        ),
    }


def construir_tabla_umbrales(
    objetivo: np.ndarray,
    probabilidades: np.ndarray,
    fechas_ns: np.ndarray,
    rendimientos: np.ndarray,
    detector: str,
    umbrales: tuple[float, ...],
    coste_operacion: float,
) -> pd.DataFrame:
    """Evalúa todos los umbrales comunes."""

    registros = [
        evaluar_umbral(
            objetivo=objetivo,
            probabilidades=probabilidades,
            fechas_ns=fechas_ns,
            rendimientos=rendimientos,
            detector=detector,
            umbral=umbral,
            coste_operacion=coste_operacion,
        )
        for umbral in umbrales
    ]

    return pd.DataFrame(
        registros
    )


def construir_tabla_coeficientes(
    modelo: SGDClassifier,
    columnas_modelo: tuple[str, ...],
) -> pd.DataFrame:
    """Construye una tabla ordenada de coeficientes binarios."""

    coeficientes = modelo.coef_[
        0
    ]

    tabla = pd.DataFrame(
        {
            "variable": list(
                columnas_modelo
            ),
            "coeficiente_clase_positiva": coeficientes,
            "coeficiente_absoluto": np.abs(
                coeficientes
            ),
        }
    )

    return tabla.sort_values(
        "coeficiente_absoluto",
        ascending=False,
    ).reset_index(
        drop=True
    )


def convertir_fechas_ns(
    fechas_ns: np.ndarray,
) -> pd.DatetimeIndex:
    """Convierte nanosegundos UTC a fechas UTC."""

    return pd.to_datetime(
        fechas_ns,
        unit="ns",
        utc=True,
    )
