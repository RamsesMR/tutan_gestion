from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_1.utilidades import (
    construir_mascara_entrada,
    factor_beneficio,
    maximo_drawdown,
    seleccionar_operaciones,
)

from cripto.corto_plazo_v4_4.configuracion import (
    COLUMNAS_PREDICCIONES_REQUERIDAS,
    COSTE,
    HORIZONTE_MAXIMO_MINUTOS,
    MODO_ENTRADA,
    RUTA_PREDICCIONES_V4_1,
    SIMBOLO,
    UMBRAL_SUBE,
)


@dataclass(frozen=True)
class ResultadoOperacion:
    indice_entrada: int
    indice_salida: int
    duracion_minutos: int
    salida_anticipada: bool
    retorno_bruto: float
    retorno_neto: float


def cargar_pliegue(
    nombre_pliegue: str,
) -> pd.DataFrame:
    """Carga las predicciones fuera de muestra ya generadas por V4.1."""

    ruta = (
        RUTA_PREDICCIONES_V4_1
        / f"{SIMBOLO}_{nombre_pliegue}.parquet"
    )

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el archivo de predicciones V4.1: {ruta}"
        )

    datos = pd.read_parquet(
        ruta
    )

    faltantes = set(
        COLUMNAS_PREDICCIONES_REQUERIDAS
    ).difference(
        datos.columns
    )

    if faltantes:
        raise ValueError(
            f"Faltan columnas en {ruta.name}: "
            + ", ".join(
                sorted(faltantes)
            )
        )

    datos = datos.copy()

    datos["fecha_apertura"] = pd.to_datetime(
        datos["fecha_apertura"],
        utc=True,
        errors="raise",
    )

    datos = (
        datos
        .sort_values(
            "fecha_apertura"
        )
        .reset_index(drop=True)
    )

    if datos[
        "fecha_apertura"
    ].duplicated().any():
        raise ValueError(
            f"{ruta.name} contiene fechas duplicadas."
        )

    columnas_numericas = (
        "precio_apertura",
        "precio_maximo",
        "precio_minimo",
        "precio_cierre",
        "probabilidad_sube",
        "probabilidad_baja",
        "objetivo_sube_4h",
    )

    for columna in columnas_numericas:
        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce",
        )

    if datos[
        list(columnas_numericas)
    ].isna().any().any():
        raise ValueError(
            f"{ruta.name} contiene valores no numéricos o nulos."
        )

    return datos


def construir_entradas_v4_1(
    datos: pd.DataFrame,
) -> np.ndarray:
    """Reproduce exactamente la entrada congelada de V4.1."""

    probabilidades_sube = datos[
        "probabilidad_sube"
    ].to_numpy(
        dtype="float64"
    )

    probabilidades_baja = datos[
        "probabilidad_baja"
    ].to_numpy(
        dtype="float64"
    )

    return construir_mascara_entrada(
        probabilidades_sube=probabilidades_sube,
        probabilidades_baja=probabilidades_baja,
        umbral_sube=UMBRAL_SUBE,
        umbral_veto_baja=None,
        modo=MODO_ENTRADA,
    )


def construir_cruces_baja(
    probabilidades_baja: np.ndarray,
    umbral_baja: float,
) -> np.ndarray:
    """Marca cruces causales del detector BAJA desde abajo."""

    sobre_umbral = (
        probabilidades_baja
        >= umbral_baja
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


def seleccionar_indices_control(
    datos: pd.DataFrame,
) -> np.ndarray:
    """Selecciona las operaciones originales de V4.1 sin solapamiento."""

    mascara = construir_entradas_v4_1(
        datos
    )

    fechas_ns = (
        datos[
            "fecha_apertura"
        ]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(
            dtype="datetime64[ns]"
        )
        .astype("int64")
    )

    duraciones = np.full(
        len(datos),
        HORIZONTE_MAXIMO_MINUTOS,
        dtype="int16",
    )

    mascara_valida = mascara.copy()

    if len(mascara_valida) > HORIZONTE_MAXIMO_MINUTOS:
        mascara_valida[
            -HORIZONTE_MAXIMO_MINUTOS:
        ] = False
    else:
        mascara_valida[:] = False

    return seleccionar_operaciones(
        fechas_ns=fechas_ns,
        mascara=mascara_valida,
        minutos_salida=duraciones,
    )


def determinar_salida(
    indice_entrada: int,
    cierres: np.ndarray,
    cruces_baja: np.ndarray | None,
    minutos_minimos: int,
    condicion_salida: str,
) -> ResultadoOperacion | None:
    """Determina una salida fija o anticipada de forma causal."""

    indice_fijo = (
        indice_entrada
        + HORIZONTE_MAXIMO_MINUTOS
    )

    if indice_fijo >= len(
        cierres
    ):
        return None

    indice_salida = indice_fijo
    salida_anticipada = False

    if cruces_baja is not None:
        inicio_busqueda = (
            indice_entrada
            + max(
                1,
                minutos_minimos,
            )
        )

        for indice in range(
            inicio_busqueda,
            indice_fijo + 1,
        ):
            if not cruces_baja[
                indice
            ]:
                continue

            retorno_bruto_actual = (
                cierres[indice]
                / cierres[indice_entrada]
                - 1.0
            )

            if (
                condicion_salida
                == "solo_beneficio_neto"
                and (
                    retorno_bruto_actual
                    - COSTE
                )
                <= 0.0
            ):
                continue

            indice_salida = indice
            salida_anticipada = (
                indice_salida
                < indice_fijo
            )
            break

    retorno_bruto = (
        cierres[indice_salida]
        / cierres[indice_entrada]
        - 1.0
    )

    retorno_neto = (
        retorno_bruto
        - COSTE
    )

    return ResultadoOperacion(
        indice_entrada=indice_entrada,
        indice_salida=indice_salida,
        duracion_minutos=(
            indice_salida
            - indice_entrada
        ),
        salida_anticipada=salida_anticipada,
        retorno_bruto=float(
            retorno_bruto
        ),
        retorno_neto=float(
            retorno_neto
        ),
    )


def simular_estrategia(
    datos: pd.DataFrame,
    modo_evaluacion: str,
    umbral_baja: float | None,
    minutos_minimos: int,
    condicion_salida: str,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
]:
    """
    Simula el control V4.1 o una salida anticipada por BAJA.

    cohorte_v4_1:
        Mantiene las mismas entradas originales. Aísla el efecto de la salida.

    reinversion:
        Permite una entrada nueva después de una salida anticipada.
        Representa la operativa real si se adopta V4.4.
    """

    if modo_evaluacion not in {
        "cohorte_v4_1",
        "reinversion",
    }:
        raise ValueError(
            f"Modo de evaluación desconocido: {modo_evaluacion}"
        )

    cierres = datos[
        "precio_cierre"
    ].to_numpy(
        dtype="float64"
    )

    probabilidades_baja = datos[
        "probabilidad_baja"
    ].to_numpy(
        dtype="float64"
    )

    objetivos = datos[
        "objetivo_sube_4h"
    ].to_numpy(
        dtype="int8"
    )

    mascara_entrada = construir_entradas_v4_1(
        datos
    )

    cruces_baja = (
        None
        if umbral_baja is None
        else construir_cruces_baja(
            probabilidades_baja=probabilidades_baja,
            umbral_baja=umbral_baja,
        )
    )

    if modo_evaluacion == "cohorte_v4_1":
        candidatos = seleccionar_indices_control(
            datos
        )
    else:
        candidatos = np.flatnonzero(
            mascara_entrada
        )

    operaciones: list[
        ResultadoOperacion
    ] = []

    siguiente_indice_permitido = 0

    for indice_entrada in candidatos:
        indice_entrada = int(
            indice_entrada
        )

        if (
            modo_evaluacion
            == "reinversion"
            and indice_entrada
            < siguiente_indice_permitido
        ):
            continue

        resultado = determinar_salida(
            indice_entrada=indice_entrada,
            cierres=cierres,
            cruces_baja=cruces_baja,
            minutos_minimos=minutos_minimos,
            condicion_salida=condicion_salida,
        )

        if resultado is None:
            continue

        operaciones.append(
            resultado
        )

        if modo_evaluacion == "reinversion":
            siguiente_indice_permitido = (
                resultado.indice_salida
            )

    if not operaciones:
        return (
            metricas_vacias(),
            pd.DataFrame(),
        )

    indices_entrada = np.asarray(
        [
            operacion.indice_entrada
            for operacion in operaciones
        ],
        dtype="int64",
    )

    indices_salida = np.asarray(
        [
            operacion.indice_salida
            for operacion in operaciones
        ],
        dtype="int64",
    )

    retornos_brutos = np.asarray(
        [
            operacion.retorno_bruto
            for operacion in operaciones
        ],
        dtype="float64",
    )

    retornos_netos = np.asarray(
        [
            operacion.retorno_neto
            for operacion in operaciones
        ],
        dtype="float64",
    )

    duraciones = np.asarray(
        [
            operacion.duracion_minutos
            for operacion in operaciones
        ],
        dtype="int64",
    )

    salidas_anticipadas = np.asarray(
        [
            operacion.salida_anticipada
            for operacion in operaciones
        ],
        dtype=bool,
    )

    fechas_entrada = datos.loc[
        indices_entrada,
        "fecha_apertura",
    ].reset_index(
        drop=True
    )

    meses = (
        fechas_entrada
        .dt.to_period("M")
        .astype(str)
    )

    retornos_mensuales = (
        pd.DataFrame(
            {
                "mes": meses,
                "retorno_neto": retornos_netos,
            }
        )
        .groupby(
            "mes",
            sort=True,
        )[
            "retorno_neto"
        ]
        .apply(
            lambda serie: float(
                np.prod(
                    1.0
                    + serie.to_numpy(
                        dtype="float64"
                    )
                )
                - 1.0
            )
        )
    )

    ganancias = retornos_netos[
        retornos_netos > 0
    ]

    concentracion_mejores_5 = 0.0

    if ganancias.size > 0:
        total_ganancias = float(
            ganancias.sum()
        )

        if total_ganancias > 0:
            concentracion_mejores_5 = float(
                np.sort(
                    ganancias
                )[-5:].sum()
                / total_ganancias
            )

    perdidas = retornos_netos[
        retornos_netos < 0
    ]

    metricas = {
        "operaciones": int(
            len(operaciones)
        ),
        "precision_clasificacion": float(
            objetivos[
                indices_entrada
            ].mean()
        ),
        "porcentaje_acierto_clasificacion": float(
            objetivos[
                indices_entrada
            ].mean()
            * 100.0
        ),
        "porcentaje_operaciones_positivas": float(
            np.mean(
                retornos_netos > 0
            )
            * 100.0
        ),
        "retorno_bruto_medio": float(
            retornos_brutos.mean()
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
        "retorno_compuesto": float(
            np.prod(
                1.0
                + np.clip(
                    retornos_netos,
                    -0.999999,
                    None,
                )
            )
            - 1.0
        ),
        "duracion_media_minutos": float(
            duraciones.mean()
        ),
        "porcentaje_salidas_anticipadas": float(
            salidas_anticipadas.mean()
            * 100.0
        ),
        "perdida_media": float(
            perdidas.mean()
            if perdidas.size
            else 0.0
        ),
        "peor_operacion": float(
            retornos_netos.min()
        ),
        "porcentaje_meses_positivos": float(
            np.mean(
                retornos_mensuales.to_numpy(
                    dtype="float64"
                )
                > 0
            )
            * 100.0
            if len(
                retornos_mensuales
            )
            else 0.0
        ),
        "concentracion_ganancias_mejores_5": (
            concentracion_mejores_5
        ),
    }

    detalle = pd.DataFrame(
        {
            "fecha_entrada": datos.loc[
                indices_entrada,
                "fecha_apertura",
            ].to_numpy(),
            "fecha_salida": datos.loc[
                indices_salida,
                "fecha_apertura",
            ].to_numpy(),
            "indice_entrada": indices_entrada,
            "indice_salida": indices_salida,
            "duracion_minutos": duraciones,
            "salida_anticipada": salidas_anticipadas,
            "probabilidad_sube_entrada": datos.loc[
                indices_entrada,
                "probabilidad_sube",
            ].to_numpy(
                dtype="float64"
            ),
            "probabilidad_baja_entrada": datos.loc[
                indices_entrada,
                "probabilidad_baja",
            ].to_numpy(
                dtype="float64"
            ),
            "objetivo_sube_4h": objetivos[
                indices_entrada
            ],
            "retorno_bruto": retornos_brutos,
            "retorno_neto": retornos_netos,
        }
    )

    return metricas, detalle


def metricas_vacias() -> dict[str, float | int]:
    """Devuelve la estructura de métricas sin operaciones."""

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
        "retorno_compuesto": 0.0,
        "duracion_media_minutos": 0.0,
        "porcentaje_salidas_anticipadas": 0.0,
        "perdida_media": 0.0,
        "peor_operacion": 0.0,
        "porcentaje_meses_positivos": 0.0,
        "concentracion_ganancias_mejores_5": 0.0,
    }


def convertir_para_json(
    valor: Any,
) -> Any:
    """Convierte tipos NumPy y pandas para JSON."""

    if isinstance(
        valor,
        (
            np.integer,
        ),
    ):
        return int(
            valor
        )

    if isinstance(
        valor,
        (
            np.floating,
        ),
    ):
        if math.isinf(
            float(valor)
        ):
            return "inf"

        return float(
            valor
        )

    if isinstance(
        valor,
        pd.Timestamp,
    ):
        return valor.isoformat()

    return valor
