from __future__ import annotations

import argparse
import gc
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from cripto.corto_plazo_v5.configuracion import (
    COLUMNAS_MODELO_V5,
    COSTES_OPERACION,
    OBJETIVO_BASE,
    OBJETIVOS_BARRERAS,
    PLIEGUES_TEMPORALES,
    RUTA_DATOS_V5,
    RUTA_DESARROLLO_V5,
    RUTA_HISTORIAL_V5,
    UMBRALES,
    VARIANTES_MODELO,
    VERSION_MODELO,
)
from cripto.corto_plazo_v5.utilidades import (
    calcular_metricas_probabilidad,
    calcular_pesos_binarios,
    evaluar_umbral,
    recorrer_lotes,
)


def construir_ruta(
    simbolo: str,
    desde: str,
    hasta: str,
) -> Path:
    """Construye una ruta V5."""

    return (
        RUTA_DATOS_V5
        / (
            f"{simbolo}_1m_v5_"
            f"{desde}_{hasta}.parquet"
        )
    )


def cargar_archivo(
    ruta: Path,
    nombre_objetivo: str,
    hasta: pd.Timestamp,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Carga variables, objetivo, fechas y retorno de salida."""

    configuracion_objetivo = OBJETIVOS_BARRERAS[
        nombre_objetivo
    ]

    columna_objetivo = (
        f"objetivo_{nombre_objetivo}"
    )

    columna_retorno = (
        f"retorno_salida_{nombre_objetivo}"
    )

    columna_fin = (
        f"fecha_fin_horizonte_{nombre_objetivo}"
    )

    columnas = [
        "fecha_apertura",
        columna_objetivo,
        columna_retorno,
        columna_fin,
        *COLUMNAS_MODELO_V5,
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
        errors="coerce",
    )

    datos[
        columna_fin
    ] = pd.to_datetime(
        datos[
            columna_fin
        ],
        utc=True,
        errors="coerce",
    )

    mascara = (
        datos[
            columna_objetivo
        ].notna()
        & datos[
            columna_retorno
        ].notna()
        & (
            datos[
                columna_fin
            ]
            < hasta
        )
    )

    datos = datos.loc[
        mascara
    ].dropna(
        subset=list(
            COLUMNAS_MODELO_V5
        )
    )

    if datos.empty:
        raise ValueError(
            f"{ruta.name}: no quedaron filas válidas."
        )

    variables = datos[
        list(
            COLUMNAS_MODELO_V5
        )
    ].to_numpy(
        dtype="float32"
    )

    objetivo = datos[
        columna_objetivo
    ].to_numpy(
        dtype="int8"
    )

    fechas_ns = (
        datos[
            "fecha_apertura"
        ]
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

    retornos = datos[
        columna_retorno
    ].to_numpy(
        dtype="float64"
    )

    return (
        variables,
        objetivo,
        fechas_ns,
        retornos,
    )


def cargar_periodos(
    simbolo: str,
    periodos,
    nombre_objetivo: str,
    hasta: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Concatena varios años."""

    hasta_ts = pd.Timestamp(
        hasta,
        tz="UTC",
    )

    bloques = []

    for desde, fin in periodos:
        ruta = construir_ruta(
            simbolo=simbolo,
            desde=desde,
            hasta=fin,
        )

        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe: {ruta}"
            )

        bloques.append(
            cargar_archivo(
                ruta=ruta,
                nombre_objetivo=nombre_objetivo,
                hasta=hasta_ts,
            )
        )

    variables = np.concatenate(
        [
            bloque[
                0
            ]
            for bloque in bloques
        ],
        axis=0,
    )

    objetivo = np.concatenate(
        [
            bloque[
                1
            ]
            for bloque in bloques
        ]
    )

    fechas_ns = np.concatenate(
        [
            bloque[
                2
            ]
            for bloque in bloques
        ]
    )

    retornos = np.concatenate(
        [
            bloque[
                3
            ]
            for bloque in bloques
        ]
    )

    return (
        variables,
        objetivo,
        fechas_ns,
        retornos,
    )


def muestreo_estratificado(
    variables: np.ndarray,
    objetivo: np.ndarray,
    maximo: int,
    semilla: int = 42,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Reduce el entrenamiento no lineal manteniendo ambas clases."""

    if len(
        objetivo
    ) <= maximo:
        return variables, objetivo

    generador = np.random.default_rng(
        semilla
    )

    indices_positivos = np.flatnonzero(
        objetivo == 1
    )

    indices_negativos = np.flatnonzero(
        objetivo == 0
    )

    proporcion_positiva = len(
        indices_positivos
    ) / len(
        objetivo
    )

    cantidad_positiva = min(
        len(
            indices_positivos
        ),
        max(
            1,
            int(
                maximo
                * proporcion_positiva
            ),
        ),
    )

    cantidad_negativa = min(
        len(
            indices_negativos
        ),
        maximo
        - cantidad_positiva,
    )

    seleccion = np.concatenate(
        [
            generador.choice(
                indices_positivos,
                size=cantidad_positiva,
                replace=False,
            ),
            generador.choice(
                indices_negativos,
                size=cantidad_negativa,
                replace=False,
            ),
        ]
    )

    generador.shuffle(
        seleccion
    )

    return (
        variables[
            seleccion
        ],
        objetivo[
            seleccion
        ],
    )


def entrenar_sgd(
    variables: np.ndarray,
    objetivo: np.ndarray,
    configuracion: dict[str, Any],
    epocas: int,
    tamano_lote: int,
):
    """Entrena el control lineal."""

    escalador = StandardScaler()

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

    pesos = calcular_pesos_binarios(
        objetivo=objetivo,
        potencia_positivo=float(
            configuracion[
                "potencia_peso_positivo"
            ]
        ),
    )

    modelo = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(
            configuracion[
                "alpha"
            ]
        ),
        learning_rate="optimal",
        class_weight=pesos,
        average=True,
        random_state=42,
    )

    primera = True

    for epoca in range(
        epocas
    ):
        generador = np.random.default_rng(
            42
            + epoca
        )

        indices = generador.permutation(
            len(
                objetivo
            )
        )

        for inicio, final in recorrer_lotes(
            len(
                indices
            ),
            tamano_lote,
        ):
            lote = indices[
                inicio:final
            ]

            x = escalador.transform(
                variables[
                    lote
                ]
            )

            y = objetivo[
                lote
            ]

            if primera:
                modelo.partial_fit(
                    x,
                    y,
                    classes=np.array(
                        [
                            0,
                            1,
                        ],
                        dtype="int8",
                    ),
                )

                primera = False
            else:
                modelo.partial_fit(
                    x,
                    y,
                )

    return modelo, escalador, pesos


def entrenar_histgb(
    variables: np.ndarray,
    objetivo: np.ndarray,
    configuracion: dict[str, Any],
):
    """Entrena un modelo no lineal sobre una muestra estratificada."""

    variables_muestra, objetivo_muestra = muestreo_estratificado(
        variables=variables,
        objetivo=objetivo,
        maximo=int(
            configuracion[
                "max_filas_entrenamiento"
            ]
        ),
    )

    pesos = calcular_pesos_binarios(
        objetivo=objetivo_muestra,
        potencia_positivo=float(
            configuracion[
                "potencia_peso_positivo"
            ]
        ),
    )

    pesos_muestra = np.where(
        objetivo_muestra == 1,
        pesos[
            1
        ],
        pesos[
            0
        ],
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
        early_stopping=False,
        random_state=42,
    )

    modelo.fit(
        variables_muestra,
        objetivo_muestra,
        sample_weight=pesos_muestra,
    )

    detalle_muestra = {
        "filas_originales": int(
            len(
                objetivo
            )
        ),
        "filas_muestra": int(
            len(
                objetivo_muestra
            )
        ),
    }

    return (
        modelo,
        None,
        pesos,
        detalle_muestra,
    )


def predecir(
    modelo,
    escalador,
    variables: np.ndarray,
    tamano_lote: int,
) -> np.ndarray:
    """Predice probabilidades por lotes."""

    probabilidades = []

    indice_positivo = int(
        np.flatnonzero(
            modelo.classes_ == 1
        )[
            0
        ]
    )

    for inicio, final in recorrer_lotes(
        len(
            variables
        ),
        tamano_lote,
    ):
        lote = variables[
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
                indice_positivo
            ]
        )

    return np.concatenate(
        probabilidades
    )


def guardar_experimento(
    identificador: str,
    simbolo: str,
    nombre_objetivo: str,
    nombre_variante: str,
    nombre_pliegue: str,
    configuracion_variante: dict[str, Any],
    epocas: int,
    modelo,
    escalador,
    pesos: dict[int, float],
    detalle_muestra: dict[str, int] | None,
    objetivo_validacion: np.ndarray,
    probabilidades: np.ndarray,
    fechas_ns: np.ndarray,
    retornos: np.ndarray,
) -> dict[str, Any]:
    """Guarda métricas y tabla completa de umbrales."""

    ruta = (
        RUTA_DESARROLLO_V5
        / simbolo
        / nombre_objetivo
        / nombre_variante
        / nombre_pliegue
        / identificador
    )

    ruta.mkdir(
        parents=True,
        exist_ok=False,
    )

    metricas = calcular_metricas_probabilidad(
        objetivo=objetivo_validacion,
        probabilidades=probabilidades,
    )

    horizonte = int(
        OBJETIVOS_BARRERAS[
            nombre_objetivo
        ][
            "horizonte_minutos"
        ]
    )

    registros = []

    for coste in COSTES_OPERACION:
        for umbral in UMBRALES:
            registros.append(
                evaluar_umbral(
                    objetivo=objetivo_validacion,
                    probabilidades=probabilidades,
                    fechas_ns=fechas_ns,
                    retornos_salida=retornos,
                    umbral=umbral,
                    horizonte_minutos=horizonte,
                    coste=coste,
                )
            )

    tabla_umbrales = pd.DataFrame(
        registros
    )

    ruta_umbrales = (
        ruta
        / "metricas_umbrales.csv"
    )

    tabla_umbrales.to_csv(
        ruta_umbrales,
        index=False,
        encoding="utf-8-sig",
    )

    if isinstance(
        modelo,
        SGDClassifier,
    ):
        coeficientes = pd.DataFrame(
            {
                "variable": COLUMNAS_MODELO_V5,
                "coeficiente": modelo.coef_[
                    0
                ],
                "coeficiente_absoluto": np.abs(
                    modelo.coef_[
                        0
                    ]
                ),
            }
        ).sort_values(
            "coeficiente_absoluto",
            ascending=False,
        )

        coeficientes.to_csv(
            ruta / "coeficientes.csv",
            index=False,
            encoding="utf-8-sig",
        )

    detalle = {
        "fecha_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "identificador": identificador,
        "version_modelo": VERSION_MODELO,
        "simbolo": simbolo,
        "objetivo": nombre_objetivo,
        "variante": nombre_variante,
        "tipo_modelo": configuracion_variante[
            "tipo"
        ],
        "pliegue": nombre_pliegue,
        "variables": len(
            COLUMNAS_MODELO_V5
        ),
        "epocas": epocas,
        "pesos": pesos,
        "detalle_muestra": detalle_muestra,
        "metricas_probabilidad": metricas,
        "ruta_umbrales": str(
            ruta_umbrales
        ),
        "uso_2025": False,
        "uso_2026": False,
    }

    (
        ruta
        / "detalle.json"
    ).write_text(
        json.dumps(
            detalle,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    fila = {
        "fecha_utc": detalle[
            "fecha_utc"
        ],
        "identificador": identificador,
        "simbolo": simbolo,
        "objetivo": nombre_objetivo,
        "variante": nombre_variante,
        "tipo_modelo": configuracion_variante[
            "tipo"
        ],
        "pliegue": nombre_pliegue,
        "variables": len(
            COLUMNAS_MODELO_V5
        ),
        "epocas": epocas,
        **metricas,
        "ruta": str(
            ruta
        ),
        "ruta_umbrales": str(
            ruta_umbrales
        ),
        "uso_2025": False,
        "uso_2026": False,
    }

    RUTA_DESARROLLO_V5.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [
            fila
        ]
    ).to_csv(
        RUTA_HISTORIAL_V5,
        mode="a",
        header=not RUTA_HISTORIAL_V5.exists(),
        index=False,
        encoding="utf-8-sig",
    )

    return fila


def resolver_nombres(
    solicitado: str,
    disponibles,
    todos: str,
):
    """Resuelve un nombre o todos."""

    if solicitado == todos:
        return list(
            disponibles
        )

    return [
        solicitado
    ]


def ejecutar(
    simbolo: str,
    objetivo_solicitado: str,
    variante_solicitada: str,
    epocas: int,
    tamano_lote: int,
) -> None:
    """Entrena modelos y pliegues de desarrollo."""

    objetivos = resolver_nombres(
        solicitado=objetivo_solicitado,
        disponibles=OBJETIVOS_BARRERAS,
        todos="TODOS",
    )

    variantes = resolver_nombres(
        solicitado=variante_solicitada,
        disponibles=VARIANTES_MODELO,
        todos="TODAS",
    )

    identificador = (
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        + "_"
        + uuid.uuid4().hex[
            :8
        ]
    )

    print(
        "\nENTRENAMIENTO V5"
    )
    print("=" * 72)
    print(
        f"Símbolo: {simbolo}"
    )
    print(
        f"Objetivos: {', '.join(objetivos)}"
    )
    print(
        f"Variantes: {', '.join(variantes)}"
    )
    print(
        "Validaciones: 2023 y 2024"
    )
    print(
        "2025 y 2026 no se utilizan."
    )

    for nombre_objetivo in objetivos:
        for nombre_variante in variantes:
            configuracion_variante = VARIANTES_MODELO[
                nombre_variante
            ]

            for nombre_pliegue, pliegue in PLIEGUES_TEMPORALES.items():
                print(
                    "\n"
                    + "=" * 72
                )
                print(
                    f"{nombre_objetivo} | {nombre_variante} | {nombre_pliegue}"
                )
                print(
                    "=" * 72
                )

                (
                    x_entrenamiento,
                    y_entrenamiento,
                    _,
                    _,
                ) = cargar_periodos(
                    simbolo=simbolo,
                    periodos=pliegue[
                        "entrenamiento"
                    ],
                    nombre_objetivo=nombre_objetivo,
                    hasta=pliegue[
                        "hasta_entrenamiento"
                    ],
                )

                (
                    x_validacion,
                    y_validacion,
                    fechas_validacion,
                    retornos_validacion,
                ) = cargar_periodos(
                    simbolo=simbolo,
                    periodos=(
                        pliegue[
                            "validacion"
                        ],
                    ),
                    nombre_objetivo=nombre_objetivo,
                    hasta=pliegue[
                        "hasta_validacion"
                    ],
                )

                print(
                    f"Entrenamiento: {len(y_entrenamiento):,}".replace(
                        ",",
                        ".",
                    )
                )
                print(
                    f"Validación: {len(y_validacion):,}".replace(
                        ",",
                        ".",
                    )
                )
                print(
                    f"Prevalencia entrenamiento: {y_entrenamiento.mean():.4f}"
                )

                if configuracion_variante[
                    "tipo"
                ] == "sgd":
                    (
                        modelo,
                        escalador,
                        pesos,
                    ) = entrenar_sgd(
                        variables=x_entrenamiento,
                        objetivo=y_entrenamiento,
                        configuracion=configuracion_variante,
                        epocas=epocas,
                        tamano_lote=tamano_lote,
                    )

                    detalle_muestra = None

                elif configuracion_variante[
                    "tipo"
                ] == "histgb":
                    (
                        modelo,
                        escalador,
                        pesos,
                        detalle_muestra,
                    ) = entrenar_histgb(
                        variables=x_entrenamiento,
                        objetivo=y_entrenamiento,
                        configuracion=configuracion_variante,
                    )

                else:
                    raise ValueError(
                        "Tipo de modelo desconocido."
                    )

                probabilidades = predecir(
                    modelo=modelo,
                    escalador=escalador,
                    variables=x_validacion,
                    tamano_lote=tamano_lote,
                )

                fila = guardar_experimento(
                    identificador=identificador,
                    simbolo=simbolo,
                    nombre_objetivo=nombre_objetivo,
                    nombre_variante=nombre_variante,
                    nombre_pliegue=nombre_pliegue,
                    configuracion_variante=configuracion_variante,
                    epocas=epocas,
                    modelo=modelo,
                    escalador=escalador,
                    pesos=pesos,
                    detalle_muestra=detalle_muestra,
                    objetivo_validacion=y_validacion,
                    probabilidades=probabilidades,
                    fechas_ns=fechas_validacion,
                    retornos=retornos_validacion,
                )

                print(
                    f"PR-AUC: {fila['pr_auc']:.4f}"
                )
                print(
                    f"PR-AUC lift: {fila['pr_auc_lift']:.4f}"
                )
                print(
                    f"ROC-AUC: {fila['roc_auc']:.4f}"
                )
                print(
                    f"Ruta: {fila['ruta']}"
                )

                del x_entrenamiento
                del y_entrenamiento
                del x_validacion
                del y_validacion
                del fechas_validacion
                del retornos_validacion
                del probabilidades
                del modelo
                del escalador
                gc.collect()


def main() -> None:
    """Punto de entrada."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--simbolo",
        default="BTCUSDT",
    )

    parser.add_argument(
        "--objetivo",
        choices=(
            *OBJETIVOS_BARRERAS.keys(),
            "TODOS",
        ),
        default=OBJETIVO_BASE,
    )

    parser.add_argument(
        "--variante",
        choices=(
            *VARIANTES_MODELO.keys(),
            "TODAS",
        ),
        default="TODAS",
    )

    parser.add_argument(
        "--epocas",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--tamano-lote",
        type=int,
        default=100000,
    )

    argumentos = parser.parse_args()

    ejecutar(
        simbolo=argumentos.simbolo,
        objetivo_solicitado=argumentos.objetivo,
        variante_solicitada=argumentos.variante,
        epocas=argumentos.epocas,
        tamano_lote=argumentos.tamano_lote,
    )


if __name__ == "__main__":
    main()
