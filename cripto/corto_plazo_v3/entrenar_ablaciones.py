from __future__ import annotations

import argparse
import gc
import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

from cripto.corto_plazo_v3.configuracion import (
    CLASES,
    NOMBRE_HORIZONTE,
    PLIEGUES_TEMPORALES,
    RUTA_DATOS_V2,
    RUTA_HISTORIAL_ABLACIONES,
    RUTA_RESULTADOS_ABLACIONES,
    SIMBOLOS,
    UMBRAL_CLASE,
    VARIANTES,
    VERSION_MODELO,
)


CLASES_NP = np.array(
    CLASES,
    dtype=object,
)


def construir_ruta_datos(
    simbolo: str,
    desde: str,
    hasta: str,
) -> Path:
    """Construye la ruta de un Parquet V2A."""

    nombre = (
        f"{simbolo}_1m_{NOMBRE_HORIZONTE}_"
        f"{desde}_{hasta}_variables.parquet"
    )

    return RUTA_DATOS_V2 / nombre


def clasificar_objetivo(
    rendimientos: pd.Series,
) -> np.ndarray:
    """Convierte el objetivo continuo en las tres clases."""

    valores = pd.to_numeric(
        rendimientos,
        errors="coerce",
    ).to_numpy(
        dtype="float64"
    )

    if not np.isfinite(
        valores
    ).all():
        raise ValueError(
            "El objetivo contiene valores nulos o infinitos."
        )

    return np.select(
        [
            valores <= -UMBRAL_CLASE,
            valores >= UMBRAL_CLASE,
        ],
        [
            "BAJA",
            "SUBE",
        ],
        default="NEUTRAL",
    )


def cargar_datos(
    ruta: Path,
    columnas_modelo: tuple[str, ...],
    desde: pd.Timestamp,
    hasta: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carga y purga un periodo sin permitir objetivos fuera del corte."""

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

    datos = (
        datos.loc[
            mascara
        ]
        .reset_index(drop=True)
    )

    if datos.empty:
        raise ValueError(
            f"{ruta.name}: no quedaron muestras dentro del corte."
        )

    variables = (
        datos[
            list(columnas_modelo)
        ]
        .to_numpy(
            dtype="float32"
        )
    )

    if not np.isfinite(
        variables
    ).all():
        raise ValueError(
            f"{ruta.name}: contiene variables no finitas."
        )

    clases = clasificar_objetivo(
        datos["rendimiento_objetivo"]
    )

    fechas = datos[
        "fecha_apertura"
    ].to_numpy()

    del datos
    gc.collect()

    return variables, clases, fechas


def recorrer_lotes(
    total: int,
    tamano_lote: int,
):
    """Genera los límites de los lotes."""

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


def ajustar_escalador(
    simbolo: str,
    configuracion_pliegue: dict[str, Any],
    columnas_modelo: tuple[str, ...],
    tamano_lote: int,
) -> tuple[StandardScaler, Counter]:
    """Ajusta el escalador y cuenta las clases de entrenamiento."""

    escalador = StandardScaler()
    conteos: Counter = Counter()

    desde = pd.Timestamp(
        configuracion_pliegue["entrenamiento_desde"],
        tz="UTC",
    )

    hasta = pd.Timestamp(
        configuracion_pliegue["entrenamiento_hasta"],
        tz="UTC",
    )

    for desde_archivo, hasta_archivo in configuracion_pliegue[
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

        variables, clases, _ = cargar_datos(
            ruta=ruta,
            columnas_modelo=columnas_modelo,
            desde=desde,
            hasta=hasta,
        )

        conteos.update(
            clases.tolist()
        )

        for inicio, final in recorrer_lotes(
            len(variables),
            tamano_lote,
        ):
            escalador.partial_fit(
                variables[inicio:final]
            )

        del variables
        del clases
        gc.collect()

    return escalador, conteos


def calcular_pesos(
    conteos: Counter,
) -> dict[str, float]:
    """Calcula pesos inversamente proporcionales."""

    total = sum(
        conteos[clase]
        for clase in CLASES
    )

    pesos: dict[str, float] = {}

    for clase in CLASES:
        cantidad = conteos[
            clase
        ]

        if cantidad == 0:
            raise ValueError(
                f"La clase {clase} no tiene muestras."
            )

        pesos[clase] = (
            total
            / (
                len(CLASES)
                * cantidad
            )
        )

    return pesos


def entrenar_modelo(
    simbolo: str,
    configuracion_pliegue: dict[str, Any],
    configuracion_variante: dict[str, Any],
    escalador: StandardScaler,
    pesos: dict[str, float],
    epocas: int,
    tamano_lote: int,
) -> SGDClassifier:
    """Entrena una variante mediante partial_fit."""

    columnas_modelo = tuple(
        configuracion_variante["columnas"]
    )

    modelo = SGDClassifier(
        loss="log_loss",
        penalty=configuracion_variante["penalty"],
        alpha=float(
            configuracion_variante["alpha"]
        ),
        l1_ratio=float(
            configuracion_variante["l1_ratio"]
        ),
        learning_rate="optimal",
        class_weight=pesos,
        average=True,
        random_state=42,
    )

    desde = pd.Timestamp(
        configuracion_pliegue["entrenamiento_desde"],
        tz="UTC",
    )

    hasta = pd.Timestamp(
        configuracion_pliegue["entrenamiento_hasta"],
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
            configuracion_pliegue[
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

            variables, clases, _ = cargar_datos(
                ruta=ruta,
                columnas_modelo=columnas_modelo,
                desde=desde,
                hasta=hasta,
            )

            generador = np.random.default_rng(
                42
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

                variables_lote = escalador.transform(
                    variables[
                        indices_lote
                    ]
                )

                clases_lote = clases[
                    indices_lote
                ]

                if primera_actualizacion:
                    modelo.partial_fit(
                        variables_lote,
                        clases_lote,
                        classes=CLASES_NP,
                    )

                    primera_actualizacion = False

                else:
                    modelo.partial_fit(
                        variables_lote,
                        clases_lote,
                    )

            del variables
            del clases
            del indices
            gc.collect()

    return modelo


def predecir_validacion(
    simbolo: str,
    configuracion_pliegue: dict[str, Any],
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
    """Predice el año de validación y conserva probabilidades."""

    desde_archivo, hasta_archivo = configuracion_pliegue[
        "archivo_validacion"
    ]

    ruta = construir_ruta_datos(
        simbolo=simbolo,
        desde=desde_archivo,
        hasta=hasta_archivo,
    )

    desde = pd.Timestamp(
        configuracion_pliegue["validacion_desde"],
        tz="UTC",
    )

    hasta = pd.Timestamp(
        configuracion_pliegue["validacion_hasta"],
        tz="UTC",
    )

    variables, reales, fechas = cargar_datos(
        ruta=ruta,
        columnas_modelo=columnas_modelo,
        desde=desde,
        hasta=hasta,
    )

    predicciones: list[np.ndarray] = []
    probabilidades: list[np.ndarray] = []

    for inicio, final in recorrer_lotes(
        len(variables),
        tamano_lote,
    ):
        variables_lote = escalador.transform(
            variables[
                inicio:final
            ]
        )

        predicciones.append(
            modelo.predict(
                variables_lote
            )
        )

        probabilidades.append(
            modelo.predict_proba(
                variables_lote
            )
        )

    del variables
    gc.collect()

    return (
        reales,
        np.concatenate(
            predicciones
        ),
        np.concatenate(
            probabilidades
        ),
        fechas,
    )


def calcular_metricas(
    reales: np.ndarray,
    predicciones: np.ndarray,
    probabilidades: np.ndarray,
    clases_modelo: np.ndarray,
) -> dict[str, Any]:
    """Calcula métricas globales y específicas de BAJA y SUBE."""

    precision, recall, f1, soporte = (
        precision_recall_fscore_support(
            reales,
            predicciones,
            labels=CLASES_NP,
            zero_division=0,
        )
    )

    indice_clase = {
        str(clase): indice
        for indice, clase in enumerate(
            clases_modelo
        )
    }

    objetivo_baja = (
        reales == "BAJA"
    ).astype(
        "int8"
    )

    objetivo_sube = (
        reales == "SUBE"
    ).astype(
        "int8"
    )

    probabilidad_baja = probabilidades[
        :,
        indice_clase["BAJA"],
    ]

    probabilidad_sube = probabilidades[
        :,
        indice_clase["SUBE"],
    ]

    metricas: dict[str, Any] = {
        "muestras": len(reales),
        "accuracy": accuracy_score(
            reales,
            predicciones,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            reales,
            predicciones,
        ),
        "f1_macro": f1_score(
            reales,
            predicciones,
            labels=CLASES_NP,
            average="macro",
            zero_division=0,
        ),
        "f1_extremos_promedio": (
            float(f1[0])
            + float(f1[2])
        ) / 2,
        "pr_auc_baja": average_precision_score(
            objetivo_baja,
            probabilidad_baja,
        ),
        "pr_auc_sube": average_precision_score(
            objetivo_sube,
            probabilidad_sube,
        ),
        "tasa_predicha_baja": float(
            np.mean(
                predicciones == "BAJA"
            )
        ),
        "tasa_predicha_neutral": float(
            np.mean(
                predicciones == "NEUTRAL"
            )
        ),
        "tasa_predicha_sube": float(
            np.mean(
                predicciones == "SUBE"
            )
        ),
    }

    for indice, clase in enumerate(
        CLASES
    ):
        nombre = clase.lower()

        metricas[
            f"precision_{nombre}"
        ] = float(
            precision[indice]
        )

        metricas[
            f"recall_{nombre}"
        ] = float(
            recall[indice]
        )

        metricas[
            f"f1_{nombre}"
        ] = float(
            f1[indice]
        )

        metricas[
            f"soporte_{nombre}"
        ] = int(
            soporte[indice]
        )

    if metricas["f1_baja"] >= metricas["f1_sube"]:
        metricas["mejor_lado"] = "BAJA"
        metricas["f1_mejor_lado"] = metricas[
            "f1_baja"
        ]
    else:
        metricas["mejor_lado"] = "SUBE"
        metricas["f1_mejor_lado"] = metricas[
            "f1_sube"
        ]

    return metricas


def calcular_metricas_mensuales(
    fechas: np.ndarray,
    reales: np.ndarray,
    predicciones: np.ndarray,
) -> pd.DataFrame:
    """Calcula estabilidad mensual de BAJA y SUBE."""

    datos = pd.DataFrame(
        {
            "fecha": pd.to_datetime(
                fechas,
                utc=True,
            ),
            "real": reales,
            "prediccion": predicciones,
        }
    )

    datos["mes"] = datos[
        "fecha"
    ].dt.to_period(
        "M"
    ).astype(
        str
    )

    registros: list[
        dict[str, Any]
    ] = []

    for mes, grupo in datos.groupby(
        "mes",
        sort=True,
    ):
        precision, recall, f1, soporte = (
            precision_recall_fscore_support(
                grupo["real"],
                grupo["prediccion"],
                labels=CLASES_NP,
                zero_division=0,
            )
        )

        registros.append(
            {
                "mes": mes,
                "muestras": len(grupo),
                "accuracy": accuracy_score(
                    grupo["real"],
                    grupo["prediccion"],
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    grupo["real"],
                    grupo["prediccion"],
                ),
                "f1_macro": f1_score(
                    grupo["real"],
                    grupo["prediccion"],
                    labels=CLASES_NP,
                    average="macro",
                    zero_division=0,
                ),
                "precision_baja": float(
                    precision[0]
                ),
                "recall_baja": float(
                    recall[0]
                ),
                "f1_baja": float(
                    f1[0]
                ),
                "soporte_baja": int(
                    soporte[0]
                ),
                "precision_sube": float(
                    precision[2]
                ),
                "recall_sube": float(
                    recall[2]
                ),
                "f1_sube": float(
                    f1[2]
                ),
                "soporte_sube": int(
                    soporte[2]
                ),
                "tasa_predicha_baja": float(
                    np.mean(
                        grupo["prediccion"] == "BAJA"
                    )
                ),
                "tasa_predicha_sube": float(
                    np.mean(
                        grupo["prediccion"] == "SUBE"
                    )
                ),
            }
        )

    return pd.DataFrame(
        registros
    )


def guardar_experimento(
    identificador_ejecucion: str,
    simbolo: str,
    nombre_pliegue: str,
    nombre_variante: str,
    configuracion_variante: dict[str, Any],
    epocas: int,
    tamano_lote: int,
    conteos: Counter,
    pesos: dict[str, float],
    modelo: SGDClassifier,
    escalador: StandardScaler,
    reales: np.ndarray,
    predicciones: np.ndarray,
    probabilidades: np.ndarray,
    fechas: np.ndarray,
    guardar_modelo: bool,
) -> dict[str, Any]:
    """Guarda métricas, matriz, meses y coeficientes."""

    fecha_utc = datetime.now(
        timezone.utc
    )

    ruta = (
        RUTA_RESULTADOS_ABLACIONES
        / simbolo
        / nombre_variante
        / nombre_pliegue
        / identificador_ejecucion
    )

    ruta.mkdir(
        parents=True,
        exist_ok=False,
    )

    metricas = calcular_metricas(
        reales=reales,
        predicciones=predicciones,
        probabilidades=probabilidades,
        clases_modelo=modelo.classes_,
    )

    metricas_mensuales = calcular_metricas_mensuales(
        fechas=fechas,
        reales=reales,
        predicciones=predicciones,
    )

    matriz = confusion_matrix(
        reales,
        predicciones,
        labels=CLASES_NP,
    )

    pd.DataFrame(
        matriz,
        index=[
            f"real_{clase}"
            for clase in CLASES
        ],
        columns=[
            f"predicha_{clase}"
            for clase in CLASES
        ],
    ).to_csv(
        ruta / "matriz_confusion.csv",
        encoding="utf-8-sig",
    )

    metricas_mensuales.to_csv(
        ruta / "metricas_mensuales.csv",
        index=False,
        encoding="utf-8-sig",
    )

    informe = classification_report(
        reales,
        predicciones,
        labels=CLASES_NP,
        output_dict=True,
        zero_division=0,
    )

    with (
        ruta / "informe_clasificacion.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            informe,
            archivo,
            ensure_ascii=False,
            indent=4,
        )

    columnas_modelo = list(
        configuracion_variante[
            "columnas"
        ]
    )

    coeficientes = pd.DataFrame(
        modelo.coef_.T,
        index=columnas_modelo,
        columns=[
            f"coeficiente_{clase}"
            for clase in modelo.classes_
        ],
    ).reset_index(
        names="variable"
    )

    coeficientes[
        "maximo_absoluto"
    ] = coeficientes[
        [
            columna
            for columna in coeficientes.columns
            if columna.startswith(
                "coeficiente_"
            )
        ]
    ].abs().max(
        axis=1
    )

    coeficientes.to_csv(
        ruta / "coeficientes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    coeficientes_cero = int(
        np.all(
            np.isclose(
                modelo.coef_,
                0.0,
                atol=1e-12,
            ),
            axis=0,
        ).sum()
    )

    detalle = {
        "fecha_utc": fecha_utc.isoformat(),
        "identificador_ejecucion": identificador_ejecucion,
        "version_modelo": VERSION_MODELO,
        "simbolo": simbolo,
        "pliegue": nombre_pliegue,
        "variante": nombre_variante,
        "grupo": configuracion_variante[
            "grupo"
        ],
        "penalty": configuracion_variante[
            "penalty"
        ],
        "alpha": configuracion_variante[
            "alpha"
        ],
        "l1_ratio": configuracion_variante[
            "l1_ratio"
        ],
        "epocas": epocas,
        "tamano_lote": tamano_lote,
        "variables": len(
            columnas_modelo
        ),
        "coeficientes_totalmente_cero": coeficientes_cero,
        "columnas_modelo": columnas_modelo,
        "conteos_entrenamiento": dict(
            conteos
        ),
        "pesos_clases": pesos,
        "metricas": metricas,
        "desviacion_mensual_f1_baja": float(
            metricas_mensuales[
                "f1_baja"
            ].std(
                ddof=0
            )
        ),
        "desviacion_mensual_f1_sube": float(
            metricas_mensuales[
                "f1_sube"
            ].std(
                ddof=0
            )
        ),
        "ruta": str(
            ruta
        ),
        "division_2025_utilizada": False,
        "division_2026_utilizada": False,
    }

    with (
        ruta / "detalle.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            detalle,
            archivo,
            ensure_ascii=False,
            indent=4,
        )

    if guardar_modelo:
        joblib.dump(
            {
                "detalle": detalle,
                "modelo": modelo,
                "escalador": escalador,
            },
            ruta / "modelo.joblib",
        )

    fila_historial = {
        "fecha_utc": fecha_utc.isoformat(),
        "identificador_ejecucion": identificador_ejecucion,
        "simbolo": simbolo,
        "pliegue": nombre_pliegue,
        "variante": nombre_variante,
        "grupo": configuracion_variante[
            "grupo"
        ],
        "penalty": configuracion_variante[
            "penalty"
        ],
        "alpha": configuracion_variante[
            "alpha"
        ],
        "l1_ratio": configuracion_variante[
            "l1_ratio"
        ],
        "epocas": epocas,
        "tamano_lote": tamano_lote,
        "variables": len(
            columnas_modelo
        ),
        "coeficientes_totalmente_cero": coeficientes_cero,
        **metricas,
        "desviacion_mensual_f1_baja": detalle[
            "desviacion_mensual_f1_baja"
        ],
        "desviacion_mensual_f1_sube": detalle[
            "desviacion_mensual_f1_sube"
        ],
        "ruta": str(
            ruta
        ),
        "division_2025_utilizada": False,
        "division_2026_utilizada": False,
    }

    RUTA_RESULTADOS_ABLACIONES.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [
            fila_historial
        ]
    ).to_csv(
        RUTA_HISTORIAL_ABLACIONES,
        mode="a",
        header=not RUTA_HISTORIAL_ABLACIONES.exists(),
        index=False,
        encoding="utf-8-sig",
    )

    return fila_historial


def obtener_variantes(
    grupo: str,
    nombres: list[str] | None,
) -> list[str]:
    """Resuelve las variantes solicitadas."""

    if nombres:
        inexistentes = set(
            nombres
        ).difference(
            VARIANTES
        )

        if inexistentes:
            raise ValueError(
                "Variantes desconocidas: "
                + ", ".join(
                    sorted(inexistentes)
                )
            )

        return nombres

    if grupo == "todo":
        return list(
            VARIANTES
        )

    return [
        nombre
        for nombre, configuracion in VARIANTES.items()
        if configuracion["grupo"] == grupo
    ]


def ejecutar(
    simbolo: str,
    grupo: str,
    nombres_variantes: list[str] | None,
    epocas: int,
    tamano_lote: int,
    guardar_modelos: bool,
) -> None:
    """Ejecuta las ablaciones solicitadas en 2023 y 2024."""

    variantes = obtener_variantes(
        grupo=grupo,
        nombres=nombres_variantes,
    )

    identificador_ejecucion = (
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
        "\nABLACIONES DE VARIABLES V3A"
    )
    print("=" * 70)
    print(
        f"Símbolo: {simbolo}"
    )
    print(
        f"Grupo: {grupo}"
    )
    print(
        f"Variantes: {', '.join(variantes)}"
    )
    print(
        f"Épocas: {epocas}"
    )
    print(
        "Validaciones: 2023 y 2024"
    )
    print(
        "2025 y 2026 no serán utilizados."
    )

    resultados: list[
        dict[str, Any]
    ] = []

    for nombre_variante in variantes:
        configuracion_variante = VARIANTES[
            nombre_variante
        ]

        columnas_modelo = tuple(
            configuracion_variante[
                "columnas"
            ]
        )

        for nombre_pliegue, configuracion_pliegue in (
            PLIEGUES_TEMPORALES.items()
        ):
            print(
                "\n"
                + "=" * 70
            )
            print(
                f"{nombre_variante} | {nombre_pliegue}"
            )
            print(
                f"Variables: {len(columnas_modelo)}"
            )
            print(
                "Penalización: "
                f"{configuracion_variante['penalty']}"
            )
            print(
                f"Alpha: {configuracion_variante['alpha']}"
            )
            print(
                f"L1 ratio: {configuracion_variante['l1_ratio']}"
            )
            print(
                "=" * 70
            )

            escalador, conteos = ajustar_escalador(
                simbolo=simbolo,
                configuracion_pliegue=configuracion_pliegue,
                columnas_modelo=columnas_modelo,
                tamano_lote=tamano_lote,
            )

            pesos = calcular_pesos(
                conteos
            )

            modelo = entrenar_modelo(
                simbolo=simbolo,
                configuracion_pliegue=configuracion_pliegue,
                configuracion_variante=configuracion_variante,
                escalador=escalador,
                pesos=pesos,
                epocas=epocas,
                tamano_lote=tamano_lote,
            )

            (
                reales,
                predicciones,
                probabilidades,
                fechas,
            ) = predecir_validacion(
                simbolo=simbolo,
                configuracion_pliegue=configuracion_pliegue,
                columnas_modelo=columnas_modelo,
                modelo=modelo,
                escalador=escalador,
                tamano_lote=tamano_lote,
            )

            fila = guardar_experimento(
                identificador_ejecucion=identificador_ejecucion,
                simbolo=simbolo,
                nombre_pliegue=nombre_pliegue,
                nombre_variante=nombre_variante,
                configuracion_variante=configuracion_variante,
                epocas=epocas,
                tamano_lote=tamano_lote,
                conteos=conteos,
                pesos=pesos,
                modelo=modelo,
                escalador=escalador,
                reales=reales,
                predicciones=predicciones,
                probabilidades=probabilidades,
                fechas=fechas,
                guardar_modelo=guardar_modelos,
            )

            resultados.append(
                fila
            )

            print(
                "\nResultado:"
            )
            print(
                f"  Accuracy: {fila['accuracy']:.4f}"
            )
            print(
                "  Balanced Accuracy: "
                f"{fila['balanced_accuracy']:.4f}"
            )
            print(
                f"  F1 macro: {fila['f1_macro']:.4f}"
            )
            print(
                f"  F1 BAJA: {fila['f1_baja']:.4f}"
            )
            print(
                f"  F1 SUBE: {fila['f1_sube']:.4f}"
            )
            print(
                f"  PR-AUC BAJA: {fila['pr_auc_baja']:.4f}"
            )
            print(
                f"  PR-AUC SUBE: {fila['pr_auc_sube']:.4f}"
            )
            print(
                f"  Mejor lado: {fila['mejor_lado']}"
            )
            print(
                "  Coeficientes totalmente en cero: "
                f"{fila['coeficientes_totalmente_cero']}"
            )

            del modelo
            del escalador
            del reales
            del predicciones
            del probabilidades
            del fechas
            gc.collect()

    ruta_ejecucion = (
        RUTA_RESULTADOS_ABLACIONES
        / f"resumen_ejecucion_{identificador_ejecucion}.csv"
    )

    pd.DataFrame(
        resultados
    ).to_csv(
        ruta_ejecucion,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nABLACIONES COMPLETADAS"
    )
    print("=" * 70)
    print(
        f"Resumen: {ruta_ejecucion}"
    )
    print(
        f"Historial: {RUTA_HISTORIAL_ABLACIONES}"
    )


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene los argumentos de consola."""

    parser = argparse.ArgumentParser(
        description=(
            "Compara variantes de 63, 52 y 46 variables "
            "y variantes ElasticNet usando validación 2023/2024."
        )
    )

    parser.add_argument(
        "--simbolo",
        required=True,
        choices=SIMBOLOS,
    )

    parser.add_argument(
        "--grupo",
        choices=(
            "reduccion",
            "elasticnet",
            "todo",
        ),
        default="reduccion",
    )

    parser.add_argument(
        "--variantes",
        nargs="+",
        default=None,
        help=(
            "Nombres concretos de variantes. "
            "Sobrescribe --grupo."
        ),
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

    parser.add_argument(
        "--guardar-modelos",
        action="store_true",
        help=(
            "Guarda los joblib de cada experimento. "
            "Por defecto solo se guardan métricas y coeficientes."
        ),
    )

    argumentos = parser.parse_args()

    if argumentos.epocas <= 0:
        parser.error(
            "--epocas debe ser mayor que cero."
        )

    if argumentos.tamano_lote <= 0:
        parser.error(
            "--tamano-lote debe ser mayor que cero."
        )

    return argumentos


def main() -> None:
    """Punto de entrada."""

    argumentos = obtener_argumentos()

    try:
        ejecutar(
            simbolo=argumentos.simbolo,
            grupo=argumentos.grupo,
            nombres_variantes=argumentos.variantes,
            epocas=argumentos.epocas,
            tamano_lote=argumentos.tamano_lote,
            guardar_modelos=argumentos.guardar_modelos,
        )

    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(
            "\nNo se pudieron ejecutar las ablaciones V3A."
        )
        print(
            f"Detalle: {error}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
