from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from cripto.corto_plazo.configuracion import (
    RUTA_DATOS_PREPARADOS,
    RUTA_PROYECTO,
)
from cripto.corto_plazo.gestionar_para_analisis import (
    actualizar_carpeta_para_analisis,
)


SIMBOLOS = (
    "BTCUSDT",
    "ETHUSDT",
)

CLASES = np.array(
    [
        "BAJA",
        "NEUTRAL",
        "SUBE",
    ],
    dtype=object,
)

CODIGOS_CLASES = np.array(
    [
        0,
        1,
        2,
    ],
    dtype=np.int8,
)

RUTA_MANIFIESTO = (
    RUTA_DATOS_PREPARADOS
    / "manifiesto_divisiones_4h.csv"
)

RUTA_MODELOS = (
    RUTA_PROYECTO
    / "modelos_entrenados"
    / "cripto"
    / "corto_plazo"
)


MESES_ES = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre",
}


def cargar_manifiesto(
    simbolo: str,
) -> pd.DataFrame:
    """Carga el manifiesto correspondiente al símbolo."""

    if not RUTA_MANIFIESTO.exists():
        raise FileNotFoundError(
            f"No existe el manifiesto: {RUTA_MANIFIESTO}"
        )

    manifiesto = pd.read_csv(
        RUTA_MANIFIESTO
    )

    columnas_requeridas = {
        "simbolo",
        "division",
        "archivo",
        "desde_archivo",
    }

    columnas_faltantes = columnas_requeridas.difference(
        manifiesto.columns
    )

    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas en el manifiesto: "
            + ", ".join(
                sorted(columnas_faltantes)
            )
        )

    manifiesto = manifiesto.loc[
        manifiesto["simbolo"] == simbolo
    ].copy()

    if manifiesto.empty:
        raise ValueError(
            f"No existen registros para {simbolo}."
        )

    return manifiesto


def obtener_rutas_validacion(
    manifiesto: pd.DataFrame,
) -> list[Path]:
    """Obtiene los archivos de validación, nunca los de prueba."""

    registros = (
        manifiesto
        .loc[
            manifiesto["division"]
            == "validacion"
        ]
        .sort_values("desde_archivo")
    )

    if registros.empty:
        raise ValueError(
            "No existen archivos para la división de validación."
        )

    rutas: list[Path] = []

    for archivo in registros["archivo"]:
        nombre_archivo = (
            str(archivo)
            .replace("\\", "/")
            .rsplit("/", 1)[-1]
        )

        ruta = (
            RUTA_DATOS_PREPARADOS
            / nombre_archivo
        )

        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe el archivo: {ruta}"
            )

        rutas.append(ruta)

    return rutas


def cargar_paquete_modelo(
    simbolo: str,
) -> dict[str, Any]:
    """Carga el modelo y sus metadatos."""

    ruta = (
        RUTA_MODELOS
        / f"modelo_base_{simbolo}_4h.joblib"
    )

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el modelo: {ruta}"
        )

    paquete = joblib.load(
        ruta
    )

    claves_requeridas = {
        "modelo",
        "escalador",
        "columnas_modelo",
        "clases",
        "umbral_clase",
    }

    claves_faltantes = claves_requeridas.difference(
        paquete
    )

    if claves_faltantes:
        raise ValueError(
            "Faltan elementos en el paquete del modelo: "
            + ", ".join(
                sorted(claves_faltantes)
            )
        )

    modelo = paquete["modelo"]

    if not hasattr(
        modelo,
        "predict_proba",
    ):
        raise ValueError(
            "El modelo no permite obtener probabilidades."
        )

    return paquete


def cargar_ajuste(
    simbolo: str,
) -> dict[str, float]:
    """Carga los multiplicadores elegidos para el símbolo."""

    ruta = (
        RUTA_MODELOS
        / f"ajuste_decision_{simbolo}_4h.json"
    )

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el ajuste de decisión: {ruta}"
        )

    with ruta.open(
        "r",
        encoding="utf-8",
    ) as archivo:
        datos = json.load(
            archivo
        )

    multiplicadores = datos.get(
        "multiplicadores",
        {},
    )

    requeridos = {
        "BAJA",
        "NEUTRAL",
        "SUBE",
    }

    faltantes = requeridos.difference(
        multiplicadores
    )

    if faltantes:
        raise ValueError(
            "Faltan multiplicadores en el ajuste: "
            + ", ".join(
                sorted(faltantes)
            )
        )

    return {
        clase: float(
            multiplicadores[clase]
        )
        for clase in requeridos
    }


def clasificar_objetivo(
    rendimientos: pd.Series,
    umbral: float,
) -> np.ndarray:
    """Convierte el rendimiento futuro en códigos de clase."""

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
            valores <= -umbral,
            valores >= umbral,
        ],
        [
            0,
            2,
        ],
        default=1,
    ).astype(
        np.int8
    )


def recorrer_lotes(
    total: int,
    tamano_lote: int,
):
    """Genera los límites de cada lote."""

    for inicio in range(
        0,
        total,
        tamano_lote,
    ):
        final = min(
            inicio + tamano_lote,
            total,
        )

        yield inicio, final


def cargar_validacion(
    rutas: list[Path],
    paquete: dict[str, Any],
    tamano_lote: int,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Carga fechas, clases reales y probabilidades de validación."""

    modelo = paquete["modelo"]
    escalador = paquete["escalador"]

    columnas_modelo = list(
        paquete["columnas_modelo"]
    )

    umbral = float(
        paquete["umbral_clase"]
    )

    clases_modelo = [
        str(clase)
        for clase in modelo.classes_
    ]

    if set(clases_modelo) != set(CLASES):
        raise ValueError(
            "Las clases del modelo no coinciden con "
            "BAJA, NEUTRAL y SUBE."
        )

    orden_columnas = [
        clases_modelo.index(
            str(clase)
        )
        for clase in CLASES
    ]

    fechas_totales: list[np.ndarray] = []
    reales_totales: list[np.ndarray] = []
    probabilidades_totales: list[np.ndarray] = []

    print("\nGENERANDO PREDICCIONES DE VALIDACIÓN")
    print("=" * 70)

    for ruta in rutas:
        print(f"Procesando: {ruta.name}")

        columnas = [
            "fecha_apertura",
            *columnas_modelo,
            "rendimiento_objetivo",
        ]

        datos = pd.read_parquet(
            ruta,
            columns=columnas,
        )

        fechas = pd.to_datetime(
            datos["fecha_apertura"],
            utc=True,
            errors="coerce",
        )

        if fechas.isna().any():
            raise ValueError(
                f"Existen fechas inválidas en: {ruta}"
            )

        variables = (
            datos[columnas_modelo]
            .to_numpy(
                dtype="float32"
            )
        )

        if not np.isfinite(
            variables
        ).all():
            raise ValueError(
                f"Existen variables no finitas en: {ruta}"
            )

        reales = clasificar_objetivo(
            rendimientos=datos[
                "rendimiento_objetivo"
            ],
            umbral=umbral,
        )

        fechas_totales.append(
            fechas.to_numpy()
        )

        reales_totales.append(
            reales
        )

        for inicio, final in recorrer_lotes(
            total=len(variables),
            tamano_lote=tamano_lote,
        ):
            variables_lote = escalador.transform(
                variables[inicio:final]
            )

            probabilidades_lote = modelo.predict_proba(
                variables_lote
            )[:, orden_columnas]

            probabilidades_totales.append(
                probabilidades_lote.astype(
                    "float32"
                )
            )

        del datos
        del variables
        del reales
        gc.collect()

    fechas_unidas = pd.to_datetime(
        np.concatenate(
            fechas_totales
        ),
        utc=True,
    )

    return (
        fechas_unidas,
        np.concatenate(
            reales_totales
        ),
        np.concatenate(
            probabilidades_totales
        ),
    )


def aplicar_decision_original(
    probabilidades: np.ndarray,
) -> np.ndarray:
    """Aplica la decisión original del modelo."""

    return np.argmax(
        probabilidades,
        axis=1,
    ).astype(
        np.int8
    )


def aplicar_decision_ajustada(
    probabilidades: np.ndarray,
    multiplicadores: dict[str, float],
) -> np.ndarray:
    """Aplica los multiplicadores seleccionados en validación."""

    puntuaciones = probabilidades.copy()

    puntuaciones[:, 0] *= multiplicadores[
        "BAJA"
    ]

    puntuaciones[:, 1] *= multiplicadores[
        "NEUTRAL"
    ]

    puntuaciones[:, 2] *= multiplicadores[
        "SUBE"
    ]

    return np.argmax(
        puntuaciones,
        axis=1,
    ).astype(
        np.int8
    )


def dividir_seguro(
    numerador: int,
    denominador: int,
) -> float:
    """Calcula un porcentaje evitando divisiones entre cero."""

    if denominador == 0:
        return 0.0

    return (
        numerador
        / denominador
        * 100
    )


def calcular_metricas_periodo(
    mes: str,
    tipo_decision: str,
    reales: np.ndarray,
    predicciones: np.ndarray,
) -> dict[str, Any]:
    """Calcula las métricas de un mes y una regla de decisión."""

    matriz = confusion_matrix(
        reales,
        predicciones,
        labels=CODIGOS_CLASES,
    )

    recalls = recall_score(
        reales,
        predicciones,
        labels=CODIGOS_CLASES,
        average=None,
        zero_division=0,
    )

    muestras = len(
        reales
    )

    reales_por_clase = [
        int(
            (reales == codigo).sum()
        )
        for codigo in CODIGOS_CLASES
    ]

    predicciones_por_clase = [
        int(
            (predicciones == codigo).sum()
        )
        for codigo in CODIGOS_CLASES
    ]

    return {
        "mes": mes,
        "tipo_decision": tipo_decision,
        "muestras": muestras,
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
            labels=CODIGOS_CLASES,
            average="macro",
            zero_division=0,
        ),
        "recall_baja": float(
            recalls[0]
        ),
        "recall_neutral": float(
            recalls[1]
        ),
        "recall_sube": float(
            recalls[2]
        ),
        "reales_baja": reales_por_clase[0],
        "reales_neutral": reales_por_clase[1],
        "reales_sube": reales_por_clase[2],
        "porcentaje_real_baja": dividir_seguro(
            reales_por_clase[0],
            muestras,
        ),
        "porcentaje_real_neutral": dividir_seguro(
            reales_por_clase[1],
            muestras,
        ),
        "porcentaje_real_sube": dividir_seguro(
            reales_por_clase[2],
            muestras,
        ),
        "predicciones_baja": predicciones_por_clase[0],
        "predicciones_neutral": predicciones_por_clase[1],
        "predicciones_sube": predicciones_por_clase[2],
        "porcentaje_predicho_baja": dividir_seguro(
            predicciones_por_clase[0],
            muestras,
        ),
        "porcentaje_predicho_neutral": dividir_seguro(
            predicciones_por_clase[1],
            muestras,
        ),
        "porcentaje_predicho_sube": dividir_seguro(
            predicciones_por_clase[2],
            muestras,
        ),
        "cm_baja_baja": int(matriz[0, 0]),
        "cm_baja_neutral": int(matriz[0, 1]),
        "cm_baja_sube": int(matriz[0, 2]),
        "cm_neutral_baja": int(matriz[1, 0]),
        "cm_neutral_neutral": int(matriz[1, 1]),
        "cm_neutral_sube": int(matriz[1, 2]),
        "cm_sube_baja": int(matriz[2, 0]),
        "cm_sube_neutral": int(matriz[2, 1]),
        "cm_sube_sube": int(matriz[2, 2]),
    }


def evaluar_por_mes(
    fechas: pd.DatetimeIndex,
    reales: np.ndarray,
    predicciones_originales: np.ndarray,
    predicciones_ajustadas: np.ndarray,
) -> pd.DataFrame:
    """Evalúa la decisión original y ajustada en cada mes."""

    periodos = (
        fechas
        .tz_convert(None)
        .to_period("M")
        .astype(str)
    )

    resultados: list[dict[str, Any]] = []

    for mes in sorted(
        np.unique(periodos)
    ):
        mascara = (
            periodos == mes
        )

        reales_mes = reales[
            mascara
        ]

        resultados.append(
            calcular_metricas_periodo(
                mes=mes,
                tipo_decision="original",
                reales=reales_mes,
                predicciones=predicciones_originales[
                    mascara
                ],
            )
        )

        resultados.append(
            calcular_metricas_periodo(
                mes=mes,
                tipo_decision="ajustada",
                reales=reales_mes,
                predicciones=predicciones_ajustadas[
                    mascara
                ],
            )
        )

    return pd.DataFrame(
        resultados
    )


def crear_comparacion(
    resultados: pd.DataFrame,
) -> pd.DataFrame:
    """Crea una fila comparativa por mes."""

    original = (
        resultados
        .loc[
            resultados["tipo_decision"]
            == "original"
        ]
        .set_index("mes")
    )

    ajustada = (
        resultados
        .loc[
            resultados["tipo_decision"]
            == "ajustada"
        ]
        .set_index("mes")
    )

    meses = original.index.intersection(
        ajustada.index
    )

    filas: list[dict[str, Any]] = []

    for mes in meses:
        fila_original = original.loc[
            mes
        ]

        fila_ajustada = ajustada.loc[
            mes
        ]

        filas.append(
            {
                "mes": mes,
                "muestras": int(
                    fila_original["muestras"]
                ),
                "accuracy_original": float(
                    fila_original["accuracy"]
                ),
                "accuracy_ajustada": float(
                    fila_ajustada["accuracy"]
                ),
                "delta_accuracy": float(
                    fila_ajustada["accuracy"]
                    - fila_original["accuracy"]
                ),
                "balanced_accuracy_original": float(
                    fila_original[
                        "balanced_accuracy"
                    ]
                ),
                "balanced_accuracy_ajustada": float(
                    fila_ajustada[
                        "balanced_accuracy"
                    ]
                ),
                "delta_balanced_accuracy": float(
                    fila_ajustada[
                        "balanced_accuracy"
                    ]
                    - fila_original[
                        "balanced_accuracy"
                    ]
                ),
                "f1_macro_original": float(
                    fila_original["f1_macro"]
                ),
                "f1_macro_ajustada": float(
                    fila_ajustada["f1_macro"]
                ),
                "delta_f1_macro": float(
                    fila_ajustada["f1_macro"]
                    - fila_original["f1_macro"]
                ),
                "recall_baja_original": float(
                    fila_original["recall_baja"]
                ),
                "recall_baja_ajustada": float(
                    fila_ajustada["recall_baja"]
                ),
                "delta_recall_baja": float(
                    fila_ajustada["recall_baja"]
                    - fila_original["recall_baja"]
                ),
                "recall_neutral_original": float(
                    fila_original[
                        "recall_neutral"
                    ]
                ),
                "recall_neutral_ajustada": float(
                    fila_ajustada[
                        "recall_neutral"
                    ]
                ),
                "delta_recall_neutral": float(
                    fila_ajustada[
                        "recall_neutral"
                    ]
                    - fila_original[
                        "recall_neutral"
                    ]
                ),
                "recall_sube_original": float(
                    fila_original["recall_sube"]
                ),
                "recall_sube_ajustada": float(
                    fila_ajustada["recall_sube"]
                ),
                "delta_recall_sube": float(
                    fila_ajustada["recall_sube"]
                    - fila_original["recall_sube"]
                ),
                "predicho_baja_original_pct": float(
                    fila_original[
                        "porcentaje_predicho_baja"
                    ]
                ),
                "predicho_baja_ajustada_pct": float(
                    fila_ajustada[
                        "porcentaje_predicho_baja"
                    ]
                ),
                "predicho_neutral_original_pct": float(
                    fila_original[
                        "porcentaje_predicho_neutral"
                    ]
                ),
                "predicho_neutral_ajustada_pct": float(
                    fila_ajustada[
                        "porcentaje_predicho_neutral"
                    ]
                ),
                "predicho_sube_original_pct": float(
                    fila_original[
                        "porcentaje_predicho_sube"
                    ]
                ),
                "predicho_sube_ajustada_pct": float(
                    fila_ajustada[
                        "porcentaje_predicho_sube"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        filas
    ).sort_values(
        "mes"
    ).reset_index(
        drop=True
    )


def nombre_mes(
    periodo: str,
) -> str:
    """Convierte 2025-01 en enero 2025."""

    anio_texto, mes_texto = periodo.split(
        "-"
    )

    return (
        f"{MESES_ES[int(mes_texto)]} "
        f"{anio_texto}"
    )


def generar_resumen_markdown(
    simbolo: str,
    multiplicadores: dict[str, float],
    comparacion: pd.DataFrame,
    ruta: Path,
) -> None:
    """Genera un resumen mensual legible y listo para enviar."""

    meses_mejora_f1 = int(
        (
            comparacion["delta_f1_macro"]
            > 0
        ).sum()
    )

    meses_mejora_balanceada = int(
        (
            comparacion[
                "delta_balanced_accuracy"
            ]
            > 0
        ).sum()
    )

    total_meses = len(
        comparacion
    )

    lineas = [
        f"# Validación mensual 2025 — {simbolo}",
        "",
        "Esta validación compara mes por mes la decisión original "
        "del modelo y la decisión ajustada.",
        "",
        "La división de prueba de 2026 no fue utilizada.",
        "",
        "## Multiplicadores evaluados",
        "",
        f"- **BAJA:** {multiplicadores['BAJA']:.2f}",
        f"- **NEUTRAL:** {multiplicadores['NEUTRAL']:.2f}",
        f"- **SUBE:** {multiplicadores['SUBE']:.2f}",
        "",
        "## Resumen de estabilidad",
        "",
        f"- El F1-score macro mejoró en **{meses_mejora_f1} de {total_meses} meses**.",
        f"- Balanced Accuracy mejoró en **{meses_mejora_balanceada} de {total_meses} meses**.",
        f"- Cambio medio de Accuracy: **{comparacion['delta_accuracy'].mean():+.4f}**.",
        f"- Cambio medio de Balanced Accuracy: **{comparacion['delta_balanced_accuracy'].mean():+.4f}**.",
        f"- Cambio medio de F1-score macro: **{comparacion['delta_f1_macro'].mean():+.4f}**.",
        "",
        "## Comparación mensual",
        "",
        "| Mes | Accuracy original | Accuracy ajustada | Δ Accuracy | Balanced original | Balanced ajustada | Δ Balanced | F1 macro original | F1 macro ajustada | Δ F1 macro |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, fila in comparacion.iterrows():
        lineas.append(
            "| "
            f"{nombre_mes(str(fila['mes']))} | "
            f"{float(fila['accuracy_original']):.4f} | "
            f"{float(fila['accuracy_ajustada']):.4f} | "
            f"{float(fila['delta_accuracy']):+.4f} | "
            f"{float(fila['balanced_accuracy_original']):.4f} | "
            f"{float(fila['balanced_accuracy_ajustada']):.4f} | "
            f"{float(fila['delta_balanced_accuracy']):+.4f} | "
            f"{float(fila['f1_macro_original']):.4f} | "
            f"{float(fila['f1_macro_ajustada']):.4f} | "
            f"{float(fila['delta_f1_macro']):+.4f} |"
        )

    lineas.extend(
        [
            "",
            "## Recall por clase",
            "",
            "| Mes | BAJA original | BAJA ajustada | NEUTRAL original | NEUTRAL ajustada | SUBE original | SUBE ajustada |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, fila in comparacion.iterrows():
        lineas.append(
            "| "
            f"{nombre_mes(str(fila['mes']))} | "
            f"{float(fila['recall_baja_original']):.4f} | "
            f"{float(fila['recall_baja_ajustada']):.4f} | "
            f"{float(fila['recall_neutral_original']):.4f} | "
            f"{float(fila['recall_neutral_ajustada']):.4f} | "
            f"{float(fila['recall_sube_original']):.4f} | "
            f"{float(fila['recall_sube_ajustada']):.4f} |"
        )

    lineas.extend(
        [
            "",
            "## Distribución de predicciones",
            "",
            "| Mes | BAJA original | BAJA ajustada | NEUTRAL original | NEUTRAL ajustada | SUBE original | SUBE ajustada |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, fila in comparacion.iterrows():
        lineas.append(
            "| "
            f"{nombre_mes(str(fila['mes']))} | "
            f"{float(fila['predicho_baja_original_pct']):.2f} % | "
            f"{float(fila['predicho_baja_ajustada_pct']):.2f} % | "
            f"{float(fila['predicho_neutral_original_pct']):.2f} % | "
            f"{float(fila['predicho_neutral_ajustada_pct']):.2f} % | "
            f"{float(fila['predicho_sube_original_pct']):.2f} % | "
            f"{float(fila['predicho_sube_ajustada_pct']):.2f} % |"
        )

    ruta.write_text(
        "\n".join(lineas) + "\n",
        encoding="utf-8",
    )


def mostrar_resultado(
    simbolo: str,
    comparacion: pd.DataFrame,
) -> None:
    """Muestra un resumen de la estabilidad mensual."""

    total_meses = len(
        comparacion
    )

    mejora_f1 = int(
        (
            comparacion["delta_f1_macro"]
            > 0
        ).sum()
    )

    mejora_balanceada = int(
        (
            comparacion[
                "delta_balanced_accuracy"
            ]
            > 0
        ).sum()
    )

    print("\nRESULTADO DE ESTABILIDAD MENSUAL")
    print("=" * 70)
    print(f"Símbolo: {simbolo}")
    print(
        f"Meses con mejora de F1-score macro: "
        f"{mejora_f1} de {total_meses}"
    )
    print(
        f"Meses con mejora de Balanced Accuracy: "
        f"{mejora_balanceada} de {total_meses}"
    )
    print(
        f"Cambio medio de Accuracy: "
        f"{comparacion['delta_accuracy'].mean():+.4f}"
    )
    print(
        f"Cambio medio de Balanced Accuracy: "
        f"{comparacion['delta_balanced_accuracy'].mean():+.4f}"
    )
    print(
        f"Cambio medio de F1-score macro: "
        f"{comparacion['delta_f1_macro'].mean():+.4f}"
    )


def validar_estabilidad_mensual(
    simbolo: str,
    tamano_lote: int,
) -> None:
    """Ejecuta la validación mensual completa sobre 2025."""

    print("\nVALIDACIÓN TEMPORAL MENSUAL")
    print("=" * 70)
    print(f"Símbolo: {simbolo}")
    print("División utilizada: validación de 2025")
    print("División de prueba de 2026: no utilizada")

    manifiesto = cargar_manifiesto(
        simbolo=simbolo
    )

    rutas_validacion = obtener_rutas_validacion(
        manifiesto=manifiesto
    )

    paquete = cargar_paquete_modelo(
        simbolo=simbolo
    )

    multiplicadores = cargar_ajuste(
        simbolo=simbolo
    )

    fechas, reales, probabilidades = cargar_validacion(
        rutas=rutas_validacion,
        paquete=paquete,
        tamano_lote=tamano_lote,
    )

    predicciones_originales = aplicar_decision_original(
        probabilidades=probabilidades
    )

    predicciones_ajustadas = aplicar_decision_ajustada(
        probabilidades=probabilidades,
        multiplicadores=multiplicadores,
    )

    resultados = evaluar_por_mes(
        fechas=fechas,
        reales=reales,
        predicciones_originales=predicciones_originales,
        predicciones_ajustadas=predicciones_ajustadas,
    )

    comparacion = crear_comparacion(
        resultados=resultados
    )

    RUTA_MODELOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_resultados = (
        RUTA_MODELOS
        / f"validacion_mensual_{simbolo}_4h.csv"
    )

    ruta_comparacion = (
        RUTA_MODELOS
        / f"comparacion_validacion_mensual_{simbolo}_4h.csv"
    )

    ruta_resumen = (
        RUTA_MODELOS
        / f"resumen_validacion_mensual_{simbolo}_4h.md"
    )

    resultados.to_csv(
        ruta_resultados,
        index=False,
        encoding="utf-8-sig",
    )

    comparacion.to_csv(
        ruta_comparacion,
        index=False,
        encoding="utf-8-sig",
    )

    generar_resumen_markdown(
        simbolo=simbolo,
        multiplicadores=multiplicadores,
        comparacion=comparacion,
        ruta=ruta_resumen,
    )

    mostrar_resultado(
        simbolo=simbolo,
        comparacion=comparacion,
    )

    print("\nArchivos generados:")
    print(f"- {ruta_resultados}")
    print(f"- {ruta_comparacion}")
    print(f"- {ruta_resumen}")

    actualizar_carpeta_para_analisis()

    print(
        "\nLa carpeta para_analisis fue actualizada."
    )
    print(
        "La división de prueba de 2026 no fue utilizada."
    )


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene los argumentos del comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Compara mes por mes la decisión original "
            "y la decisión ajustada usando validación de 2025."
        )
    )

    parser.add_argument(
        "--simbolo",
        required=True,
        choices=SIMBOLOS,
    )

    parser.add_argument(
        "--tamano-lote",
        type=int,
        default=100000,
    )

    argumentos = parser.parse_args()

    if argumentos.tamano_lote <= 0:
        parser.error(
            "--tamano-lote debe ser mayor que cero."
        )

    return argumentos


def main() -> None:
    """Punto de entrada."""

    argumentos = obtener_argumentos()

    try:
        validar_estabilidad_mensual(
            simbolo=argumentos.simbolo,
            tamano_lote=argumentos.tamano_lote,
        )
    except (
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(
            "\nNo se pudo validar la estabilidad mensual."
        )
        print(f"Detalle: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
