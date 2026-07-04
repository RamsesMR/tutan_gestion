from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cripto.corto_plazo.gestionar_para_analisis import (
    actualizar_carpeta_para_analisis,
)
from cripto.corto_plazo.validar_estabilidad_mensual import (
    CODIGOS_CLASES,
    RUTA_MODELOS,
    SIMBOLOS,
    aplicar_decision_original,
    calcular_metricas_periodo,
    cargar_manifiesto,
    cargar_paquete_modelo,
    cargar_validacion,
    nombre_mes,
    obtener_rutas_validacion,
)


def crear_valores_rango(
    minimo: float,
    maximo: float,
    paso: float,
) -> np.ndarray:
    """Crea un rango decimal estable e incluye ambos extremos."""

    cantidad = int(
        round(
            (maximo - minimo)
            / paso
        )
    )

    valores = [
        round(
            minimo + indice * paso,
            6,
        )
        for indice in range(
            cantidad + 1
        )
    ]

    return np.array(
        valores,
        dtype="float64",
    )


def aplicar_multiplicadores(
    probabilidades: np.ndarray,
    multiplicador_baja: float,
    multiplicador_sube: float,
) -> np.ndarray:
    """
    Aplica multiplicadores a BAJA y SUBE.

    NEUTRAL conserva un multiplicador de 1. La lógica mantiene el mismo
    desempate que np.argmax con el orden BAJA, NEUTRAL y SUBE.
    """

    puntuacion_baja = (
        probabilidades[:, 0]
        * multiplicador_baja
    )

    puntuacion_neutral = probabilidades[
        :,
        1,
    ]

    puntuacion_sube = (
        probabilidades[:, 2]
        * multiplicador_sube
    )

    predicciones = np.zeros(
        len(probabilidades),
        dtype=np.int8,
    )

    mascara_neutral = (
        (puntuacion_neutral > puntuacion_baja)
        & (puntuacion_neutral >= puntuacion_sube)
    )

    mascara_sube = (
        (puntuacion_sube > puntuacion_baja)
        & (puntuacion_sube > puntuacion_neutral)
    )

    predicciones[
        mascara_neutral
    ] = 1

    predicciones[
        mascara_sube
    ] = 2

    return predicciones


def calcular_metricas_busqueda(
    reales: np.ndarray,
    predicciones: np.ndarray,
) -> tuple[float, float, float]:
    """Calcula Accuracy, Balanced Accuracy y F1-score macro."""

    codigos = (
        reales.astype(
            np.int64,
            copy=False,
        )
        * 3
        + predicciones.astype(
            np.int64,
            copy=False,
        )
    )

    matriz = np.bincount(
        codigos,
        minlength=9,
    ).reshape(
        3,
        3,
    )

    total = int(
        matriz.sum()
    )

    if total == 0:
        raise ValueError(
            "No existen muestras para calcular métricas."
        )

    diagonal = np.diag(
        matriz
    ).astype(
        "float64"
    )

    reales_por_clase = matriz.sum(
        axis=1
    ).astype(
        "float64"
    )

    predichas_por_clase = matriz.sum(
        axis=0
    ).astype(
        "float64"
    )

    recalls = np.divide(
        diagonal,
        reales_por_clase,
        out=np.zeros_like(
            diagonal
        ),
        where=(
            reales_por_clase
            != 0
        ),
    )

    precisiones = np.divide(
        diagonal,
        predichas_por_clase,
        out=np.zeros_like(
            diagonal
        ),
        where=(
            predichas_por_clase
            != 0
        ),
    )

    f1_por_clase = np.divide(
        2
        * precisiones
        * recalls,
        precisiones
        + recalls,
        out=np.zeros_like(
            diagonal
        ),
        where=(
            precisiones
            + recalls
            != 0
        ),
    )

    accuracy = float(
        diagonal.sum()
        / total
    )

    balanced_accuracy = float(
        recalls.mean()
    )

    f1_macro = float(
        f1_por_clase.mean()
    )

    return (
        accuracy,
        balanced_accuracy,
        f1_macro,
    )


def evaluar_combinacion(
    reales: np.ndarray,
    probabilidades: np.ndarray,
    multiplicador_baja: float,
    multiplicador_sube: float,
) -> dict[str, float]:
    """Evalúa una combinación concreta de multiplicadores."""

    predicciones = aplicar_multiplicadores(
        probabilidades=probabilidades,
        multiplicador_baja=multiplicador_baja,
        multiplicador_sube=multiplicador_sube,
    )

    (
        accuracy,
        balanced_accuracy,
        f1_macro,
    ) = calcular_metricas_busqueda(
        reales=reales,
        predicciones=predicciones,
    )

    return {
        "multiplicador_baja": float(
            multiplicador_baja
        ),
        "multiplicador_neutral": 1.0,
        "multiplicador_sube": float(
            multiplicador_sube
        ),
        "accuracy": accuracy,
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "f1_macro": f1_macro,
    }


def evaluar_rejilla(
    reales: np.ndarray,
    probabilidades: np.ndarray,
    valores_baja: np.ndarray,
    valores_sube: np.ndarray,
    combinaciones_evaluadas: set[
        tuple[float, float]
    ],
) -> list[dict[str, float]]:
    """Evalúa una rejilla de multiplicadores sin repetir combinaciones."""

    resultados: list[
        dict[str, float]
    ] = []

    for valor_baja in valores_baja:
        for valor_sube in valores_sube:
            clave = (
                round(
                    float(
                        valor_baja
                    ),
                    6,
                ),
                round(
                    float(
                        valor_sube
                    ),
                    6,
                ),
            )

            if clave in combinaciones_evaluadas:
                continue

            combinaciones_evaluadas.add(
                clave
            )

            resultados.append(
                evaluar_combinacion(
                    reales=reales,
                    probabilidades=probabilidades,
                    multiplicador_baja=clave[0],
                    multiplicador_sube=clave[1],
                )
            )

    return resultados


def seleccionar_mejor(
    resultados: pd.DataFrame,
    accuracy_minima: float,
) -> pd.Series:
    """
    Selecciona la combinación con mayor F1-score macro.

    En caso de empate prioriza Balanced Accuracy y después Accuracy.
    """

    elegibles = resultados.loc[
        resultados["accuracy"]
        >= accuracy_minima
    ].copy()

    if elegibles.empty:
        raise ValueError(
            "Ninguna combinación respetó la pérdida "
            "máxima permitida de Accuracy."
        )

    elegibles = elegibles.sort_values(
        [
            "f1_macro",
            "balanced_accuracy",
            "accuracy",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    return elegibles.iloc[0]


def buscar_mejor_ajuste(
    reales: np.ndarray,
    probabilidades: np.ndarray,
    multiplicador_minimo: float,
    multiplicador_maximo: float,
    paso_grueso: float,
    paso_fino: float,
    perdida_maxima_accuracy: float,
) -> dict[str, float]:
    """Busca los multiplicadores utilizando únicamente el periodo anterior."""

    predicciones_originales = (
        aplicar_decision_original(
            probabilidades=probabilidades
        )
    )

    (
        accuracy_original,
        balanced_original,
        f1_original,
    ) = calcular_metricas_busqueda(
        reales=reales,
        predicciones=predicciones_originales,
    )

    accuracy_minima = (
        accuracy_original
        - perdida_maxima_accuracy
    )

    valores_gruesos = crear_valores_rango(
        minimo=multiplicador_minimo,
        maximo=multiplicador_maximo,
        paso=paso_grueso,
    )

    combinaciones_evaluadas: set[
        tuple[float, float]
    ] = set()

    resultados = evaluar_rejilla(
        reales=reales,
        probabilidades=probabilidades,
        valores_baja=valores_gruesos,
        valores_sube=valores_gruesos,
        combinaciones_evaluadas=combinaciones_evaluadas,
    )

    resultados_gruesos = pd.DataFrame(
        resultados
    )

    mejor_grueso = seleccionar_mejor(
        resultados=resultados_gruesos,
        accuracy_minima=accuracy_minima,
    )

    centro_baja = float(
        mejor_grueso[
            "multiplicador_baja"
        ]
    )

    centro_sube = float(
        mejor_grueso[
            "multiplicador_sube"
        ]
    )

    minimo_baja = max(
        multiplicador_minimo,
        centro_baja
        - paso_grueso,
    )

    maximo_baja = min(
        multiplicador_maximo,
        centro_baja
        + paso_grueso,
    )

    minimo_sube = max(
        multiplicador_minimo,
        centro_sube
        - paso_grueso,
    )

    maximo_sube = min(
        multiplicador_maximo,
        centro_sube
        + paso_grueso,
    )

    valores_finos_baja = crear_valores_rango(
        minimo=minimo_baja,
        maximo=maximo_baja,
        paso=paso_fino,
    )

    valores_finos_sube = crear_valores_rango(
        minimo=minimo_sube,
        maximo=maximo_sube,
        paso=paso_fino,
    )

    resultados.extend(
        evaluar_rejilla(
            reales=reales,
            probabilidades=probabilidades,
            valores_baja=valores_finos_baja,
            valores_sube=valores_finos_sube,
            combinaciones_evaluadas=combinaciones_evaluadas,
        )
    )

    resultados_totales = pd.DataFrame(
        resultados
    )

    mejor = seleccionar_mejor(
        resultados=resultados_totales,
        accuracy_minima=accuracy_minima,
    )

    return {
        "multiplicador_baja": float(
            mejor[
                "multiplicador_baja"
            ]
        ),
        "multiplicador_neutral": 1.0,
        "multiplicador_sube": float(
            mejor[
                "multiplicador_sube"
            ]
        ),
        "accuracy_ajuste_original": (
            accuracy_original
        ),
        "balanced_accuracy_ajuste_original": (
            balanced_original
        ),
        "f1_macro_ajuste_original": (
            f1_original
        ),
        "accuracy_ajuste_seleccionada": float(
            mejor["accuracy"]
        ),
        "balanced_accuracy_ajuste_seleccionada": float(
            mejor[
                "balanced_accuracy"
            ]
        ),
        "f1_macro_ajuste_seleccionada": float(
            mejor[
                "f1_macro"
            ]
        ),
    }


def construir_comparacion(
    fila_original: dict[str, Any],
    fila_ajustada: dict[str, Any],
    datos_ajuste: dict[str, Any],
) -> dict[str, Any]:
    """Construye una fila comparativa del mes evaluado."""

    return {
        "mes_evaluado": fila_original[
            "mes"
        ],
        "desde_ajuste": datos_ajuste[
            "desde_ajuste"
        ],
        "hasta_ajuste": datos_ajuste[
            "hasta_ajuste"
        ],
        "muestras_ajuste": datos_ajuste[
            "muestras_ajuste"
        ],
        "muestras_evaluacion": fila_original[
            "muestras"
        ],
        "multiplicador_baja": datos_ajuste[
            "multiplicador_baja"
        ],
        "multiplicador_neutral": 1.0,
        "multiplicador_sube": datos_ajuste[
            "multiplicador_sube"
        ],
        "accuracy_original": fila_original[
            "accuracy"
        ],
        "accuracy_ajustada": fila_ajustada[
            "accuracy"
        ],
        "delta_accuracy": (
            fila_ajustada["accuracy"]
            - fila_original["accuracy"]
        ),
        "balanced_accuracy_original": fila_original[
            "balanced_accuracy"
        ],
        "balanced_accuracy_ajustada": fila_ajustada[
            "balanced_accuracy"
        ],
        "delta_balanced_accuracy": (
            fila_ajustada[
                "balanced_accuracy"
            ]
            - fila_original[
                "balanced_accuracy"
            ]
        ),
        "f1_macro_original": fila_original[
            "f1_macro"
        ],
        "f1_macro_ajustada": fila_ajustada[
            "f1_macro"
        ],
        "delta_f1_macro": (
            fila_ajustada["f1_macro"]
            - fila_original["f1_macro"]
        ),
        "recall_baja_original": fila_original[
            "recall_baja"
        ],
        "recall_baja_ajustada": fila_ajustada[
            "recall_baja"
        ],
        "delta_recall_baja": (
            fila_ajustada["recall_baja"]
            - fila_original["recall_baja"]
        ),
        "recall_neutral_original": fila_original[
            "recall_neutral"
        ],
        "recall_neutral_ajustada": fila_ajustada[
            "recall_neutral"
        ],
        "delta_recall_neutral": (
            fila_ajustada[
                "recall_neutral"
            ]
            - fila_original[
                "recall_neutral"
            ]
        ),
        "recall_sube_original": fila_original[
            "recall_sube"
        ],
        "recall_sube_ajustada": fila_ajustada[
            "recall_sube"
        ],
        "delta_recall_sube": (
            fila_ajustada["recall_sube"]
            - fila_original["recall_sube"]
        ),
        "predicho_baja_original_pct": fila_original[
            "porcentaje_predicho_baja"
        ],
        "predicho_baja_ajustada_pct": fila_ajustada[
            "porcentaje_predicho_baja"
        ],
        "predicho_neutral_original_pct": fila_original[
            "porcentaje_predicho_neutral"
        ],
        "predicho_neutral_ajustada_pct": fila_ajustada[
            "porcentaje_predicho_neutral"
        ],
        "predicho_sube_original_pct": fila_original[
            "porcentaje_predicho_sube"
        ],
        "predicho_sube_ajustada_pct": fila_ajustada[
            "porcentaje_predicho_sube"
        ],
    }


def generar_resumen_markdown(
    simbolo: str,
    comparacion: pd.DataFrame,
    mes_inicial: int,
    perdida_maxima_accuracy: float,
    ruta: Path,
) -> None:
    """Genera un resumen legible de la validación walk-forward."""

    total_meses = len(
        comparacion
    )

    meses_mejora_f1 = int(
        (
            comparacion[
                "delta_f1_macro"
            ]
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

    meses_mejora_accuracy = int(
        (
            comparacion[
                "delta_accuracy"
            ]
            > 0
        ).sum()
    )

    lineas = [
        f"# Validación walk-forward 2025 — {simbolo}",
        "",
        "Cada mes fue evaluado con multiplicadores elegidos "
        "exclusivamente mediante los meses anteriores.",
        "",
        "Antes de ajustar cada mes se excluyeron las últimas "
        "cuatro horas del periodo anterior para impedir que el "
        "objetivo cruce la frontera temporal.",
        "",
        "La división de prueba de 2026 no fue utilizada.",
        "",
        "## Configuración",
        "",
        f"- **Primer mes evaluado:** {nombre_mes(f'2025-{mes_inicial:02d}')}",
        "- **Criterio:** mayor F1-score macro; después Balanced Accuracy y Accuracy.",
        f"- **Pérdida máxima de Accuracy durante el ajuste:** {perdida_maxima_accuracy:.4f}",
        "",
        "## Resumen",
        "",
        f"- F1-score macro mejoró en **{meses_mejora_f1} de {total_meses} meses**.",
        f"- Balanced Accuracy mejoró en **{meses_mejora_balanceada} de {total_meses} meses**.",
        f"- Accuracy mejoró en **{meses_mejora_accuracy} de {total_meses} meses**.",
        f"- Cambio medio de Accuracy: **{comparacion['delta_accuracy'].mean():+.4f}**.",
        f"- Cambio medio de Balanced Accuracy: **{comparacion['delta_balanced_accuracy'].mean():+.4f}**.",
        f"- Cambio medio de F1-score macro: **{comparacion['delta_f1_macro'].mean():+.4f}**.",
        "",
        "## Comparación mensual fuera de muestra",
        "",
        "| Mes evaluado | Multiplicador BAJA | Multiplicador SUBE | Accuracy original | Accuracy ajustada | Δ Accuracy | Balanced original | Balanced ajustada | Δ Balanced | F1 original | F1 ajustada | Δ F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, fila in comparacion.iterrows():
        lineas.append(
            "| "
            f"{nombre_mes(str(fila['mes_evaluado']))} | "
            f"{float(fila['multiplicador_baja']):.2f} | "
            f"{float(fila['multiplicador_sube']):.2f} | "
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
            f"{nombre_mes(str(fila['mes_evaluado']))} | "
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
            f"{nombre_mes(str(fila['mes_evaluado']))} | "
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


def validar_ajuste_walk_forward(
    simbolo: str,
    tamano_lote: int,
    mes_inicial: int,
    multiplicador_minimo: float,
    multiplicador_maximo: float,
    paso_grueso: float,
    paso_fino: float,
    perdida_maxima_accuracy: float,
) -> None:
    """Ejecuta el ajuste y evaluación walk-forward dentro de 2025."""

    print("\nVALIDACIÓN WALK-FORWARD 2025")
    print("=" * 70)
    print(f"Símbolo: {simbolo}")
    print(
        f"Primer mes evaluado: "
        f"{nombre_mes(f'2025-{mes_inicial:02d}')}"
    )
    print(
        "La prueba de 2026 no será utilizada."
    )

    manifiesto = cargar_manifiesto(
        simbolo=simbolo
    )

    rutas_validacion = obtener_rutas_validacion(
        manifiesto=manifiesto
    )

    paquete = cargar_paquete_modelo(
        simbolo=simbolo
    )

    fechas, reales, probabilidades = cargar_validacion(
        rutas=rutas_validacion,
        paquete=paquete,
        tamano_lote=tamano_lote,
    )

    resultados_detallados: list[
        dict[str, Any]
    ] = []

    comparaciones: list[
        dict[str, Any]
    ] = []

    for mes in range(
        mes_inicial,
        13,
    ):
        periodo = (
            f"2025-{mes:02d}"
        )

        inicio_mes = pd.Timestamp(
            year=2025,
            month=mes,
            day=1,
            tz="UTC",
        )

        if mes == 12:
            fin_mes = pd.Timestamp(
                year=2026,
                month=1,
                day=1,
                tz="UTC",
            )
        else:
            fin_mes = pd.Timestamp(
                year=2025,
                month=mes + 1,
                day=1,
                tz="UTC",
            )

        limite_ajuste = (
            inicio_mes
            - pd.Timedelta(
                hours=4
            )
        )

        mascara_ajuste = (
            fechas
            < limite_ajuste
        )

        mascara_evaluacion = (
            (fechas >= inicio_mes)
            & (fechas < fin_mes)
        )

        if not mascara_ajuste.any():
            raise ValueError(
                f"No existen muestras anteriores para ajustar {periodo}."
            )

        if not mascara_evaluacion.any():
            raise ValueError(
                f"No existen muestras para evaluar {periodo}."
            )

        reales_ajuste = reales[
            mascara_ajuste
        ]

        probabilidades_ajuste = probabilidades[
            mascara_ajuste
        ]

        reales_mes = reales[
            mascara_evaluacion
        ]

        probabilidades_mes = probabilidades[
            mascara_evaluacion
        ]

        print(
            f"\nAJUSTANDO PARA "
            f"{nombre_mes(periodo).upper()}"
        )
        print("-" * 70)
        print(
            f"Muestras de ajuste: "
            f"{len(reales_ajuste):,}".replace(",", ".")
        )
        print(
            f"Muestras de evaluación: "
            f"{len(reales_mes):,}".replace(",", ".")
        )

        mejor = buscar_mejor_ajuste(
            reales=reales_ajuste,
            probabilidades=probabilidades_ajuste,
            multiplicador_minimo=multiplicador_minimo,
            multiplicador_maximo=multiplicador_maximo,
            paso_grueso=paso_grueso,
            paso_fino=paso_fino,
            perdida_maxima_accuracy=perdida_maxima_accuracy,
        )

        predicciones_originales = (
            aplicar_decision_original(
                probabilidades=probabilidades_mes
            )
        )

        predicciones_ajustadas = aplicar_multiplicadores(
            probabilidades=probabilidades_mes,
            multiplicador_baja=mejor[
                "multiplicador_baja"
            ],
            multiplicador_sube=mejor[
                "multiplicador_sube"
            ],
        )

        fila_original = calcular_metricas_periodo(
            mes=periodo,
            tipo_decision="original",
            reales=reales_mes,
            predicciones=predicciones_originales,
        )

        fila_ajustada = calcular_metricas_periodo(
            mes=periodo,
            tipo_decision="ajustada_walk_forward",
            reales=reales_mes,
            predicciones=predicciones_ajustadas,
        )

        desde_ajuste = fechas[
            mascara_ajuste
        ].min().isoformat()

        hasta_ajuste = fechas[
            mascara_ajuste
        ].max().isoformat()

        datos_ajuste = {
            "desde_ajuste": desde_ajuste,
            "hasta_ajuste": hasta_ajuste,
            "muestras_ajuste": len(
                reales_ajuste
            ),
            **mejor,
        }

        for fila in (
            fila_original,
            fila_ajustada,
        ):
            fila.update(
                {
                    "desde_ajuste": desde_ajuste,
                    "hasta_ajuste": hasta_ajuste,
                    "muestras_ajuste": len(
                        reales_ajuste
                    ),
                    "multiplicador_baja": mejor[
                        "multiplicador_baja"
                    ],
                    "multiplicador_neutral": 1.0,
                    "multiplicador_sube": mejor[
                        "multiplicador_sube"
                    ],
                }
            )

            resultados_detallados.append(
                fila
            )

        comparaciones.append(
            construir_comparacion(
                fila_original=fila_original,
                fila_ajustada=fila_ajustada,
                datos_ajuste=datos_ajuste,
            )
        )

        print(
            f"Multiplicadores elegidos: "
            f"BAJA {mejor['multiplicador_baja']:.2f} | "
            f"NEUTRAL 1.00 | "
            f"SUBE {mejor['multiplicador_sube']:.2f}"
        )
        print(
            f"F1-score macro original: "
            f"{fila_original['f1_macro']:.4f}"
        )
        print(
            f"F1-score macro ajustada: "
            f"{fila_ajustada['f1_macro']:.4f}"
        )
        print(
            f"Δ F1-score macro: "
            f"{fila_ajustada['f1_macro'] - fila_original['f1_macro']:+.4f}"
        )

    resultados = pd.DataFrame(
        resultados_detallados
    )

    comparacion = pd.DataFrame(
        comparaciones
    ).sort_values(
        "mes_evaluado"
    ).reset_index(
        drop=True
    )

    RUTA_MODELOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_resultados = (
        RUTA_MODELOS
        / f"validacion_walk_forward_{simbolo}_4h.csv"
    )

    ruta_comparacion = (
        RUTA_MODELOS
        / f"comparacion_walk_forward_{simbolo}_4h.csv"
    )

    ruta_resumen = (
        RUTA_MODELOS
        / f"resumen_walk_forward_{simbolo}_4h.md"
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
        comparacion=comparacion,
        mes_inicial=mes_inicial,
        perdida_maxima_accuracy=perdida_maxima_accuracy,
        ruta=ruta_resumen,
    )

    print("\nRESULTADO WALK-FORWARD")
    print("=" * 70)
    print(
        f"Meses con mejora de F1-score macro: "
        f"{int((comparacion['delta_f1_macro'] > 0).sum())} "
        f"de {len(comparacion)}"
    )
    print(
        f"Meses con mejora de Balanced Accuracy: "
        f"{int((comparacion['delta_balanced_accuracy'] > 0).sum())} "
        f"de {len(comparacion)}"
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
            "Ajusta multiplicadores con meses anteriores "
            "y evalúa cada mes futuro dentro de 2025."
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

    parser.add_argument(
        "--mes-inicial",
        type=int,
        default=4,
        help=(
            "Primer mes de 2025 que será evaluado. "
            "El valor predeterminado es abril."
        ),
    )

    parser.add_argument(
        "--multiplicador-minimo",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--multiplicador-maximo",
        type=float,
        default=3.00,
    )

    parser.add_argument(
        "--paso-grueso",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--paso-fino",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--max-perdida-accuracy",
        type=float,
        default=0.05,
    )

    argumentos = parser.parse_args()

    if argumentos.tamano_lote <= 0:
        parser.error(
            "--tamano-lote debe ser mayor que cero."
        )

    if not (
        2
        <= argumentos.mes_inicial
        <= 12
    ):
        parser.error(
            "--mes-inicial debe estar entre 2 y 12."
        )

    if argumentos.multiplicador_minimo <= 0:
        parser.error(
            "--multiplicador-minimo debe ser mayor que cero."
        )

    if (
        argumentos.multiplicador_maximo
        < argumentos.multiplicador_minimo
    ):
        parser.error(
            "--multiplicador-maximo debe ser mayor "
            "o igual que --multiplicador-minimo."
        )

    if argumentos.paso_grueso <= 0:
        parser.error(
            "--paso-grueso debe ser mayor que cero."
        )

    if argumentos.paso_fino <= 0:
        parser.error(
            "--paso-fino debe ser mayor que cero."
        )

    if not (
        0
        <= argumentos.max_perdida_accuracy
        < 1
    ):
        parser.error(
            "--max-perdida-accuracy debe estar entre cero y uno."
        )

    return argumentos


def main() -> None:
    """Punto de entrada."""

    argumentos = obtener_argumentos()

    try:
        validar_ajuste_walk_forward(
            simbolo=argumentos.simbolo,
            tamano_lote=argumentos.tamano_lote,
            mes_inicial=argumentos.mes_inicial,
            multiplicador_minimo=(
                argumentos.multiplicador_minimo
            ),
            multiplicador_maximo=(
                argumentos.multiplicador_maximo
            ),
            paso_grueso=argumentos.paso_grueso,
            paso_fino=argumentos.paso_fino,
            perdida_maxima_accuracy=(
                argumentos.max_perdida_accuracy
            ),
        )
    except (
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(
            "\nNo se pudo completar la validación walk-forward."
        )
        print(f"Detalle: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
