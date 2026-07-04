from __future__ import annotations

import argparse
import gc
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd

from cripto.corto_plazo_v4.configuracion import (
    DETECTORES,
    PLIEGUES_TEMPORALES,
    RUTA_HISTORIAL_DESARROLLO,
    RUTA_RESULTADOS_DESARROLLO,
    SIMBOLOS,
    UMBRALES_PROBABILIDAD,
    VARIANTES,
    VERSION_MODELO,
)
from cripto.corto_plazo_v4.utilidades import (
    ajustar_escalador_y_contar,
    calcular_metricas_generales,
    calcular_pesos_binarios,
    construir_tabla_coeficientes,
    construir_tabla_umbrales,
    entrenar_modelo_binario,
    predecir_periodo,
)


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
                    sorted(
                        inexistentes
                    )
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


def obtener_detectores(
    detector: str,
) -> list[str]:
    """Resuelve BAJA, SUBE o AMBOS."""

    if detector == "AMBOS":
        return list(
            DETECTORES
        )

    return [
        detector
    ]


def guardar_experimento(
    identificador_ejecucion: str,
    simbolo: str,
    detector: str,
    nombre_variante: str,
    configuracion_variante: dict[str, Any],
    nombre_pliegue: str,
    epocas: int,
    tamano_lote: int,
    coste_operacion: float,
    conteos,
    pesos: dict[int, float],
    modelo,
    escalador,
    objetivo,
    probabilidades,
    fechas_ns,
    rendimientos,
    guardar_modelo: bool,
) -> dict[str, Any]:
    """Guarda métricas, umbrales y coeficientes de un experimento."""

    fecha_utc = datetime.now(
        timezone.utc
    )

    ruta = (
        RUTA_RESULTADOS_DESARROLLO
        / simbolo
        / detector
        / nombre_variante
        / nombre_pliegue
        / identificador_ejecucion
    )

    ruta.mkdir(
        parents=True,
        exist_ok=False,
    )

    metricas_generales = calcular_metricas_generales(
        objetivo=objetivo,
        probabilidades=probabilidades,
    )

    tabla_umbrales = construir_tabla_umbrales(
        objetivo=objetivo,
        probabilidades=probabilidades,
        fechas_ns=fechas_ns,
        rendimientos=rendimientos,
        detector=detector,
        umbrales=UMBRALES_PROBABILIDAD,
        coste_operacion=coste_operacion,
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

    tabla_coeficientes = construir_tabla_coeficientes(
        modelo=modelo,
        columnas_modelo=tuple(
            configuracion_variante["columnas"]
        ),
    )

    ruta_coeficientes = (
        ruta
        / "coeficientes.csv"
    )

    tabla_coeficientes.to_csv(
        ruta_coeficientes,
        index=False,
        encoding="utf-8-sig",
    )

    mejor_f1 = tabla_umbrales.sort_values(
        [
            "f1",
            "precision",
        ],
        ascending=[
            False,
            False,
        ],
    ).iloc[
        0
    ]

    detalle = {
        "fecha_utc": fecha_utc.isoformat(),
        "identificador_ejecucion": identificador_ejecucion,
        "version_modelo": VERSION_MODELO,
        "simbolo": simbolo,
        "detector": detector,
        "variante": nombre_variante,
        "grupo": configuracion_variante["grupo"],
        "pliegue": nombre_pliegue,
        "variables": len(
            configuracion_variante["columnas"]
        ),
        "columnas_modelo": list(
            configuracion_variante["columnas"]
        ),
        "penalty": configuracion_variante["penalty"],
        "alpha": configuracion_variante["alpha"],
        "potencia_peso_positivo": configuracion_variante[
            "potencia_peso_positivo"
        ],
        "pesos_clases": {
            str(
                clase
            ): float(
                peso
            )
            for clase, peso in pesos.items()
        },
        "conteos_entrenamiento": {
            str(
                clase
            ): int(
                cantidad
            )
            for clase, cantidad in conteos.items()
        },
        "epocas": epocas,
        "tamano_lote": tamano_lote,
        "coste_operacion": coste_operacion,
        "metricas_generales": metricas_generales,
        "mejor_umbral_f1": float(
            mejor_f1["umbral"]
        ),
        "mejor_f1": float(
            mejor_f1["f1"]
        ),
        "precision_mejor_f1": float(
            mejor_f1["precision"]
        ),
        "recall_mejor_f1": float(
            mejor_f1["recall"]
        ),
        "ruta_metricas_umbrales": str(
            ruta_umbrales
        ),
        "ruta_coeficientes": str(
            ruta_coeficientes
        ),
        "division_2025_utilizada": False,
        "division_2026_utilizada": False,
    }

    ruta_detalle = (
        ruta
        / "detalle.json"
    )

    ruta_detalle.write_text(
        json.dumps(
            detalle,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
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
        "detector": detector,
        "variante": nombre_variante,
        "grupo": configuracion_variante["grupo"],
        "pliegue": nombre_pliegue,
        "variables": len(
            configuracion_variante["columnas"]
        ),
        "penalty": configuracion_variante["penalty"],
        "alpha": configuracion_variante["alpha"],
        "potencia_peso_positivo": configuracion_variante[
            "potencia_peso_positivo"
        ],
        "peso_clase_negativa": pesos[0],
        "peso_clase_positiva": pesos[1],
        "epocas": epocas,
        "tamano_lote": tamano_lote,
        "coste_operacion": coste_operacion,
        **metricas_generales,
        "mejor_umbral_f1": detalle[
            "mejor_umbral_f1"
        ],
        "mejor_f1": detalle[
            "mejor_f1"
        ],
        "precision_mejor_f1": detalle[
            "precision_mejor_f1"
        ],
        "recall_mejor_f1": detalle[
            "recall_mejor_f1"
        ],
        "ruta": str(
            ruta
        ),
        "ruta_metricas_umbrales": str(
            ruta_umbrales
        ),
        "ruta_coeficientes": str(
            ruta_coeficientes
        ),
        "division_2025_utilizada": False,
        "division_2026_utilizada": False,
    }

    RUTA_RESULTADOS_DESARROLLO.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [
            fila_historial
        ]
    ).to_csv(
        RUTA_HISTORIAL_DESARROLLO,
        mode="a",
        header=not RUTA_HISTORIAL_DESARROLLO.exists(),
        index=False,
        encoding="utf-8-sig",
    )

    return fila_historial


def ejecutar(
    simbolo: str,
    detector_solicitado: str,
    grupo: str,
    nombres_variantes: list[str] | None,
    epocas: int,
    tamano_lote: int,
    coste_operacion: float,
    guardar_modelos: bool,
) -> None:
    """Ejecuta detectores, variantes y pliegues."""

    variantes = obtener_variantes(
        grupo=grupo,
        nombres=nombres_variantes,
    )

    detectores = obtener_detectores(
        detector=detector_solicitado,
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
        "\nDETECTORES BINARIOS V4"
    )
    print("=" * 72)
    print(
        f"Símbolo: {simbolo}"
    )
    print(
        f"Detectores: {', '.join(detectores)}"
    )
    print(
        f"Variantes: {', '.join(variantes)}"
    )
    print(
        f"Épocas: {epocas}"
    )
    print(
        f"Coste por operación: {coste_operacion:.4%}"
    )
    print(
        "Validaciones internas: 2023 y 2024"
    )
    print(
        "2025 y 2026 no se utilizarán."
    )

    resultados: list[
        dict[str, Any]
    ] = []

    for detector in detectores:
        for nombre_variante in variantes:
            configuracion_variante = VARIANTES[
                nombre_variante
            ]

            columnas_modelo = tuple(
                configuracion_variante["columnas"]
            )

            for nombre_pliegue, configuracion_pliegue in (
                PLIEGUES_TEMPORALES.items()
            ):
                print(
                    "\n"
                    + "=" * 72
                )
                print(
                    f"{simbolo} | {detector} | "
                    f"{nombre_variante} | {nombre_pliegue}"
                )
                print(
                    f"Variables: {len(columnas_modelo)}"
                )
                print(
                    "Potencia del peso positivo: "
                    f"{configuracion_variante['potencia_peso_positivo']}"
                )
                print(
                    "=" * 72
                )

                escalador, conteos = ajustar_escalador_y_contar(
                    simbolo=simbolo,
                    detector=detector,
                    configuracion_periodo=configuracion_pliegue,
                    columnas_modelo=columnas_modelo,
                    tamano_lote=tamano_lote,
                )

                pesos = calcular_pesos_binarios(
                    conteos=conteos,
                    potencia_peso_positivo=float(
                        configuracion_variante[
                            "potencia_peso_positivo"
                        ]
                    ),
                )

                print(
                    "  Distribución entrenamiento:"
                )
                print(
                    f"    Negativos: {conteos[0]:,}".replace(
                        ",",
                        ".",
                    )
                )
                print(
                    f"    Positivos: {conteos[1]:,}".replace(
                        ",",
                        ".",
                    )
                )
                print(
                    "  Pesos:"
                )
                print(
                    f"    Negativa: {pesos[0]:.6f}"
                )
                print(
                    f"    Positiva: {pesos[1]:.6f}"
                )

                modelo = entrenar_modelo_binario(
                    simbolo=simbolo,
                    detector=detector,
                    configuracion_periodo=configuracion_pliegue,
                    configuracion_variante=configuracion_variante,
                    escalador=escalador,
                    pesos=pesos,
                    epocas=epocas,
                    tamano_lote=tamano_lote,
                )

                (
                    objetivo,
                    probabilidades,
                    fechas_ns,
                    rendimientos,
                ) = predecir_periodo(
                    simbolo=simbolo,
                    detector=detector,
                    configuracion_periodo=configuracion_pliegue,
                    columnas_modelo=columnas_modelo,
                    modelo=modelo,
                    escalador=escalador,
                    tamano_lote=tamano_lote,
                )

                fila = guardar_experimento(
                    identificador_ejecucion=identificador_ejecucion,
                    simbolo=simbolo,
                    detector=detector,
                    nombre_variante=nombre_variante,
                    configuracion_variante=configuracion_variante,
                    nombre_pliegue=nombre_pliegue,
                    epocas=epocas,
                    tamano_lote=tamano_lote,
                    coste_operacion=coste_operacion,
                    conteos=conteos,
                    pesos=pesos,
                    modelo=modelo,
                    escalador=escalador,
                    objetivo=objetivo,
                    probabilidades=probabilidades,
                    fechas_ns=fechas_ns,
                    rendimientos=rendimientos,
                    guardar_modelo=guardar_modelos,
                )

                resultados.append(
                    fila
                )

                print(
                    "\n  Resultado:"
                )
                print(
                    f"    Prevalencia: {fila['prevalencia']:.4f}"
                )
                print(
                    f"    PR-AUC: {fila['pr_auc']:.4f}"
                )
                print(
                    f"    PR-AUC lift: {fila['pr_auc_lift']:.4f}"
                )
                print(
                    f"    ROC-AUC: {fila['roc_auc']:.4f}"
                )
                print(
                    f"    F1 a 0.50: {fila['f1_050']:.4f}"
                )
                print(
                    "    Mejor umbral por F1: "
                    f"{fila['mejor_umbral_f1']:.2f}"
                )
                print(
                    f"    Mejor F1: {fila['mejor_f1']:.4f}"
                )
                print(
                    "    Precision en mejor F1: "
                    f"{fila['precision_mejor_f1']:.4f}"
                )
                print(
                    "    Recall en mejor F1: "
                    f"{fila['recall_mejor_f1']:.4f}"
                )

                del modelo
                del escalador
                del objetivo
                del probabilidades
                del fechas_ns
                del rendimientos
                gc.collect()

    ruta_resumen = (
        RUTA_RESULTADOS_DESARROLLO
        / f"resumen_ejecucion_{identificador_ejecucion}.csv"
    )

    pd.DataFrame(
        resultados
    ).to_csv(
        ruta_resumen,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nEJECUCIÓN V4 COMPLETADA"
    )
    print("=" * 72)
    print(
        f"Resumen: {ruta_resumen}"
    )
    print(
        f"Historial: {RUTA_HISTORIAL_DESARROLLO}"
    )


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene argumentos de consola."""

    parser = argparse.ArgumentParser(
        description=(
            "Entrena detectores binarios especializados de BAJA y SUBE."
        )
    )

    parser.add_argument(
        "--simbolo",
        required=True,
        choices=SIMBOLOS,
    )

    parser.add_argument(
        "--detector",
        choices=(
            "BAJA",
            "SUBE",
            "AMBOS",
        ),
        default="AMBOS",
    )

    parser.add_argument(
        "--grupo",
        choices=(
            "base",
            "pesos",
            "todo",
        ),
        default="base",
    )

    parser.add_argument(
        "--variantes",
        nargs="+",
        default=None,
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
        "--coste-operacion",
        type=float,
        default=0.0,
        help=(
            "Coste total de entrada y salida como fracción. "
            "Ejemplo: 0.001 equivale a 0,10%%."
        ),
    )

    parser.add_argument(
        "--guardar-modelos",
        action="store_true",
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

    if argumentos.coste_operacion < 0:
        parser.error(
            "--coste-operacion no puede ser negativo."
        )

    return argumentos


def main() -> None:
    """Punto de entrada."""

    argumentos = obtener_argumentos()

    try:
        ejecutar(
            simbolo=argumentos.simbolo,
            detector_solicitado=argumentos.detector,
            grupo=argumentos.grupo,
            nombres_variantes=argumentos.variantes,
            epocas=argumentos.epocas,
            tamano_lote=argumentos.tamano_lote,
            coste_operacion=argumentos.coste_operacion,
            guardar_modelos=argumentos.guardar_modelos,
        )

    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(
            "\nNo se pudieron entrenar los detectores V4."
        )
        print(
            f"Detalle: {error}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
