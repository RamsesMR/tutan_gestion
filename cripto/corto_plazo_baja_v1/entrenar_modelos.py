from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from cripto.corto_plazo_baja_v1.configuracion import (
    COSTE_BASE,
    EPOCAS_SGD,
    OBJETIVOS,
    PLIEGUES_TEMPORALES,
    RUTA_HISTORIAL,
    RUTA_MODELOS_DESARROLLO,
    RUTA_RESULTADOS,
    TAMANO_LOTE,
    UMBRALES,
    VARIANTES,
    VERSION_MODELO,
)
from cripto.corto_plazo_baja_v1.utilidades_modelo import (
    cargar_periodo,
    entrenar_histgb,
    entrenar_sgd,
    metricas_clasificacion,
    metricas_discriminacion,
    predecir,
    simular_short,
)


def parsear_nombres(
    valor: str,
    disponibles: dict,
    etiqueta: str,
) -> list[str]:
    if valor.strip().lower() == "todos":
        return list(
            disponibles
        )

    nombres = [
        elemento.strip()
        for elemento in valor.split(",")
        if elemento.strip()
    ]

    desconocidos = set(
        nombres
    ).difference(
        disponibles
    )

    if desconocidos:
        raise ValueError(
            f"{etiqueta} desconocidos: "
            f"{sorted(desconocidos)}"
        )

    return nombres


def eliminar_experimento_previo(
    historial: pd.DataFrame,
    objetivo: str,
    variante: str,
    pliegue: str,
) -> pd.DataFrame:
    if historial.empty:
        return historial

    mascara = ~(
        (
            historial[
                "objetivo"
            ]
            == objetivo
        )
        & (
            historial[
                "variante"
            ]
            == variante
        )
        & (
            historial[
                "pliegue"
            ]
            == pliegue
        )
    )

    return historial.loc[
        mascara
    ].copy()


def experimento_completo(
    historial: pd.DataFrame,
    objetivo: str,
    variante: str,
    pliegue: str,
) -> bool:
    if historial.empty:
        return False

    filas = historial.loc[
        (
            historial[
                "objetivo"
            ]
            == objetivo
        )
        & (
            historial[
                "variante"
            ]
            == variante
        )
        & (
            historial[
                "pliegue"
            ]
            == pliegue
        )
    ]

    return (
        len(
            filas[
                "umbral"
            ].unique()
        )
        == len(
            UMBRALES
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--objetivos",
        default="todos",
    )

    parser.add_argument(
        "--variantes",
        default="todos",
    )

    parser.add_argument(
        "--epocas",
        type=int,
        default=EPOCAS_SGD,
    )

    parser.add_argument(
        "--tamano-lote",
        type=int,
        default=TAMANO_LOTE,
    )

    parser.add_argument(
        "--sobrescribir",
        action="store_true",
    )

    argumentos = parser.parse_args()

    objetivos = parsear_nombres(
        argumentos.objetivos,
        OBJETIVOS,
        "Objetivos",
    )

    variantes = parsear_nombres(
        argumentos.variantes,
        VARIANTES,
        "Variantes",
    )

    RUTA_MODELOS_DESARROLLO.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUTA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    if RUTA_HISTORIAL.exists():
        historial = pd.read_csv(
            RUTA_HISTORIAL
        )
    else:
        historial = pd.DataFrame()

    print(
        "\nENTRENAMIENTO BAJA V1"
    )
    print("=" * 72)
    print(
        f"Objetivos: {objetivos}"
    )
    print(
        f"Variantes: {variantes}"
    )
    print(
        f"Coste base: {COSTE_BASE:.4%}"
    )
    print(
        "2025 y 2026 no participan en selección."
    )

    for objetivo in objetivos:
        for nombre_variante in variantes:
            configuracion = VARIANTES[
                nombre_variante
            ]

            columnas = tuple(
                configuracion[
                    "columnas"
                ]
            )

            for nombre_pliegue, pliegue in (
                PLIEGUES_TEMPORALES.items()
            ):
                carpeta_modelo = (
                    RUTA_MODELOS_DESARROLLO
                    / objetivo
                    / nombre_variante
                    / nombre_pliegue
                )

                ruta_modelo = (
                    carpeta_modelo
                    / "modelo.joblib"
                )

                ruta_escalador = (
                    carpeta_modelo
                    / "escalador.joblib"
                )

                ruta_detalle = (
                    carpeta_modelo
                    / "detalle.json"
                )

                completo = experimento_completo(
                    historial=historial,
                    objetivo=objetivo,
                    variante=nombre_variante,
                    pliegue=nombre_pliegue,
                )

                algun_artefacto = any(
                    (
                        ruta_modelo.exists(),
                        ruta_escalador.exists(),
                        ruta_detalle.exists(),
                    )
                )

                artefactos_ok = (
                    ruta_modelo.exists()
                    and ruta_detalle.exists()
                    and (
                        str(
                            configuracion[
                                "tipo"
                            ]
                        )
                        != "sgd"
                        or ruta_escalador.exists()
                    )
                )

                if (
                    completo
                    and artefactos_ok
                    and not argumentos.sobrescribir
                ):
                    print(
                        f"Conservado: {objetivo} | "
                        f"{nombre_variante} | {nombre_pliegue}"
                    )

                    continue

                if argumentos.sobrescribir:
                    historial = (
                        eliminar_experimento_previo(
                            historial=historial,
                            objetivo=objetivo,
                            variante=nombre_variante,
                            pliegue=nombre_pliegue,
                        )
                    )
                elif (
                    completo
                    or algun_artefacto
                ):
                    raise RuntimeError(
                        "Existe una ejecución parcial o inconsistente. "
                        "Usa --sobrescribir para regenerar: "
                        f"{objetivo} | {nombre_variante} | "
                        f"{nombre_pliegue}"
                    )

                print(
                    "\n" + "#" * 72
                )
                print(
                    f"{objetivo} | {nombre_variante} | "
                    f"{nombre_pliegue}"
                )
                print(
                    f"Variables: {len(columnas)}"
                )
                print(
                    "#" * 72
                )

                tipo = str(
                    configuracion[
                        "tipo"
                    ]
                )

                if tipo == "sgd":
                    (
                        modelo,
                        escalador,
                        conteos,
                    ) = entrenar_sgd(
                        periodos=tuple(
                            pliegue[
                                "entrenamiento"
                            ]
                        ),
                        columnas_modelo=columnas,
                        objetivo=objetivo,
                        hasta_entrenamiento=str(
                            pliegue[
                                "hasta_entrenamiento"
                            ]
                        ),
                        configuracion=configuracion,
                        epocas=argumentos.epocas,
                        tamano_lote=(
                            argumentos.tamano_lote
                        ),
                    )
                elif tipo == "histgb":
                    (
                        modelo,
                        escalador,
                        conteos,
                    ) = entrenar_histgb(
                        periodos=tuple(
                            pliegue[
                                "entrenamiento"
                            ]
                        ),
                        columnas_modelo=columnas,
                        objetivo=objetivo,
                        hasta_entrenamiento=str(
                            pliegue[
                                "hasta_entrenamiento"
                            ]
                        ),
                        configuracion=configuracion,
                    )
                else:
                    raise ValueError(
                        f"Tipo de modelo desconocido: {tipo}"
                    )

                x_validacion, y_validacion, datos = (
                    cargar_periodo(
                        periodo=tuple(
                            pliegue[
                                "validacion"
                            ]
                        ),
                        columnas_modelo=columnas,
                        objetivo=objetivo,
                        hasta_exclusivo=str(
                            pliegue[
                                "hasta_validacion"
                            ]
                        ),
                    )
                )

                probabilidades = predecir(
                    modelo=modelo,
                    escalador=escalador,
                    x=x_validacion,
                    tamano_lote=(
                        argumentos.tamano_lote
                    ),
                )

                discriminacion = metricas_discriminacion(
                    y=y_validacion,
                    probabilidades=probabilidades,
                )

                filas_experimento = []

                for umbral in UMBRALES:
                    clasificacion = (
                        metricas_clasificacion(
                            y=y_validacion,
                            probabilidades=probabilidades,
                            umbral=umbral,
                        )
                    )

                    estrategia, _ = (
                        simular_short(
                            datos=datos,
                            probabilidades=probabilidades,
                            objetivo=objetivo,
                            umbral=umbral,
                            coste=COSTE_BASE,
                        )
                    )

                    filas_experimento.append(
                        {
                            "version": VERSION_MODELO,
                            "fecha_utc": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "objetivo": objetivo,
                            "grupo_objetivo": OBJETIVOS[
                                objetivo
                            ]["grupo"],
                            "variante": nombre_variante,
                            "grupo_variante": configuracion[
                                "grupo"
                            ],
                            "tipo_modelo": tipo,
                            "variables": len(
                                columnas
                            ),
                            "pliegue": nombre_pliegue,
                            "umbral": umbral,
                            "coste": COSTE_BASE,
                            "filas_validacion": len(
                                datos
                            ),
                            "negativos_entrenamiento": int(
                                conteos[0]
                            ),
                            "positivos_entrenamiento": int(
                                conteos[1]
                            ),
                            **discriminacion,
                            **clasificacion,
                            **estrategia,
                        }
                    )

                historial = pd.concat(
                    [
                        historial,
                        pd.DataFrame(
                            filas_experimento
                        ),
                    ],
                    ignore_index=True,
                )

                historial.to_csv(
                    RUTA_HISTORIAL,
                    index=False,
                    encoding="utf-8-sig",
                )

                carpeta_modelo.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                joblib.dump(
                    modelo,
                    ruta_modelo,
                )

                if escalador is not None:
                    joblib.dump(
                        escalador,
                        ruta_escalador,
                    )

                mejor_local = max(
                    filas_experimento,
                    key=lambda fila: (
                        float(
                            fila[
                                "retorno_neto_medio"
                            ]
                        ),
                        float(
                            fila[
                                "factor_beneficio"
                            ]
                        ),
                    ),
                )

                ruta_detalle.write_text(
                    json.dumps(
                        {
                            "fecha_utc": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "version": VERSION_MODELO,
                            "objetivo": objetivo,
                            "variante": nombre_variante,
                            "pliegue": nombre_pliegue,
                            "tipo_modelo": tipo,
                            "variables": list(
                                columnas
                            ),
                            "conteos_entrenamiento": {
                                "0": int(
                                    conteos[0]
                                ),
                                "1": int(
                                    conteos[1]
                                ),
                            },
                            "mejor_umbral_local_por_retorno": {
                                clave: valor
                                for clave, valor
                                in mejor_local.items()
                                if clave not in {
                                    "fecha_utc",
                                }
                            },
                            "uso_2025_para_seleccion": False,
                            "uso_2026_para_seleccion": False,
                        },
                        ensure_ascii=False,
                        indent=4,
                        default=str,
                    ),
                    encoding="utf-8",
                )

                fila_base = next(
                    fila
                    for fila in filas_experimento
                    if abs(
                        float(
                            fila["umbral"]
                        )
                        - 0.50
                    )
                    < 1e-12
                )

                print(
                    f"PR-AUC: {fila_base['pr_auc']:.4f}"
                )
                print(
                    f"ROC-AUC: {fila_base['roc_auc']:.4f}"
                )
                print(
                    "Precisión a 0,50: "
                    f"{fila_base['precision']:.4f}"
                )
                print(
                    "Porcentaje de acierto a 0,50: "
                    f"{fila_base['porcentaje_acierto']:.2f} %"
                )
                print(
                    "Mejor retorno local: "
                    f"{mejor_local['retorno_neto_medio']:.4%} "
                    f"| PF {mejor_local['factor_beneficio']:.4f} "
                    f"| umbral {mejor_local['umbral']:.2f}"
                )

                del modelo
                del escalador
                del x_validacion
                del y_validacion
                del probabilidades
                del datos
                gc.collect()

    print(
        "\nENTRENAMIENTO BAJA V1 COMPLETADO"
    )
    print(
        f"- {RUTA_HISTORIAL}"
    )


if __name__ == "__main__":
    main()
