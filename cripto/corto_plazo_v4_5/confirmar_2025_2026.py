from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_1.utilidades import (
    calcular_salida_fija,
    construir_mascara_entrada,
    convertir_fechas_ns,
    evaluar_estrategia,
)
from cripto.corto_plazo_v4_5.configuracion_confirmacion import (
    ESTRATEGIA_CONGELADA,
    PERIODOS_CONFIRMACION,
    RUTA_ARTEFACTOS,
    RUTA_RESULTADOS_CONFIRMACION,
    VARIANTES_CONFIRMACION,
    VERSION_CONFIRMACION,
)
from cripto.corto_plazo_v4_5.utilidades_modelo import (
    ajustar_escalador,
    entrenar_modelo,
    predecir_validacion,
)


def valor_json(
    valor: Any,
) -> Any:
    if isinstance(
        valor,
        np.bool_,
    ):
        return bool(
            valor
        )

    if isinstance(
        valor,
        np.integer,
    ):
        return int(
            valor
        )

    if isinstance(
        valor,
        np.floating,
    ):
        if np.isnan(
            valor
        ):
            return None

        if np.isinf(
            valor
        ):
            return (
                "inf"
                if valor > 0
                else "-inf"
            )

        return float(
            valor
        )

    return valor


def evaluar_probabilidades(
    probabilidades: np.ndarray,
    objetivo: np.ndarray,
    datos: pd.DataFrame,
    umbral: float,
) -> tuple[
    dict[str, float | int],
    pd.DataFrame,
]:
    probabilidades_baja = np.zeros(
        len(
            probabilidades
        ),
        dtype="float64",
    )

    mascara = construir_mascara_entrada(
        probabilidades_sube=probabilidades,
        probabilidades_baja=probabilidades_baja,
        umbral_sube=umbral,
        umbral_veto_baja=None,
        modo=str(
            ESTRATEGIA_CONGELADA[
                "modo_entrada"
            ]
        ),
    )

    cierres = datos[
        "precio_cierre"
    ].to_numpy(
        dtype="float64"
    )

    retornos_brutos, duraciones = calcular_salida_fija(
        cierres=cierres,
        minutos=int(
            ESTRATEGIA_CONGELADA[
                "horizonte_salida_minutos"
            ]
        ),
    )

    metricas = evaluar_estrategia(
        fechas_ns=convertir_fechas_ns(
            datos[
                "fecha_apertura"
            ]
        ),
        mascara_entrada=mascara,
        retornos_brutos=retornos_brutos,
        minutos_salida=duraciones,
        objetivo_sube_4h=objetivo,
        coste=float(
            ESTRATEGIA_CONGELADA[
                "coste"
            ]
        ),
    )

    predicciones = pd.DataFrame(
        {
            "fecha_apertura": datos[
                "fecha_apertura"
            ],
            "fecha_objetivo": datos[
                "fecha_objetivo"
            ],
            "precio_cierre": datos[
                "precio_cierre"
            ].astype(
                "float64"
            ),
            "rendimiento_objetivo": datos[
                "rendimiento_objetivo"
            ].astype(
                "float64"
            ),
            "objetivo_sube_4h": objetivo.astype(
                "int8"
            ),
            "probabilidad_sube": probabilidades.astype(
                "float32"
            ),
            "cruce_entrada": mascara.astype(
                "int8"
            ),
        }
    )

    return (
        metricas,
        predicciones,
    )


def division_segura(
    numerador: float,
    denominador: float,
) -> float | None:
    if abs(
        denominador
    ) < 1e-15:
        return None

    return float(
        numerador
        / denominador
    )


def comparar_periodo(
    periodo: str,
    metricas_control: dict[str, float | int],
    metricas_candidata: dict[str, float | int],
    operaciones_minimas: int,
) -> dict[str, Any]:
    retorno_control = float(
        metricas_control[
            "retorno_neto_medio"
        ]
    )

    retorno_candidata = float(
        metricas_candidata[
            "retorno_neto_medio"
        ]
    )

    factor_control = float(
        metricas_control[
            "factor_beneficio"
        ]
    )

    factor_candidata = float(
        metricas_candidata[
            "factor_beneficio"
        ]
    )

    drawdown_control = float(
        metricas_control[
            "maximo_drawdown"
        ]
    )

    drawdown_candidata = float(
        metricas_candidata[
            "maximo_drawdown"
        ]
    )

    ratio_retorno = division_segura(
        retorno_candidata,
        retorno_control,
    )

    ratio_factor = division_segura(
        factor_candidata,
        factor_control,
    )

    candidata_sana = bool(
        int(
            metricas_candidata[
                "operaciones"
            ]
        )
        >= operaciones_minimas
        and retorno_candidata > 0
        and factor_candidata > 1
    )

    supera_estricto = bool(
        candidata_sana
        and retorno_candidata
        >= retorno_control
        and factor_candidata
        >= factor_control
        and (
            drawdown_candidata
            - drawdown_control
        )
        >= -0.01
    )

    no_degrada_grave = bool(
        candidata_sana
        and (
            ratio_retorno is None
            or ratio_retorno
            >= 0.90
        )
        and (
            ratio_factor is None
            or ratio_factor
            >= 0.95
        )
        and (
            drawdown_candidata
            - drawdown_control
        )
        >= -0.02
    )

    return {
        "periodo": periodo,
        "operaciones_minimas_exigidas": operaciones_minimas,
        "operaciones_control": int(
            metricas_control[
                "operaciones"
            ]
        ),
        "operaciones_v45": int(
            metricas_candidata[
                "operaciones"
            ]
        ),
        "precision_control": float(
            metricas_control[
                "precision_clasificacion"
            ]
        ),
        "precision_v45": float(
            metricas_candidata[
                "precision_clasificacion"
            ]
        ),
        "porcentaje_acierto_control": float(
            metricas_control[
                "porcentaje_acierto_clasificacion"
            ]
        ),
        "porcentaje_acierto_v45": float(
            metricas_candidata[
                "porcentaje_acierto_clasificacion"
            ]
        ),
        "positivas_control": float(
            metricas_control[
                "porcentaje_operaciones_positivas"
            ]
        ),
        "positivas_v45": float(
            metricas_candidata[
                "porcentaje_operaciones_positivas"
            ]
        ),
        "retorno_neto_medio_control": retorno_control,
        "retorno_neto_medio_v45": retorno_candidata,
        "mejora_relativa_retorno": (
            None
            if ratio_retorno is None
            else ratio_retorno
            - 1.0
        ),
        "factor_beneficio_control": factor_control,
        "factor_beneficio_v45": factor_candidata,
        "mejora_relativa_factor": (
            None
            if ratio_factor is None
            else ratio_factor
            - 1.0
        ),
        "drawdown_control": drawdown_control,
        "drawdown_v45": drawdown_candidata,
        "mejora_drawdown_absoluta": (
            drawdown_candidata
            - drawdown_control
        ),
        "candidata_sana": candidata_sana,
        "supera_control_estricto": supera_estricto,
        "no_degrada_grave": no_degrada_grave,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epocas",
        type=int,
        default=int(
            ESTRATEGIA_CONGELADA[
                "epocas"
            ]
        ),
    )

    parser.add_argument(
        "--tamano-lote",
        type=int,
        default=int(
            ESTRATEGIA_CONGELADA[
                "tamano_lote"
            ]
        ),
    )

    parser.add_argument(
        "--sobrescribir",
        action="store_true",
    )

    argumentos = parser.parse_args()

    if argumentos.epocas != int(
        ESTRATEGIA_CONGELADA[
            "epocas"
        ]
    ):
        raise ValueError(
            "La confirmación debe conservar exactamente 3 épocas."
        )

    if argumentos.tamano_lote != int(
        ESTRATEGIA_CONGELADA[
            "tamano_lote"
        ]
    ):
        raise ValueError(
            "La confirmación debe conservar el lote 100000."
        )

    RUTA_ARTEFACTOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUTA_RESULTADOS_CONFIRMACION.mkdir(
        parents=True,
        exist_ok=True,
    )

    resultados: dict[
        str,
        dict[
            str,
            dict[str, Any]
        ],
    ] = {}

    filas_metricas = []

    print(
        "\nCONFIRMACIÓN CONGELADA V4.5-SPOT"
    )
    print("=" * 72)
    print(
        "Control: V4.1, 63 variables y umbral 0,44."
    )
    print(
        "Candidata: V4.5-Spot, 90 variables y umbral 0,42."
    )
    print(
        "Entrada: cruce desde abajo."
    )
    print(
        "Salida: 480 minutos. Coste: 0,10 %."
    )
    print(
        "2025 y 2026 no se utilizaron para seleccionar V4.5."
    )

    for (
        nombre_periodo,
        configuracion_periodo,
    ) in PERIODOS_CONFIRMACION.items():
        resultados[
            nombre_periodo
        ] = {}

        print(
            "\n" + "=" * 72
        )
        print(
            nombre_periodo
        )
        print(
            "Entrenamiento: "
            f"{configuracion_periodo['entrenamiento_desde']} "
            f"a {configuracion_periodo['entrenamiento_hasta']}"
        )
        print(
            "Evaluación: "
            f"{configuracion_periodo['validacion_desde']} "
            f"a {configuracion_periodo['validacion_hasta']}"
        )
        print(
            "=" * 72
        )

        for (
            nombre_variante,
            configuracion_variante,
        ) in VARIANTES_CONFIRMACION.items():
            carpeta = (
                RUTA_ARTEFACTOS
                / nombre_periodo
                / nombre_variante
            )

            ruta_modelo = (
                carpeta
                / "modelo.joblib"
            )

            ruta_escalador = (
                carpeta
                / "escalador.joblib"
            )

            ruta_predicciones = (
                carpeta
                / "predicciones.parquet"
            )

            ruta_detalle = (
                carpeta
                / "detalle.json"
            )

            existentes = [
                ruta_modelo.exists(),
                ruta_escalador.exists(),
                ruta_predicciones.exists(),
                ruta_detalle.exists(),
            ]

            if (
                any(
                    existentes
                )
                and not argumentos.sobrescribir
            ):
                if not all(
                    existentes
                ):
                    raise RuntimeError(
                        "Existe una confirmación parcial. "
                        "Usa --sobrescribir para regenerar: "
                        f"{carpeta}"
                    )

                detalle_existente = json.loads(
                    ruta_detalle.read_text(
                        encoding="utf-8"
                    )
                )

                resultados[
                    nombre_periodo
                ][
                    nombre_variante
                ] = detalle_existente[
                    "metricas"
                ]

                filas_metricas.append(
                    {
                        "periodo": nombre_periodo,
                        "variante": nombre_variante,
                        "umbral": float(
                            configuracion_variante[
                                "umbral"
                            ]
                        ),
                        **detalle_existente[
                            "metricas"
                        ],
                    }
                )

                print(
                    f"\nConservado: "
                    f"{nombre_periodo} | {nombre_variante}"
                )

                continue

            carpeta.mkdir(
                parents=True,
                exist_ok=True,
            )

            columnas = tuple(
                configuracion_variante[
                    "columnas"
                ]
            )

            etapa = str(
                configuracion_variante[
                    "etapa_datos"
                ]
            )

            umbral = float(
                configuracion_variante[
                    "umbral"
                ]
            )

            print(
                f"\n{nombre_variante}: "
                f"{len(columnas)} variables | "
                f"umbral {umbral:.2f}"
            )
            print("-" * 72)

            escalador, conteos = ajustar_escalador(
                simbolo="BTCUSDT",
                configuracion_periodo=configuracion_periodo,
                columnas_modelo=columnas,
                etapa=etapa,
                tamano_lote=argumentos.tamano_lote,
            )

            modelo = entrenar_modelo(
                simbolo="BTCUSDT",
                configuracion_periodo=configuracion_periodo,
                columnas_modelo=columnas,
                etapa=etapa,
                escalador=escalador,
                epocas=argumentos.epocas,
                tamano_lote=argumentos.tamano_lote,
            )

            (
                probabilidades,
                objetivo,
                datos,
            ) = predecir_validacion(
                simbolo="BTCUSDT",
                configuracion_periodo=configuracion_periodo,
                columnas_modelo=columnas,
                etapa=etapa,
                modelo=modelo,
                escalador=escalador,
                tamano_lote=argumentos.tamano_lote,
            )

            metricas, predicciones = evaluar_probabilidades(
                probabilidades=probabilidades,
                objetivo=objetivo,
                datos=datos,
                umbral=umbral,
            )

            joblib.dump(
                modelo,
                ruta_modelo,
            )

            joblib.dump(
                escalador,
                ruta_escalador,
            )

            predicciones.to_parquet(
                ruta_predicciones,
                index=False,
            )

            detalle = {
                "fecha_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "version": VERSION_CONFIRMACION,
                "periodo": nombre_periodo,
                "periodo_entrenamiento": {
                    "desde": configuracion_periodo[
                        "entrenamiento_desde"
                    ],
                    "hasta": configuracion_periodo[
                        "entrenamiento_hasta"
                    ],
                },
                "periodo_evaluacion": {
                    "desde": configuracion_periodo[
                        "validacion_desde"
                    ],
                    "hasta": configuracion_periodo[
                        "validacion_hasta"
                    ],
                },
                "periodo_conocido_en_el_proyecto": bool(
                    configuracion_periodo[
                        "periodo_conocido_en_el_proyecto"
                    ]
                ),
                "usado_para_seleccionar_v4_5": False,
                "variante": nombre_variante,
                "variables": len(
                    columnas
                ),
                "umbral": umbral,
                "estrategia_congelada": ESTRATEGIA_CONGELADA,
                "conteos_entrenamiento": {
                    str(
                        clase
                    ): int(
                        cantidad
                    )
                    for clase, cantidad
                    in conteos.items()
                },
                "filas_evaluadas": len(
                    datos
                ),
                "metricas": metricas,
                "ruta_modelo": str(
                    ruta_modelo
                ),
                "ruta_escalador": str(
                    ruta_escalador
                ),
                "ruta_predicciones": str(
                    ruta_predicciones
                ),
            }

            ruta_detalle.write_text(
                json.dumps(
                    detalle,
                    ensure_ascii=False,
                    indent=4,
                    default=str,
                ),
                encoding="utf-8",
            )

            resultados[
                nombre_periodo
            ][
                nombre_variante
            ] = metricas

            filas_metricas.append(
                {
                    "periodo": nombre_periodo,
                    "variante": nombre_variante,
                    "umbral": umbral,
                    **metricas,
                }
            )

            print(
                f"Filas: {len(datos):,}".replace(
                    ",",
                    ".",
                )
            )
            print(
                "Precisión: "
                f"{metricas['precision_clasificacion']:.4f}"
            )
            print(
                "Porcentaje de acierto: "
                f"{metricas['porcentaje_acierto_clasificacion']:.2f} %"
            )
            print(
                "Operaciones positivas: "
                f"{metricas['porcentaje_operaciones_positivas']:.2f} %"
            )
            print(
                f"Operaciones: {metricas['operaciones']}"
            )
            print(
                "Retorno neto medio: "
                f"{metricas['retorno_neto_medio']:.4%}"
            )
            print(
                "Factor de beneficio: "
                f"{metricas['factor_beneficio']:.4f}"
            )
            print(
                "Drawdown: "
                f"{metricas['maximo_drawdown']:.2%}"
            )

            del modelo
            del escalador
            del probabilidades
            del objetivo
            del datos
            del predicciones
            gc.collect()

    comparaciones = []

    for (
        nombre_periodo,
        configuracion_periodo,
    ) in PERIODOS_CONFIRMACION.items():
        comparacion = comparar_periodo(
            periodo=nombre_periodo,
            metricas_control=resultados[
                nombre_periodo
            ][
                "v41_control_63"
            ],
            metricas_candidata=resultados[
                nombre_periodo
            ][
                "v45_spot_90"
            ],
            operaciones_minimas=int(
                configuracion_periodo[
                    "operaciones_minimas"
                ]
            ),
        )

        comparaciones.append(
            comparacion
        )

    supera_ambos = all(
        comparacion[
            "supera_control_estricto"
        ]
        for comparacion
        in comparaciones
    )

    sana_ambos = all(
        comparacion[
            "candidata_sana"
        ]
        for comparacion
        in comparaciones
    )

    no_degrada_ambos = all(
        comparacion[
            "no_degrada_grave"
        ]
        for comparacion
        in comparaciones
    )

    supera_al_menos_uno = any(
        comparacion[
            "supera_control_estricto"
        ]
        for comparacion
        in comparaciones
    )

    if supera_ambos:
        decision = (
            "v4_5_spot_supera_v4_1_en_2025_y_2026"
        )
    elif (
        sana_ambos
        and no_degrada_ambos
        and supera_al_menos_uno
    ):
        decision = (
            "v4_5_spot_confirmada_pero_no_supera_"
            "v4_1_de_forma_concluyente"
        )
    elif sana_ambos:
        decision = (
            "v4_5_spot_es_rentable_pero_no_"
            "justifica_sustituir_v4_1"
        )
    else:
        decision = (
            "v4_5_spot_no_confirma_robustez_"
            "fuera_de_desarrollo"
        )

    tabla_metricas = pd.DataFrame(
        filas_metricas
    )

    ruta_metricas = (
        RUTA_RESULTADOS_CONFIRMACION
        / "metricas_confirmacion.csv"
    )

    tabla_metricas.to_csv(
        ruta_metricas,
        index=False,
        encoding="utf-8-sig",
    )

    tabla_comparacion = pd.DataFrame(
        comparaciones
    )

    ruta_comparacion = (
        RUTA_RESULTADOS_CONFIRMACION
        / "comparacion_v45_vs_v41.csv"
    )

    tabla_comparacion.to_csv(
        ruta_comparacion,
        index=False,
        encoding="utf-8-sig",
    )

    detalle_final = {
        "fecha_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "version": VERSION_CONFIRMACION,
        "decision": decision,
        "campeona_antes_de_confirmacion": "V4.1",
        "candidata_evaluada": "V4.5-Spot",
        "parametros_congelados": {
            "v41_control": {
                "variables": 63,
                "umbral": 0.44,
            },
            "v45_spot": {
                "variables": 90,
                "umbral": 0.42,
            },
            **ESTRATEGIA_CONGELADA,
        },
        "comparacion_misma_muestra": True,
        "periodos_conocidos_en_el_proyecto": True,
        "periodos_usados_para_seleccionar_v4_5": False,
        "resultados": {
            periodo: {
                variante: {
                    clave: valor_json(
                        valor
                    )
                    for clave, valor
                    in metricas.items()
                }
                for variante, metricas
                in resultados_periodo.items()
            }
            for periodo, resultados_periodo
            in resultados.items()
        },
        "comparaciones": [
            {
                clave: valor_json(
                    valor
                )
                for clave, valor
                in comparacion.items()
            }
            for comparacion
            in comparaciones
        ],
        "promocion_automatica": False,
        "requiere_revision_humana": True,
    }

    ruta_decision = (
        RUTA_RESULTADOS_CONFIRMACION
        / "decision_confirmacion.json"
    )

    ruta_decision.write_text(
        json.dumps(
            detalle_final,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\nCOMPARACIÓN FINAL V4.5-SPOT VS V4.1"
    )
    print("=" * 72)

    for comparacion in comparaciones:
        print(
            f"\n{comparacion['periodo']}"
        )
        print("-" * 72)
        print(
            "Retorno V4.1: "
            f"{comparacion['retorno_neto_medio_control']:.4%}"
        )
        print(
            "Retorno V4.5: "
            f"{comparacion['retorno_neto_medio_v45']:.4%}"
        )
        print(
            "PF V4.1: "
            f"{comparacion['factor_beneficio_control']:.4f}"
        )
        print(
            "PF V4.5: "
            f"{comparacion['factor_beneficio_v45']:.4f}"
        )
        print(
            "DD V4.1: "
            f"{comparacion['drawdown_control']:.2%}"
        )
        print(
            "DD V4.5: "
            f"{comparacion['drawdown_v45']:.2%}"
        )
        print(
            "Supera control estricto: "
            f"{comparacion['supera_control_estricto']}"
        )

    print(
        f"\nDECISIÓN AUTOMÁTICA: {decision}"
    )
    print(
        "Ningún modelo fue promovido automáticamente."
    )
    print(
        f"- {ruta_metricas}"
    )
    print(
        f"- {ruta_comparacion}"
    )
    print(
        f"- {ruta_decision}"
    )


if __name__ == "__main__":
    main()
