from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_6_macro.configuracion import (
    BASES,
    RUTA_RESULTADOS,
    RUTA_SELECCION,
    SERIES_MACRO,
)


PLIEGUES = (
    "validacion_2023",
    "validacion_2024",
)


def fila_resultado(
    resultados: pd.DataFrame,
    variante: str,
    pliegue: str,
    umbral: float,
) -> pd.Series:
    filas = resultados.loc[
        (resultados["variante"] == variante)
        & (resultados["pliegue"] == pliegue)
        & np.isclose(
            resultados["umbral"].astype(float),
            umbral,
        )
    ]

    if len(filas) != 1:
        raise ValueError(
            f"No existe un resultado único para "
            f"{variante}, {pliegue}, {umbral}."
        )

    return filas.iloc[0]


def main() -> None:
    ruta_resultados = (
        RUTA_RESULTADOS
        / "resultados_desarrollo.csv"
    )

    ruta_estabilidad = (
        RUTA_RESULTADOS
        / "estabilidad_coeficientes_macro.csv"
    )

    if not ruta_resultados.exists():
        raise FileNotFoundError(
            f"No existe: {ruta_resultados}"
        )

    if not ruta_estabilidad.exists():
        raise FileNotFoundError(
            f"No existe: {ruta_estabilidad}"
        )

    resultados = pd.read_csv(
        ruta_resultados
    )

    estabilidad = pd.read_csv(
        ruta_estabilidad
    )

    registros: list[
        dict[str, Any]
    ] = []

    decisiones = {}

    for base, configuracion_base in BASES.items():
        umbral = float(
            configuracion_base[
                "umbral"
            ]
        )

        variante_control = (
            f"{base}_control_macro"
        )

        control_por_pliegue = {
            pliegue: fila_resultado(
                resultados,
                variante_control,
                pliegue,
                umbral,
            )
            for pliegue in PLIEGUES
        }

        control_retorno_minimo = min(
            float(
                fila[
                    "retorno_neto_medio"
                ]
            )
            for fila in control_por_pliegue.values()
        )

        control_factor_minimo = min(
            float(
                fila[
                    "factor_beneficio"
                ]
            )
            for fila in control_por_pliegue.values()
        )

        control_drawdown_peor = min(
            float(
                fila[
                    "maximo_drawdown"
                ]
            )
            for fila in control_por_pliegue.values()
        )

        aprobadas = []

        for clave_macro, detalle_macro in SERIES_MACRO.items():
            variante = (
                f"{base}_{clave_macro}"
            )

            filas_candidata = {
                pliegue: fila_resultado(
                    resultados,
                    variante,
                    pliegue,
                    umbral,
                )
                for pliegue in PLIEGUES
            }

            retorno_minimo = min(
                float(
                    fila[
                        "retorno_neto_medio"
                    ]
                )
                for fila in filas_candidata.values()
            )

            factor_minimo = min(
                float(
                    fila[
                        "factor_beneficio"
                    ]
                )
                for fila in filas_candidata.values()
            )

            drawdown_peor = min(
                float(
                    fila[
                        "maximo_drawdown"
                    ]
                )
                for fila in filas_candidata.values()
            )

            operaciones_minimas = min(
                int(
                    fila["operaciones"]
                )
                for fila in filas_candidata.values()
            )

            mediana_minima = min(
                float(
                    fila[
                        "retorno_neto_mediano"
                    ]
                )
                for fila in filas_candidata.values()
            )

            retencion_por_ano = min(
                (
                    float(
                        filas_candidata[
                            pliegue
                        ][
                            "retorno_neto_medio"
                        ]
                    )
                    / float(
                        control_por_pliegue[
                            pliegue
                        ][
                            "retorno_neto_medio"
                        ]
                    )
                )
                for pliegue in PLIEGUES
            )

            fila_estabilidad = estabilidad.loc[
                estabilidad[
                    "variante"
                ]
                == variante
            ]

            if len(
                fila_estabilidad
            ) != 1:
                raise ValueError(
                    f"No existe estabilidad única para {variante}."
                )

            signo_estable = bool(
                fila_estabilidad.iloc[
                    0
                ][
                    "signo_estable"
                ]
            )

            mejora_retorno = (
                retorno_minimo
                / control_retorno_minimo
                - 1.0
            )

            mejora_factor = (
                factor_minimo
                / control_factor_minimo
                - 1.0
            )

            mejora_drawdown = (
                drawdown_peor
                - control_drawdown_peor
            )

            mejora_material = bool(
                mejora_retorno >= 0.03
                or mejora_factor >= 0.03
                or mejora_drawdown >= 0.005
            )

            no_degrada = bool(
                retencion_por_ano >= 0.90
                and factor_minimo
                >= control_factor_minimo * 0.98
                and mejora_drawdown >= -0.01
            )

            aprobada = bool(
                operaciones_minimas >= 50
                and retorno_minimo > 0
                and mediana_minima >= 0
                and signo_estable
                and mejora_material
                and no_degrada
            )

            registro = {
                "base": base,
                "variante": variante,
                "clave_macro": clave_macro,
                "variable_macro": detalle_macro[
                    "columna"
                ],
                "umbral_congelado": umbral,
                "operaciones_minimas": operaciones_minimas,
                "retorno_minimo": retorno_minimo,
                "factor_minimo": factor_minimo,
                "drawdown_peor": drawdown_peor,
                "mediana_minima": mediana_minima,
                "retencion_retorno_peor_ano": retencion_por_ano,
                "mejora_relativa_retorno_minimo": mejora_retorno,
                "mejora_relativa_factor_minimo": mejora_factor,
                "mejora_drawdown_absoluta": mejora_drawdown,
                "signo_coeficiente_estable": signo_estable,
                "mejora_material": mejora_material,
                "no_degrada": no_degrada,
                "aprobada_preliminar": aprobada,
            }

            registros.append(registro)

            if aprobada:
                aprobadas.append(
                    {
                        "clave_macro": clave_macro,
                        "variable_macro": detalle_macro[
                            "columna"
                        ],
                        "variante": variante,
                    }
                )

        decisiones[base] = {
            "base": base,
            "variante_control": variante_control,
            "umbral_congelado": umbral,
            "variables_aprobadas_preliminares": aprobadas,
            "requiere_revision_manual_antes_de_combinar": True,
        }

    tabla = pd.DataFrame(
        registros
    )

    RUTA_SELECCION.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_csv = (
        RUTA_SELECCION
        / "seleccion_variables_individuales.csv"
    )

    tabla.to_csv(
        ruta_csv,
        index=False,
        encoding="utf-8-sig",
    )

    ruta_json = (
        RUTA_SELECCION
        / "decision_variables_individuales.json"
    )

    ruta_json.write_text(
        json.dumps(
            {
                "decision": (
                    "fase_individual_completada"
                ),
                "uso_2025_para_seleccion": False,
                "uso_2026_para_seleccion": False,
                "bases": decisiones,
                "nota": (
                    "No se construye una combinación automáticamente. "
                    "Las candidatas deben revisarse antes de la fase 2 "
                    "y confirmarse con datos vintage de ALFRED antes "
                    "de cualquier promoción."
                ),
            },
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\nSELECCIÓN PRELIMINAR DE VARIABLES MACRO"
    )
    print("=" * 72)

    for base, decision in decisiones.items():
        aprobadas = decision[
            "variables_aprobadas_preliminares"
        ]

        print(
            f"\n{base}: {len(aprobadas)} aprobadas preliminarmente"
        )

        for aprobada in aprobadas:
            print(
                f"  - {aprobada['variable_macro']}"
            )

    print(
        "\nNo se construyó ninguna combinación automáticamente."
    )
    print(
        "2025 y 2026 no fueron utilizados."
    )
    print(f"- {ruta_csv}")
    print(f"- {ruta_json}")


if __name__ == "__main__":
    main()
