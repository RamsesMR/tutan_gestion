from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_6_macro.configuracion import (
    PLIEGUES_TEMPORALES,
    RUTA_MODELOS_DESARROLLO,
    RUTA_RESULTADOS,
    VARIANTES,
)


def signo(
    valor: float,
) -> int:
    if valor > 0:
        return 1

    if valor < 0:
        return -1

    return 0


def main() -> None:
    carpeta_operaciones = (
        RUTA_RESULTADOS
        / "operaciones_umbral_base"
    )

    registros_coeficientes = []
    registros_concentracion = []

    for nombre_variante, configuracion in VARIANTES.items():
        variable_macro = configuracion[
            "variable_macro"
        ]

        if variable_macro is None:
            continue

        columnas = list(
            configuracion[
                "columnas"
            ]
        )

        indice_macro = columnas.index(
            variable_macro
        )

        coeficientes = {}

        for nombre_pliegue in PLIEGUES_TEMPORALES:
            ruta_modelo = (
                RUTA_MODELOS_DESARROLLO
                / nombre_variante
                / nombre_pliegue
                / "modelo.joblib"
            )

            if not ruta_modelo.exists():
                raise FileNotFoundError(
                    f"No existe: {ruta_modelo}"
                )

            modelo = joblib.load(
                ruta_modelo
            )

            coeficiente = float(
                modelo.coef_[
                    0,
                    indice_macro,
                ]
            )

            coeficientes[
                nombre_pliegue
            ] = coeficiente

            ruta_operaciones = (
                carpeta_operaciones
                / f"{nombre_variante}_{nombre_pliegue}.parquet"
            )

            operaciones = pd.read_parquet(
                ruta_operaciones
            )

            if operaciones.empty:
                meses = 0
                meses_positivos = 0
                maximo_peso_mes = 0.0
            else:
                operaciones[
                    "fecha_entrada"
                ] = pd.to_datetime(
                    operaciones[
                        "fecha_entrada"
                    ],
                    utc=True,
                    errors="raise",
                )

                operaciones[
                    "mes"
                ] = operaciones[
                    "fecha_entrada"
                ].dt.to_period("M").astype(str)

                mensual = (
                    operaciones.groupby(
                        "mes",
                        as_index=False,
                    )[
                        "retorno_neto"
                    ]
                    .sum()
                )

                meses = len(mensual)
                meses_positivos = int(
                    (
                        mensual[
                            "retorno_neto"
                        ]
                        > 0
                    ).sum()
                )

                suma_absoluta = float(
                    mensual[
                        "retorno_neto"
                    ].abs().sum()
                )

                maximo_peso_mes = (
                    float(
                        mensual[
                            "retorno_neto"
                        ].abs().max()
                        / suma_absoluta
                    )
                    if suma_absoluta > 0
                    else 0.0
                )

            registros_concentracion.append(
                {
                    "variante": nombre_variante,
                    "variable_macro": variable_macro,
                    "pliegue": nombre_pliegue,
                    "meses_con_operaciones": meses,
                    "meses_positivos": meses_positivos,
                    "porcentaje_meses_positivos": (
                        meses_positivos
                        / meses
                        * 100.0
                        if meses > 0
                        else 0.0
                    ),
                    "peso_maximo_mes_retorno_absoluto": maximo_peso_mes,
                }
            )

        estable = bool(
            signo(
                coeficientes[
                    "validacion_2023"
                ]
            )
            != 0
            and signo(
                coeficientes[
                    "validacion_2023"
                ]
            )
            == signo(
                coeficientes[
                    "validacion_2024"
                ]
            )
        )

        registros_coeficientes.append(
            {
                "variante": nombre_variante,
                "base": configuracion[
                    "base"
                ],
                "variable_macro": variable_macro,
                "coeficiente_validacion_2023": coeficientes[
                    "validacion_2023"
                ],
                "coeficiente_validacion_2024": coeficientes[
                    "validacion_2024"
                ],
                "signo_estable": estable,
            }
        )

    coeficientes = pd.DataFrame(
        registros_coeficientes
    )

    concentracion = pd.DataFrame(
        registros_concentracion
    )

    ruta_coeficientes = (
        RUTA_RESULTADOS
        / "estabilidad_coeficientes_macro.csv"
    )

    ruta_concentracion = (
        RUTA_RESULTADOS
        / "concentracion_temporal_operaciones.csv"
    )

    coeficientes.to_csv(
        ruta_coeficientes,
        index=False,
        encoding="utf-8-sig",
    )

    concentracion.to_csv(
        ruta_concentracion,
        index=False,
        encoding="utf-8-sig",
    )

    ruta_json = (
        RUTA_RESULTADOS
        / "detalle_estabilidad_macro.json"
    )

    ruta_json.write_text(
        json.dumps(
            {
                "coeficientes": coeficientes.to_dict(
                    orient="records"
                ),
                "concentracion": concentracion.to_dict(
                    orient="records"
                ),
            },
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\nANÁLISIS DE ESTABILIDAD MACRO COMPLETADO"
    )
    print("=" * 72)
    print(
        coeficientes.to_string(
            index=False
        )
    )
    print(f"\n- {ruta_coeficientes}")
    print(f"- {ruta_concentracion}")
    print(f"- {ruta_json}")


if __name__ == "__main__":
    main()
