from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_6_macro.configuracion import (
    PLIEGUES_TEMPORALES,
    SIMBOLO_OPERATIVO,
    VARIANTES,
)
from cripto.corto_plazo_v4_6_macro.configuracion_robustez import (
    COMPARACIONES_ROBUSTEZ,
    CONCENTRACION_MENSUAL_MAXIMA,
    EMPEORAMIENTO_MAXIMO_DRAWDOWN,
    OPERACIONES_MINIMAS,
    RETENCION_MINIMA_FACTOR_ANUAL,
    RETENCION_MINIMA_RETORNO_ANUAL,
    RUTA_MODELOS_ROBUSTEZ,
    RUTA_PREDICCIONES_ROBUSTEZ,
    RUTA_RESULTADOS_ROBUSTEZ,
    SEMILLAS_ROBUSTEZ,
)
from cripto.corto_plazo_v4_6_macro.evaluar_desarrollo import (
    simular,
)


def division_segura(
    numerador: float,
    denominador: float,
) -> float:
    if denominador == 0:
        return float("nan")

    return float(
        numerador / denominador
    )


def concentracion_mensual(
    operaciones: pd.DataFrame,
) -> float:
    if operaciones.empty:
        return 0.0

    tabla = operaciones.copy()

    tabla[
        "fecha_entrada"
    ] = pd.to_datetime(
        tabla["fecha_entrada"],
        utc=True,
        errors="raise",
    )

    tabla[
        "mes"
    ] = tabla[
        "fecha_entrada"
    ].dt.to_period(
        "M"
    ).astype(str)

    mensual = (
        tabla.groupby(
            "mes",
            as_index=False,
        )[
            "retorno_neto"
        ]
        .sum()
    )

    total_absoluto = float(
        mensual[
            "retorno_neto"
        ].abs().sum()
    )

    if total_absoluto == 0:
        return 0.0

    return float(
        mensual[
            "retorno_neto"
        ].abs().max()
        / total_absoluto
    )


def signo(
    valor: float,
) -> int:
    if valor > 0:
        return 1

    if valor < 0:
        return -1

    return 0


def ruta_predicciones(
    semilla: int,
    variante: str,
    pliegue: str,
):
    return (
        RUTA_PREDICCIONES_ROBUSTEZ
        / f"semilla_{semilla}"
        / variante
        / (
            f"{SIMBOLO_OPERATIVO}_"
            f"{pliegue}.parquet"
        )
    )


def ruta_modelo(
    semilla: int,
    variante: str,
    pliegue: str,
):
    return (
        RUTA_MODELOS_ROBUSTEZ
        / f"semilla_{semilla}"
        / variante
        / pliegue
        / "modelo.joblib"
    )


def coeficiente_macro(
    semilla: int,
    variante: str,
    pliegue: str,
    variable_macro: str,
) -> float:
    columnas = list(
        VARIANTES[
            variante
        ]["columnas"]
    )

    indice = columnas.index(
        variable_macro
    )

    modelo = joblib.load(
        ruta_modelo(
            semilla,
            variante,
            pliegue,
        )
    )

    return float(
        modelo.coef_[
            0,
            indice,
        ]
    )


def valor_json(
    valor: Any,
) -> Any:
    if isinstance(
        valor,
        np.bool_,
    ):
        return bool(valor)

    if isinstance(
        valor,
        np.integer,
    ):
        return int(valor)

    if isinstance(
        valor,
        np.floating,
    ):
        if np.isnan(valor):
            return None

        if np.isinf(valor):
            return (
                "inf"
                if valor > 0
                else "-inf"
            )

        return float(valor)

    return valor


def main() -> None:
    RUTA_RESULTADOS_ROBUSTEZ.mkdir(
        parents=True,
        exist_ok=True,
    )

    registros_detalle = []
    registros_semilla = []

    print(
        "\nEVALUACIÓN PAREADA DE ROBUSTEZ V4.6"
    )
    print("=" * 72)

    for (
        nombre_comparacion,
        configuracion,
    ) in COMPARACIONES_ROBUSTEZ.items():
        control = str(
            configuracion["control"]
        )

        candidata = str(
            configuracion["candidata"]
        )

        umbral = float(
            configuracion["umbral"]
        )

        variable_macro = str(
            configuracion["variable_macro"]
        )

        print(
            "\n" + "#" * 72
        )
        print(
            f"{nombre_comparacion} | "
            f"umbral {umbral:.2f}"
        )
        print(
            "#" * 72
        )

        for semilla in SEMILLAS_ROBUSTEZ:
            filas_semilla = []

            for pliegue in PLIEGUES_TEMPORALES:
                ruta_control = ruta_predicciones(
                    semilla,
                    control,
                    pliegue,
                )

                ruta_candidata = ruta_predicciones(
                    semilla,
                    candidata,
                    pliegue,
                )

                if not ruta_control.exists():
                    raise FileNotFoundError(
                        f"No existe: {ruta_control}"
                    )

                if not ruta_candidata.exists():
                    raise FileNotFoundError(
                        f"No existe: {ruta_candidata}"
                    )

                datos_control = pd.read_parquet(
                    ruta_control
                )

                datos_candidata = pd.read_parquet(
                    ruta_candidata
                )

                (
                    metricas_control,
                    operaciones_control,
                ) = simular(
                    datos_control,
                    umbral,
                )

                (
                    metricas_candidata,
                    operaciones_candidata,
                ) = simular(
                    datos_candidata,
                    umbral,
                )

                coeficiente = coeficiente_macro(
                    semilla=semilla,
                    variante=candidata,
                    pliegue=pliegue,
                    variable_macro=variable_macro,
                )

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

                registro = {
                    "comparacion": nombre_comparacion,
                    "semilla": semilla,
                    "pliegue": pliegue,
                    "control": control,
                    "candidata": candidata,
                    "umbral": umbral,
                    "variable_macro": variable_macro,
                    "operaciones_control": int(
                        metricas_control[
                            "operaciones"
                        ]
                    ),
                    "operaciones_candidata": int(
                        metricas_candidata[
                            "operaciones"
                        ]
                    ),
                    "precision_control": float(
                        metricas_control[
                            "precision"
                        ]
                    ),
                    "precision_candidata": float(
                        metricas_candidata[
                            "precision"
                        ]
                    ),
                    "positivas_control": float(
                        metricas_control[
                            "porcentaje_operaciones_positivas"
                        ]
                    ),
                    "positivas_candidata": float(
                        metricas_candidata[
                            "porcentaje_operaciones_positivas"
                        ]
                    ),
                    "retorno_control": retorno_control,
                    "retorno_candidata": retorno_candidata,
                    "retencion_retorno": division_segura(
                        retorno_candidata,
                        retorno_control,
                    ),
                    "mejora_relativa_retorno": (
                        division_segura(
                            retorno_candidata,
                            retorno_control,
                        )
                        - 1.0
                    ),
                    "factor_control": factor_control,
                    "factor_candidata": factor_candidata,
                    "retencion_factor": division_segura(
                        factor_candidata,
                        factor_control,
                    ),
                    "mejora_relativa_factor": (
                        division_segura(
                            factor_candidata,
                            factor_control,
                        )
                        - 1.0
                    ),
                    "drawdown_control": drawdown_control,
                    "drawdown_candidata": drawdown_candidata,
                    "mejora_drawdown_absoluta": (
                        drawdown_candidata
                        - drawdown_control
                    ),
                    "retorno_mediano_control": float(
                        metricas_control[
                            "retorno_neto_mediano"
                        ]
                    ),
                    "retorno_mediano_candidata": float(
                        metricas_candidata[
                            "retorno_neto_mediano"
                        ]
                    ),
                    "coeficiente_macro": coeficiente,
                    "signo_coeficiente": signo(
                        coeficiente
                    ),
                    "concentracion_mensual_control": (
                        concentracion_mensual(
                            operaciones_control
                        )
                    ),
                    "concentracion_mensual_candidata": (
                        concentracion_mensual(
                            operaciones_candidata
                        )
                    ),
                }

                registros_detalle.append(
                    registro
                )

                filas_semilla.append(
                    registro
                )

            tabla_semilla = pd.DataFrame(
                filas_semilla
            )

            retencion_retorno_peor_ano = float(
                tabla_semilla[
                    "retencion_retorno"
                ].min()
            )

            retencion_factor_peor_ano = float(
                tabla_semilla[
                    "retencion_factor"
                ].min()
            )

            mejora_retorno_media = float(
                tabla_semilla[
                    "mejora_relativa_retorno"
                ].mean()
            )

            mejora_factor_media = float(
                tabla_semilla[
                    "mejora_relativa_factor"
                ].mean()
            )

            mejora_drawdown_peor = float(
                tabla_semilla[
                    "mejora_drawdown_absoluta"
                ].min()
            )

            operaciones_minimas = int(
                tabla_semilla[
                    "operaciones_candidata"
                ].min()
            )

            retorno_minimo = float(
                tabla_semilla[
                    "retorno_candidata"
                ].min()
            )

            factor_minimo = float(
                tabla_semilla[
                    "factor_candidata"
                ].min()
            )

            mediana_minima = float(
                tabla_semilla[
                    "retorno_mediano_candidata"
                ].min()
            )

            signos = tabla_semilla[
                "signo_coeficiente"
            ].tolist()

            signo_estable = bool(
                len(
                    set(signos)
                ) == 1
                and signos[0] != 0
            )

            concentracion_maxima = float(
                tabla_semilla[
                    "concentracion_mensual_candidata"
                ].max()
            )

            aprueba_semilla = bool(
                operaciones_minimas
                >= OPERACIONES_MINIMAS
                and retorno_minimo > 0.0
                and factor_minimo > 1.0
                and mediana_minima >= 0.0
                and retencion_retorno_peor_ano
                >= RETENCION_MINIMA_RETORNO_ANUAL
                and retencion_factor_peor_ano
                >= RETENCION_MINIMA_FACTOR_ANUAL
                and mejora_retorno_media > 0.0
                and mejora_factor_media >= 0.0
                and mejora_drawdown_peor
                >= EMPEORAMIENTO_MAXIMO_DRAWDOWN
                and signo_estable
                and concentracion_maxima
                <= CONCENTRACION_MENSUAL_MAXIMA
            )

            resumen = {
                "comparacion": nombre_comparacion,
                "semilla": semilla,
                "control": control,
                "candidata": candidata,
                "umbral": umbral,
                "variable_macro": variable_macro,
                "operaciones_minimas_candidata": (
                    operaciones_minimas
                ),
                "retorno_minimo_candidata": retorno_minimo,
                "factor_minimo_candidata": factor_minimo,
                "mediana_minima_candidata": mediana_minima,
                "retencion_retorno_peor_ano": (
                    retencion_retorno_peor_ano
                ),
                "retencion_factor_peor_ano": (
                    retencion_factor_peor_ano
                ),
                "mejora_retorno_media": mejora_retorno_media,
                "mejora_factor_media": mejora_factor_media,
                "mejora_drawdown_peor": mejora_drawdown_peor,
                "coeficiente_validacion_2023": float(
                    tabla_semilla.loc[
                        tabla_semilla[
                            "pliegue"
                        ]
                        == "validacion_2023",
                        "coeficiente_macro",
                    ].iloc[0]
                ),
                "coeficiente_validacion_2024": float(
                    tabla_semilla.loc[
                        tabla_semilla[
                            "pliegue"
                        ]
                        == "validacion_2024",
                        "coeficiente_macro",
                    ].iloc[0]
                ),
                "signo_estable": signo_estable,
                "concentracion_mensual_maxima": (
                    concentracion_maxima
                ),
                "aprueba_semilla": aprueba_semilla,
            }

            registros_semilla.append(
                resumen
            )

            print(
                f"Semilla {semilla}: "
                f"retorno medio {mejora_retorno_media:+.2%}, "
                f"factor {mejora_factor_media:+.2%}, "
                f"peor DD {mejora_drawdown_peor:+.2%}, "
                f"aprueba={aprueba_semilla}"
            )

    detalle = pd.DataFrame(
        registros_detalle
    )

    resumen_semillas = pd.DataFrame(
        registros_semilla
    )

    ruta_detalle = (
        RUTA_RESULTADOS_ROBUSTEZ
        / "resultados_pareados_por_pliegue.csv"
    )

    ruta_resumen = (
        RUTA_RESULTADOS_ROBUSTEZ
        / "resumen_pareado_por_semilla.csv"
    )

    detalle.to_csv(
        ruta_detalle,
        index=False,
        encoding="utf-8-sig",
    )

    resumen_semillas.to_csv(
        ruta_resumen,
        index=False,
        encoding="utf-8-sig",
    )

    ruta_json = (
        RUTA_RESULTADOS_ROBUSTEZ
        / "detalle_robustez_semillas.json"
    )

    ruta_json.write_text(
        json.dumps(
            {
                "uso_2025_para_seleccion": False,
                "uso_2026_para_seleccion": False,
                "resultados_por_pliegue": [
                    {
                        clave: valor_json(valor)
                        for clave, valor
                        in registro.items()
                    }
                    for registro
                    in registros_detalle
                ],
                "resumen_por_semilla": [
                    {
                        clave: valor_json(valor)
                        for clave, valor
                        in registro.items()
                    }
                    for registro
                    in registros_semilla
                ],
            },
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\nEVALUACIÓN DE ROBUSTEZ COMPLETADA"
    )
    print(
        f"- {ruta_detalle}"
    )
    print(
        f"- {ruta_resumen}"
    )
    print(
        f"- {ruta_json}"
    )


if __name__ == "__main__":
    main()
