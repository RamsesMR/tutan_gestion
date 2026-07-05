from __future__ import annotations

import json
from typing import Any

import pandas as pd

from cripto.corto_plazo_v4_5.configuracion import (
    RUTA_RESULTADOS,
    RUTA_SELECCION_FUTUROS,
    VARIANTE_CONTROL_FUTUROS,
    VARIANTES_ABLACION_FUTUROS,
)


PLIEGUES = (
    "validacion_2023",
    "validacion_2024",
)


def resumir(
    grupo: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "variante": str(
            grupo["variante"].iloc[0]
        ),
        "umbral": float(
            grupo["umbral"].iloc[0]
        ),
        "operaciones_minimas": int(
            grupo["operaciones"].min()
        ),
        "retorno_neto_medio_minimo": float(
            grupo["retorno_neto_medio"].min()
        ),
        "retorno_neto_mediano_minimo": float(
            grupo["retorno_neto_mediano"].min()
        ),
        "factor_beneficio_minimo": float(
            grupo["factor_beneficio"].min()
        ),
        "drawdown_peor": float(
            grupo["maximo_drawdown"].min()
        ),
        "operaciones_positivas_minimas": float(
            grupo[
                "porcentaje_operaciones_positivas"
            ].min()
        ),
    }


def es_valido(
    fila: pd.Series,
) -> bool:
    return bool(
        int(
            fila["operaciones_minimas"]
        )
        >= 50
        and float(
            fila["retorno_neto_medio_minimo"]
        )
        > 0.0
        and float(
            fila["retorno_neto_mediano_minimo"]
        )
        >= 0.0
        and float(
            fila["factor_beneficio_minimo"]
        )
        > 1.0
    )


def elegir_mejor(
    tabla: pd.DataFrame,
) -> pd.Series:
    validos = tabla.loc[
        tabla["cumple_filtros"].astype(bool)
    ].copy()

    if validos.empty:
        raise ValueError(
            "No existe ninguna configuración válida en la tabla recibida."
        )

    validos = validos.sort_values(
        [
            "retorno_neto_medio_minimo",
            "factor_beneficio_minimo",
            "drawdown_peor",
            "operaciones_minimas",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )

    return validos.iloc[0]


def main() -> None:
    ruta_resultados = (
        RUTA_RESULTADOS
        / "resultados_desarrollo.csv"
    )

    if not ruta_resultados.exists():
        raise FileNotFoundError(
            f"No existe: {ruta_resultados}"
        )

    resultados = pd.read_csv(
        ruta_resultados
    )

    variantes_necesarias = {
        VARIANTE_CONTROL_FUTUROS,
        *VARIANTES_ABLACION_FUTUROS,
    }

    faltantes = variantes_necesarias.difference(
        set(
            resultados["variante"].unique()
        )
    )

    if faltantes:
        raise ValueError(
            "Faltan resultados de variantes de ablación: "
            + ", ".join(
                sorted(faltantes)
            )
        )

    resumenes: list[
        dict[str, Any]
    ] = []

    for (
        variante,
        umbral,
    ), grupo in resultados.loc[
        resultados["variante"].isin(
            variantes_necesarias
        )
    ].groupby(
        [
            "variante",
            "umbral",
        ],
        sort=True,
    ):
        if set(
            grupo["pliegue"]
        ) != set(
            PLIEGUES
        ):
            continue

        resumenes.append(
            resumir(
                grupo
            )
        )

    resumen = pd.DataFrame(
        resumenes
    )

    resumen["cumple_filtros"] = resumen.apply(
        es_valido,
        axis=1,
    )

    control_tabla = resumen.loc[
        resumen["variante"]
        == VARIANTE_CONTROL_FUTUROS
    ].copy()

    control = elegir_mejor(
        control_tabla
    )

    candidatos = resumen.loc[
        resumen["variante"].isin(
            VARIANTES_ABLACION_FUTUROS
        )
        & resumen[
            "cumple_filtros"
        ].astype(bool)
    ].copy()

    candidatos["retencion_retorno"] = (
        candidatos[
            "retorno_neto_medio_minimo"
        ]
        / float(
            control[
                "retorno_neto_medio_minimo"
            ]
        )
    )

    candidatos["retencion_factor_beneficio"] = (
        candidatos[
            "factor_beneficio_minimo"
        ]
        / float(
            control[
                "factor_beneficio_minimo"
            ]
        )
    )

    candidatos["mejora_drawdown_absoluta"] = (
        candidatos[
            "drawdown_peor"
        ]
        - float(
            control[
                "drawdown_peor"
            ]
        )
    )

    candidatos["mejora_material"] = (
        (
            candidatos[
                "retencion_retorno"
            ]
            >= 1.05
        )
        |
        (
            candidatos[
                "retencion_factor_beneficio"
            ]
            >= 1.05
        )
        |
        (
            candidatos[
                "mejora_drawdown_absoluta"
            ]
            >= 0.01
        )
    )

    candidatos["promovible"] = (
        (
            candidatos[
                "retencion_retorno"
            ]
            >= 0.95
        )
        & (
            candidatos[
                "retencion_factor_beneficio"
            ]
            >= 0.95
        )
        & (
            candidatos[
                "mejora_drawdown_absoluta"
            ]
            >= -0.01
        )
        & candidatos[
            "mejora_material"
        ].astype(bool)
    )

    RUTA_SELECCION_FUTUROS.mkdir(
        parents=True,
        exist_ok=True,
    )

    resumen.to_csv(
        RUTA_SELECCION_FUTUROS
        / "resumen_todas_las_configuraciones.csv",
        index=False,
        encoding="utf-8-sig",
    )

    candidatos.to_csv(
        RUTA_SELECCION_FUTUROS
        / "comparacion_candidatos_futuros.csv",
        index=False,
        encoding="utf-8-sig",
    )

    promovibles = candidatos.loc[
        candidatos[
            "promovible"
        ].astype(bool)
    ].copy()

    if promovibles.empty:
        decision = {
            "decision": "mantener_v4_5_spot",
            "motivo": (
                "Ninguna familia de futuros mejoró de forma material "
                "el control Spot sobre exactamente la misma muestra "
                "sin degradar demasiado retorno, factor o drawdown."
            ),
            "control": {
                "variante": str(
                    control["variante"]
                ),
                "umbral": float(
                    control["umbral"]
                ),
                "retorno_neto_medio_minimo": float(
                    control[
                        "retorno_neto_medio_minimo"
                    ]
                ),
                "factor_beneficio_minimo": float(
                    control[
                        "factor_beneficio_minimo"
                    ]
                ),
                "drawdown_peor": float(
                    control["drawdown_peor"]
                ),
            },
            "uso_2025_para_seleccion": False,
            "uso_2026_para_seleccion": False,
        }
    else:
        promovibles = promovibles.sort_values(
            [
                "retorno_neto_medio_minimo",
                "factor_beneficio_minimo",
                "drawdown_peor",
                "operaciones_minimas",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )

        mejor = promovibles.iloc[0]

        decision = {
            "decision": "candidata_futuros_seleccionada",
            "variante": str(
                mejor["variante"]
            ),
            "umbral": float(
                mejor["umbral"]
            ),
            "retorno_neto_medio_minimo": float(
                mejor[
                    "retorno_neto_medio_minimo"
                ]
            ),
            "factor_beneficio_minimo": float(
                mejor[
                    "factor_beneficio_minimo"
                ]
            ),
            "drawdown_peor": float(
                mejor["drawdown_peor"]
            ),
            "retencion_retorno": float(
                mejor["retencion_retorno"]
            ),
            "retencion_factor_beneficio": float(
                mejor[
                    "retencion_factor_beneficio"
                ]
            ),
            "mejora_drawdown_absoluta": float(
                mejor[
                    "mejora_drawdown_absoluta"
                ]
            ),
            "control": {
                "variante": str(
                    control["variante"]
                ),
                "umbral": float(
                    control["umbral"]
                ),
            },
            "uso_2025_para_seleccion": False,
            "uso_2026_para_seleccion": False,
        }

    ruta_decision = (
        RUTA_SELECCION_FUTUROS
        / "decision_ablacion_futuros.json"
    )

    ruta_decision.write_text(
        json.dumps(
            decision,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\nSELECCIÓN DE ABLACIÓN DE FUTUROS V4.5"
    )
    print("=" * 72)
    print(
        "Control Spot sobre la misma muestra: "
        f"{control['variante']} con umbral {float(control['umbral']):.2f}"
    )
    print(
        f"Decisión: {decision['decision']}"
    )

    if (
        decision["decision"]
        == "candidata_futuros_seleccionada"
    ):
        print(
            "Candidata: "
            f"{decision['variante']} con umbral {decision['umbral']:.2f}"
        )

    print(
        f"- {ruta_decision}"
    )
    print(
        "\nLa selección original de V4.5-Spot no fue sobrescrita."
    )
    print(
        "2025 y 2026 no fueron utilizados."
    )


if __name__ == "__main__":
    main()
