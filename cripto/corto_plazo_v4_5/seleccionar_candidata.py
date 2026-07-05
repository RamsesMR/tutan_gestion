from __future__ import annotations

import json

import pandas as pd

from cripto.corto_plazo_v4_5.configuracion import (
    RUTA_RESULTADOS,
    RUTA_SELECCION,
    UMBRAL_SUBE_CONGELADO,
)


def resumir_candidato(
    grupo: pd.DataFrame,
) -> dict:
    return {
        "variante": grupo["variante"].iloc[0],
        "umbral": float(grupo["umbral"].iloc[0]),
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


def main() -> None:
    ruta = (
        RUTA_RESULTADOS
        / "resultados_desarrollo.csv"
    )

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe: {ruta}"
        )

    resultados = pd.read_csv(ruta)

    resumenes = []

    for (
        variante,
        umbral,
    ), grupo in resultados.groupby(
        [
            "variante",
            "umbral",
        ],
        sort=True,
    ):
        if grupo["pliegue"].nunique() != 2:
            continue

        resumenes.append(
            resumir_candidato(grupo)
        )

    resumen = pd.DataFrame(resumenes)

    resumen["cumple_filtros"] = (
        (resumen["operaciones_minimas"] >= 50)
        & (
            resumen["retorno_neto_medio_minimo"]
            > 0
        )
        & (
            resumen["retorno_neto_mediano_minimo"]
            >= 0
        )
        & (
            resumen["factor_beneficio_minimo"]
            > 1
        )
    )

    RUTA_SELECCION.mkdir(
        parents=True,
        exist_ok=True,
    )

    resumen.to_csv(
        RUTA_SELECCION / "resumen_candidatos.csv",
        index=False,
        encoding="utf-8-sig",
    )

    control = resumen.loc[
        (resumen["variante"] == "control_63")
        & (
            resumen["umbral"]
            == UMBRAL_SUBE_CONGELADO
        )
    ]

    if control.empty:
        raise ValueError(
            "No existe el control_63 con umbral 0,44."
        )

    control = control.iloc[0]

    nuevos = resumen.loc[
        (resumen["variante"] != "control_63")
        & resumen["cumple_filtros"]
    ].copy()

    if nuevos.empty:
        decision = {
            "decision": "mantener_v4_1",
            "motivo": (
                "Ninguna variante con variables nuevas cumplió "
                "los filtros mínimos en ambos pliegues."
            ),
            "uso_2025_para_seleccion": False,
            "uso_2026_para_seleccion": False,
        }
    else:
        nuevos["mejora_retorno_minimo"] = (
            nuevos["retorno_neto_medio_minimo"]
            - float(
                control["retorno_neto_medio_minimo"]
            )
        )

        nuevos["mejora_drawdown"] = (
            nuevos["drawdown_peor"]
            - float(
                control["drawdown_peor"]
            )
        )

        nuevos["promovible"] = (
            (
                nuevos["retorno_neto_medio_minimo"]
                >= float(
                    control[
                        "retorno_neto_medio_minimo"
                    ]
                )
                * 1.05
            )
            |
            (
                (
                    nuevos[
                        "retorno_neto_medio_minimo"
                    ]
                    >= float(
                        control[
                            "retorno_neto_medio_minimo"
                        ]
                    )
                    * 0.95
                )
                & (
                    nuevos["mejora_drawdown"]
                    >= 0.01
                )
            )
        )

        promovibles = nuevos.loc[
            nuevos["promovible"]
        ].copy()

        if promovibles.empty:
            decision = {
                "decision": "mantener_v4_1",
                "motivo": (
                    "Las variables nuevas generaron candidatos válidos, "
                    "pero ninguno mejoró suficientemente el peor retorno "
                    "o el drawdown frente al control."
                ),
                "uso_2025_para_seleccion": False,
                "uso_2026_para_seleccion": False,
            }
        else:
            promovibles = promovibles.sort_values(
                [
                    "retorno_neto_medio_minimo",
                    "factor_beneficio_minimo",
                    "drawdown_peor",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )

            mejor = promovibles.iloc[0]

            decision = {
                "decision": "candidata_v4_5_seleccionada",
                "variante": mejor["variante"],
                "umbral": float(mejor["umbral"]),
                "retorno_neto_medio_minimo": float(
                    mejor["retorno_neto_medio_minimo"]
                ),
                "factor_beneficio_minimo": float(
                    mejor["factor_beneficio_minimo"]
                ),
                "drawdown_peor": float(
                    mejor["drawdown_peor"]
                ),
                "uso_2025_para_seleccion": False,
                "uso_2026_para_seleccion": False,
            }

    (
        RUTA_SELECCION / "decision_v4_5.json"
    ).write_text(
        json.dumps(
            decision,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print("\nSELECCIÓN V4.5")
    print("=" * 72)
    print(f"Decisión: {decision['decision']}")
    print(
        f"- {RUTA_SELECCION / 'resumen_candidatos.csv'}"
    )
    print(
        f"- {RUTA_SELECCION / 'decision_v4_5.json'}"
    )
    print("\n2025 y 2026 no fueron utilizados.")


if __name__ == "__main__":
    main()
