from __future__ import annotations

import json

import pandas as pd

from cripto.corto_plazo_v4_4.configuracion import (
    PLIEGUES,
    RUTA_COMPARACION,
    RUTA_LABORATORIO,
    RUTA_SELECCION,
)


def porcentaje(
    valor: float,
) -> str:
    """Formatea un valor decimal como porcentaje."""

    return f"{valor * 100:.4f} %"


def main() -> None:
    """Genera una comparación legible entre V4.1 y la decisión V4.4."""

    ruta_decision = (
        RUTA_SELECCION
        / "decision_v4_4.json"
    )

    ruta_resultados = (
        RUTA_LABORATORIO
        / "resultados_laboratorio.csv"
    )

    if not ruta_decision.exists():
        raise FileNotFoundError(
            f"No existe: {ruta_decision}"
        )

    if not ruta_resultados.exists():
        raise FileNotFoundError(
            f"No existe: {ruta_resultados}"
        )

    decision = json.loads(
        ruta_decision.read_text(
            encoding="utf-8"
        )
    )

    resultados = pd.read_csv(
        ruta_resultados
    )

    control = resultados.loc[
        (
            resultados[
                "modo_evaluacion"
            ]
            == "cohorte_v4_1"
        )
        & resultados[
            "es_control_v4_1"
        ].astype(bool)
    ].copy()

    lineas = [
        "# Comparación V4.1 frente a V4.4",
        "",
        f"- **Decisión:** {decision['decision']}",
        "- **Entrada:** cruce SUBE 0,44.",
        "- **Veto previo:** ninguno.",
        "- **Coste:** 0,10 %.",
        "- **2025 utilizado:** no.",
        "- **2026 utilizado:** no.",
        "",
    ]

    if (
        decision[
            "decision"
        ]
        == "mantener_v4_1"
    ):
        lineas.extend(
            [
                "## Resultado",
                "",
                decision[
                    "motivo"
                ],
                "",
                "V4.1 permanece como campeona.",
                "",
            ]
        )
    else:
        estrategia_id = decision[
            "estrategia_id"
        ]

        candidata = resultados.loc[
            (
                resultados[
                    "modo_evaluacion"
                ]
                == "cohorte_v4_1"
            )
            & (
                resultados[
                    "estrategia_id"
                ]
                == estrategia_id
            )
        ].copy()

        lineas.extend(
            [
                "## Estrategia candidata",
                "",
                f"- **ID:** {estrategia_id}",
                f"- **Umbral BAJA:** {decision['umbral_baja']}",
                f"- **Minutos mínimos:** {decision['minutos_minimos']}",
                f"- **Condición:** {decision['condicion_salida']}",
                "",
                "| Pliegue | Modelo | Retorno neto medio | Factor beneficio | Drawdown | Positivas |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )

        for pliegue in PLIEGUES:
            fila_control = control.loc[
                control[
                    "pliegue"
                ]
                == pliegue
            ].iloc[0]

            fila_candidata = candidata.loc[
                candidata[
                    "pliegue"
                ]
                == pliegue
            ].iloc[0]

            lineas.append(
                "| "
                f"{pliegue} | V4.1 | "
                f"{porcentaje(float(fila_control['retorno_neto_medio']))} | "
                f"{float(fila_control['factor_beneficio']):.4f} | "
                f"{porcentaje(float(fila_control['maximo_drawdown']))} | "
                f"{float(fila_control['porcentaje_operaciones_positivas']):.2f} % |"
            )

            lineas.append(
                "| "
                f"{pliegue} | V4.4 | "
                f"{porcentaje(float(fila_candidata['retorno_neto_medio']))} | "
                f"{float(fila_candidata['factor_beneficio']):.4f} | "
                f"{porcentaje(float(fila_candidata['maximo_drawdown']))} | "
                f"{float(fila_candidata['porcentaje_operaciones_positivas']):.2f} % |"
            )

        lineas.append("")

    RUTA_COMPARACION.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta = (
        RUTA_COMPARACION
        / "comparacion_v4_1_vs_v4_4.md"
    )

    ruta.write_text(
        "\n".join(
            lineas
        ),
        encoding="utf-8",
    )

    print(
        "\nCOMPARACIÓN GENERADA"
    )
    print("=" * 72)
    print(
        f"- {ruta}"
    )


if __name__ == "__main__":
    main()
