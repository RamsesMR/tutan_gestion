from __future__ import annotations

import json
from typing import Any

import pandas as pd

from cripto.corto_plazo_v4_5.configuracion import (
    RUTA_COMPARACION_FUTUROS,
    RUTA_RESULTADOS,
    RUTA_SELECCION_FUTUROS,
)


PLIEGUES = (
    "validacion_2023",
    "validacion_2024",
)


def porcentaje_decimal(
    valor: float,
) -> str:
    return f"{valor * 100:.4f} %"


def porcentaje_normal(
    valor: float,
) -> str:
    return f"{valor:.2f} %"


def obtener_filas(
    resultados: pd.DataFrame,
    variante: str,
    umbral: float,
) -> pd.DataFrame:
    filas = resultados.loc[
        (resultados["variante"] == variante)
        & (
            resultados["umbral"].astype(float)
            == float(umbral)
        )
    ].copy()

    if set(
        filas["pliegue"]
    ) != set(
        PLIEGUES
    ):
        raise ValueError(
            f"{variante} con umbral {umbral} no contiene ambos pliegues."
        )

    return filas


def main() -> None:
    ruta_decision = (
        RUTA_SELECCION_FUTUROS
        / "decision_ablacion_futuros.json"
    )

    ruta_resultados = (
        RUTA_RESULTADOS
        / "resultados_desarrollo.csv"
    )

    if not ruta_decision.exists():
        raise FileNotFoundError(
            f"No existe: {ruta_decision}"
        )

    if not ruta_resultados.exists():
        raise FileNotFoundError(
            f"No existe: {ruta_resultados}"
        )

    decision: dict[str, Any] = json.loads(
        ruta_decision.read_text(
            encoding="utf-8"
        )
    )

    resultados = pd.read_csv(
        ruta_resultados
    )

    control = decision["control"]

    filas_control = obtener_filas(
        resultados,
        str(
            control["variante"]
        ),
        float(
            control["umbral"]
        ),
    )

    RUTA_COMPARACION_FUTUROS.mkdir(
        parents=True,
        exist_ok=True,
    )

    lineas = [
        "# Ablación de variables de futuros — V4.5",
        "",
        "## Control experimental",
        "",
        f"- **Variante:** `{control['variante']}`",
        f"- **Umbral:** `{float(control['umbral']):.2f}`",
        "- **Muestra:** exactamente las mismas fechas que las variantes "
        "de futuros.",
        "- **Selección:** únicamente 2023 y 2024.",
        "- **2025 utilizado:** no.",
        "- **2026 utilizado:** no.",
        "",
        f"## Decisión: `{decision['decision']}`",
        "",
    ]

    registros_csv: list[
        dict[str, Any]
    ] = []

    if (
        decision["decision"]
        == "candidata_futuros_seleccionada"
    ):
        variante = str(
            decision["variante"]
        )
        umbral = float(
            decision["umbral"]
        )

        filas_candidata = obtener_filas(
            resultados,
            variante,
            umbral,
        )

        lineas.extend(
            [
                f"- **Candidata:** `{variante}`",
                f"- **Umbral:** `{umbral:.2f}`",
                "",
                "| Pliegue | Modelo | Operaciones | Precisión | Positivas | "
                "Retorno neto medio | Factor beneficio | Drawdown |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for pliegue in PLIEGUES:
            fila_control = filas_control.loc[
                filas_control["pliegue"]
                == pliegue
            ].iloc[0]

            fila_candidata = filas_candidata.loc[
                filas_candidata["pliegue"]
                == pliegue
            ].iloc[0]

            for nombre, fila in (
                ("Control Spot", fila_control),
                ("Candidata futuros", fila_candidata),
            ):
                lineas.append(
                    "| "
                    f"{pliegue} | "
                    f"{nombre} | "
                    f"{int(fila['operaciones'])} | "
                    f"{porcentaje_normal(float(fila['porcentaje_acierto']))} | "
                    f"{porcentaje_normal(float(fila['porcentaje_operaciones_positivas']))} | "
                    f"{porcentaje_decimal(float(fila['retorno_neto_medio']))} | "
                    f"{float(fila['factor_beneficio']):.4f} | "
                    f"{porcentaje_decimal(float(fila['maximo_drawdown']))} |"
                )

            registros_csv.append(
                {
                    "pliegue": pliegue,
                    "control_variante": str(
                        control["variante"]
                    ),
                    "control_umbral": float(
                        control["umbral"]
                    ),
                    "candidata_variante": variante,
                    "candidata_umbral": umbral,
                    "retorno_control": float(
                        fila_control[
                            "retorno_neto_medio"
                        ]
                    ),
                    "retorno_candidata": float(
                        fila_candidata[
                            "retorno_neto_medio"
                        ]
                    ),
                    "factor_control": float(
                        fila_control[
                            "factor_beneficio"
                        ]
                    ),
                    "factor_candidata": float(
                        fila_candidata[
                            "factor_beneficio"
                        ]
                    ),
                    "drawdown_control": float(
                        fila_control[
                            "maximo_drawdown"
                        ]
                    ),
                    "drawdown_candidata": float(
                        fila_candidata[
                            "maximo_drawdown"
                        ]
                    ),
                }
            )
    else:
        lineas.extend(
            [
                decision["motivo"],
                "",
                "**La candidata congelada V4.5-Spot continúa siendo "
                "la opción principal.**",
                "",
            ]
        )

    ruta_md = (
        RUTA_COMPARACION_FUTUROS
        / "comparacion_ablacion_futuros.md"
    )

    ruta_md.write_text(
        "\n".join(
            lineas
        ),
        encoding="utf-8",
    )

    if registros_csv:
        pd.DataFrame(
            registros_csv
        ).to_csv(
            RUTA_COMPARACION_FUTUROS
            / "comparacion_directa_control_vs_futuros.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(
        "\nCOMPARACIÓN DE ABLACIÓN DE FUTUROS GENERADA"
    )
    print("=" * 72)
    print(
        f"Decisión: {decision['decision']}"
    )
    print(
        f"- {ruta_md}"
    )


if __name__ == "__main__":
    main()
