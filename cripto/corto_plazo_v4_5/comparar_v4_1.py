from __future__ import annotations

import json
from typing import Any

import pandas as pd

from cripto.corto_plazo_v4_1.configuracion import (
    RUTA_SELECCION as RUTA_SELECCION_V4_1,
)
from cripto.corto_plazo_v4_5.configuracion import (
    RUTA_COMPARACION,
    RUTA_RESULTADOS,
    RUTA_SELECCION,
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


def cargar_v4_1_original() -> pd.DataFrame:
    if not RUTA_SELECCION_V4_1.exists():
        raise FileNotFoundError(
            f"No existe la estrategia original V4.1: "
            f"{RUTA_SELECCION_V4_1}"
        )

    seleccion = pd.read_csv(
        RUTA_SELECCION_V4_1
    )

    if seleccion.empty:
        raise ValueError(
            "La selección original de V4.1 está vacía."
        )

    fila = seleccion.iloc[0]

    registros = []

    for anio, pliegue in (
        ("2023", "validacion_2023"),
        ("2024", "validacion_2024"),
    ):
        registros.append(
            {
                "modelo": "V4.1 original",
                "variante": "v4_63_sin_pesos",
                "umbral": float(
                    fila["umbral_sube"]
                ),
                "pliegue": pliegue,
                "operaciones": int(
                    fila[f"operaciones_{anio}"]
                ),
                "precision": float(
                    fila[
                        f"precision_clasificacion_{anio}"
                    ]
                ),
                "porcentaje_acierto": float(
                    fila[
                        f"porcentaje_acierto_clasificacion_{anio}"
                    ]
                ),
                "porcentaje_operaciones_positivas": float(
                    fila[
                        f"porcentaje_operaciones_positivas_{anio}"
                    ]
                ),
                "retorno_neto_medio": float(
                    fila[
                        f"retorno_neto_medio_{anio}"
                    ]
                ),
                "retorno_neto_mediano": float(
                    fila[
                        f"retorno_neto_mediano_{anio}"
                    ]
                ),
                "factor_beneficio": float(
                    fila[
                        f"factor_beneficio_{anio}"
                    ]
                ),
                "maximo_drawdown": float(
                    fila[
                        f"maximo_drawdown_{anio}"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        registros
    )


def cargar_candidata_v4_5(
    resultados: pd.DataFrame,
    decision: dict[str, Any],
) -> pd.DataFrame:
    if (
        decision.get("decision")
        != "candidata_v4_5_seleccionada"
    ):
        raise ValueError(
            "No existe una candidata V4.5 seleccionada."
        )

    variante = str(
        decision["variante"]
    )
    umbral = float(
        decision["umbral"]
    )

    candidata = resultados.loc[
        (resultados["variante"] == variante)
        & (
            resultados["umbral"].astype(float)
            == umbral
        )
    ].copy()

    if set(
        candidata["pliegue"]
    ) != set(
        PLIEGUES
    ):
        raise ValueError(
            "La candidata V4.5 no contiene los dos pliegues."
        )

    candidata["modelo"] = "V4.5 candidata"

    return candidata[
        [
            "modelo",
            "variante",
            "umbral",
            "pliegue",
            "operaciones",
            "precision",
            "porcentaje_acierto",
            "porcentaje_operaciones_positivas",
            "retorno_neto_medio",
            "retorno_neto_mediano",
            "factor_beneficio",
            "maximo_drawdown",
        ]
    ].copy()


def construir_comparacion(
    v4_1: pd.DataFrame,
    v4_5: pd.DataFrame,
) -> pd.DataFrame:
    registros = []

    for pliegue in PLIEGUES:
        fila_v4_1 = v4_1.loc[
            v4_1["pliegue"] == pliegue
        ].iloc[0]

        fila_v4_5 = v4_5.loc[
            v4_5["pliegue"] == pliegue
        ].iloc[0]

        registros.append(
            {
                "pliegue": pliegue,
                "operaciones_v4_1": int(
                    fila_v4_1["operaciones"]
                ),
                "operaciones_v4_5": int(
                    fila_v4_5["operaciones"]
                ),
                "precision_v4_1": float(
                    fila_v4_1["precision"]
                ),
                "precision_v4_5": float(
                    fila_v4_5["precision"]
                ),
                "positivas_v4_1": float(
                    fila_v4_1[
                        "porcentaje_operaciones_positivas"
                    ]
                ),
                "positivas_v4_5": float(
                    fila_v4_5[
                        "porcentaje_operaciones_positivas"
                    ]
                ),
                "retorno_v4_1": float(
                    fila_v4_1["retorno_neto_medio"]
                ),
                "retorno_v4_5": float(
                    fila_v4_5["retorno_neto_medio"]
                ),
                "mejora_relativa_retorno": (
                    float(
                        fila_v4_5[
                            "retorno_neto_medio"
                        ]
                    )
                    / float(
                        fila_v4_1[
                            "retorno_neto_medio"
                        ]
                    )
                    - 1.0
                ),
                "factor_v4_1": float(
                    fila_v4_1["factor_beneficio"]
                ),
                "factor_v4_5": float(
                    fila_v4_5["factor_beneficio"]
                ),
                "mejora_relativa_factor": (
                    float(
                        fila_v4_5[
                            "factor_beneficio"
                        ]
                    )
                    / float(
                        fila_v4_1[
                            "factor_beneficio"
                        ]
                    )
                    - 1.0
                ),
                "drawdown_v4_1": float(
                    fila_v4_1["maximo_drawdown"]
                ),
                "drawdown_v4_5": float(
                    fila_v4_5["maximo_drawdown"]
                ),
                "mejora_drawdown_absoluta": (
                    float(
                        fila_v4_5[
                            "maximo_drawdown"
                        ]
                    )
                    - float(
                        fila_v4_1[
                            "maximo_drawdown"
                        ]
                    )
                ),
            }
        )

    return pd.DataFrame(
        registros
    )


def main() -> None:
    ruta_resultados = (
        RUTA_RESULTADOS
        / "resultados_desarrollo.csv"
    )

    ruta_decision = (
        RUTA_SELECCION
        / "decision_v4_5.json"
    )

    if not ruta_resultados.exists():
        raise FileNotFoundError(
            f"No existe: {ruta_resultados}"
        )

    if not ruta_decision.exists():
        raise FileNotFoundError(
            f"No existe: {ruta_decision}"
        )

    resultados = pd.read_csv(
        ruta_resultados
    )

    decision = json.loads(
        ruta_decision.read_text(
            encoding="utf-8"
        )
    )

    v4_1 = cargar_v4_1_original()

    v4_5 = cargar_candidata_v4_5(
        resultados,
        decision,
    )

    comparacion = construir_comparacion(
        v4_1,
        v4_5,
    )

    RUTA_COMPARACION.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_csv = (
        RUTA_COMPARACION
        / "comparacion_directa_v4_1_vs_v4_5.csv"
    )

    comparacion.to_csv(
        ruta_csv,
        index=False,
        encoding="utf-8-sig",
    )

    variante = str(
        decision["variante"]
    )
    umbral = float(
        decision["umbral"]
    )

    lineas = [
        "# Comparación directa V4.1 original frente a V4.5",
        "",
        "## Configuración comparada",
        "",
        "- **V4.1 original:** detector `v4_63_sin_pesos`, "
        "63 variables y umbral `0,44`.",
        f"- **V4.5 candidata:** variante `{variante}`, "
        f"90 variables y umbral `{umbral:.2f}`.",
        "- **Entrada:** cruce desde abajo.",
        "- **Salida:** fija a 480 minutos.",
        "- **Coste:** 0,10 %.",
        "- **Selección:** 2023 y 2024.",
        "- **2025 utilizado para selección:** no.",
        "- **2026 utilizado para selección:** no.",
        "",
        "## Resultados",
        "",
        "| Pliegue | Modelo | Operaciones | Precisión | Positivas | "
        "Retorno neto medio | Factor beneficio | Drawdown |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for pliegue in PLIEGUES:
        fila_v4_1 = v4_1.loc[
            v4_1["pliegue"] == pliegue
        ].iloc[0]

        fila_v4_5 = v4_5.loc[
            v4_5["pliegue"] == pliegue
        ].iloc[0]

        for fila in (
            fila_v4_1,
            fila_v4_5,
        ):
            lineas.append(
                "| "
                f"{pliegue} | "
                f"{fila['modelo']} | "
                f"{int(fila['operaciones'])} | "
                f"{porcentaje_normal(float(fila['porcentaje_acierto']))} | "
                f"{porcentaje_normal(float(fila['porcentaje_operaciones_positivas']))} | "
                f"{porcentaje_decimal(float(fila['retorno_neto_medio']))} | "
                f"{float(fila['factor_beneficio']):.4f} | "
                f"{porcentaje_decimal(float(fila['maximo_drawdown']))} |"
            )

    lineas.extend(
        [
            "",
            "## Diferencias de V4.5 respecto a V4.1",
            "",
            "| Pliegue | Mejora relativa retorno | "
            "Mejora relativa factor | Mejora drawdown |",
            "|---|---:|---:|---:|",
        ]
    )

    for _, fila in comparacion.iterrows():
        lineas.append(
            "| "
            f"{fila['pliegue']} | "
            f"{porcentaje_decimal(float(fila['mejora_relativa_retorno']))} | "
            f"{porcentaje_decimal(float(fila['mejora_relativa_factor']))} | "
            f"{porcentaje_decimal(float(fila['mejora_drawdown_absoluta']))} |"
        )

    lineas.extend(
        [
            "",
            "## Decisión",
            "",
            "**V4.5 queda seleccionada como candidata de desarrollo, "
            "pero todavía no sustituye a V4.1 hasta completar las "
            "confirmaciones posteriores con la configuración congelada.**",
            "",
        ]
    )

    ruta_md = (
        RUTA_COMPARACION
        / "comparacion_v4_1_vs_v4_5.md"
    )

    ruta_md.write_text(
        "\n".join(
            lineas
        ),
        encoding="utf-8",
    )

    print(
        "\nCOMPARACIÓN DIRECTA V4.1 VS V4.5"
    )
    print("=" * 72)
    print(
        f"Candidata V4.5: {variante}, umbral {umbral:.2f}"
    )

    for _, fila in comparacion.iterrows():
        print(
            f"\n{fila['pliegue']}"
        )
        print("-" * 72)
        print(
            "Mejora relativa del retorno: "
            f"{fila['mejora_relativa_retorno']:.2%}"
        )
        print(
            "Mejora relativa del factor de beneficio: "
            f"{fila['mejora_relativa_factor']:.2%}"
        )
        print(
            "Mejora absoluta del drawdown: "
            f"{fila['mejora_drawdown_absoluta']:.2%}"
        )

    print("\nArchivos generados:")
    print(f"- {ruta_csv}")
    print(f"- {ruta_md}")


if __name__ == "__main__":
    main()
