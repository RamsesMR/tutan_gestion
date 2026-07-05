from __future__ import annotations

import json

import pandas as pd

from cripto.corto_plazo_v4_5.configuracion import (
    RUTA_COMPARACION,
    RUTA_RESULTADOS,
    RUTA_SELECCION,
)


def porcentaje_decimal(
    valor: float,
) -> str:
    return f"{valor * 100:.4f} %"


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

    congelados = resultados.loc[
        resultados["umbral"] == 0.44
    ].copy()

    lineas = [
        "# Comparación V4.1 frente a V4.5",
        "",
        "- Periodos de selección: 2023 y 2024.",
        "- Umbral congelado principal: 0,44.",
        "- Entrada: cruce desde abajo.",
        "- Salida: 480 minutos.",
        "- Coste: 0,10 %.",
        "- 2025 utilizado para selección: no.",
        "- 2026 utilizado para selección: no.",
        "",
        "## Resultados con estrategia congelada",
        "",
        "| Variante | Pliegue | Operaciones | Precisión | Positivas | Retorno neto medio | Factor beneficio | Drawdown |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, fila in congelados.sort_values(
        [
            "variante",
            "pliegue",
        ]
    ).iterrows():
        lineas.append(
            "| "
            f"{fila['variante']} | "
            f"{fila['pliegue']} | "
            f"{int(fila['operaciones'])} | "
            f"{float(fila['porcentaje_acierto']):.2f} % | "
            f"{float(fila['porcentaje_operaciones_positivas']):.2f} % | "
            f"{porcentaje_decimal(float(fila['retorno_neto_medio']))} | "
            f"{float(fila['factor_beneficio']):.4f} | "
            f"{porcentaje_decimal(float(fila['maximo_drawdown']))} |"
        )

    lineas.extend(
        [
            "",
            "## Decisión automática",
            "",
            f"**{decision['decision']}**",
            "",
        ]
    )

    if "motivo" in decision:
        lineas.append(
            decision["motivo"]
        )
        lineas.append("")

    RUTA_COMPARACION.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_salida = (
        RUTA_COMPARACION
        / "comparacion_v4_1_vs_v4_5.md"
    )

    ruta_salida.write_text(
        "\n".join(lineas),
        encoding="utf-8",
    )

    print("\nCOMPARACIÓN V4.5 GENERADA")
    print("=" * 72)
    print(f"- {ruta_salida}")


if __name__ == "__main__":
    main()
