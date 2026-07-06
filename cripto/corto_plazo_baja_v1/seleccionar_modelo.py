from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from cripto.corto_plazo_baja_v1.configuracion import (
    DRAWDOWN_MINIMO_ADMITIDO,
    FACTOR_BENEFICIO_MINIMO,
    OBJETIVO_BASE,
    OPERACIONES_MINIMAS_POR_PLIEGUE,
    PORCENTAJE_POSITIVAS_MINIMO,
    RUTA_HISTORIAL,
    RUTA_SELECCION,
)


PLIEGUES_REQUERIDOS = {
    "validacion_2023",
    "validacion_2024",
}


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
    if not RUTA_HISTORIAL.exists():
        raise FileNotFoundError(
            f"No existe: {RUTA_HISTORIAL}"
        )

    historial = pd.read_csv(
        RUTA_HISTORIAL
    )

    registros = []

    claves = [
        "objetivo",
        "variante",
        "umbral",
    ]

    for (
        objetivo,
        variante,
        umbral,
    ), grupo in historial.groupby(
        claves,
        sort=False,
    ):
        pliegues = set(
            grupo[
                "pliegue"
            ].astype(str)
        )

        if pliegues != PLIEGUES_REQUERIDOS:
            continue

        if len(grupo) != 2:
            continue

        operaciones_minimas = int(
            grupo[
                "operaciones"
            ].min()
        )

        retorno_minimo = float(
            grupo[
                "retorno_neto_medio"
            ].min()
        )

        retorno_mediano_minimo = float(
            grupo[
                "retorno_neto_mediano"
            ].min()
        )

        factor_minimo = float(
            grupo[
                "factor_beneficio"
            ].min()
        )

        drawdown_peor = float(
            grupo[
                "maximo_drawdown"
            ].min()
        )

        positivas_minimas = float(
            grupo[
                "porcentaje_operaciones_positivas"
            ].min()
        )

        precision_minima = float(
            grupo[
                "precision_operaciones"
            ].min()
        )

        pr_auc_minimo = float(
            grupo[
                "pr_auc"
            ].min()
        )

        lift_minimo = float(
            grupo[
                "pr_auc_lift"
            ].min()
        )

        concentracion_maxima = float(
            grupo[
                "concentracion_mensual_maxima"
            ].max()
        )

        aprobado = bool(
            operaciones_minimas
            >= OPERACIONES_MINIMAS_POR_PLIEGUE
            and retorno_minimo > 0.0
            and retorno_mediano_minimo
            >= 0.0
            and factor_minimo
            >= FACTOR_BENEFICIO_MINIMO
            and drawdown_peor
            >= DRAWDOWN_MINIMO_ADMITIDO
            and positivas_minimas
            >= PORCENTAJE_POSITIVAS_MINIMO
            and concentracion_maxima
            <= 0.35
        )

        registros.append(
            {
                "objetivo": objetivo,
                "variante": variante,
                "umbral": float(
                    umbral
                ),
                "operaciones_minimas": operaciones_minimas,
                "retorno_minimo": retorno_minimo,
                "retorno_mediano_minimo": retorno_mediano_minimo,
                "factor_minimo": factor_minimo,
                "drawdown_peor": drawdown_peor,
                "positivas_minimas": positivas_minimas,
                "precision_operaciones_minima": precision_minima,
                "pr_auc_minimo": pr_auc_minimo,
                "pr_auc_lift_minimo": lift_minimo,
                "concentracion_mensual_maxima": concentracion_maxima,
                "aprobado": aprobado,
            }
        )

    tabla = pd.DataFrame(
        registros
    )

    if tabla.empty:
        raise ValueError(
            "No hay configuraciones completas para seleccionar."
        )

    tabla = tabla.sort_values(
        [
            "aprobado",
            "retorno_minimo",
            "factor_minimo",
            "drawdown_peor",
            "operaciones_minimas",
            "pr_auc_lift_minimo",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    aprobadas = tabla.loc[
        tabla[
            "aprobado"
        ].astype(bool)
    ].copy()

    if aprobadas.empty:
        decision = {
            "decision": (
                "ninguna_configuracion_baja_aprobada"
            ),
            "candidata_principal": None,
            "siguiente_paso": (
                "Revisar resultados por objetivo antes de añadir "
                "nuevas variables. No usar 2025."
            ),
        }
    else:
        mejor = aprobadas.iloc[
            0
        ]

        mejor_objetivo_base = (
            aprobadas.loc[
                aprobadas[
                    "objetivo"
                ]
                == OBJETIVO_BASE
            ]
            .head(1)
        )

        decision = {
            "decision": (
                "candidata_baja_v1_seleccionada_"
                "provisionalmente"
            ),
            "candidata_principal": {
                clave: valor_json(
                    mejor[clave]
                )
                for clave in tabla.columns
            },
            "mejor_objetivo_base": (
                None
                if mejor_objetivo_base.empty
                else {
                    clave: valor_json(
                        mejor_objetivo_base.iloc[
                            0
                        ][clave]
                    )
                    for clave in tabla.columns
                }
            ),
            "siguiente_paso": (
                "Revisar matrices, operaciones y estabilidad por "
                "umbral. Después ejecutar robustez por semillas "
                "antes de usar 2025."
            ),
        }

    decision.update(
        {
            "fecha_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "uso_2025_para_seleccion": False,
            "uso_2026_para_seleccion": False,
            "promocion_automatica": False,
        }
    )

    RUTA_SELECCION.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_csv = (
        RUTA_SELECCION
        / "seleccion_baja_v1.csv"
    )

    tabla.to_csv(
        ruta_csv,
        index=False,
        encoding="utf-8-sig",
    )

    ruta_json = (
        RUTA_SELECCION
        / "decision_baja_v1.json"
    )

    ruta_json.write_text(
        json.dumps(
            decision,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    lineas = [
        "# Selección BAJA V1",
        "",
        "| Objetivo | Variante | Umbral | Ops mín. | "
        "Retorno mín. | PF mín. | DD peor | Positivas mín. | Aprobado |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, fila in tabla.head(
        30
    ).iterrows():
        lineas.append(
            "| "
            f"{fila['objetivo']} | "
            f"{fila['variante']} | "
            f"{float(fila['umbral']):.2f} | "
            f"{int(fila['operaciones_minimas'])} | "
            f"{float(fila['retorno_minimo']):.4%} | "
            f"{float(fila['factor_minimo']):.4f} | "
            f"{float(fila['drawdown_peor']):.2%} | "
            f"{float(fila['positivas_minimas']):.2f} % | "
            f"{bool(fila['aprobado'])} |"
        )

    lineas.extend(
        [
            "",
            "## Decisión",
            "",
            f"`{decision['decision']}`",
            "",
            "No se utilizaron 2025 ni 2026 y no se promovió "
            "automáticamente ningún modelo.",
            "",
        ]
    )

    ruta_md = (
        RUTA_SELECCION
        / "informe_seleccion_baja_v1.md"
    )

    ruta_md.write_text(
        "\n".join(
            lineas
        ),
        encoding="utf-8",
    )

    print(
        "\nSELECCIÓN BAJA V1 COMPLETADA"
    )
    print("=" * 72)
    print(
        tabla.head(
            20
        ).to_string(
            index=False
        )
    )
    print(
        f"\nDECISIÓN: {decision['decision']}"
    )
    print(
        f"- {ruta_csv}"
    )
    print(
        f"- {ruta_json}"
    )
    print(
        f"- {ruta_md}"
    )


if __name__ == "__main__":
    main()
