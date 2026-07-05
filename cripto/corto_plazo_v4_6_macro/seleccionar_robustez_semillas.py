from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_6_macro.configuracion_robustez import (
    COMPARACIONES_ROBUSTEZ,
    CONCENTRACION_MENSUAL_MAXIMA,
    EMPEORAMIENTO_MAXIMO_DRAWDOWN,
    OPERACIONES_MINIMAS,
    RUTA_RESULTADOS_ROBUSTEZ,
    RUTA_SELECCION_ROBUSTEZ,
    SEMILLAS_MINIMAS_APROBADAS,
    SEMILLAS_ROBUSTEZ,
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
    ruta_resumen = (
        RUTA_RESULTADOS_ROBUSTEZ
        / "resumen_pareado_por_semilla.csv"
    )

    if not ruta_resumen.exists():
        raise FileNotFoundError(
            f"No existe: {ruta_resumen}"
        )

    resumen = pd.read_csv(
        ruta_resumen
    )

    esperadas = set(
        COMPARACIONES_ROBUSTEZ
    )

    presentes = set(
        resumen[
            "comparacion"
        ].unique()
    )

    faltantes = esperadas.difference(
        presentes
    )

    if faltantes:
        raise ValueError(
            "Faltan comparaciones: "
            + ", ".join(
                sorted(faltantes)
            )
        )

    registros = []

    for nombre_comparacion in COMPARACIONES_ROBUSTEZ:
        grupo = resumen.loc[
            resumen[
                "comparacion"
            ]
            == nombre_comparacion
        ].copy()

        if set(
            grupo["semilla"].astype(int)
        ) != set(
            SEMILLAS_ROBUSTEZ
        ):
            raise ValueError(
                f"{nombre_comparacion} no contiene "
                "todas las semillas configuradas."
            )

        semillas_aprobadas = int(
            grupo[
                "aprueba_semilla"
            ].astype(bool).sum()
        )

        signos_estables = int(
            grupo[
                "signo_estable"
            ].astype(bool).sum()
        )

        mejora_retorno_mediana = float(
            grupo[
                "mejora_retorno_media"
            ].median()
        )

        mejora_factor_mediana = float(
            grupo[
                "mejora_factor_media"
            ].median()
        )

        mejora_drawdown_mediana = float(
            grupo[
                "mejora_drawdown_peor"
            ].median()
        )

        operaciones_mediana = float(
            grupo[
                "operaciones_minimas_candidata"
            ].median()
        )

        concentracion_mediana = float(
            grupo[
                "concentracion_mensual_maxima"
            ].median()
        )

        retencion_retorno_mediana = float(
            grupo[
                "retencion_retorno_peor_ano"
            ].median()
        )

        retencion_factor_mediana = float(
            grupo[
                "retencion_factor_peor_ano"
            ].median()
        )

        robusta = bool(
            semillas_aprobadas
            >= SEMILLAS_MINIMAS_APROBADAS
            and signos_estables
            >= SEMILLAS_MINIMAS_APROBADAS
            and mejora_retorno_mediana > 0.0
            and mejora_factor_mediana > 0.0
            and mejora_drawdown_mediana
            >= EMPEORAMIENTO_MAXIMO_DRAWDOWN
            and operaciones_mediana
            >= OPERACIONES_MINIMAS
            and concentracion_mediana
            <= CONCENTRACION_MENSUAL_MAXIMA
        )

        configuracion = COMPARACIONES_ROBUSTEZ[
            nombre_comparacion
        ]

        registros.append(
            {
                "comparacion": nombre_comparacion,
                "control": configuracion[
                    "control"
                ],
                "candidata": configuracion[
                    "candidata"
                ],
                "umbral": configuracion[
                    "umbral"
                ],
                "variable_macro": configuracion[
                    "variable_macro"
                ],
                "semillas_totales": len(
                    SEMILLAS_ROBUSTEZ
                ),
                "semillas_aprobadas": semillas_aprobadas,
                "signos_estables": signos_estables,
                "mejora_retorno_mediana": (
                    mejora_retorno_mediana
                ),
                "mejora_factor_mediana": (
                    mejora_factor_mediana
                ),
                "mejora_drawdown_mediana": (
                    mejora_drawdown_mediana
                ),
                "retencion_retorno_mediana_peor_ano": (
                    retencion_retorno_mediana
                ),
                "retencion_factor_mediana_peor_ano": (
                    retencion_factor_mediana
                ),
                "operaciones_minimas_mediana": (
                    operaciones_mediana
                ),
                "concentracion_mensual_mediana": (
                    concentracion_mediana
                ),
                "robusta": robusta,
            }
        )

    tabla = pd.DataFrame(
        registros
    )

    robustas = tabla.loc[
        tabla[
            "robusta"
        ].astype(bool)
    ].copy()

    if robustas.empty:
        decision = {
            "decision": (
                "ninguna_candidata_macro_supera_"
                "la_prueba_de_semillas"
            ),
            "candidatas_robustas": [],
            "siguiente_paso": (
                "Mantener V4.1 y V4.5-Spot sin macro."
            ),
        }
    else:
        robustas = robustas.sort_values(
            [
                "semillas_aprobadas",
                "mejora_retorno_mediana",
                "mejora_factor_mediana",
                "mejora_drawdown_mediana",
                "operaciones_minimas_mediana",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
            ],
        )

        mejor = robustas.iloc[0]

        decision = {
            "decision": (
                "candidatas_macro_robustas_"
                "seleccionadas_provisionalmente"
            ),
            "candidata_principal": {
                clave: valor_json(
                    mejor[clave]
                )
                for clave in tabla.columns
            },
            "candidatas_robustas": [
                {
                    clave: valor_json(
                        fila[clave]
                    )
                    for clave in tabla.columns
                }
                for _, fila
                in robustas.iterrows()
            ],
            "siguiente_paso": (
                "Confirmar la candidata principal con "
                "datos vintage de ALFRED antes de usar "
                "2025 o 2026."
            ),
        }

    decision.update(
        {
            "semillas": list(
                SEMILLAS_ROBUSTEZ
            ),
            "uso_2025_para_seleccion": False,
            "uso_2026_para_seleccion": False,
            "promocion_automatica": False,
        }
    )

    RUTA_SELECCION_ROBUSTEZ.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_csv = (
        RUTA_SELECCION_ROBUSTEZ
        / "resumen_final_robustez.csv"
    )

    tabla.to_csv(
        ruta_csv,
        index=False,
        encoding="utf-8-sig",
    )

    ruta_json = (
        RUTA_SELECCION_ROBUSTEZ
        / "decision_robustez_semillas.json"
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
        "# Robustez por semillas — V4.6-Macro",
        "",
        "| Comparación | Semillas aprobadas | "
        "Mejora mediana retorno | Mejora mediana factor | "
        "Mejora mediana drawdown | Robusta |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, fila in tabla.iterrows():
        lineas.append(
            "| "
            f"{fila['comparacion']} | "
            f"{int(fila['semillas_aprobadas'])}/"
            f"{int(fila['semillas_totales'])} | "
            f"{float(fila['mejora_retorno_mediana']):.2%} | "
            f"{float(fila['mejora_factor_mediana']):.2%} | "
            f"{float(fila['mejora_drawdown_mediana']):.2%} | "
            f"{bool(fila['robusta'])} |"
        )

    lineas.extend(
        [
            "",
            "## Decisión",
            "",
            f"`{decision['decision']}`",
            "",
            "No se utilizaron 2025 ni 2026 y no se "
            "promovió automáticamente ningún modelo.",
            "",
        ]
    )

    ruta_md = (
        RUTA_SELECCION_ROBUSTEZ
        / "informe_robustez_semillas.md"
    )

    ruta_md.write_text(
        "\n".join(
            lineas
        ),
        encoding="utf-8",
    )

    print(
        "\nSELECCIÓN DE ROBUSTEZ COMPLETADA"
    )
    print("=" * 72)
    print(
        tabla[
            [
                "comparacion",
                "semillas_aprobadas",
                "mejora_retorno_mediana",
                "mejora_factor_mediana",
                "mejora_drawdown_mediana",
                "robusta",
            ]
        ].to_string(
            index=False
        )
    )
    print(
        f"\nDecisión: {decision['decision']}"
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
