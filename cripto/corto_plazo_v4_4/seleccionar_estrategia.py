from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_4.configuracion import (
    MEJORA_MINIMA_DRAWDOWN_ABSOLUTA,
    PLIEGUES,
    RETENCION_MINIMA_FACTOR_BENEFICIO,
    RETENCION_MINIMA_RETORNO,
    RUTA_LABORATORIO,
    RUTA_SELECCION,
    VERSION_MODELO,
)
from cripto.corto_plazo_v4_4.utilidades import (
    convertir_para_json,
)


def main() -> None:
    """Selecciona V4.4 únicamente si mejora riesgo sin destruir ventaja."""

    ruta_resultados = (
        RUTA_LABORATORIO
        / "resultados_laboratorio.csv"
    )

    if not ruta_resultados.exists():
        raise FileNotFoundError(
            f"No existe: {ruta_resultados}"
        )

    resultados = pd.read_csv(
        ruta_resultados
    )

    primarios = resultados.loc[
        resultados[
            "modo_evaluacion"
        ]
        == "cohorte_v4_1"
    ].copy()

    controles = primarios.loc[
        primarios[
            "es_control_v4_1"
        ].astype(bool)
    ].set_index(
        "pliegue"
    )

    if set(
        controles.index
    ) != set(
        PLIEGUES
    ):
        raise ValueError(
            "No existe un control V4.1 completo para ambos pliegues."
        )

    candidatos = primarios.loc[
        ~primarios[
            "es_control_v4_1"
        ].astype(bool)
    ].copy()

    evaluaciones = []

    for estrategia_id, grupo in candidatos.groupby(
        "estrategia_id",
        sort=True,
    ):
        grupo = grupo.set_index(
            "pliegue"
        )

        if set(
            grupo.index
        ) != set(
            PLIEGUES
        ):
            continue

        cumple_todos = True
        mejoras_drawdown = []
        retenciones_retorno = []
        retenciones_factor = []

        for pliegue in PLIEGUES:
            candidato = grupo.loc[
                pliegue
            ]

            control = controles.loc[
                pliegue
            ]

            retencion_retorno = (
                float(
                    candidato[
                        "retorno_neto_medio"
                    ]
                )
                / float(
                    control[
                        "retorno_neto_medio"
                    ]
                )
                if float(
                    control[
                        "retorno_neto_medio"
                    ]
                )
                > 0
                else 0.0
            )

            retencion_factor = (
                float(
                    candidato[
                        "factor_beneficio"
                    ]
                )
                / float(
                    control[
                        "factor_beneficio"
                    ]
                )
                if float(
                    control[
                        "factor_beneficio"
                    ]
                )
                > 0
                else 0.0
            )

            mejora_drawdown = (
                float(
                    candidato[
                        "maximo_drawdown"
                    ]
                )
                - float(
                    control[
                        "maximo_drawdown"
                    ]
                )
            )

            cumple_pliegue = (
                retencion_retorno
                >= RETENCION_MINIMA_RETORNO
                and retencion_factor
                >= RETENCION_MINIMA_FACTOR_BENEFICIO
                and float(
                    candidato[
                        "factor_beneficio"
                    ]
                )
                > 1.0
                and float(
                    candidato[
                        "retorno_neto_mediano"
                    ]
                )
                >= 0.0
                and mejora_drawdown
                >= 0.0
                and int(
                    candidato[
                        "operaciones"
                    ]
                )
                == int(
                    control[
                        "operaciones"
                    ]
                )
            )

            cumple_todos = (
                cumple_todos
                and cumple_pliegue
            )

            mejoras_drawdown.append(
                mejora_drawdown
            )

            retenciones_retorno.append(
                retencion_retorno
            )

            retenciones_factor.append(
                retencion_factor
            )

        mejora_minima_drawdown = min(
            mejoras_drawdown
        )

        cumple_mejora_minima = (
            max(
                mejoras_drawdown
            )
            >= MEJORA_MINIMA_DRAWDOWN_ABSOLUTA
            and mejora_minima_drawdown
            >= 0.0
        )

        primera = grupo.iloc[
            0
        ]

        evaluaciones.append(
            {
                "estrategia_id": estrategia_id,
                "umbral_baja": primera[
                    "umbral_baja"
                ],
                "minutos_minimos": int(
                    primera[
                        "minutos_minimos"
                    ]
                ),
                "condicion_salida": primera[
                    "condicion_salida"
                ],
                "cumple_filtros": (
                    cumple_todos
                    and cumple_mejora_minima
                ),
                "retencion_minima_retorno": min(
                    retenciones_retorno
                ),
                "retencion_minima_factor_beneficio": min(
                    retenciones_factor
                ),
                "mejora_minima_drawdown": mejora_minima_drawdown,
                "mejora_media_drawdown": float(
                    np.mean(
                        mejoras_drawdown
                    )
                ),
                "retorno_neto_medio_minimo": float(
                    grupo[
                        "retorno_neto_medio"
                    ].min()
                ),
                "factor_beneficio_minimo": float(
                    grupo[
                        "factor_beneficio"
                    ].min()
                ),
                "drawdown_peor": float(
                    grupo[
                        "maximo_drawdown"
                    ].min()
                ),
            }
        )

    resumen = pd.DataFrame(
        evaluaciones
    )

    RUTA_SELECCION.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_resumen = (
        RUTA_SELECCION
        / "resumen_candidatos.csv"
    )

    resumen.to_csv(
        ruta_resumen,
        index=False,
        encoding="utf-8-sig",
    )

    validos = resumen.loc[
        resumen[
            "cumple_filtros"
        ].astype(bool)
    ].copy()

    if validos.empty:
        decision = {
            "version": VERSION_MODELO,
            "decision": "mantener_v4_1",
            "motivo": (
                "Ninguna salida BAJA mejoró el drawdown en desarrollo "
                "manteniendo al menos el 90 % del retorno y el 95 % "
                "del factor de beneficio de V4.1 en ambos pliegues."
            ),
            "uso_2025": False,
            "uso_2026": False,
        }

        seleccion = pd.DataFrame()
    else:
        validos = validos.sort_values(
            [
                "mejora_minima_drawdown",
                "retencion_minima_retorno",
                "factor_beneficio_minimo",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )

        mejor = validos.iloc[
            0
        ]

        decision = {
            "version": VERSION_MODELO,
            "decision": "candidata_v4_4_seleccionada",
            "estrategia_id": mejor[
                "estrategia_id"
            ],
            "umbral_baja": convertir_para_json(
                mejor[
                    "umbral_baja"
                ]
            ),
            "minutos_minimos": int(
                mejor[
                    "minutos_minimos"
                ]
            ),
            "condicion_salida": mejor[
                "condicion_salida"
            ],
            "uso_2025": False,
            "uso_2026": False,
        }

        seleccion = primarios.loc[
            primarios[
                "estrategia_id"
            ]
            == mejor[
                "estrategia_id"
            ]
        ].copy()

    ruta_decision = (
        RUTA_SELECCION
        / "decision_v4_4.json"
    )

    ruta_decision.write_text(
        json.dumps(
            decision,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    if not seleccion.empty:
        seleccion.to_csv(
            RUTA_SELECCION
            / "estrategia_seleccionada.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(
        "\nSELECCIÓN V4.4"
    )
    print("=" * 72)
    print(
        f"Decisión: {decision['decision']}"
    )
    print(
        f"- {ruta_resumen}"
    )
    print(
        f"- {ruta_decision}"
    )
    print(
        "\n2025 y 2026 no fueron utilizados."
    )


if __name__ == "__main__":
    main()
