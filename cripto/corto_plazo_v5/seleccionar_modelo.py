from __future__ import annotations

from pathlib import Path

import pandas as pd

from cripto.corto_plazo_v5.configuracion import (
    COSTE_BASE,
    OPERACIONES_MINIMAS_POR_PLIEGUE,
    RUTA_HISTORIAL_V5,
    RUTA_RESUMEN_V5,
    RUTA_SELECCION_V5,
)


def ultimas_ejecuciones(
    historial: pd.DataFrame,
) -> pd.DataFrame:
    """Conserva el experimento más reciente por combinación."""

    historial = historial.copy()

    historial[
        "fecha_utc"
    ] = pd.to_datetime(
        historial[
            "fecha_utc"
        ],
        utc=True,
        errors="coerce",
    )

    return (
        historial
        .sort_values(
            "fecha_utc"
        )
        .groupby(
            [
                "simbolo",
                "objetivo",
                "variante",
                "pliegue",
            ],
            as_index=False,
        )
        .tail(
            1
        )
        .reset_index(
            drop=True
        )
    )


def cargar_umbrales(
    fila: pd.Series,
) -> pd.DataFrame:
    """Carga la tabla de un experimento."""

    ruta = Path(
        fila[
            "ruta_umbrales"
        ]
    )

    tabla = pd.read_csv(
        ruta
    )

    tabla[
        "simbolo"
    ] = fila[
        "simbolo"
    ]

    tabla[
        "objetivo"
    ] = fila[
        "objetivo"
    ]

    tabla[
        "variante"
    ] = fila[
        "variante"
    ]

    tabla[
        "tipo_modelo"
    ] = fila[
        "tipo_modelo"
    ]

    tabla[
        "pliegue"
    ] = fila[
        "pliegue"
    ]

    tabla[
        "pr_auc"
    ] = fila[
        "pr_auc"
    ]

    tabla[
        "pr_auc_lift"
    ] = fila[
        "pr_auc_lift"
    ]

    return tabla


def combinar_pliegues(
    tablas: pd.DataFrame,
    simbolo: str,
    objetivo: str,
    variante: str,
) -> pd.DataFrame:
    """Combina 2023 y 2024 en el mismo umbral y coste."""

    subconjunto = tablas.loc[
        (tablas["simbolo"] == simbolo)
        & (tablas["objetivo"] == objetivo)
        & (tablas["variante"] == variante)
    ].copy()

    pliegues = sorted(
        subconjunto[
            "pliegue"
        ].unique()
    )

    if len(
        pliegues
    ) != 2:
        raise ValueError(
            f"{simbolo} {objetivo} {variante} no tiene dos pliegues."
        )

    claves = [
        "umbral",
        "coste",
    ]

    columnas_excluir = {
        *claves,
        "simbolo",
        "objetivo",
        "variante",
        "tipo_modelo",
        "pliegue",
    }

    metricas = [
        columna
        for columna in subconjunto.columns
        if columna not in columnas_excluir
    ]

    primera = subconjunto.loc[
        subconjunto[
            "pliegue"
        ] == pliegues[
            0
        ],
        [
            *claves,
            *metricas,
        ],
    ].rename(
        columns={
            columna: f"{columna}_{pliegues[0]}"
            for columna in metricas
        }
    )

    segunda = subconjunto.loc[
        subconjunto[
            "pliegue"
        ] == pliegues[
            1
        ],
        [
            *claves,
            *metricas,
        ],
    ].rename(
        columns={
            columna: f"{columna}_{pliegues[1]}"
            for columna in metricas
        }
    )

    combinado = primera.merge(
        segunda,
        on=claves,
        validate="one_to_one",
    )

    for metrica in metricas:
        columnas = [
            f"{metrica}_{pliegues[0]}",
            f"{metrica}_{pliegues[1]}",
        ]

        combinado[
            f"{metrica}_minimo"
        ] = combinado[
            columnas
        ].min(
            axis=1
        )

        combinado[
            f"{metrica}_medio"
        ] = combinado[
            columnas
        ].mean(
            axis=1
        )

        combinado[
            f"{metrica}_diferencia"
        ] = (
            combinado[
                columnas[
                    0
                ]
            ]
            - combinado[
                columnas[
                    1
                ]
            ]
        ).abs()

    combinado[
        "simbolo"
    ] = simbolo

    combinado[
        "objetivo"
    ] = objetivo

    combinado[
        "variante"
    ] = variante

    combinado[
        "tipo_modelo"
    ] = subconjunto[
        "tipo_modelo"
    ].iloc[
        0
    ]

    return combinado


def main() -> None:
    """Selecciona una configuración robusta y rentable."""

    if not RUTA_HISTORIAL_V5.exists():
        raise FileNotFoundError(
            f"No existe: {RUTA_HISTORIAL_V5}"
        )

    historial = ultimas_ejecuciones(
        pd.read_csv(
            RUTA_HISTORIAL_V5
        )
    )

    tablas = pd.concat(
        [
            cargar_umbrales(
                fila
            )
            for _, fila in historial.iterrows()
        ],
        ignore_index=True,
    )

    combinaciones = []

    for (
        simbolo,
        objetivo,
        variante,
    ), _ in tablas.groupby(
        [
            "simbolo",
            "objetivo",
            "variante",
        ]
    ):
        combinaciones.append(
            combinar_pliegues(
                tablas=tablas,
                simbolo=simbolo,
                objetivo=objetivo,
                variante=variante,
            )
        )

    comparacion = pd.concat(
        combinaciones,
        ignore_index=True,
    )

    base = comparacion.loc[
        comparacion[
            "coste"
        ].round(
            6
        )
        == round(
            COSTE_BASE,
            6,
        )
    ].copy()

    robustez_coste_alto = comparacion.loc[
        comparacion[
            "coste"
        ].round(
            6
        )
        == 0.002
    ][
        [
            "simbolo",
            "objetivo",
            "variante",
            "umbral",
            "retorno_neto_medio_minimo",
            "factor_beneficio_minimo",
        ]
    ].rename(
        columns={
            "retorno_neto_medio_minimo": (
                "retorno_neto_medio_minimo_coste_020"
            ),
            "factor_beneficio_minimo": (
                "factor_beneficio_minimo_coste_020"
            ),
        }
    )

    base = base.merge(
        robustez_coste_alto,
        on=[
            "simbolo",
            "objetivo",
            "variante",
            "umbral",
        ],
        how="left",
        validate="one_to_one",
    )

    base[
        "cumple_minimo"
    ] = (
        (
            base[
                "operaciones_no_solapadas_minimo"
            ]
            >= OPERACIONES_MINIMAS_POR_PLIEGUE
        )
        & (
            base[
                "precision_lift_minimo"
            ]
            > 1.0
        )
        & (
            base[
                "retorno_neto_medio_minimo"
            ]
            > 0.0
        )
        & (
            base[
                "retorno_neto_mediano_minimo"
            ]
            > 0.0
        )
        & (
            base[
                "factor_beneficio_minimo"
            ]
            > 1.0
        )
        & (
            base[
                "pr_auc_lift_minimo"
            ]
            > 1.0
        )
    )

    base[
        "puntuacion"
    ] = (
        5.0
        * base[
            "retorno_neto_medio_minimo"
        ]
        + 1.0
        * base[
            "retorno_neto_mediano_minimo"
        ]
        + 0.08
        * (
            base[
                "factor_beneficio_minimo"
            ]
            - 1.0
        )
        + 0.03
        * (
            base[
                "precision_lift_minimo"
            ]
            - 1.0
        )
        + 0.02
        * (
            base[
                "pr_auc_lift_minimo"
            ]
            - 1.0
        )
        + 0.02
        * base[
            "maximo_drawdown_minimo"
        ]
        - 0.20
        * base[
            "retorno_neto_medio_diferencia"
        ]
    )

    elegibles = base.loc[
        base[
            "cumple_minimo"
        ]
    ].copy()

    fuente = (
        elegibles
        if not elegibles.empty
        else base
    )

    seleccion = (
        fuente
        .sort_values(
            [
                "puntuacion",
                "operaciones_no_solapadas_minimo",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .groupby(
            [
                "simbolo",
            ],
            as_index=False,
        )
        .head(
            1
        )
        .reset_index(
            drop=True
        )
    )

    RUTA_RESUMEN_V5.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparacion.to_csv(
        RUTA_RESUMEN_V5
        / "comparacion_completa_v5.csv",
        index=False,
        encoding="utf-8-sig",
    )

    base.sort_values(
        "puntuacion",
        ascending=False,
    ).to_csv(
        RUTA_RESUMEN_V5
        / "ranking_v5.csv",
        index=False,
        encoding="utf-8-sig",
    )

    seleccion.to_csv(
        RUTA_SELECCION_V5,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nSELECCIÓN V5"
    )
    print("=" * 72)

    for _, fila in seleccion.iterrows():
        print(
            f"{fila['simbolo']}: "
            f"{fila['objetivo']} | "
            f"{fila['variante']} | "
            f"umbral {fila['umbral']:.2f}"
        )
        print(
            f"  Cumple mínimo: {bool(fila['cumple_minimo'])}"
        )
        print(
            f"  Retorno neto mínimo: "
            f"{fila['retorno_neto_medio_minimo']:.4%}"
        )
        print(
            f"  Factor beneficio mínimo: "
            f"{fila['factor_beneficio_minimo']:.3f}"
        )
        print(
            f"  Operaciones mínimas: "
            f"{int(fila['operaciones_no_solapadas_minimo'])}"
        )

    print(
        f"\n- {RUTA_SELECCION_V5}"
    )


if __name__ == "__main__":
    main()
