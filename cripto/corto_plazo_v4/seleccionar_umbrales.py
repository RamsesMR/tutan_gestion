from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4.configuracion import (
    DETECTORES,
    PERFILES_UMBRALES,
    RUTA_HISTORIAL_DESARROLLO,
    RUTA_RESUMEN_DESARROLLO,
    RUTA_SELECCION_UMBRALES,
    SIMBOLOS,
)


def seleccionar_ultimos(
    datos: pd.DataFrame,
) -> pd.DataFrame:
    """Conserva el último experimento por combinación."""

    datos = datos.copy()

    datos["fecha_utc"] = pd.to_datetime(
        datos["fecha_utc"],
        utc=True,
        errors="coerce",
    )

    return (
        datos
        .sort_values(
            "fecha_utc"
        )
        .groupby(
            [
                "simbolo",
                "detector",
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


def cargar_tabla_fila(
    fila: pd.Series,
) -> pd.DataFrame:
    """Carga una tabla de umbrales y agrega identificadores."""

    ruta = Path(
        fila[
            "ruta_metricas_umbrales"
        ]
    )

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe: {ruta}"
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
        "detector"
    ] = fila[
        "detector"
    ]

    tabla[
        "variante"
    ] = fila[
        "variante"
    ]

    tabla[
        "pliegue"
    ] = fila[
        "pliegue"
    ]

    tabla[
        "variables"
    ] = fila[
        "variables"
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


def agregar_variante(
    tablas: pd.DataFrame,
    simbolo: str,
    detector: str,
    variante: str,
) -> pd.DataFrame:
    """Combina 2023 y 2024 en el mismo umbral."""

    subconjunto = tablas.loc[
        (tablas["simbolo"] == simbolo)
        & (tablas["detector"] == detector)
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
            f"{simbolo} {detector} {variante} "
            "no tiene exactamente dos pliegues."
        )

    primera = subconjunto.loc[
        subconjunto[
            "pliegue"
        ] == pliegues[
            0
        ]
    ].copy()

    segunda = subconjunto.loc[
        subconjunto[
            "pliegue"
        ] == pliegues[
            1
        ]
    ].copy()

    columnas_identidad = {
        "simbolo",
        "detector",
        "variante",
        "pliegue",
        "variables",
    }

    columnas_metricas = [
        columna
        for columna in primera.columns
        if columna not in columnas_identidad
        and columna != "umbral"
    ]

    primera = primera[
        [
            "umbral",
            *columnas_metricas,
        ]
    ].rename(
        columns={
            columna: f"{columna}_{pliegues[0]}"
            for columna in columnas_metricas
        }
    )

    segunda = segunda[
        [
            "umbral",
            *columnas_metricas,
        ]
    ].rename(
        columns={
            columna: f"{columna}_{pliegues[1]}"
            for columna in columnas_metricas
        }
    )

    combinado = primera.merge(
        segunda,
        on="umbral",
        how="inner",
        validate="one_to_one",
    )

    combinado[
        "simbolo"
    ] = simbolo

    combinado[
        "detector"
    ] = detector

    combinado[
        "variante"
    ] = variante

    combinado[
        "variables"
    ] = int(
        subconjunto[
            "variables"
        ].iloc[
            0
        ]
    )

    for metrica in (
        "precision",
        "recall",
        "f1",
        "precision_lift",
        "operaciones_no_solapadas",
        "precision_objetivo_no_solapada",
        "tasa_retorno_positivo_no_solapada",
        "retorno_neto_medio_no_solapado",
        "factor_beneficio_no_solapado",
        "maximo_drawdown_no_solapado",
        "pr_auc",
        "pr_auc_lift",
    ):
        columna_a = f"{metrica}_{pliegues[0]}"
        columna_b = f"{metrica}_{pliegues[1]}"

        combinado[
            f"{metrica}_minimo"
        ] = combinado[
            [
                columna_a,
                columna_b,
            ]
        ].min(
            axis=1
        )

        combinado[
            f"{metrica}_medio"
        ] = combinado[
            [
                columna_a,
                columna_b,
            ]
        ].mean(
            axis=1
        )

        combinado[
            f"{metrica}_diferencia"
        ] = (
            combinado[
                columna_a
            ]
            - combinado[
                columna_b
            ]
        ).abs()

    return combinado


def elegir_por_perfil(
    candidatos: pd.DataFrame,
    perfil: str,
    operaciones_minimas: int,
    recall_minimo_precision: float,
) -> pd.Series:
    """Elige un candidato y marca si cumple los filtros."""

    candidatos = candidatos.copy()

    filtro_comun = (
        candidatos[
            "operaciones_no_solapadas_minimo"
        ]
        >= operaciones_minimas
    )

    if perfil == "equilibrado":
        candidatos[
            "puntuacion_perfil"
        ] = (
            candidatos[
                "f1_minimo"
            ]
            + 0.20
            * (
                candidatos[
                    "precision_lift_minimo"
                ]
                - 1.0
            )
            + 0.10
            * (
                candidatos[
                    "pr_auc_lift_minimo"
                ]
                - 1.0
            )
            - 0.15
            * candidatos[
                "f1_diferencia"
            ]
        )

        mascara_elegible = (
            filtro_comun
            & (
                candidatos[
                    "precision_lift_minimo"
                ]
                > 1.0
            )
        )

    elif perfil == "precision":
        candidatos[
            "puntuacion_perfil"
        ] = (
            candidatos[
                "precision_minimo"
            ]
            + 0.10
            * candidatos[
                "recall_minimo"
            ]
            + 0.05
            * (
                candidatos[
                    "pr_auc_lift_minimo"
                ]
                - 1.0
            )
        )

        mascara_elegible = (
            filtro_comun
            & (
                candidatos[
                    "recall_minimo"
                ]
                >= recall_minimo_precision
            )
            & (
                candidatos[
                    "precision_lift_minimo"
                ]
                > 1.0
            )
        )

    elif perfil == "operativo":
        candidatos[
            "puntuacion_perfil"
        ] = (
            candidatos[
                "retorno_neto_medio_no_solapado_minimo"
            ]
            + 0.20
            * candidatos[
                "precision_objetivo_no_solapada_minimo"
            ]
            + 0.05
            * candidatos[
                "tasa_retorno_positivo_no_solapada_minimo"
            ]
            + 0.02
            * candidatos[
                "maximo_drawdown_no_solapado_minimo"
            ]
        )

        mascara_elegible = (
            filtro_comun
            & (
                candidatos[
                    "precision_lift_minimo"
                ]
                > 1.0
            )
            & (
                candidatos[
                    "retorno_neto_medio_no_solapado_minimo"
                ]
                > 0.0
            )
            & (
                candidatos[
                    "factor_beneficio_no_solapado_minimo"
                ]
                > 1.0
            )
        )

    else:
        raise ValueError(
            f"Perfil desconocido: {perfil}"
        )

    elegibles = candidatos.loc[
        mascara_elegible
    ].copy()

    cumple_filtros = not elegibles.empty

    fuente = (
        elegibles
        if cumple_filtros
        else candidatos
    )

    seleccionado = fuente.sort_values(
        [
            "puntuacion_perfil",
            "variables",
        ],
        ascending=[
            False,
            True,
        ],
    ).iloc[
        0
    ].copy()

    seleccionado[
        "perfil"
    ] = perfil

    seleccionado[
        "cumple_filtros"
    ] = bool(
        cumple_filtros
    )

    return seleccionado

def ejecutar(
    operaciones_minimas: int,
    recall_minimo_precision: float,
) -> None:
    """Construye comparaciones y selecciona umbrales."""

    if not RUTA_HISTORIAL_DESARROLLO.exists():
        raise FileNotFoundError(
            f"No existe: {RUTA_HISTORIAL_DESARROLLO}"
        )

    historial = pd.read_csv(
        RUTA_HISTORIAL_DESARROLLO
    )

    historial = seleccionar_ultimos(
        historial
    )

    tablas = pd.concat(
        [
            cargar_tabla_fila(
                fila
            )
            for _, fila in historial.iterrows()
        ],
        ignore_index=True,
    )

    RUTA_RESUMEN_DESARROLLO.mkdir(
        parents=True,
        exist_ok=True,
    )

    selecciones: list[
        pd.Series
    ] = []

    lineas = [
        "# Selección de umbrales V4",
        "",
        (
            "Los umbrales se seleccionan usando únicamente las "
            "validaciones de 2023 y 2024."
        ),
        "",
    ]

    for simbolo in SIMBOLOS:
        for detector in DETECTORES:
            variantes = sorted(
                tablas.loc[
                    (tablas["simbolo"] == simbolo)
                    & (tablas["detector"] == detector),
                    "variante",
                ].unique()
            )

            comparaciones = pd.concat(
                [
                    agregar_variante(
                        tablas=tablas,
                        simbolo=simbolo,
                        detector=detector,
                        variante=variante,
                    )
                    for variante in variantes
                ],
                ignore_index=True,
            )

            ruta_comparacion = (
                RUTA_RESUMEN_DESARROLLO
                / f"comparacion_umbrales_{simbolo}_{detector}.csv"
            )

            comparaciones.to_csv(
                ruta_comparacion,
                index=False,
                encoding="utf-8-sig",
            )

            lineas.extend(
                [
                    f"## {simbolo} — {detector}",
                    "",
                ]
            )

            for perfil in PERFILES_UMBRALES:
                seleccionado = elegir_por_perfil(
                    candidatos=comparaciones,
                    perfil=perfil,
                    operaciones_minimas=operaciones_minimas,
                    recall_minimo_precision=recall_minimo_precision,
                )

                selecciones.append(
                    seleccionado
                )

                lineas.extend(
                    [
                        f"### {perfil}",
                        "",
                        (
                            f"- Variante: "
                            f"`{seleccionado['variante']}`"
                        ),
                        (
                            f"- Umbral: "
                            f"{seleccionado['umbral']:.2f}"
                        ),
                        (
                            f"- Precision mínima: "
                            f"{seleccionado['precision_minimo']:.4f}"
                        ),
                        (
                            f"- Recall mínimo: "
                            f"{seleccionado['recall_minimo']:.4f}"
                        ),
                        (
                            f"- F1 mínimo: "
                            f"{seleccionado['f1_minimo']:.4f}"
                        ),
                        (
                            f"- PR-AUC lift mínimo: "
                            f"{seleccionado['pr_auc_lift_minimo']:.4f}"
                        ),
                        (
                            f"- Operaciones no solapadas mínimas: "
                            f"{int(seleccionado['operaciones_no_solapadas_minimo'])}"
                        ),
                        (
                            f"- Cumple filtros: "
                            f"{bool(seleccionado['cumple_filtros'])}"
                        ),
                        "",
                    ]
                )

                print(
                    f"{simbolo} {detector} {perfil}: "
                    f"{seleccionado['variante']} "
                    f"@ {seleccionado['umbral']:.2f} "
                    f"| cumple={bool(seleccionado['cumple_filtros'])}"
                )

            print(
                f"- {ruta_comparacion}"
            )

    seleccion = pd.DataFrame(
        selecciones
    )

    seleccion.to_csv(
        RUTA_SELECCION_UMBRALES,
        index=False,
        encoding="utf-8-sig",
    )

    ruta_markdown = (
        RUTA_RESUMEN_DESARROLLO
        / "resumen_seleccion_umbrales_v4.md"
    )

    ruta_markdown.write_text(
        "\n".join(
            lineas
        ),
        encoding="utf-8",
    )

    print(
        "\nSelección completa:"
    )
    print(
        f"- {RUTA_SELECCION_UMBRALES}"
    )
    print(
        f"- {ruta_markdown}"
    )


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene argumentos."""

    parser = argparse.ArgumentParser(
        description=(
            "Selecciona variantes y umbrales estables entre 2023 y 2024."
        )
    )

    parser.add_argument(
        "--operaciones-minimas",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--recall-minimo-precision",
        type=float,
        default=0.03,
    )

    argumentos = parser.parse_args()

    if argumentos.operaciones_minimas <= 0:
        parser.error(
            "--operaciones-minimas debe ser mayor que cero."
        )

    if not 0 <= argumentos.recall_minimo_precision <= 1:
        parser.error(
            "--recall-minimo-precision debe estar entre 0 y 1."
        )

    return argumentos


def main() -> None:
    """Punto de entrada."""

    argumentos = obtener_argumentos()

    try:
        ejecutar(
            operaciones_minimas=argumentos.operaciones_minimas,
            recall_minimo_precision=argumentos.recall_minimo_precision,
        )

    except (
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(
            "\nNo se pudieron seleccionar los umbrales V4."
        )
        print(
            f"Detalle: {error}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
