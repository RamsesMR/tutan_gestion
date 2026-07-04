from __future__ import annotations

import pandas as pd

from cripto.corto_plazo_v4.configuracion import (
    DETECTORES,
    RUTA_HISTORIAL_DESARROLLO,
    RUTA_RESUMEN_DESARROLLO,
    SIMBOLOS,
)


def seleccionar_ultimas_ejecuciones(
    datos: pd.DataFrame,
) -> pd.DataFrame:
    """Conserva el resultado más reciente de cada combinación."""

    datos = datos.copy()

    datos["fecha_utc"] = pd.to_datetime(
        datos["fecha_utc"],
        utc=True,
        errors="coerce",
    )

    if datos["fecha_utc"].isna().any():
        raise ValueError(
            "El historial contiene fechas inválidas."
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


def crear_ranking(
    datos: pd.DataFrame,
    simbolo: str,
    detector: str,
) -> pd.DataFrame:
    """Agrega 2023 y 2024 por variante."""

    subconjunto = datos.loc[
        (datos["simbolo"] == simbolo)
        & (datos["detector"] == detector)
    ].copy()

    if subconjunto.empty:
        raise ValueError(
            f"No existen resultados para {simbolo} {detector}."
        )

    ranking = (
        subconjunto.groupby(
            [
                "simbolo",
                "detector",
                "variante",
                "grupo",
                "variables",
                "potencia_peso_positivo",
            ],
            as_index=False,
        )
        .agg(
            pliegues=(
                "pliegue",
                "nunique",
            ),
            prevalencia_media=(
                "prevalencia",
                "mean",
            ),
            pr_auc_media=(
                "pr_auc",
                "mean",
            ),
            pr_auc_minimo=(
                "pr_auc",
                "min",
            ),
            pr_auc_lift_medio=(
                "pr_auc_lift",
                "mean",
            ),
            pr_auc_lift_minimo=(
                "pr_auc_lift",
                "min",
            ),
            roc_auc_media=(
                "roc_auc",
                "mean",
            ),
            roc_auc_minimo=(
                "roc_auc",
                "min",
            ),
            precision_050_media=(
                "precision_050",
                "mean",
            ),
            recall_050_media=(
                "recall_050",
                "mean",
            ),
            f1_050_media=(
                "f1_050",
                "mean",
            ),
            mejor_f1_medio=(
                "mejor_f1",
                "mean",
            ),
            mejor_f1_minimo=(
                "mejor_f1",
                "min",
            ),
            desviacion_pr_auc=(
                "pr_auc",
                "std",
            ),
            desviacion_mejor_f1=(
                "mejor_f1",
                "std",
            ),
        )
    )

    ranking[
        "puntuacion"
    ] = (
        ranking[
            "pr_auc_lift_minimo"
        ]
        + 0.35
        * ranking[
            "mejor_f1_minimo"
        ]
        + 0.15
        * ranking[
            "roc_auc_minimo"
        ]
        - 0.10
        * ranking[
            "desviacion_pr_auc"
        ].fillna(
            0.0
        )
    )

    return ranking.sort_values(
        [
            "puntuacion",
            "variables",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )


def main() -> None:
    """Genera rankings de detectores."""

    if not RUTA_HISTORIAL_DESARROLLO.exists():
        raise FileNotFoundError(
            f"No existe: {RUTA_HISTORIAL_DESARROLLO}"
        )

    datos = pd.read_csv(
        RUTA_HISTORIAL_DESARROLLO
    )

    datos = seleccionar_ultimas_ejecuciones(
        datos
    )

    RUTA_RESUMEN_DESARROLLO.mkdir(
        parents=True,
        exist_ok=True,
    )

    lineas = [
        "# Resumen de detectores binarios V4",
        "",
        (
            "Los modelos se comparan con validaciones de 2023 y 2024. "
            "No se utilizan 2025 ni 2026 para seleccionar configuraciones."
        ),
        "",
    ]

    for simbolo in SIMBOLOS:
        for detector in DETECTORES:
            ranking = crear_ranking(
                datos=datos,
                simbolo=simbolo,
                detector=detector,
            )

            ruta = (
                RUTA_RESUMEN_DESARROLLO
                / f"ranking_{simbolo}_{detector}.csv"
            )

            ranking.to_csv(
                ruta,
                index=False,
                encoding="utf-8-sig",
            )

            mejor = ranking.iloc[
                0
            ]

            lineas.extend(
                [
                    f"## {simbolo} — {detector}",
                    "",
                    f"- Variante: `{mejor['variante']}`",
                    f"- Variables: {int(mejor['variables'])}",
                    (
                        f"- PR-AUC medio: "
                        f"{mejor['pr_auc_media']:.4f}"
                    ),
                    (
                        f"- PR-AUC lift mínimo: "
                        f"{mejor['pr_auc_lift_minimo']:.4f}"
                    ),
                    (
                        f"- Mejor F1 medio: "
                        f"{mejor['mejor_f1_medio']:.4f}"
                    ),
                    (
                        f"- ROC-AUC medio: "
                        f"{mejor['roc_auc_media']:.4f}"
                    ),
                    "",
                ]
            )

            print(
                f"\n{simbolo} {detector}"
            )
            print(
                f"Mejor variante: {mejor['variante']}"
            )
            print(
                f"PR-AUC lift mínimo: "
                f"{mejor['pr_auc_lift_minimo']:.4f}"
            )
            print(
                f"- {ruta}"
            )

    ruta_markdown = (
        RUTA_RESUMEN_DESARROLLO
        / "resumen_detectores_v4.md"
    )

    ruta_markdown.write_text(
        "\n".join(
            lineas
        ),
        encoding="utf-8",
    )

    print(
        "\nResumen general:"
    )
    print(
        f"- {ruta_markdown}"
    )


if __name__ == "__main__":
    main()
