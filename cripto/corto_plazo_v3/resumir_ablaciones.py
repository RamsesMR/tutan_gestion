from __future__ import annotations

from pathlib import Path

import pandas as pd

from cripto.corto_plazo_v3.configuracion import (
    RUTA_HISTORIAL_ABLACIONES,
    RUTA_RESULTADOS_ABLACIONES,
    SIMBOLOS,
)


def seleccionar_ultima_ejecucion(
    datos: pd.DataFrame,
) -> pd.DataFrame:
    """Conserva la ejecución más reciente de cada combinación."""

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

    datos = (
        datos
        .sort_values("fecha_utc")
        .groupby(
            [
                "simbolo",
                "variante",
                "pliegue",
            ],
            as_index=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )

    return datos


def crear_resumen_simbolo(
    datos: pd.DataFrame,
    simbolo: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrega los dos pliegues de cada variante."""

    datos_simbolo = datos.loc[
        datos["simbolo"] == simbolo
    ].copy()

    if datos_simbolo.empty:
        raise ValueError(
            f"No existen resultados para {simbolo}."
        )

    resumen = (
        datos_simbolo.groupby(
            [
                "simbolo",
                "variante",
                "grupo",
                "penalty",
                "alpha",
                "l1_ratio",
                "variables",
            ],
            as_index=False,
        )
        .agg(
            pliegues=(
                "pliegue",
                "nunique",
            ),
            accuracy_media=(
                "accuracy",
                "mean",
            ),
            balanced_accuracy_media=(
                "balanced_accuracy",
                "mean",
            ),
            f1_macro_media=(
                "f1_macro",
                "mean",
            ),
            f1_extremos_media=(
                "f1_extremos_promedio",
                "mean",
            ),
            precision_baja_media=(
                "precision_baja",
                "mean",
            ),
            recall_baja_media=(
                "recall_baja",
                "mean",
            ),
            f1_baja_media=(
                "f1_baja",
                "mean",
            ),
            f1_baja_minimo=(
                "f1_baja",
                "min",
            ),
            pr_auc_baja_media=(
                "pr_auc_baja",
                "mean",
            ),
            precision_sube_media=(
                "precision_sube",
                "mean",
            ),
            recall_sube_media=(
                "recall_sube",
                "mean",
            ),
            f1_sube_media=(
                "f1_sube",
                "mean",
            ),
            f1_sube_minimo=(
                "f1_sube",
                "min",
            ),
            pr_auc_sube_media=(
                "pr_auc_sube",
                "mean",
            ),
            tasa_baja_media=(
                "tasa_predicha_baja",
                "mean",
            ),
            tasa_sube_media=(
                "tasa_predicha_sube",
                "mean",
            ),
            desviacion_entre_pliegues_f1_baja=(
                "f1_baja",
                "std",
            ),
            desviacion_entre_pliegues_f1_sube=(
                "f1_sube",
                "std",
            ),
            desviacion_mensual_f1_baja_media=(
                "desviacion_mensual_f1_baja",
                "mean",
            ),
            desviacion_mensual_f1_sube_media=(
                "desviacion_mensual_f1_sube",
                "mean",
            ),
            coeficientes_cero_media=(
                "coeficientes_totalmente_cero",
                "mean",
            ),
        )
    )

    resumen[
        "mejor_lado"
    ] = resumen.apply(
        lambda fila: (
            "BAJA"
            if fila["f1_baja_media"]
            >= fila["f1_sube_media"]
            else "SUBE"
        ),
        axis=1,
    )

    resumen[
        "f1_mejor_lado"
    ] = resumen[
        [
            "f1_baja_media",
            "f1_sube_media",
        ]
    ].max(
        axis=1
    )

    resumen[
        "f1_peor_lado"
    ] = resumen[
        [
            "f1_baja_media",
            "f1_sube_media",
        ]
    ].min(
        axis=1
    )

    # Ranking equilibrado: útil cuando queremos conservar ambos lados.
    resumen[
        "puntuacion_equilibrada"
    ] = (
        resumen["f1_extremos_media"]
        + 0.50
        * resumen[
            "balanced_accuracy_media"
        ]
        - 0.20
        * (
            resumen[
                "desviacion_entre_pliegues_f1_baja"
            ].fillna(0)
            + resumen[
                "desviacion_entre_pliegues_f1_sube"
            ].fillna(0)
        )
    )

    # Ranking especializado: acepta que una variante destaque en un lado.
    resumen[
        "puntuacion_especializada"
    ] = (
        resumen["f1_mejor_lado"]
        + 0.35
        * resumen[
            [
                "pr_auc_baja_media",
                "pr_auc_sube_media",
            ]
        ].max(
            axis=1
        )
        - 0.15
        * resumen[
            [
                "desviacion_entre_pliegues_f1_baja",
                "desviacion_entre_pliegues_f1_sube",
            ]
        ].max(
            axis=1
        ).fillna(0)
    )

    ranking_equilibrado = resumen.sort_values(
        [
            "puntuacion_equilibrada",
            "variables",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    ranking_especializado = resumen.sort_values(
        [
            "puntuacion_especializada",
            "variables",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    return (
        ranking_equilibrado,
        ranking_especializado,
    )


def main() -> None:
    """Resume los experimentos más recientes."""

    if not RUTA_HISTORIAL_ABLACIONES.exists():
        raise FileNotFoundError(
            f"No existe el historial: {RUTA_HISTORIAL_ABLACIONES}"
        )

    datos = pd.read_csv(
        RUTA_HISTORIAL_ABLACIONES
    )

    datos = seleccionar_ultima_ejecucion(
        datos
    )

    ruta_salida = (
        RUTA_RESULTADOS_ABLACIONES
        / "resumen"
    )

    ruta_salida.mkdir(
        parents=True,
        exist_ok=True,
    )

    lineas = [
        "# Resumen de ablaciones V3A",
        "",
        (
            "Los rankings usan validaciones internas de 2023 y 2024. "
            "No se utilizaron 2025 ni 2026 para seleccionar variables."
        ),
        "",
    ]

    for simbolo in SIMBOLOS:
        (
            ranking_equilibrado,
            ranking_especializado,
        ) = crear_resumen_simbolo(
            datos=datos,
            simbolo=simbolo,
        )

        ruta_equilibrado = (
            ruta_salida
            / f"ranking_equilibrado_{simbolo}.csv"
        )

        ruta_especializado = (
            ruta_salida
            / f"ranking_especializado_{simbolo}.csv"
        )

        ranking_equilibrado.to_csv(
            ruta_equilibrado,
            index=False,
            encoding="utf-8-sig",
        )

        ranking_especializado.to_csv(
            ruta_especializado,
            index=False,
            encoding="utf-8-sig",
        )

        mejor_equilibrado = ranking_equilibrado.iloc[
            0
        ]

        mejor_especializado = ranking_especializado.iloc[
            0
        ]

        lineas.extend(
            [
                f"## {simbolo}",
                "",
                "### Mejor variante equilibrada",
                "",
                (
                    f"- Variante: `{mejor_equilibrado['variante']}`"
                ),
                (
                    f"- Variables: {int(mejor_equilibrado['variables'])}"
                ),
                (
                    f"- F1 BAJA medio: "
                    f"{mejor_equilibrado['f1_baja_media']:.4f}"
                ),
                (
                    f"- F1 SUBE medio: "
                    f"{mejor_equilibrado['f1_sube_media']:.4f}"
                ),
                (
                    f"- F1 extremos medio: "
                    f"{mejor_equilibrado['f1_extremos_media']:.4f}"
                ),
                "",
                "### Mejor variante especializada",
                "",
                (
                    f"- Variante: `{mejor_especializado['variante']}`"
                ),
                (
                    f"- Lado dominante: "
                    f"{mejor_especializado['mejor_lado']}"
                ),
                (
                    f"- F1 del lado dominante: "
                    f"{mejor_especializado['f1_mejor_lado']:.4f}"
                ),
                (
                    f"- Variables: {int(mejor_especializado['variables'])}"
                ),
                "",
            ]
        )

        print(
            f"\n{simbolo}"
        )
        print(
            "Mejor equilibrada: "
            f"{mejor_equilibrado['variante']}"
        )
        print(
            "Mejor especializada: "
            f"{mejor_especializado['variante']} "
            f"({mejor_especializado['mejor_lado']})"
        )
        print(
            f"- {ruta_equilibrado}"
        )
        print(
            f"- {ruta_especializado}"
        )

    ruta_markdown = (
        ruta_salida
        / "resumen_ablaciones_v3a.md"
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
