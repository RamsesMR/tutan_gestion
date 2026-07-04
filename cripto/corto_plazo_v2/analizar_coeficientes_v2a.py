from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from cripto.corto_plazo_v2.configuracion import (
    COLUMNAS_CRUZADAS,
    COLUMNAS_MODELO_V1,
    NOMBRE_HORIZONTE,
    RUTA_MODELOS_V2,
    SIMBOLOS,
)


RUTA_SALIDA = (
    RUTA_MODELOS_V2
    / "analisis_coeficientes"
)


def clasificar_grupo(
    variable: str,
) -> str:
    """Asigna cada variable a un grupo interpretable."""

    if variable in COLUMNAS_CRUZADAS:
        if variable.startswith(
            "otro_rendimiento_"
        ):
            return "rendimiento_otro"

        if variable.startswith(
            "divergencia_rendimiento_"
        ):
            return "divergencia_rendimiento"

        if variable.startswith(
            "otro_volatilidad_"
        ):
            return "volatilidad_otro"

        if variable.startswith(
            "ratio_volatilidad_"
        ):
            return "volatilidad_relativa"

        if variable.startswith(
            "correlacion_"
        ):
            return "correlacion"

        if variable.startswith(
            "beta_"
        ):
            return "beta"

        if variable.startswith(
            "residual_"
        ):
            return "residual"

        if variable.startswith(
            "zscore_"
        ):
            return "zscore_divergencia"

        return "cruzadas_otras"

    if variable.startswith(
        (
            "rango_",
            "cuerpo_",
            "mecha_",
        )
    ):
        return "forma_vela"

    if variable.startswith(
        "proporcion_compradora_"
    ):
        return "presion_compradora"

    if variable.startswith(
        "rendimiento_"
    ):
        return "rendimiento_propio"

    if variable.startswith(
        "volatilidad_"
    ):
        return "volatilidad_propia"

    if variable.startswith(
        "desviacion_media_"
    ):
        return "desviacion_media"

    if variable.startswith(
        "volumen_relativo_"
    ):
        return "volumen_relativo"

    if variable.startswith(
        "operaciones_relativas_"
    ):
        return "operaciones_relativas"

    if variable.endswith(
        (
            "_seno",
            "_coseno",
        )
    ):
        return "tiempo"

    return "otros"


def guardar_json(
    contenido: dict[str, Any],
    ruta: Path,
) -> None:
    """Guarda un JSON legible."""

    with ruta.open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            contenido,
            archivo,
            ensure_ascii=False,
            indent=4,
        )


def analizar_simbolo(
    simbolo: str,
) -> None:
    """Analiza los coeficientes estandarizados del SGDClassifier."""

    ruta_modelo = (
        RUTA_MODELOS_V2
        / f"modelo_base_{simbolo}_{NOMBRE_HORIZONTE}.joblib"
    )

    if not ruta_modelo.exists():
        raise FileNotFoundError(
            f"No existe: {ruta_modelo}"
        )

    paquete = joblib.load(
        ruta_modelo
    )

    modelo = paquete[
        "modelo"
    ]

    columnas = list(
        paquete[
            "columnas_modelo"
        ]
    )

    if not hasattr(
        modelo,
        "coef_",
    ):
        raise ValueError(
            f"El modelo de {simbolo} no expone coeficientes."
        )

    coeficientes = np.asarray(
        modelo.coef_,
        dtype="float64",
    )

    clases = [
        str(clase)
        for clase in modelo.classes_
    ]

    if coeficientes.shape != (
        len(clases),
        len(columnas),
    ):
        raise ValueError(
            f"{simbolo}: dimensiones de coeficientes inesperadas."
        )

    datos = pd.DataFrame(
        {
            "variable": columnas,
            "grupo": [
                clasificar_grupo(
                    variable
                )
                for variable in columnas
            ],
            "es_cruzada": [
                variable in COLUMNAS_CRUZADAS
                for variable in columnas
            ],
        }
    )

    for indice, clase in enumerate(
        clases
    ):
        datos[
            f"coeficiente_{clase}"
        ] = coeficientes[
            indice
        ]

        datos[
            f"coeficiente_absoluto_{clase}"
        ] = np.abs(
            coeficientes[
                indice
            ]
        )

    columnas_absolutas = [
        f"coeficiente_absoluto_{clase}"
        for clase in clases
    ]

    datos[
        "importancia_media_absoluta"
    ] = datos[
        columnas_absolutas
    ].mean(
        axis=1
    )

    datos[
        "importancia_maxima_absoluta"
    ] = datos[
        columnas_absolutas
    ].max(
        axis=1
    )

    datos = datos.sort_values(
        "importancia_media_absoluta",
        ascending=False,
    ).reset_index(
        drop=True
    )

    grupos = (
        datos.groupby(
            "grupo",
            as_index=False,
        )
        .agg(
            variables=(
                "variable",
                "count",
            ),
            importancia_media=(
                "importancia_media_absoluta",
                "mean",
            ),
            importancia_total=(
                "importancia_media_absoluta",
                "sum",
            ),
            importancia_maxima=(
                "importancia_maxima_absoluta",
                "max",
            ),
        )
        .sort_values(
            "importancia_total",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    RUTA_SALIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_variables = (
        RUTA_SALIDA
        / f"coeficientes_variables_{simbolo}_{NOMBRE_HORIZONTE}.csv"
    )

    ruta_grupos = (
        RUTA_SALIDA
        / f"importancia_grupos_{simbolo}_{NOMBRE_HORIZONTE}.csv"
    )

    ruta_resumen = (
        RUTA_SALIDA
        / f"resumen_coeficientes_{simbolo}_{NOMBRE_HORIZONTE}.md"
    )

    datos.to_csv(
        ruta_variables,
        index=False,
        encoding="utf-8-sig",
    )

    grupos.to_csv(
        ruta_grupos,
        index=False,
        encoding="utf-8-sig",
    )

    lineas = [
        f"# Coeficientes V2A — {simbolo}",
        "",
        (
            "Los coeficientes corresponden a variables previamente "
            "estandarizadas. Un valor alto indica mayor influencia lineal, "
            "pero no demuestra causalidad."
        ),
        "",
        "## Variables con mayor importancia media absoluta",
        "",
        "| Posición | Variable | Grupo | Cruzada | Importancia |",
        "|---:|---|---|---|---:|",
    ]

    for posicion, fila in enumerate(
        datos.head(20).itertuples(
            index=False
        ),
        start=1,
    ):
        lineas.append(
            f"| {posicion} | {fila.variable} | {fila.grupo} | "
            f"{'sí' if fila.es_cruzada else 'no'} | "
            f"{fila.importancia_media_absoluta:.6f} |"
        )

    lineas.extend(
        [
            "",
            "## Importancia por grupo",
            "",
            "| Grupo | Variables | Media | Total | Máxima |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for fila in grupos.itertuples(
        index=False
    ):
        lineas.append(
            f"| {fila.grupo} | {fila.variables} | "
            f"{fila.importancia_media:.6f} | "
            f"{fila.importancia_total:.6f} | "
            f"{fila.importancia_maxima:.6f} |"
        )

    for clase in clases:
        lineas.extend(
            [
                "",
                f"## Coeficientes más positivos para {clase}",
                "",
                "| Variable | Grupo | Coeficiente |",
                "|---|---|---:|",
            ]
        )

        positivos = datos.sort_values(
            f"coeficiente_{clase}",
            ascending=False,
        ).head(
            10
        )

        for fila in positivos.itertuples(
            index=False
        ):
            lineas.append(
                f"| {fila.variable} | {fila.grupo} | "
                f"{getattr(fila, f'coeficiente_{clase}'):.6f} |"
            )

        lineas.extend(
            [
                "",
                f"## Coeficientes más negativos para {clase}",
                "",
                "| Variable | Grupo | Coeficiente |",
                "|---|---|---:|",
            ]
        )

        negativos = datos.sort_values(
            f"coeficiente_{clase}",
            ascending=True,
        ).head(
            10
        )

        for fila in negativos.itertuples(
            index=False
        ):
            lineas.append(
                f"| {fila.variable} | {fila.grupo} | "
                f"{getattr(fila, f'coeficiente_{clase}'):.6f} |"
            )

    ruta_resumen.write_text(
        "\n".join(
            lineas
        ),
        encoding="utf-8",
    )

    print(
        f"\n{simbolo}"
    )
    print(
        f"- {ruta_variables}"
    )
    print(
        f"- {ruta_grupos}"
    )
    print(
        f"- {ruta_resumen}"
    )


def main() -> None:
    """Analiza ambos modelos V2A."""

    print(
        "\nANÁLISIS DE COEFICIENTES V2A"
    )
    print("=" * 70)

    for simbolo in SIMBOLOS:
        analizar_simbolo(
            simbolo
        )


if __name__ == "__main__":
    main()
