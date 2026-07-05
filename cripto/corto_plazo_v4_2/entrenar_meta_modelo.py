from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cripto.corto_plazo_v4_2.configuracion import (
    COLUMNAS_META,
    COSTE_TOTAL,
    RUTA_BASES,
    RUTA_MODELOS,
    RUTA_POLITICA,
    RUTA_RESULTADOS_META,
    UMBRAL_SUBE_BASE,
)
from cripto.corto_plazo_v4_2.utilidades import (
    aplicar_rearme,
    calcular_retorno_fijo,
    convertir_fechas_ns,
    cruces_desde_abajo,
    evaluar_indices,
    seleccionar_no_solapadas,
)


def construir_mascara_politica(
    datos: pd.DataFrame,
    politica: dict,
) -> np.ndarray:
    """Reproduce exactamente la política V4.2 seleccionada."""

    probabilidades = datos[
        "probabilidad_sube"
    ].to_numpy(
        dtype="float64"
    )

    cruces_base = cruces_desde_abajo(
        probabilidades,
        UMBRAL_SUBE_BASE,
    )

    cruces = aplicar_rearme(
        cruces=cruces_base,
        probabilidades=probabilidades,
        umbral_rearme=politica[
            "umbral_rearme"
        ],
    )

    mascara = cruces.copy()

    mascara &= (
        datos[
            "distancia_umbral"
        ].to_numpy(
            dtype="float64"
        )
        >= float(
            politica[
                "distancia_minima"
            ]
        )
    )

    diferencia_minima = politica[
        "diferencia_minima"
    ]

    if diferencia_minima is not None:
        mascara &= (
            datos[
                "diferencia_sube_baja"
            ].to_numpy(
                dtype="float64"
            )
            >= float(
                diferencia_minima
            )
        )

    if bool(
        politica[
            "exige_pendiente_positiva_5m"
        ]
    ):
        mascara &= (
            datos[
                "pendiente_sube_5m"
            ].to_numpy(
                dtype="float64"
            )
            > 0
        )

    return mascara


def preparar_senales(
    nombre: str,
    politica: dict,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    """Prepara solo las señales que cumplen la política seleccionada."""

    datos = pd.read_parquet(
        RUTA_BASES
        / f"{nombre}.parquet"
    )

    horizonte = int(
        politica[
            "horizonte"
        ]
    )

    retornos = calcular_retorno_fijo(
        cierres=datos[
            "precio_cierre"
        ].to_numpy(
            dtype="float64"
        ),
        horizonte=horizonte,
    )

    mascara = construir_mascara_politica(
        datos=datos,
        politica=politica,
    )

    mascara &= np.isfinite(
        retornos
    )

    candidatos = datos.loc[
        mascara,
        [
            "fecha_apertura",
            *COLUMNAS_META,
        ],
    ].copy()

    candidatos[
        "indice_original"
    ] = np.flatnonzero(
        mascara
    )

    candidatos[
        "retorno_neto"
    ] = (
        retornos[
            mascara
        ]
        - COSTE_TOTAL
    )

    candidatos[
        "objetivo_meta"
    ] = (
        candidatos[
            "retorno_neto"
        ]
        > 0
    ).astype(
        "int8"
    )

    fechas_ns = convertir_fechas_ns(
        datos[
            "fecha_apertura"
        ]
    )

    return (
        candidatos.reset_index(
            drop=True
        ),
        fechas_ns,
        retornos,
    )


def evaluar_umbral_meta(
    candidatos: pd.DataFrame,
    probabilidades_meta: np.ndarray,
    fechas_ns: np.ndarray,
    retornos: np.ndarray,
    politica: dict,
    umbral_meta: float,
) -> dict:
    """Evalúa las señales aceptadas por el meta-modelo sin solapamientos."""

    aceptadas = (
        probabilidades_meta
        >= umbral_meta
    )

    mascara_global = np.zeros(
        len(
            retornos
        ),
        dtype=bool,
    )

    indices_globales = candidatos.loc[
        aceptadas,
        "indice_original",
    ].to_numpy(
        dtype="int64"
    )

    mascara_global[
        indices_globales
    ] = True

    indices = seleccionar_no_solapadas(
        fechas_ns=fechas_ns,
        mascara=mascara_global,
        horizonte=int(
            politica[
                "horizonte"
            ]
        ),
        enfriamiento=int(
            politica[
                "enfriamiento"
            ]
        ),
    )

    objetivo_dummy = (
        retornos > 0
    ).astype(
        "int8"
    )

    metricas = evaluar_indices(
        indices=indices,
        retorno_bruto=retornos,
        objetivo_sube=objetivo_dummy,
        coste=COSTE_TOTAL,
    )

    return {
        "umbral_meta": float(
            round(
                umbral_meta,
                2,
            )
        ),
        **metricas,
    }


def main() -> None:
    """Entrena en 2023 y valida en 2024 sobre la política V4.2."""

    if not RUTA_POLITICA.exists():
        raise FileNotFoundError(
            f"No existe: {RUTA_POLITICA}"
        )

    politica = json.loads(
        RUTA_POLITICA.read_text(
            encoding="utf-8"
        )
    )

    (
        entrenamiento,
        _,
        _,
    ) = preparar_senales(
        nombre="validacion_2023",
        politica=politica,
    )

    (
        validacion,
        fechas_2024,
        retornos_2024,
    ) = preparar_senales(
        nombre="validacion_2024",
        politica=politica,
    )

    if entrenamiento[
        "objetivo_meta"
    ].nunique() < 2:
        raise RuntimeError(
            "El entrenamiento del meta-modelo necesita ambas clases."
        )

    modelo = Pipeline(
        steps=[
            (
                "imputador",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "escalador",
                StandardScaler(),
            ),
            (
                "clasificador",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=1e-4,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    modelo.fit(
        entrenamiento[
            list(
                COLUMNAS_META
            )
        ],
        entrenamiento[
            "objetivo_meta"
        ],
    )

    probabilidades_meta = modelo.predict_proba(
        validacion[
            list(
                COLUMNAS_META
            )
        ]
    )[
        :,
        1,
    ]

    registros = []

    for umbral in np.arange(
        0.40,
        0.76,
        0.02,
    ):
        registros.append(
            evaluar_umbral_meta(
                candidatos=validacion,
                probabilidades_meta=probabilidades_meta,
                fechas_ns=fechas_2024,
                retornos=retornos_2024,
                politica=politica,
                umbral_meta=float(
                    umbral
                ),
            )
        )

    resultados = pd.DataFrame(
        registros
    )

    resultados[
        "cumple_minimo_operaciones"
    ] = (
        resultados[
            "operaciones"
        ]
        >= 40
    )

    elegibles = resultados.loc[
        resultados[
            "cumple_minimo_operaciones"
        ]
        & (
            resultados[
                "retorno_neto_medio"
            ]
            > 0
        )
        & (
            resultados[
                "retorno_neto_mediano"
            ]
            >= 0
        )
        & (
            resultados[
                "factor_beneficio"
            ]
            > 1
        )
    ].copy()

    if elegibles.empty:
        seleccion = resultados.sort_values(
            [
                "retorno_neto_medio",
                "operaciones",
            ],
            ascending=[
                False,
                False,
            ],
        ).head(
            1
        ).copy()

        seleccion[
            "cumple_filtros_meta"
        ] = False

    else:
        elegibles[
            "puntuacion_meta"
        ] = (
            10.0
            * elegibles[
                "retorno_neto_medio"
            ]
            + 0.10
            * (
                elegibles[
                    "factor_beneficio"
                ]
                - 1.0
            )
            + 0.001
            * elegibles[
                "porcentaje_positivas"
            ]
            + 0.04
            * elegibles[
                "maximo_drawdown"
            ]
        )

        seleccion = elegibles.sort_values(
            [
                "puntuacion_meta",
                "operaciones",
            ],
            ascending=[
                False,
                False,
            ],
        ).head(
            1
        ).copy()

        seleccion[
            "cumple_filtros_meta"
        ] = True

    RUTA_RESULTADOS_META.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resultados.to_csv(
        RUTA_RESULTADOS_META,
        index=False,
        encoding="utf-8-sig",
    )

    ruta_seleccion = (
        RUTA_RESULTADOS_META.parent
        / "seleccion_meta_modelo.csv"
    )

    seleccion.to_csv(
        ruta_seleccion,
        index=False,
        encoding="utf-8-sig",
    )

    RUTA_MODELOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_modelo = (
        RUTA_MODELOS
        / "meta_modelo_2023.joblib"
    )

    joblib.dump(
        modelo,
        ruta_modelo,
    )

    fila = seleccion.iloc[
        0
    ]

    print(
        "\nMETA-MODELO V4.2 CORREGIDO"
    )
    print(
        "=" * 72
    )
    print(
        f"Señales candidatas 2023: {len(entrenamiento)}"
    )
    print(
        f"Señales candidatas 2024: {len(validacion)}"
    )
    print(
        f"Umbral meta seleccionado: {fila['umbral_meta']:.2f}"
    )
    print(
        f"Cumple filtros: {bool(fila['cumple_filtros_meta'])}"
    )
    print(
        f"Operaciones 2024: {int(fila['operaciones'])}"
    )
    print(
        f"Precisión operativa: {fila['precision']:.4f}"
    )
    print(
        f"Porcentaje de acierto: {fila['porcentaje_acierto']:.2f} %"
    )
    print(
        f"Operaciones positivas: {fila['porcentaje_positivas']:.2f} %"
    )
    print(
        f"Retorno neto medio: {fila['retorno_neto_medio']:.4%}"
    )
    print(
        f"Retorno neto mediano: {fila['retorno_neto_mediano']:.4%}"
    )
    print(
        f"Factor beneficio: {fila['factor_beneficio']:.3f}"
    )
    print(
        f"Drawdown: {fila['maximo_drawdown']:.2%}"
    )
    print(
        f"\n- {RUTA_RESULTADOS_META}"
    )
    print(
        f"- {ruta_seleccion}"
    )
    print(
        f"- {ruta_modelo}"
    )


if __name__ == "__main__":
    main()
