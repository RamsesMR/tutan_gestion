from __future__ import annotations

import json
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
    calcular_retorno_fijo,
    cruces_desde_abajo,
)


def preparar_senales(
    nombre: str,
    horizonte: int,
) -> pd.DataFrame:
    datos = pd.read_parquet(RUTA_BASES / f"{nombre}.parquet")
    probabilidades = datos["probabilidad_sube"].to_numpy(float)
    cruces = cruces_desde_abajo(probabilidades, UMBRAL_SUBE_BASE)
    retornos = calcular_retorno_fijo(
        datos["precio_cierre"].to_numpy(float),
        horizonte,
    )

    mascara = cruces & np.isfinite(retornos)

    senales = datos.loc[
        mascara,
        [
            "fecha_apertura",
            *COLUMNAS_META,
        ],
    ].copy()

    senales["retorno_neto"] = retornos[mascara] - COSTE_TOTAL
    senales["objetivo_meta"] = (
        senales["retorno_neto"] > 0
    ).astype("int8")

    return senales.reset_index(drop=True)


def main() -> None:
    politica = json.loads(RUTA_POLITICA.read_text(encoding="utf-8"))
    horizonte = int(politica["horizonte"])

    entrenamiento = preparar_senales(
        "validacion_2023",
        horizonte,
    )

    validacion = preparar_senales(
        "validacion_2024",
        horizonte,
    )

    modelo = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="median")),
            ("escalador", StandardScaler()),
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
        entrenamiento[list(COLUMNAS_META)],
        entrenamiento["objetivo_meta"],
    )

    probabilidades = modelo.predict_proba(
        validacion[list(COLUMNAS_META)]
    )[:, 1]

    registros = []

    for umbral in np.arange(0.40, 0.76, 0.02):
        mascara = probabilidades >= umbral
        seleccionadas = validacion.loc[mascara]

        if seleccionadas.empty:
            continue

        netos = seleccionadas["retorno_neto"].to_numpy(float)

        registros.append(
            {
                "umbral_meta": float(round(umbral, 2)),
                "operaciones": int(len(seleccionadas)),
                "porcentaje_positivas": float((netos > 0).mean() * 100),
                "retorno_neto_medio": float(netos.mean()),
                "retorno_neto_mediano": float(np.median(netos)),
            }
        )

    resultados = pd.DataFrame(registros)
    resultados.to_csv(
        RUTA_RESULTADOS_META,
        index=False,
        encoding="utf-8-sig",
    )

    RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        modelo,
        RUTA_MODELOS / "meta_modelo_2023.joblib",
    )

    print("\nMETA-MODELO V4.2")
    print("=" * 72)
    print(
        f"Señales entrenamiento 2023: {len(entrenamiento)}"
    )
    print(
        f"Señales validación 2024: {len(validacion)}"
    )
    print(f"- {RUTA_RESULTADOS_META}")
    print(f"- {RUTA_MODELOS / 'meta_modelo_2023.joblib'}")


if __name__ == "__main__":
    main()
