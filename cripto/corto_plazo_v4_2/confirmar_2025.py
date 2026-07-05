from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cripto.corto_plazo_v4.configuracion import VARIANTES
from cripto.corto_plazo_v4.utilidades import (
    ajustar_escalador_y_contar,
    calcular_pesos_binarios,
    construir_ruta_datos,
    entrenar_modelo_binario,
    predecir_periodo,
)
from cripto.corto_plazo_v4_2.configuracion import (
    COLUMNAS_META,
    COSTE_TOTAL,
    RUTA_BASES,
    RUTA_CONFIRMACIONES,
    RUTA_DATOS_V2,
    RUTA_MODELOS,
    RUTA_POLITICA,
    SIMBOLO,
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


PERIODO_CONFIRMACION_2025 = {
    "entrenamiento_desde": "2021-01-01",
    "entrenamiento_hasta": "2025-01-01",
    "archivos_entrenamiento": (
        ("2021-01-01", "2022-01-01"),
        ("2022-01-01", "2023-01-01"),
        ("2023-01-01", "2024-01-01"),
        ("2024-01-01", "2025-01-01"),
    ),
    "validacion_desde": "2025-01-01",
    "validacion_hasta": "2026-01-01",
    "archivo_validacion": (
        "2025-01-01",
        "2026-01-01",
    ),
}

POLITICA_CONGELADA = {
    "horizonte": 720,
    "umbral_rearme": 0.40,
    "enfriamiento": 60,
    "distancia_minima": 0.01,
    "diferencia_minima": None,
    "exige_pendiente_positiva_5m": False,
}

UMBRAL_META_CONGELADO = 0.48

VARIANTE_SUBE = "v4_63_sin_pesos"
VARIANTE_BAJA = "v4_63_moderado"


def validar_congelacion() -> None:
    """Comprueba que política y umbral siguen siendo los aprobados."""

    if not RUTA_POLITICA.exists():
        raise FileNotFoundError(
            f"No existe: {RUTA_POLITICA}"
        )

    politica = json.loads(
        RUTA_POLITICA.read_text(
            encoding="utf-8"
        )
    )

    for clave, valor_esperado in POLITICA_CONGELADA.items():
        valor_real = politica.get(
            clave
        )

        if isinstance(
            valor_esperado,
            float,
        ):
            if not np.isclose(
                float(
                    valor_real
                ),
                valor_esperado,
            ):
                raise RuntimeError(
                    f"La política cambió en {clave}: "
                    f"{valor_real} != {valor_esperado}"
                )
        else:
            if valor_real != valor_esperado:
                raise RuntimeError(
                    f"La política cambió en {clave}: "
                    f"{valor_real} != {valor_esperado}"
                )

    ruta_umbral = (
        RUTA_POLITICA.parent
        / "seleccion_meta_modelo.csv"
    )

    if not ruta_umbral.exists():
        raise FileNotFoundError(
            f"No existe: {ruta_umbral}"
        )

    seleccion_meta = pd.read_csv(
        ruta_umbral
    ).iloc[
        0
    ]

    umbral_real = float(
        seleccion_meta[
            "umbral_meta"
        ]
    )

    if not np.isclose(
        umbral_real,
        UMBRAL_META_CONGELADO,
    ):
        raise RuntimeError(
            "El umbral meta guardado no coincide "
            f"con {UMBRAL_META_CONGELADO:.2f}."
        )


def entrenar_y_predecir_detector(
    detector: str,
    variante: str,
    epocas: int,
    tamano_lote: int,
) -> np.ndarray:
    """Entrena un detector V4 con 2021-2024 y predice 2025."""

    configuracion_variante = VARIANTES[
        variante
    ]

    columnas = tuple(
        configuracion_variante[
            "columnas"
        ]
    )

    escalador, conteos = ajustar_escalador_y_contar(
        simbolo=SIMBOLO,
        detector=detector,
        configuracion_periodo=PERIODO_CONFIRMACION_2025,
        columnas_modelo=columnas,
        tamano_lote=tamano_lote,
    )

    pesos = calcular_pesos_binarios(
        conteos=conteos,
        potencia_peso_positivo=float(
            configuracion_variante[
                "potencia_peso_positivo"
            ]
        ),
    )

    modelo = entrenar_modelo_binario(
        simbolo=SIMBOLO,
        detector=detector,
        configuracion_periodo=PERIODO_CONFIRMACION_2025,
        configuracion_variante=configuracion_variante,
        escalador=escalador,
        pesos=pesos,
        epocas=epocas,
        tamano_lote=tamano_lote,
    )

    (
        _,
        probabilidades,
        _,
        _,
    ) = predecir_periodo(
        simbolo=SIMBOLO,
        detector=detector,
        configuracion_periodo=PERIODO_CONFIRMACION_2025,
        columnas_modelo=columnas,
        modelo=modelo,
        escalador=escalador,
        tamano_lote=tamano_lote,
    )

    return probabilidades


def cargar_datos_2025() -> pd.DataFrame:
    """Carga precios y variables de 2025 con la misma purga temporal."""

    desde, hasta = PERIODO_CONFIRMACION_2025[
        "archivo_validacion"
    ]

    ruta_precios = construir_ruta_datos(
        simbolo=SIMBOLO,
        desde=desde,
        hasta=hasta,
    )

    ruta_v2 = (
        RUTA_DATOS_V2
        / f"{SIMBOLO}_1m_4h_{desde}_{hasta}_variables.parquet"
    )

    if not ruta_v2.exists():
        raise FileNotFoundError(
            f"No existe: {ruta_v2}"
        )

    precios = pd.read_parquet(
        ruta_precios,
        columns=[
            "fecha_apertura",
            "fecha_objetivo",
            "precio_cierre",
            "rendimiento_objetivo",
        ],
    )

    columnas_v2 = [
        "fecha_apertura",
        *[
            columna
            for columna in COLUMNAS_META
            if columna not in {
                "probabilidad_sube",
                "probabilidad_baja",
                "diferencia_sube_baja",
                "pendiente_sube_3m",
                "pendiente_sube_5m",
                "pendiente_sube_10m",
                "distancia_umbral",
            }
        ],
    ]

    variables = pd.read_parquet(
        ruta_v2,
        columns=columnas_v2,
    )

    for marco in (
        precios,
        variables,
    ):
        marco[
            "fecha_apertura"
        ] = pd.to_datetime(
            marco[
                "fecha_apertura"
            ],
            utc=True,
            errors="raise",
        )

    precios[
        "fecha_objetivo"
    ] = pd.to_datetime(
        precios[
            "fecha_objetivo"
        ],
        utc=True,
        errors="raise",
    )

    desde_ts = pd.Timestamp(
        PERIODO_CONFIRMACION_2025[
            "validacion_desde"
        ],
        tz="UTC",
    )

    hasta_ts = pd.Timestamp(
        PERIODO_CONFIRMACION_2025[
            "validacion_hasta"
        ],
        tz="UTC",
    )

    mascara = (
        (
            precios[
                "fecha_apertura"
            ]
            >= desde_ts
        )
        & (
            precios[
                "fecha_apertura"
            ]
            < hasta_ts
        )
        & (
            precios[
                "fecha_objetivo"
            ]
            > precios[
                "fecha_apertura"
            ]
        )
        & (
            precios[
                "fecha_objetivo"
            ]
            < hasta_ts
        )
    )

    precios = precios.loc[
        mascara
    ].reset_index(
        drop=True
    )

    base = precios.merge(
        variables,
        on="fecha_apertura",
        how="inner",
        validate="one_to_one",
    )

    return base


def completar_variables_meta(
    base: pd.DataFrame,
    probabilidades_sube: np.ndarray,
    probabilidades_baja: np.ndarray,
) -> pd.DataFrame:
    """Añade probabilidades y variables derivadas para el meta-modelo."""

    if not (
        len(
            base
        )
        == len(
            probabilidades_sube
        )
        == len(
            probabilidades_baja
        )
    ):
        raise RuntimeError(
            "Las probabilidades y la base 2025 "
            "no tienen la misma longitud."
        )

    base = base.copy()

    base[
        "probabilidad_sube"
    ] = probabilidades_sube

    base[
        "probabilidad_baja"
    ] = probabilidades_baja

    base[
        "diferencia_sube_baja"
    ] = (
        base[
            "probabilidad_sube"
        ]
        - base[
            "probabilidad_baja"
        ]
    )

    for minutos in (
        3,
        5,
        10,
    ):
        base[
            f"pendiente_sube_{minutos}m"
        ] = (
            base[
                "probabilidad_sube"
            ]
            - base[
                "probabilidad_sube"
            ].shift(
                minutos
            )
        )

    base[
        "distancia_umbral"
    ] = (
        base[
            "probabilidad_sube"
        ]
        - UMBRAL_SUBE_BASE
    )

    base = base.dropna(
        subset=list(
            COLUMNAS_META
        )
    ).reset_index(
        drop=True
    )

    return base


def construir_mascara_politica(
    datos: pd.DataFrame,
) -> np.ndarray:
    """Aplica exactamente la política V4.2 congelada."""

    probabilidades = datos[
        "probabilidad_sube"
    ].to_numpy(
        dtype="float64"
    )

    cruces = cruces_desde_abajo(
        probabilidades,
        UMBRAL_SUBE_BASE,
    )

    cruces = aplicar_rearme(
        cruces=cruces,
        probabilidades=probabilidades,
        umbral_rearme=POLITICA_CONGELADA[
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
        >= POLITICA_CONGELADA[
            "distancia_minima"
        ]
    )

    return mascara


def preparar_senales_entrenamiento(
    nombre: str,
) -> pd.DataFrame:
    """Obtiene señales candidatas de 2023 o 2024."""

    datos = pd.read_parquet(
        RUTA_BASES
        / f"{nombre}.parquet"
    )

    retornos = calcular_retorno_fijo(
        cierres=datos[
            "precio_cierre"
        ].to_numpy(
            dtype="float64"
        ),
        horizonte=POLITICA_CONGELADA[
            "horizonte"
        ],
    )

    mascara = construir_mascara_politica(
        datos
    )

    mascara &= np.isfinite(
        retornos
    )

    senales = datos.loc[
        mascara,
        list(
            COLUMNAS_META
        ),
    ].copy()

    senales[
        "retorno_neto"
    ] = (
        retornos[
            mascara
        ]
        - COSTE_TOTAL
    )

    senales[
        "objetivo_meta"
    ] = (
        senales[
            "retorno_neto"
        ]
        > 0
    ).astype(
        "int8"
    )

    return senales.reset_index(
        drop=True
    )


def entrenar_meta_modelo() -> Pipeline:
    """Reentrena el meta-modelo con 2023 y 2024 completos."""

    datos_2023 = preparar_senales_entrenamiento(
        "validacion_2023"
    )

    datos_2024 = preparar_senales_entrenamiento(
        "validacion_2024"
    )

    entrenamiento = pd.concat(
        [
            datos_2023,
            datos_2024,
        ],
        ignore_index=True,
    )

    if entrenamiento[
        "objetivo_meta"
    ].nunique() < 2:
        raise RuntimeError(
            "El meta-modelo necesita ambas clases."
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

    print(
        f"Señales meta 2023: {len(datos_2023)}"
    )
    print(
        f"Señales meta 2024: {len(datos_2024)}"
    )
    print(
        f"Señales meta totales: {len(entrenamiento)}"
    )

    return modelo


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epocas",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--tamano-lote",
        type=int,
        default=100000,
    )

    argumentos = parser.parse_args()

    validar_congelacion()

    print(
        "\nCONFIRMACIÓN V4.2 EN 2025"
    )
    print(
        "=" * 72
    )
    print(
        "Política y umbral meta congelados."
    )
    print(
        "Detector SUBE: v4_63_sin_pesos"
    )
    print(
        "Detector BAJA: v4_63_moderado"
    )
    print(
        "Horizonte: 720 minutos"
    )
    print(
        "Rearme: 0,40"
    )
    print(
        "Entrada efectiva mínima: 0,45"
    )
    print(
        "Enfriamiento: 60 minutos"
    )
    print(
        "Umbral meta: 0,48"
    )
    print(
        "Coste total: 0,10 %"
    )

    probabilidades_sube = entrenar_y_predecir_detector(
        detector="SUBE",
        variante=VARIANTE_SUBE,
        epocas=argumentos.epocas,
        tamano_lote=argumentos.tamano_lote,
    )

    probabilidades_baja = entrenar_y_predecir_detector(
        detector="BAJA",
        variante=VARIANTE_BAJA,
        epocas=argumentos.epocas,
        tamano_lote=argumentos.tamano_lote,
    )

    base_2025 = cargar_datos_2025()

    base_2025 = completar_variables_meta(
        base=base_2025,
        probabilidades_sube=probabilidades_sube,
        probabilidades_baja=probabilidades_baja,
    )

    modelo_meta = entrenar_meta_modelo()

    mascara_politica = construir_mascara_politica(
        base_2025
    )

    retornos = calcular_retorno_fijo(
        cierres=base_2025[
            "precio_cierre"
        ].to_numpy(
            dtype="float64"
        ),
        horizonte=POLITICA_CONGELADA[
            "horizonte"
        ],
    )

    mascara_politica &= np.isfinite(
        retornos
    )

    candidatos = base_2025.loc[
        mascara_politica,
        [
            "fecha_apertura",
            *COLUMNAS_META,
        ],
    ].copy()

    candidatos[
        "indice_original"
    ] = np.flatnonzero(
        mascara_politica
    )

    probabilidades_meta = modelo_meta.predict_proba(
        candidatos[
            list(
                COLUMNAS_META
            )
        ]
    )[
        :,
        1,
    ]

    aceptadas = (
        probabilidades_meta
        >= UMBRAL_META_CONGELADO
    )

    mascara_final = np.zeros(
        len(
            base_2025
        ),
        dtype=bool,
    )

    indices_aceptados = candidatos.loc[
        aceptadas,
        "indice_original",
    ].to_numpy(
        dtype="int64"
    )

    mascara_final[
        indices_aceptados
    ] = True

    fechas_ns = convertir_fechas_ns(
        base_2025[
            "fecha_apertura"
        ]
    )

    indices = seleccionar_no_solapadas(
        fechas_ns=fechas_ns,
        mascara=mascara_final,
        horizonte=POLITICA_CONGELADA[
            "horizonte"
        ],
        enfriamiento=POLITICA_CONGELADA[
            "enfriamiento"
        ],
    )

    objetivo_sube = (
        base_2025[
            "rendimiento_objetivo"
        ].to_numpy(
            dtype="float64"
        )
        >= 0.005
    ).astype(
        "int8"
    )

    metricas = evaluar_indices(
        indices=indices,
        retorno_bruto=retornos,
        objetivo_sube=objetivo_sube,
        coste=COSTE_TOTAL,
    )

    fecha = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    ruta_salida = (
        RUTA_CONFIRMACIONES
        / "confirmacion_2025"
        / fecha
    )

    ruta_salida.mkdir(
        parents=True,
        exist_ok=False,
    )

    joblib.dump(
        modelo_meta,
        ruta_salida
        / "meta_modelo_2023_2024.joblib",
    )

    detalle = {
        "fecha_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "periodo_entrenamiento_detector": {
            "desde": "2021-01-01",
            "hasta": "2025-01-01",
        },
        "periodo_entrenamiento_meta": [
            "2023",
            "2024",
        ],
        "periodo_confirmacion": {
            "desde": "2025-01-01",
            "hasta": "2026-01-01",
        },
        "politica_congelada": POLITICA_CONGELADA,
        "umbral_meta_congelado": UMBRAL_META_CONGELADO,
        "variante_sube": VARIANTE_SUBE,
        "variante_baja": VARIANTE_BAJA,
        "filas_evaluadas": len(
            base_2025
        ),
        "senales_politica": int(
            mascara_politica.sum()
        ),
        "senales_aceptadas_meta": int(
            aceptadas.sum()
        ),
        "metricas_2025": metricas,
    }

    (
        ruta_salida
        / "detalle.json"
    ).write_text(
        json.dumps(
            detalle,
            ensure_ascii=False,
            indent=4,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        "\nRESULTADOS V4.2 EN 2025"
    )
    print(
        "=" * 72
    )
    print(
        f"Filas evaluadas: {len(base_2025):,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"Señales de política: {int(mascara_politica.sum())}"
    )
    print(
        f"Señales aceptadas por meta-modelo: {int(aceptadas.sum())}"
    )
    print(
        f"Operaciones no solapadas: {metricas['operaciones']}"
    )
    print(
        f"Precisión clasificatoria: {metricas['precision']:.4f}"
    )
    print(
        f"Porcentaje de acierto clasificatorio: "
        f"{metricas['porcentaje_acierto']:.2f} %"
    )
    print(
        f"Operaciones positivas netas: "
        f"{metricas['porcentaje_positivas']:.2f} %"
    )
    print(
        f"Retorno neto medio: "
        f"{metricas['retorno_neto_medio']:.4%}"
    )
    print(
        f"Retorno neto mediano: "
        f"{metricas['retorno_neto_mediano']:.4%}"
    )
    print(
        f"Factor beneficio: "
        f"{metricas['factor_beneficio']:.3f}"
    )
    print(
        f"Drawdown: "
        f"{metricas['maximo_drawdown']:.2%}"
    )
    print(
        f"\n- {ruta_salida}"
    )


if __name__ == "__main__":
    main()
