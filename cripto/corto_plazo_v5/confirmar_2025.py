from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from cripto.corto_plazo_v5.configuracion import (
    COLUMNAS_MODELO_V5,
    COSTE_BASE,
    OBJETIVOS_BARRERAS,
    PERIODO_CONFIRMACION_2025,
    RUTA_CONFIRMACION_2025,
    RUTA_SELECCION_V5,
    VARIANTES_MODELO,
    VERSION_MODELO,
)
from cripto.corto_plazo_v5.entrenar_modelos import (
    cargar_periodos,
    entrenar_histgb,
    entrenar_sgd,
    predecir,
)
from cripto.corto_plazo_v5.utilidades import (
    calcular_metricas_probabilidad,
    evaluar_umbral,
)


def main() -> None:
    """Confirma la configuración elegida en 2025 sin reajustarla."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--simbolo",
        default="BTCUSDT",
    )

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

    if not RUTA_SELECCION_V5.exists():
        raise FileNotFoundError(
            f"No existe: {RUTA_SELECCION_V5}"
        )

    seleccion = pd.read_csv(
        RUTA_SELECCION_V5
    )

    fila = seleccion.loc[
        seleccion[
            "simbolo"
        ] == argumentos.simbolo
    ]

    if fila.empty:
        raise ValueError(
            f"No hay selección para {argumentos.simbolo}."
        )

    fila = fila.iloc[
        0
    ]

    nombre_objetivo = str(
        fila[
            "objetivo"
        ]
    )

    nombre_variante = str(
        fila[
            "variante"
        ]
    )

    umbral = float(
        fila[
            "umbral"
        ]
    )

    configuracion_variante = VARIANTES_MODELO[
        nombre_variante
    ]

    print(
        "\nCONFIRMACIÓN V5 EN 2025"
    )
    print("=" * 72)
    print(
        f"Símbolo: {argumentos.simbolo}"
    )
    print(
        f"Objetivo congelado: {nombre_objetivo}"
    )
    print(
        f"Variante congelada: {nombre_variante}"
    )
    print(
        f"Umbral congelado: {umbral:.2f}"
    )
    print(
        "2025 es confirmación secundaria; 2026 no se utiliza."
    )

    (
        x_entrenamiento,
        y_entrenamiento,
        _,
        _,
    ) = cargar_periodos(
        simbolo=argumentos.simbolo,
        periodos=PERIODO_CONFIRMACION_2025[
            "entrenamiento"
        ],
        nombre_objetivo=nombre_objetivo,
        hasta=PERIODO_CONFIRMACION_2025[
            "hasta_entrenamiento"
        ],
    )

    (
        x_validacion,
        y_validacion,
        fechas_validacion,
        retornos_validacion,
    ) = cargar_periodos(
        simbolo=argumentos.simbolo,
        periodos=(
            PERIODO_CONFIRMACION_2025[
                "validacion"
            ],
        ),
        nombre_objetivo=nombre_objetivo,
        hasta=PERIODO_CONFIRMACION_2025[
            "hasta_validacion"
        ],
    )

    if configuracion_variante[
        "tipo"
    ] == "sgd":
        (
            modelo,
            escalador,
            pesos,
        ) = entrenar_sgd(
            variables=x_entrenamiento,
            objetivo=y_entrenamiento,
            configuracion=configuracion_variante,
            epocas=argumentos.epocas,
            tamano_lote=argumentos.tamano_lote,
        )

        detalle_muestra = None

    else:
        (
            modelo,
            escalador,
            pesos,
            detalle_muestra,
        ) = entrenar_histgb(
            variables=x_entrenamiento,
            objetivo=y_entrenamiento,
            configuracion=configuracion_variante,
        )

    probabilidades = predecir(
        modelo=modelo,
        escalador=escalador,
        variables=x_validacion,
        tamano_lote=argumentos.tamano_lote,
    )

    metricas_probabilidad = calcular_metricas_probabilidad(
        objetivo=y_validacion,
        probabilidades=probabilidades,
    )

    horizonte = int(
        OBJETIVOS_BARRERAS[
            nombre_objetivo
        ][
            "horizonte_minutos"
        ]
    )

    metricas_operativas = evaluar_umbral(
        objetivo=y_validacion,
        probabilidades=probabilidades,
        fechas_ns=fechas_validacion,
        retornos_salida=retornos_validacion,
        umbral=umbral,
        horizonte_minutos=horizonte,
        coste=COSTE_BASE,
    )

    identificador = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    ruta = (
        RUTA_CONFIRMACION_2025
        / argumentos.simbolo
        / identificador
    )

    ruta.mkdir(
        parents=True,
        exist_ok=False,
    )

    detalle = {
        "fecha_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "version_modelo": VERSION_MODELO,
        "simbolo": argumentos.simbolo,
        "objetivo": nombre_objetivo,
        "variante": nombre_variante,
        "umbral": umbral,
        "coste": COSTE_BASE,
        "variables": len(
            COLUMNAS_MODELO_V5
        ),
        "pesos": pesos,
        "detalle_muestra": detalle_muestra,
        "metricas_probabilidad": metricas_probabilidad,
        "metricas_operativas": metricas_operativas,
        "confirmacion_secundaria": True,
        "prueba_totalmente_intacta": False,
        "uso_2026": False,
    }

    (
        ruta
        / "detalle.json"
    ).write_text(
        json.dumps(
            detalle,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    joblib.dump(
        {
            "detalle": detalle,
            "modelo": modelo,
            "escalador": escalador,
        },
        ruta / "modelo.joblib",
    )

    pd.DataFrame(
        {
            "fecha_apertura": pd.to_datetime(
                fechas_validacion,
                unit="ns",
                utc=True,
            ),
            "objetivo": y_validacion,
            "probabilidad": probabilidades,
            "retorno_salida": retornos_validacion,
        }
    ).to_parquet(
        ruta
        / "predicciones_2025.parquet",
        index=False,
    )

    print(
        "\nResultado:"
    )
    print(
        f"PR-AUC lift: {metricas_probabilidad['pr_auc_lift']:.4f}"
    )
    print(
        f"Precision: {metricas_operativas['precision']:.4f}"
    )
    print(
        f"Recall: {metricas_operativas['recall']:.4f}"
    )
    print(
        f"Operaciones: {metricas_operativas['operaciones_no_solapadas']}"
    )
    print(
        f"Retorno neto medio: "
        f"{metricas_operativas['retorno_neto_medio']:.4%}"
    )
    print(
        f"Factor beneficio: "
        f"{metricas_operativas['factor_beneficio']:.3f}"
    )
    print(
        f"Drawdown: "
        f"{metricas_operativas['maximo_drawdown']:.2%}"
    )
    print(
        f"- {ruta}"
    )


if __name__ == "__main__":
    main()
