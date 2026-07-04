from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4.configuracion import (
    PLIEGUES_TEMPORALES,
    VARIANTES,
)
from cripto.corto_plazo_v4.utilidades import (
    ajustar_escalador_y_contar,
    calcular_pesos_binarios,
    construir_ruta_datos,
    entrenar_modelo_binario,
    predecir_periodo,
)
from cripto.corto_plazo_v4_1.configuracion import (
    COLUMNAS_BAJA,
    COLUMNAS_SUBE,
    RUTA_PREDICCIONES_DESARROLLO,
    SIMBOLO,
    UMBRAL_CLASE_SUBE,
    VARIANTE_BAJA,
    VARIANTE_SUBE,
    VERSION_MODELO,
)


def cargar_validacion_completa(
    configuracion_periodo: dict,
) -> pd.DataFrame:
    """Carga precios con la misma purga temporal usada por V4."""

    desde_archivo, hasta_archivo = configuracion_periodo[
        "archivo_validacion"
    ]

    ruta = construir_ruta_datos(
        simbolo=SIMBOLO,
        desde=desde_archivo,
        hasta=hasta_archivo,
    )

    columnas = [
        "fecha_apertura",
        "fecha_objetivo",
        "precio_apertura",
        "precio_maximo",
        "precio_minimo",
        "precio_cierre",
        "rendimiento_objetivo",
    ]

    datos = pd.read_parquet(
        ruta,
        columns=columnas,
    )

    datos[
        "fecha_apertura"
    ] = pd.to_datetime(
        datos[
            "fecha_apertura"
        ],
        utc=True,
        errors="raise",
    )

    datos[
        "fecha_objetivo"
    ] = pd.to_datetime(
        datos[
            "fecha_objetivo"
        ],
        utc=True,
        errors="raise",
    )

    desde = pd.Timestamp(
        configuracion_periodo[
            "validacion_desde"
        ],
        tz="UTC",
    )

    hasta = pd.Timestamp(
        configuracion_periodo[
            "validacion_hasta"
        ],
        tz="UTC",
    )

    mascara = (
        (
            datos[
                "fecha_apertura"
            ]
            >= desde
        )
        & (
            datos[
                "fecha_apertura"
            ]
            < hasta
        )
        & (
            datos[
                "fecha_objetivo"
            ]
            > datos[
                "fecha_apertura"
            ]
        )
        & (
            datos[
                "fecha_objetivo"
            ]
            < hasta
        )
    )

    return datos.loc[
        mascara
    ].reset_index(
        drop=True
    )


def entrenar_y_predecir(
    detector: str,
    variante: str,
    configuracion_periodo: dict,
    epocas: int,
    tamano_lote: int,
) -> np.ndarray:
    """Entrena una variante V4 y devuelve probabilidades OOS."""

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
        configuracion_periodo=configuracion_periodo,
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
        configuracion_periodo=configuracion_periodo,
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
        configuracion_periodo=configuracion_periodo,
        columnas_modelo=columnas,
        modelo=modelo,
        escalador=escalador,
        tamano_lote=tamano_lote,
    )

    return probabilidades


def main() -> None:
    """Genera probabilidades fuera de muestra de 2023 y 2024."""

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

    parser.add_argument(
        "--sobrescribir",
        action="store_true",
    )

    argumentos = parser.parse_args()

    RUTA_PREDICCIONES_DESARROLLO.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nGENERACIÓN DE PREDICCIONES V4.1"
    )
    print("=" * 72)
    print(
        f"SUBE: {VARIANTE_SUBE}"
    )
    print(
        f"BAJA para veto: {VARIANTE_BAJA}"
    )
    print(
        "2025 y 2026 no se utilizan."
    )

    for nombre_pliegue, configuracion_periodo in PLIEGUES_TEMPORALES.items():
        ruta_salida = (
            RUTA_PREDICCIONES_DESARROLLO
            / f"{SIMBOLO}_{nombre_pliegue}.parquet"
        )

        if ruta_salida.exists() and not argumentos.sobrescribir:
            raise FileExistsError(
                f"Ya existe: {ruta_salida}. Usa --sobrescribir."
            )

        print(
            "\n"
            + "=" * 72
        )
        print(
            nombre_pliegue
        )
        print(
            "=" * 72
        )

        probabilidades_sube = entrenar_y_predecir(
            detector="SUBE",
            variante=VARIANTE_SUBE,
            configuracion_periodo=configuracion_periodo,
            epocas=argumentos.epocas,
            tamano_lote=argumentos.tamano_lote,
        )

        probabilidades_baja = entrenar_y_predecir(
            detector="BAJA",
            variante=VARIANTE_BAJA,
            configuracion_periodo=configuracion_periodo,
            epocas=argumentos.epocas,
            tamano_lote=argumentos.tamano_lote,
        )

        datos = cargar_validacion_completa(
            configuracion_periodo
        )

        if not (
            len(
                datos
            )
            == len(
                probabilidades_sube
            )
            == len(
                probabilidades_baja
            )
        ):
            raise RuntimeError(
                "Las predicciones y los precios no tienen la misma longitud."
            )

        datos[
            "probabilidad_sube"
        ] = probabilidades_sube.astype(
            "float32"
        )

        datos[
            "probabilidad_baja"
        ] = probabilidades_baja.astype(
            "float32"
        )

        datos[
            "objetivo_sube_4h"
        ] = (
            datos[
                "rendimiento_objetivo"
            ]
            >= UMBRAL_CLASE_SUBE
        ).astype(
            "int8"
        )

        datos.to_parquet(
            ruta_salida,
            index=False,
        )

        precision_048 = float(
            datos.loc[
                datos[
                    "probabilidad_sube"
                ]
                >= 0.48,
                "objetivo_sube_4h",
            ].mean()
        )

        print(
            f"Filas: {len(datos):,}".replace(
                ",",
                ".",
            )
        )
        print(
            f"Precisión a umbral 0,48: {precision_048:.4f}"
        )
        print(
            f"Porcentaje de acierto a 0,48: {precision_048:.2%}"
        )
        print(
            f"- {ruta_salida}"
        )

        del datos
        del probabilidades_sube
        del probabilidades_baja
        gc.collect()

    detalle = {
        "fecha_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "version": VERSION_MODELO,
        "simbolo": SIMBOLO,
        "variante_sube": VARIANTE_SUBE,
        "variante_baja": VARIANTE_BAJA,
        "epocas": argumentos.epocas,
        "tamano_lote": argumentos.tamano_lote,
        "uso_2025": False,
        "uso_2026": False,
    }

    (
        RUTA_PREDICCIONES_DESARROLLO
        / "detalle_generacion.json"
    ).write_text(
        json.dumps(
            detalle,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
