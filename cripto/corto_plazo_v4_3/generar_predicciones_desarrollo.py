from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone

import pandas as pd

from cripto.corto_plazo_v4_3.configuracion import (
    PLIEGUES_TEMPORALES,
    RUTA_PREDICCIONES_DESARROLLO,
    SIMBOLO,
    UMBRAL_CLASE_SUBE,
    UMBRAL_SUBE,
    VARIANTE_SUBE,
    VERSION_MODELO,
)
from cripto.corto_plazo_v4_3.entrenamiento import (
    cargar_validacion_completa,
    entrenar_y_predecir_sube,
)


def main() -> None:
    """Genera probabilidades 2023/2024 usando el histórico ampliado."""

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

    if argumentos.epocas <= 0:
        parser.error(
            "--epocas debe ser mayor que cero."
        )

    if argumentos.tamano_lote <= 0:
        parser.error(
            "--tamano-lote debe ser mayor que cero."
        )

    RUTA_PREDICCIONES_DESARROLLO.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nGENERACIÓN DE PREDICCIONES V4.3"
    )
    print("=" * 72)
    print(
        f"Variante SUBE: {VARIANTE_SUBE}"
    )
    print(
        "Única diferencia frente a V4.1: entrenamiento desde 2017."
    )
    print(
        "La estrategia y sus parámetros permanecen congelados."
    )

    detalles = {}

    for nombre_pliegue, configuracion_periodo in PLIEGUES_TEMPORALES.items():
        ruta_salida = (
            RUTA_PREDICCIONES_DESARROLLO
            / f"{SIMBOLO}_{nombre_pliegue}.parquet"
        )

        if ruta_salida.exists() and not argumentos.sobrescribir:
            raise FileExistsError(
                f"Ya existe: {ruta_salida}. "
                "Usa --sobrescribir para regenerarlo."
            )

        print("\n" + "=" * 72)
        print(nombre_pliegue)
        print("=" * 72)

        probabilidades, detalle_entrenamiento = entrenar_y_predecir_sube(
            configuracion_periodo=configuracion_periodo,
            epocas=argumentos.epocas,
            tamano_lote=argumentos.tamano_lote,
        )

        datos = cargar_validacion_completa(
            configuracion_periodo=configuracion_periodo
        )

        if len(datos) != len(probabilidades):
            raise RuntimeError(
                "Las predicciones y los precios no tienen la misma longitud."
            )

        datos[
            "probabilidad_sube"
        ] = probabilidades.astype(
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

        senales_umbral = datos.loc[
            datos[
                "probabilidad_sube"
            ]
            >= UMBRAL_SUBE,
            "objetivo_sube_4h",
        ]

        precision_umbral = (
            float(
                senales_umbral.mean()
            )
            if not senales_umbral.empty
            else 0.0
        )

        print(
            f"Filas: {len(datos):,}".replace(
                ",",
                ".",
            )
        )
        print(
            f"Precisión bruta a umbral {UMBRAL_SUBE:.2f}: "
            f"{precision_umbral:.4f}"
        )
        print(
            "Porcentaje de acierto bruto: "
            f"{precision_umbral:.2%}"
        )
        print(
            f"- {ruta_salida}"
        )

        detalles[
            nombre_pliegue
        ] = {
            "configuracion_periodo": configuracion_periodo,
            "filas": len(datos),
            "precision_bruta_umbral": precision_umbral,
            **detalle_entrenamiento,
        }

        del datos
        del probabilidades
        gc.collect()

    detalle = {
        "fecha_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "version": VERSION_MODELO,
        "simbolo": SIMBOLO,
        "variante_sube": VARIANTE_SUBE,
        "epocas": argumentos.epocas,
        "tamano_lote": argumentos.tamano_lote,
        "historico_ampliado_desde": "2017-08-17",
        "estrategia_modificada": False,
        "uso_2025_para_seleccion": False,
        "uso_2026_para_seleccion": False,
        "pliegues": detalles,
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
