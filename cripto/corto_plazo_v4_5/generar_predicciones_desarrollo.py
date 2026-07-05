from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone

import joblib
import numpy as np

from cripto.corto_plazo_v4_5.configuracion import (
    PLIEGUES_TEMPORALES,
    RUTA_MODELOS_DESARROLLO,
    RUTA_PREDICCIONES,
    SIMBOLO_OPERATIVO,
    UMBRAL_SUBE_CONGELADO,
    VARIANTES,
    VERSION_MODELO,
)
from cripto.corto_plazo_v4_5.utilidades_modelo import (
    ajustar_escalador,
    entrenar_modelo,
    predecir_validacion,
)


def parsear_variantes(
    valor: str,
) -> list[str]:
    if valor.strip().lower() == "todas":
        return list(VARIANTES)

    variantes = [
        item.strip()
        for item in valor.split(",")
        if item.strip()
    ]

    desconocidas = set(variantes).difference(
        VARIANTES
    )

    if desconocidas:
        raise ValueError(
            f"Variantes desconocidas: {sorted(desconocidas)}"
        )

    return variantes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variantes",
        default="todas",
        help=(
            "todas o lista separada por comas. "
            "Ejemplo: control_63,flujo_spot"
        ),
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
    parser.add_argument(
        "--sobrescribir",
        action="store_true",
    )
    argumentos = parser.parse_args()

    variantes = parsear_variantes(
        argumentos.variantes
    )

    RUTA_MODELOS_DESARROLLO.mkdir(
        parents=True,
        exist_ok=True,
    )
    RUTA_PREDICCIONES.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_detalle = (
        RUTA_PREDICCIONES
        / "detalle_generacion.json"
    )

    if ruta_detalle.exists():
        try:
            detalle = json.loads(
                ruta_detalle.read_text(
                    encoding="utf-8"
                )
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"El manifiesto existente no es JSON válido: {ruta_detalle}"
            ) from error
    else:
        detalle = {
            "variantes": {},
        }

    detalle.update(
        {
            "fecha_utc_ultima_ejecucion": datetime.now(
                timezone.utc
            ).isoformat(),
            "version": VERSION_MODELO,
            "simbolo": SIMBOLO_OPERATIVO,
            "epocas_ultima_ejecucion": argumentos.epocas,
            "tamano_lote_ultima_ejecucion": argumentos.tamano_lote,
            "uso_2025_para_seleccion": False,
            "uso_2026_para_seleccion": False,
        }
    )

    detalle.setdefault(
        "variantes",
        {},
    )

    print("\nGENERACIÓN DE PREDICCIONES V4.5")
    print("=" * 72)
    print(
        "Única diferencia estudiada: nuevas variables históricas."
    )
    print(
        "Algoritmo: SGD logístico, sin pesos, 3 épocas por defecto."
    )

    for nombre_variante in variantes:
        configuracion_variante = VARIANTES[
            nombre_variante
        ]
        columnas = tuple(
            configuracion_variante["columnas"]
        )
        etapa = str(
            configuracion_variante["etapa_datos"]
        )

        detalle["variantes"][nombre_variante] = {
            "columnas": len(columnas),
            "etapa": etapa,
            "pliegues": {},
        }

        print("\n" + "#" * 72)
        print(
            f"{nombre_variante}: {len(columnas)} variables"
        )
        print("#" * 72)

        for nombre_pliegue, configuracion_periodo in (
            PLIEGUES_TEMPORALES.items()
        ):
            carpeta_variante = (
                RUTA_PREDICCIONES
                / nombre_variante
            )
            carpeta_variante.mkdir(
                parents=True,
                exist_ok=True,
            )

            ruta_predicciones = (
                carpeta_variante
                / f"{SIMBOLO_OPERATIVO}_{nombre_pliegue}.parquet"
            )

            carpeta_modelo = (
                RUTA_MODELOS_DESARROLLO
                / nombre_variante
                / nombre_pliegue
            )
            carpeta_modelo.mkdir(
                parents=True,
                exist_ok=True,
            )

            if (
                ruta_predicciones.exists()
                and not argumentos.sobrescribir
            ):
                raise FileExistsError(
                    f"Ya existe: {ruta_predicciones}. "
                    "Usa --sobrescribir."
                )

            print("\n" + "=" * 72)
            print(nombre_pliegue)
            print("=" * 72)

            escalador, conteos = ajustar_escalador(
                simbolo=SIMBOLO_OPERATIVO,
                configuracion_periodo=configuracion_periodo,
                columnas_modelo=columnas,
                etapa=etapa,
                tamano_lote=argumentos.tamano_lote,
            )

            modelo = entrenar_modelo(
                simbolo=SIMBOLO_OPERATIVO,
                configuracion_periodo=configuracion_periodo,
                columnas_modelo=columnas,
                etapa=etapa,
                escalador=escalador,
                epocas=argumentos.epocas,
                tamano_lote=argumentos.tamano_lote,
            )

            probabilidades, objetivo, datos = predecir_validacion(
                simbolo=SIMBOLO_OPERATIVO,
                configuracion_periodo=configuracion_periodo,
                columnas_modelo=columnas,
                etapa=etapa,
                modelo=modelo,
                escalador=escalador,
                tamano_lote=argumentos.tamano_lote,
            )

            datos = datos.copy()
            datos["probabilidad_sube"] = probabilidades.astype(
                "float32"
            )
            datos["objetivo_sube_4h"] = objetivo.astype(
                "int8"
            )

            columnas_salida = [
                "fecha_apertura",
                "fecha_objetivo",
                "precio_apertura",
                "precio_maximo",
                "precio_minimo",
                "precio_cierre",
                "rendimiento_objetivo",
                "probabilidad_sube",
                "objetivo_sube_4h",
            ]

            datos[
                columnas_salida
            ].to_parquet(
                ruta_predicciones,
                index=False,
            )

            joblib.dump(
                modelo,
                carpeta_modelo / "modelo.joblib",
            )
            joblib.dump(
                escalador,
                carpeta_modelo / "escalador.joblib",
            )

            mascara = (
                probabilidades
                >= UMBRAL_SUBE_CONGELADO
            )

            precision = float(
                objetivo[mascara].mean()
                if mascara.any()
                else 0.0
            )

            detalle["variantes"][
                nombre_variante
            ]["pliegues"][nombre_pliegue] = {
                "filas": len(datos),
                "conteos_entrenamiento": {
                    str(clase): int(valor)
                    for clase, valor in conteos.items()
                },
                "precision_bruta_umbral_0_44": precision,
                "porcentaje_acierto_bruto": precision * 100.0,
                "archivo": str(ruta_predicciones),
            }

            print(
                f"Filas: {len(datos):,}".replace(",", ".")
            )
            print(
                f"Precisión bruta a umbral 0,44: {precision:.4f}"
            )
            print(
                f"Porcentaje de acierto bruto: {precision:.2%}"
            )
            print(f"- {ruta_predicciones}")

            del datos
            del probabilidades
            del objetivo
            del modelo
            del escalador
            gc.collect()

    ruta_detalle.write_text(
        json.dumps(
            detalle,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print("\nPREDICCIONES V4.5 GENERADAS")


if __name__ == "__main__":
    main()
