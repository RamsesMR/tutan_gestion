from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from cripto.corto_plazo_v4_6_macro.configuracion import (
    PLIEGUES_TEMPORALES,
    SIMBOLO_OPERATIVO,
    VARIANTES,
    VERSION_MODELO,
)
from cripto.corto_plazo_v4_6_macro.configuracion_robustez import (
    RUTA_ESCALADORES_ROBUSTEZ,
    RUTA_MODELOS_ROBUSTEZ,
    RUTA_PREDICCIONES_ROBUSTEZ,
    SEMILLAS_ROBUSTEZ,
    VARIANTES_ROBUSTEZ,
)
from cripto.corto_plazo_v4_6_macro.utilidades_modelo import (
    ajustar_escalador,
    entrenar_modelo,
    predecir_validacion,
)


def parsear_semillas(
    valor: str,
) -> list[int]:
    if valor.strip().lower() == "todas":
        return list(
            SEMILLAS_ROBUSTEZ
        )

    semillas = [
        int(elemento.strip())
        for elemento in valor.split(",")
        if elemento.strip()
    ]

    if not semillas:
        raise ValueError(
            "Debes indicar al menos una semilla."
        )

    if len(
        set(semillas)
    ) != len(
        semillas
    ):
        raise ValueError(
            "Las semillas no pueden repetirse."
        )

    return semillas


def artefactos_completos(
    ruta_predicciones: Path,
    ruta_modelo: Path,
) -> bool:
    existen = (
        ruta_predicciones.exists(),
        ruta_modelo.exists(),
    )

    if all(existen):
        return True

    if any(existen):
        raise RuntimeError(
            "Existe una ejecución parcial. "
            "Usa --sobrescribir para regenerarla: "
            f"{ruta_predicciones.parent}"
        )

    return False


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--semillas",
        default="todas",
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

    semillas = parsear_semillas(
        argumentos.semillas
    )

    RUTA_MODELOS_ROBUSTEZ.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUTA_ESCALADORES_ROBUSTEZ.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUTA_PREDICCIONES_ROBUSTEZ.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_manifiesto = (
        RUTA_PREDICCIONES_ROBUSTEZ
        / "detalle_generacion_robustez.json"
    )

    if ruta_manifiesto.exists():
        try:
            manifiesto = json.loads(
                ruta_manifiesto.read_text(
                    encoding="utf-8"
                )
            )
        except json.JSONDecodeError:
            manifiesto = {
                "ejecuciones": {},
            }
    else:
        manifiesto = {
            "ejecuciones": {},
        }

    manifiesto.update(
        {
            "version": VERSION_MODELO,
            "fecha_utc_ultima_ejecucion": datetime.now(
                timezone.utc
            ).isoformat(),
            "semillas_configuradas": list(
                SEMILLAS_ROBUSTEZ
            ),
            "variantes": list(
                VARIANTES_ROBUSTEZ
            ),
            "epocas": argumentos.epocas,
            "tamano_lote": argumentos.tamano_lote,
            "uso_2025_para_seleccion": False,
            "uso_2026_para_seleccion": False,
        }
    )

    manifiesto.setdefault(
        "ejecuciones",
        {},
    )

    print(
        "\nROBUSTEZ POR SEMILLAS V4.6-MACRO"
    )
    print("=" * 72)
    print(
        f"Semillas: {semillas}"
    )
    print(
        "Cinco variantes únicas; los controles se reutilizan "
        "en las comparaciones."
    )

    for semilla in semillas:
        clave_semilla = str(
            semilla
        )

        manifiesto[
            "ejecuciones"
        ].setdefault(
            clave_semilla,
            {},
        )

        print(
            "\n" + "=" * 72
        )
        print(
            f"SEMILLA {semilla}"
        )
        print(
            "=" * 72
        )

        for nombre_variante in VARIANTES_ROBUSTEZ:
            configuracion = VARIANTES[
                nombre_variante
            ]

            base = str(
                configuracion["base"]
            )

            columnas = tuple(
                configuracion["columnas"]
            )

            print(
                "\n" + "#" * 72
            )
            print(
                f"{nombre_variante}: "
                f"{len(columnas)} variables"
            )
            print(
                f"Base: {base}"
            )
            print(
                "#" * 72
            )

            manifiesto[
                "ejecuciones"
            ][clave_semilla].setdefault(
                nombre_variante,
                {},
            )

            for (
                nombre_pliegue,
                configuracion_periodo,
            ) in PLIEGUES_TEMPORALES.items():
                carpeta_predicciones = (
                    RUTA_PREDICCIONES_ROBUSTEZ
                    / f"semilla_{semilla}"
                    / nombre_variante
                )

                carpeta_modelo = (
                    RUTA_MODELOS_ROBUSTEZ
                    / f"semilla_{semilla}"
                    / nombre_variante
                    / nombre_pliegue
                )

                carpeta_predicciones.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                carpeta_modelo.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                ruta_predicciones = (
                    carpeta_predicciones
                    / (
                        f"{SIMBOLO_OPERATIVO}_"
                        f"{nombre_pliegue}.parquet"
                    )
                )

                ruta_modelo = (
                    carpeta_modelo
                    / "modelo.joblib"
                )

                carpeta_escalador = (
                    RUTA_ESCALADORES_ROBUSTEZ
                    / nombre_variante
                    / nombre_pliegue
                )

                carpeta_escalador.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                ruta_escalador = (
                    carpeta_escalador
                    / "escalador.joblib"
                )

                ruta_conteos = (
                    carpeta_escalador
                    / "conteos_entrenamiento.json"
                )

                if (
                    not argumentos.sobrescribir
                    and artefactos_completos(
                        ruta_predicciones,
                        ruta_modelo,
                    )
                ):
                    print(
                        f"  Conservado: {nombre_pliegue}"
                    )

                    manifiesto[
                        "ejecuciones"
                    ][clave_semilla][
                        nombre_variante
                    ][nombre_pliegue] = {
                        "estado": "conservado",
                        "predicciones": str(
                            ruta_predicciones
                        ),
                    }

                    continue

                print(
                    f"\n  {nombre_pliegue}"
                )
                print(
                    "  " + "-" * 68
                )

                if (
                    ruta_escalador.exists()
                    and ruta_conteos.exists()
                    and not argumentos.sobrescribir
                ):
                    escalador = joblib.load(
                        ruta_escalador
                    )

                    conteos_guardados = json.loads(
                        ruta_conteos.read_text(
                            encoding="utf-8"
                        )
                    )

                    conteos = {
                        int(clase): int(valor)
                        for clase, valor
                        in conteos_guardados.items()
                    }

                    print(
                        "  Escalador común conservado."
                    )
                else:
                    escalador, conteos = (
                        ajustar_escalador(
                            base=base,
                            simbolo=SIMBOLO_OPERATIVO,
                            configuracion_periodo=(
                                configuracion_periodo
                            ),
                            columnas_modelo=columnas,
                            tamano_lote=(
                                argumentos.tamano_lote
                            ),
                        )
                    )

                    joblib.dump(
                        escalador,
                        ruta_escalador,
                    )

                    ruta_conteos.write_text(
                        json.dumps(
                            {
                                str(clase): int(valor)
                                for clase, valor
                                in conteos.items()
                            },
                            ensure_ascii=False,
                            indent=4,
                        ),
                        encoding="utf-8",
                    )

                    print(
                        "  Escalador común generado."
                    )

                modelo = entrenar_modelo(
                    base=base,
                    simbolo=SIMBOLO_OPERATIVO,
                    configuracion_periodo=(
                        configuracion_periodo
                    ),
                    columnas_modelo=columnas,
                    escalador=escalador,
                    epocas=argumentos.epocas,
                    tamano_lote=(
                        argumentos.tamano_lote
                    ),
                    semilla=semilla,
                )

                (
                    probabilidades,
                    objetivo,
                    datos,
                ) = predecir_validacion(
                    base=base,
                    simbolo=SIMBOLO_OPERATIVO,
                    configuracion_periodo=(
                        configuracion_periodo
                    ),
                    columnas_modelo=columnas,
                    modelo=modelo,
                    escalador=escalador,
                    tamano_lote=(
                        argumentos.tamano_lote
                    ),
                )

                datos = datos.copy()

                datos[
                    "probabilidad_sube"
                ] = probabilidades.astype(
                    "float32"
                )

                datos[
                    "objetivo_sube_4h"
                ] = objetivo.astype(
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
                    ruta_modelo,
                )

                manifiesto[
                    "ejecuciones"
                ][clave_semilla][
                    nombre_variante
                ][nombre_pliegue] = {
                    "estado": "generado",
                    "filas": len(datos),
                    "conteos_entrenamiento": {
                        str(clase): int(valor)
                        for clase, valor
                        in conteos.items()
                    },
                    "predicciones": str(
                        ruta_predicciones
                    ),
                    "modelo": str(
                        ruta_modelo
                    ),
                    "escalador_comun": str(
                        ruta_escalador
                    ),
                }

                print(
                    f"  Filas: {len(datos):,}".replace(
                        ",",
                        ".",
                    )
                )
                print(
                    f"  - {ruta_predicciones}"
                )

                del datos
                del probabilidades
                del objetivo
                del modelo
                del escalador
                gc.collect()

                ruta_manifiesto.write_text(
                    json.dumps(
                        manifiesto,
                        ensure_ascii=False,
                        indent=4,
                    ),
                    encoding="utf-8",
                )

    ruta_manifiesto.write_text(
        json.dumps(
            manifiesto,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\nROBUSTEZ POR SEMILLAS GENERADA"
    )
    print(
        f"- {ruta_manifiesto}"
    )


if __name__ == "__main__":
    main()
