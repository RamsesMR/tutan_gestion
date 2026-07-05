from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_4.configuracion import (
    COSTE,
    HORIZONTE_MAXIMO_MINUTOS,
    PLIEGUES,
    RUTA_DIAGNOSTICO,
    UMBRALES_BAJA,
    VERSION_MODELO,
)
from cripto.corto_plazo_v4_4.utilidades import (
    construir_cruces_baja,
    cargar_pliegue,
    convertir_para_json,
    seleccionar_indices_control,
)


def analizar_operacion(
    datos: pd.DataFrame,
    indice_entrada: int,
    cierres: np.ndarray,
    maximos: np.ndarray,
    minimos: np.ndarray,
    probabilidades_baja: np.ndarray,
    cruces_por_umbral: dict[float, np.ndarray],
) -> dict:
    """Analiza una operación original de V4.1 durante sus 480 minutos."""

    indice_salida = (
        indice_entrada
        + HORIZONTE_MAXIMO_MINUTOS
    )

    precio_entrada = cierres[
        indice_entrada
    ]

    retorno_fijo_bruto = (
        cierres[indice_salida]
        / precio_entrada
        - 1.0
    )

    retorno_fijo_neto = (
        retorno_fijo_bruto
        - COSTE
    )

    tramo = slice(
        indice_entrada + 1,
        indice_salida + 1,
    )

    retornos_maximos = (
        maximos[tramo]
        / precio_entrada
        - 1.0
    )

    retornos_minimos = (
        minimos[tramo]
        / precio_entrada
        - 1.0
    )

    probabilidades_tramo = probabilidades_baja[
        indice_entrada:
        indice_salida + 1
    ]

    indice_maximo_baja_relativo = int(
        np.argmax(
            probabilidades_tramo
        )
    )

    registro = {
        "fecha_entrada": datos.loc[
            indice_entrada,
            "fecha_apertura",
        ],
        "fecha_salida_fija": datos.loc[
            indice_salida,
            "fecha_apertura",
        ],
        "indice_entrada": indice_entrada,
        "indice_salida_fija": indice_salida,
        "probabilidad_sube_entrada": float(
            datos.loc[
                indice_entrada,
                "probabilidad_sube",
            ]
        ),
        "probabilidad_baja_entrada": float(
            probabilidades_baja[
                indice_entrada
            ]
        ),
        "probabilidad_baja_maxima": float(
            probabilidades_tramo[
                indice_maximo_baja_relativo
            ]
        ),
        "minuto_maximo_baja": (
            indice_maximo_baja_relativo
        ),
        "objetivo_sube_4h": int(
            datos.loc[
                indice_entrada,
                "objetivo_sube_4h",
            ]
        ),
        "retorno_fijo_bruto": float(
            retorno_fijo_bruto
        ),
        "retorno_fijo_neto": float(
            retorno_fijo_neto
        ),
        "operacion_positiva": bool(
            retorno_fijo_neto > 0
        ),
        "excursion_favorable_maxima": float(
            retornos_maximos.max()
        ),
        "excursion_adversa_maxima": float(
            retornos_minimos.min()
        ),
    }

    for umbral in UMBRALES_BAJA:
        cruces = cruces_por_umbral[
            umbral
        ]

        indices_cruce = np.flatnonzero(
            cruces[
                indice_entrada + 1:
                indice_salida + 1
            ]
        )

        sufijo = str(
            umbral
        ).replace(
            ".",
            "_",
        )

        if indices_cruce.size == 0:
            registro[
                f"cruza_baja_{sufijo}"
            ] = False

            registro[
                f"minuto_cruce_baja_{sufijo}"
            ] = np.nan

            registro[
                f"retorno_neto_en_cruce_{sufijo}"
            ] = np.nan
        else:
            minuto_cruce = int(
                indices_cruce[0]
                + 1
            )

            indice_cruce = (
                indice_entrada
                + minuto_cruce
            )

            retorno_cruce_neto = (
                cierres[indice_cruce]
                / precio_entrada
                - 1.0
                - COSTE
            )

            registro[
                f"cruza_baja_{sufijo}"
            ] = True

            registro[
                f"minuto_cruce_baja_{sufijo}"
            ] = minuto_cruce

            registro[
                f"retorno_neto_en_cruce_{sufijo}"
            ] = float(
                retorno_cruce_neto
            )

    return registro


def main() -> None:
    """Diagnostica si BAJA anticipa pérdidas en las operaciones V4.1."""

    RUTA_DIAGNOSTICO.mkdir(
        parents=True,
        exist_ok=True,
    )

    registros = []

    print(
        "\nDIAGNÓSTICO DEL DETECTOR BAJA PARA V4.4"
    )
    print("=" * 72)

    for pliegue in PLIEGUES:
        datos = cargar_pliegue(
            pliegue
        )

        indices = seleccionar_indices_control(
            datos
        )

        cierres = datos[
            "precio_cierre"
        ].to_numpy(
            dtype="float64"
        )

        maximos = datos[
            "precio_maximo"
        ].to_numpy(
            dtype="float64"
        )

        minimos = datos[
            "precio_minimo"
        ].to_numpy(
            dtype="float64"
        )

        probabilidades_baja = datos[
            "probabilidad_baja"
        ].to_numpy(
            dtype="float64"
        )

        cruces_por_umbral = {
            umbral: construir_cruces_baja(
                probabilidades_baja=probabilidades_baja,
                umbral_baja=umbral,
            )
            for umbral in UMBRALES_BAJA
        }

        print(
            f"{pliegue}: {len(indices)} operaciones originales V4.1."
        )

        for indice in indices:
            registro = analizar_operacion(
                datos=datos,
                indice_entrada=int(
                    indice
                ),
                cierres=cierres,
                maximos=maximos,
                minimos=minimos,
                probabilidades_baja=probabilidades_baja,
                cruces_por_umbral=cruces_por_umbral,
            )

            registro[
                "pliegue"
            ] = pliegue

            registros.append(
                registro
            )

    detalle = pd.DataFrame(
        registros
    )

    ruta_detalle = (
        RUTA_DIAGNOSTICO
        / "diagnostico_operaciones.csv"
    )

    detalle.to_csv(
        ruta_detalle,
        index=False,
        encoding="utf-8-sig",
    )

    resumenes = []

    for pliegue in PLIEGUES:
        datos_pliegue = detalle.loc[
            detalle["pliegue"]
            == pliegue
        ]

        for umbral in UMBRALES_BAJA:
            sufijo = str(
                umbral
            ).replace(
                ".",
                "_",
            )

            columna_cruce = (
                f"cruza_baja_{sufijo}"
            )

            columna_retorno = (
                f"retorno_neto_en_cruce_{sufijo}"
            )

            cruzadas = datos_pliegue.loc[
                datos_pliegue[
                    columna_cruce
                ].astype(bool)
            ]

            perdedoras = datos_pliegue.loc[
                ~datos_pliegue[
                    "operacion_positiva"
                ].astype(bool)
            ]

            ganadoras = datos_pliegue.loc[
                datos_pliegue[
                    "operacion_positiva"
                ].astype(bool)
            ]

            resumenes.append(
                {
                    "pliegue": pliegue,
                    "umbral_baja": umbral,
                    "operaciones": len(
                        datos_pliegue
                    ),
                    "operaciones_con_cruce": len(
                        cruzadas
                    ),
                    "porcentaje_operaciones_con_cruce": float(
                        len(cruzadas)
                        / len(datos_pliegue)
                        * 100.0
                    ),
                    "porcentaje_perdedoras_con_cruce": float(
                        perdedoras[
                            columna_cruce
                        ].astype(bool).mean()
                        * 100.0
                        if len(perdedoras)
                        else 0.0
                    ),
                    "porcentaje_ganadoras_con_cruce": float(
                        ganadoras[
                            columna_cruce
                        ].astype(bool).mean()
                        * 100.0
                        if len(ganadoras)
                        else 0.0
                    ),
                    "precision_alerta_perdida": float(
                        (
                            ~cruzadas[
                                "operacion_positiva"
                            ].astype(bool)
                        ).mean()
                        * 100.0
                        if len(cruzadas)
                        else 0.0
                    ),
                    "retorno_neto_medio_fijo_cruzadas": float(
                        cruzadas[
                            "retorno_fijo_neto"
                        ].mean()
                        if len(cruzadas)
                        else 0.0
                    ),
                    "retorno_neto_medio_en_cruce": float(
                        cruzadas[
                            columna_retorno
                        ].mean()
                        if len(cruzadas)
                        else 0.0
                    ),
                }
            )

    resumen = pd.DataFrame(
        resumenes
    )

    ruta_resumen = (
        RUTA_DIAGNOSTICO
        / "resumen_diagnostico.csv"
    )

    resumen.to_csv(
        ruta_resumen,
        index=False,
        encoding="utf-8-sig",
    )

    detalle_json = {
        "version": VERSION_MODELO,
        "uso_2025": False,
        "uso_2026": False,
        "operaciones_analizadas": len(
            detalle
        ),
        "resumen": [
            {
                clave: convertir_para_json(
                    valor
                )
                for clave, valor in registro.items()
            }
            for registro in resumen.to_dict(
                orient="records"
            )
        ],
    }

    (
        RUTA_DIAGNOSTICO
        / "detalle_diagnostico.json"
    ).write_text(
        json.dumps(
            detalle_json,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\nArchivos generados:"
    )
    print(
        f"- {ruta_detalle}"
    )
    print(
        f"- {ruta_resumen}"
    )
    print(
        "\nNo se reentrenó ni modificó V4.1."
    )


if __name__ == "__main__":
    main()
