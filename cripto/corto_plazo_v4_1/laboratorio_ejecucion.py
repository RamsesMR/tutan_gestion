from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_1.configuracion import (
    BARRERAS,
    COSTES,
    MODOS_ENTRADA,
    RUTA_PREDICCIONES_DESARROLLO,
    RUTA_RESULTADOS_LABORATORIO,
    SIMBOLO,
    SALIDAS_FIJAS_MINUTOS,
    UMBRALES_SUBE,
    UMBRALES_VETO_BAJA,
)
from cripto.corto_plazo_v4_1.utilidades import (
    calcular_salida_barrera,
    calcular_salida_fija,
    construir_mascara_entrada,
    convertir_fechas_ns,
    evaluar_estrategia,
)


PLIEGUES = (
    "validacion_2023",
    "validacion_2024",
)


def cargar_pliegue(
    nombre_pliegue: str,
):
    """Carga un Parquet de probabilidades fuera de muestra."""

    ruta = (
        RUTA_PREDICCIONES_DESARROLLO
        / f"{SIMBOLO}_{nombre_pliegue}.parquet"
    )

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe: {ruta}"
        )

    datos = pd.read_parquet(
        ruta
    )

    return {
        "fechas_ns": convertir_fechas_ns(
            datos[
                "fecha_apertura"
            ]
        ),
        "aperturas": datos[
            "precio_apertura"
        ].to_numpy(
            dtype="float64"
        ),
        "maximos": datos[
            "precio_maximo"
        ].to_numpy(
            dtype="float64"
        ),
        "minimos": datos[
            "precio_minimo"
        ].to_numpy(
            dtype="float64"
        ),
        "cierres": datos[
            "precio_cierre"
        ].to_numpy(
            dtype="float64"
        ),
        "probabilidades_sube": datos[
            "probabilidad_sube"
        ].to_numpy(
            dtype="float64"
        ),
        "probabilidades_baja": datos[
            "probabilidad_baja"
        ].to_numpy(
            dtype="float64"
        ),
        "objetivo_sube_4h": datos[
            "objetivo_sube_4h"
        ].to_numpy(
            dtype="int8"
        ),
    }


def construir_salidas(
    datos: dict,
):
    """Precalcula retornos y duraciones de todas las salidas."""

    salidas = []

    for minutos in SALIDAS_FIJAS_MINUTOS:
        retornos, duraciones = calcular_salida_fija(
            cierres=datos[
                "cierres"
            ],
            minutos=minutos,
        )

        salidas.append(
            {
                "tipo_salida": "fija",
                "nombre_salida": f"fija_{minutos}m",
                "horizonte_salida": minutos,
                "retornos": retornos,
                "duraciones": duraciones,
            }
        )

    for barrera in BARRERAS:
        retornos, duraciones = calcular_salida_barrera(
            cierres=datos[
                "cierres"
            ],
            maximos=datos[
                "maximos"
            ],
            minimos=datos[
                "minimos"
            ],
            take_profit=float(
                barrera[
                    "take_profit"
                ]
            ),
            stop_loss=float(
                barrera[
                    "stop_loss"
                ]
            ),
            horizonte=int(
                barrera[
                    "horizonte"
                ]
            ),
        )

        salidas.append(
            {
                "tipo_salida": "barrera",
                "nombre_salida": barrera[
                    "nombre"
                ],
                "horizonte_salida": int(
                    barrera[
                        "horizonte"
                    ]
                ),
                "retornos": retornos,
                "duraciones": duraciones,
            }
        )

    return salidas


def main() -> None:
    """Explora reglas de entrada y salida usando solo 2023/2024."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modo-rapido",
        action="store_true",
        help="Prueba una rejilla menor para validar el flujo.",
    )

    argumentos = parser.parse_args()

    umbrales_sube = (
        (
            0.46,
            0.48,
            0.50,
            0.52,
            0.54,
            0.56,
        )
        if argumentos.modo_rapido
        else UMBRALES_SUBE
    )

    modos = (
        (
            "nivel",
            "cruce",
            "confirmacion_3m",
            "creciente_3m",
        )
        if argumentos.modo_rapido
        else MODOS_ENTRADA
    )

    vetos = (
        (
            None,
            0.30,
            0.40,
            0.50,
        )
        if argumentos.modo_rapido
        else UMBRALES_VETO_BAJA
    )

    registros = []

    print(
        "\nLABORATORIO DE EJECUCIÓN V4.1"
    )
    print("=" * 72)
    print(
        "Busca reglas robustas con 2023 y 2024."
    )
    print(
        "La precisión y el porcentaje de acierto se guardan explícitamente."
    )

    for nombre_pliegue in PLIEGUES:
        print(
            f"\nProcesando {nombre_pliegue}..."
        )

        datos = cargar_pliegue(
            nombre_pliegue
        )

        salidas = construir_salidas(
            datos
        )

        total_combinaciones = (
            len(
                umbrales_sube
            )
            * len(
                modos
            )
            * len(
                vetos
            )
            * len(
                salidas
            )
            * len(
                COSTES
            )
        )

        procesadas = 0

        for (
            umbral_sube,
            modo,
            veto,
        ) in itertools.product(
            umbrales_sube,
            modos,
            vetos,
        ):
            mascara = construir_mascara_entrada(
                probabilidades_sube=datos[
                    "probabilidades_sube"
                ],
                probabilidades_baja=datos[
                    "probabilidades_baja"
                ],
                umbral_sube=float(
                    umbral_sube
                ),
                umbral_veto_baja=veto,
                modo=modo,
            )

            for salida in salidas:
                for coste in COSTES:
                    metricas = evaluar_estrategia(
                        fechas_ns=datos[
                            "fechas_ns"
                        ],
                        mascara_entrada=mascara,
                        retornos_brutos=salida[
                            "retornos"
                        ],
                        minutos_salida=salida[
                            "duraciones"
                        ],
                        objetivo_sube_4h=datos[
                            "objetivo_sube_4h"
                        ],
                        coste=float(
                            coste
                        ),
                    )

                    registros.append(
                        {
                            "pliegue": nombre_pliegue,
                            "umbral_sube": float(
                                umbral_sube
                            ),
                            "modo_entrada": modo,
                            "umbral_veto_baja": (
                                np.nan
                                if veto is None
                                else float(
                                    veto
                                )
                            ),
                            "usa_veto_baja": (
                                veto is not None
                            ),
                            "tipo_salida": salida[
                                "tipo_salida"
                            ],
                            "nombre_salida": salida[
                                "nombre_salida"
                            ],
                            "horizonte_salida": salida[
                                "horizonte_salida"
                            ],
                            "coste": float(
                                coste
                            ),
                            **metricas,
                        }
                    )

                    procesadas += 1

        print(
            f"Combinaciones: {procesadas:,} de {total_combinaciones:,}".replace(
                ",",
                ".",
            )
        )

    resultados = pd.DataFrame(
        registros
    )

    RUTA_RESULTADOS_LABORATORIO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resultados.to_csv(
        RUTA_RESULTADOS_LABORATORIO,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nLaboratorio finalizado:"
    )
    print(
        f"- {RUTA_RESULTADOS_LABORATORIO}"
    )


if __name__ == "__main__":
    main()
