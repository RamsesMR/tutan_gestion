from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_6_macro.configuracion import (
    COSTE,
    HORIZONTE_SALIDA_MINUTOS,
    PLIEGUES_TEMPORALES,
    RUTA_PREDICCIONES,
    RUTA_RESULTADOS,
    SIMBOLO_OPERATIVO,
    UMBRALES_LABORATORIO,
    VARIANTES,
    VERSION_MODELO,
)


def factor_beneficio(
    retornos: np.ndarray,
) -> float:
    ganancias = retornos[
        retornos > 0
    ].sum()

    perdidas = -retornos[
        retornos < 0
    ].sum()

    if perdidas == 0:
        return (
            float("inf")
            if ganancias > 0
            else 0.0
        )

    return float(
        ganancias / perdidas
    )


def maximo_drawdown(
    retornos: np.ndarray,
) -> float:
    if retornos.size == 0:
        return 0.0

    curva = np.cumprod(
        1.0
        + np.clip(
            retornos,
            -0.999999,
            None,
        )
    )

    maximos = np.maximum.accumulate(
        curva
    )

    drawdowns = (
        curva / maximos - 1.0
    )

    return float(
        drawdowns.min()
    )


def construir_cruces(
    probabilidades: np.ndarray,
    umbral: float,
) -> np.ndarray:
    if probabilidades.size == 0:
        return np.asarray(
            [],
            dtype=bool,
        )

    sobre = (
        probabilidades
        >= umbral
    )

    anterior = np.empty_like(
        sobre
    )

    anterior[0] = False
    anterior[1:] = sobre[:-1]

    return sobre & ~anterior


def simular(
    datos: pd.DataFrame,
    umbral: float,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
]:
    datos = (
        datos.sort_values(
            "fecha_apertura"
        )
        .reset_index(drop=True)
        .copy()
    )

    fechas = pd.to_datetime(
        datos["fecha_apertura"],
        utc=True,
        errors="raise",
    )

    probabilidades = datos[
        "probabilidad_sube"
    ].to_numpy(dtype="float64")

    objetivos = datos[
        "objetivo_sube_4h"
    ].to_numpy(dtype="int8")

    cierres = datos[
        "precio_cierre"
    ].to_numpy(dtype="float64")

    cruces = construir_cruces(
        probabilidades,
        umbral,
    )

    indice_por_fecha = {
        fecha.value: indice
        for indice, fecha in enumerate(
            fechas
        )
    }

    operaciones = []

    siguiente_fecha_permitida = (
        pd.Timestamp.min.tz_localize(
            "UTC"
        )
    )

    for indice_entrada in np.flatnonzero(
        cruces
    ):
        fecha_entrada = fechas.iloc[
            indice_entrada
        ]

        if (
            fecha_entrada
            < siguiente_fecha_permitida
        ):
            continue

        fecha_salida = (
            fecha_entrada
            + pd.Timedelta(
                minutes=HORIZONTE_SALIDA_MINUTOS
            )
        )

        indice_salida = indice_por_fecha.get(
            fecha_salida.value
        )

        if indice_salida is None:
            continue

        retorno_bruto = (
            cierres[
                indice_salida
            ]
            / cierres[
                indice_entrada
            ]
            - 1.0
        )

        retorno_neto = (
            retorno_bruto
            - COSTE
        )

        operaciones.append(
            {
                "indice_entrada": int(
                    indice_entrada
                ),
                "fecha_entrada": fecha_entrada,
                "fecha_salida": fecha_salida,
                "objetivo": int(
                    objetivos[
                        indice_entrada
                    ]
                ),
                "probabilidad_entrada": float(
                    probabilidades[
                        indice_entrada
                    ]
                ),
                "retorno_bruto": float(
                    retorno_bruto
                ),
                "retorno_neto": float(
                    retorno_neto
                ),
            }
        )

        siguiente_fecha_permitida = (
            fecha_salida
        )

    if not operaciones:
        return (
            {
                "operaciones": 0,
                "precision": 0.0,
                "porcentaje_acierto": 0.0,
                "porcentaje_operaciones_positivas": 0.0,
                "retorno_neto_medio": 0.0,
                "retorno_neto_mediano": 0.0,
                "factor_beneficio": 0.0,
                "maximo_drawdown": 0.0,
                "retorno_compuesto": 0.0,
            },
            pd.DataFrame(),
        )

    tabla = pd.DataFrame(
        operaciones
    )

    retornos = tabla[
        "retorno_neto"
    ].to_numpy(dtype="float64")

    precision = float(
        tabla["objetivo"].mean()
    )

    metricas = {
        "operaciones": len(tabla),
        "precision": precision,
        "porcentaje_acierto": (
            precision * 100.0
        ),
        "porcentaje_operaciones_positivas": float(
            np.mean(
                retornos > 0
            )
            * 100.0
        ),
        "retorno_neto_medio": float(
            retornos.mean()
        ),
        "retorno_neto_mediano": float(
            np.median(
                retornos
            )
        ),
        "factor_beneficio": factor_beneficio(
            retornos
        ),
        "maximo_drawdown": maximo_drawdown(
            retornos
        ),
        "retorno_compuesto": float(
            np.prod(
                1.0
                + np.clip(
                    retornos,
                    -0.999999,
                    None,
                )
            )
            - 1.0
        ),
    }

    return metricas, tabla


def valor_json(
    valor: Any,
) -> Any:
    if isinstance(
        valor,
        np.integer,
    ):
        return int(valor)

    if isinstance(
        valor,
        np.floating,
    ):
        if np.isinf(valor):
            return "inf"

        return float(valor)

    return valor


def main() -> None:
    RUTA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    carpeta_operaciones = (
        RUTA_RESULTADOS
        / "operaciones_umbral_base"
    )

    carpeta_operaciones.mkdir(
        parents=True,
        exist_ok=True,
    )

    registros = []

    print(
        "\nEVALUACIÓN DE DESARROLLO V4.6-MACRO"
    )
    print("=" * 72)
    print(
        "Entrada por cruce, salida 480m y coste 0,10 %."
    )
    print(
        "La selección inicial usa el umbral congelado de cada base."
    )

    for nombre_variante, configuracion in VARIANTES.items():
        carpeta = (
            RUTA_PREDICCIONES
            / nombre_variante
        )

        if not carpeta.exists():
            print(
                f"Se omite {nombre_variante}: no tiene predicciones."
            )
            continue

        umbral_base = float(
            configuracion[
                "umbral_base"
            ]
        )

        for nombre_pliegue in PLIEGUES_TEMPORALES:
            ruta = (
                carpeta
                / f"{SIMBOLO_OPERATIVO}_{nombre_pliegue}.parquet"
            )

            if not ruta.exists():
                raise FileNotFoundError(
                    f"Falta una predicción: {ruta}"
                )

            datos = pd.read_parquet(
                ruta
            )

            tabla_base = None
            metricas_base = None

            for umbral in UMBRALES_LABORATORIO:
                metricas, operaciones = simular(
                    datos,
                    umbral,
                )

                registros.append(
                    {
                        "variante": nombre_variante,
                        "base": configuracion[
                            "base"
                        ],
                        "variable_macro": configuracion[
                            "variable_macro"
                        ],
                        "pliegue": nombre_pliegue,
                        "umbral": umbral,
                        "es_umbral_base": bool(
                            np.isclose(
                                umbral,
                                umbral_base,
                            )
                        ),
                        **metricas,
                    }
                )

                if np.isclose(
                    umbral,
                    umbral_base,
                ):
                    tabla_base = operaciones
                    metricas_base = metricas

            if metricas_base is None:
                raise RuntimeError(
                    "El umbral base no está en UMBRALES_LABORATORIO."
                )

            ruta_operaciones = (
                carpeta_operaciones
                / f"{nombre_variante}_{nombre_pliegue}.parquet"
            )

            tabla_base.to_parquet(
                ruta_operaciones,
                index=False,
            )

            print(
                f"\n{nombre_variante} | {nombre_pliegue} "
                f"| umbral {umbral_base:.2f}"
            )
            print("-" * 72)
            print(
                f"Operaciones: {metricas_base['operaciones']}"
            )
            print(
                f"Precisión: {metricas_base['precision']:.4f}"
            )
            print(
                "Porcentaje de acierto: "
                f"{metricas_base['porcentaje_acierto']:.2f} %"
            )
            print(
                "Operaciones positivas: "
                f"{metricas_base['porcentaje_operaciones_positivas']:.2f} %"
            )
            print(
                "Retorno neto medio: "
                f"{metricas_base['retorno_neto_medio']:.4%}"
            )
            print(
                "Factor de beneficio: "
                f"{metricas_base['factor_beneficio']:.4f}"
            )
            print(
                "Drawdown máximo: "
                f"{metricas_base['maximo_drawdown']:.2%}"
            )

    resultados = pd.DataFrame(
        registros
    )

    ruta_csv = (
        RUTA_RESULTADOS
        / "resultados_desarrollo.csv"
    )

    resultados.to_csv(
        ruta_csv,
        index=False,
        encoding="utf-8-sig",
    )

    ruta_json = (
        RUTA_RESULTADOS
        / "detalle_desarrollo.json"
    )

    ruta_json.write_text(
        json.dumps(
            {
                "version": VERSION_MODELO,
                "uso_2025_para_seleccion": False,
                "uso_2026_para_seleccion": False,
                "resultados": [
                    {
                        clave: valor_json(valor)
                        for clave, valor in registro.items()
                    }
                    for registro in registros
                ],
            },
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\nEVALUACIÓN V4.6-MACRO COMPLETADA"
    )
    print(f"- {ruta_csv}")


if __name__ == "__main__":
    main()
