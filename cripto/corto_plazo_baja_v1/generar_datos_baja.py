from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cripto.corto_plazo_baja_v1.configuracion import (
    COLUMNAS_COMPLETAS_101,
    MAXIMO_CONTEXTO_FUTURO,
    OBJETIVOS,
    PERIODOS_DESARROLLO,
    PERIODOS_FUENTE,
    RUTA_DATOS_BAJA,
    RUTA_DATOS_V5,
    SIMBOLO,
)


COLUMNAS_OHLC = (
    "fecha_apertura",
    "apertura",
    "maximo",
    "minimo",
    "cierre",
)


def ruta_fuente_v5(
    simbolo: str,
    desde: str,
    hasta: str,
) -> Path:
    return (
        RUTA_DATOS_V5
        / f"{simbolo}_1m_v5_{desde}_{hasta}.parquet"
    )


def ruta_salida_baja(
    simbolo: str,
    desde: str,
    hasta: str,
) -> Path:
    return (
        RUTA_DATOS_BAJA
        / f"{simbolo}_1m_baja_v1_{desde}_{hasta}.parquet"
    )


def cargar_fuente(
    simbolo: str,
    desde: str,
    hasta: str,
) -> pd.DataFrame:
    ruta = ruta_fuente_v5(
        simbolo,
        desde,
        hasta,
    )

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe la fuente V5: {ruta}\n"
            "Ejecuta primero la generación de datos V5."
        )

    columnas = [
        *COLUMNAS_OHLC,
        *COLUMNAS_COMPLETAS_101,
    ]

    datos = pd.read_parquet(
        ruta,
        columns=list(
            dict.fromkeys(columnas)
        ),
    )

    datos["fecha_apertura"] = pd.to_datetime(
        datos["fecha_apertura"],
        utc=True,
        errors="raise",
    )

    for columna in (
        "apertura",
        "maximo",
        "minimo",
        "cierre",
    ):
        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="raise",
        )

    datos = (
        datos.drop_duplicates(
            subset=["fecha_apertura"],
            keep="last",
        )
        .sort_values("fecha_apertura")
        .reset_index(drop=True)
    )

    return datos


def grupos_continuos(
    datos: pd.DataFrame,
) -> pd.Series:
    diferencia = datos[
        "fecha_apertura"
    ].diff()

    nuevo_segmento = (
        diferencia.isna()
        | (
            diferencia
            != pd.Timedelta(minutes=1)
        )
    )

    return nuevo_segmento.cumsum()


def agregar_objetivo_cierre_segmento(
    datos: pd.DataFrame,
    nombre: str,
    umbral_baja: float,
    horizonte_objetivo: int,
    horizonte_salida: int,
) -> pd.DataFrame:
    datos = datos.reset_index(
        drop=True
    ).copy()

    cierre = datos[
        "cierre"
    ].to_numpy(
        dtype="float64"
    )

    n = len(datos)

    objetivo = np.full(
        n,
        np.nan,
        dtype="float64",
    )

    retorno_salida = np.full(
        n,
        np.nan,
        dtype="float64",
    )

    minutos_salida = np.full(
        n,
        -1,
        dtype="int16",
    )

    limite_objetivo = (
        n - horizonte_objetivo
    )

    if limite_objetivo > 0:
        indices = np.arange(
            limite_objetivo
        )

        rendimiento_futuro = (
            cierre[
                indices
                + horizonte_objetivo
            ]
            / cierre[indices]
            - 1.0
        )

        objetivo[indices] = (
            rendimiento_futuro
            <= -umbral_baja
        ).astype(
            "int8"
        )

    limite_salida = (
        n - horizonte_salida
    )

    if limite_salida > 0:
        indices = np.arange(
            limite_salida
        )

        # PnL lineal de una posición corta USD-M:
        # (entrada - salida) / entrada.
        retorno_salida[indices] = (
            cierre[indices]
            - cierre[
                indices
                + horizonte_salida
            ]
        ) / cierre[indices]

        minutos_salida[indices] = (
            horizonte_salida
        )

    valido = (
        np.isfinite(objetivo)
        & np.isfinite(retorno_salida)
    )

    resultado = np.full(
        n,
        None,
        dtype=object,
    )

    resultado[
        valido
        & (
            objetivo == 1
        )
    ] = "BAJA"

    resultado[
        valido
        & (
            objetivo == 0
        )
    ] = "NO_BAJA"

    datos[
        f"objetivo_{nombre}"
    ] = objetivo

    datos[
        f"resultado_{nombre}"
    ] = resultado

    datos[
        f"retorno_salida_{nombre}"
    ] = retorno_salida.astype(
        "float32"
    )

    datos[
        f"minutos_salida_{nombre}"
    ] = minutos_salida

    horizonte_maximo = max(
        horizonte_objetivo,
        horizonte_salida,
    )

    datos[
        f"fecha_fin_horizonte_{nombre}"
    ] = (
        datos["fecha_apertura"]
        + pd.to_timedelta(
            horizonte_maximo,
            unit="m",
        )
    )

    return datos


def agregar_objetivo_barreras_segmento(
    datos: pd.DataFrame,
    nombre: str,
    take_profit: float,
    stop_loss: float,
    horizonte: int,
) -> pd.DataFrame:
    """Etiqueta una operación short. En empate intravela gana el stop."""

    datos = datos.reset_index(
        drop=True
    ).copy()

    cierre = datos[
        "cierre"
    ].to_numpy(
        dtype="float64"
    )

    maximo = datos[
        "maximo"
    ].to_numpy(
        dtype="float64"
    )

    minimo = datos[
        "minimo"
    ].to_numpy(
        dtype="float64"
    )

    n = len(datos)

    codigo = np.full(
        n,
        -1,
        dtype="int8",
    )

    minutos_salida = np.full(
        n,
        -1,
        dtype="int16",
    )

    retorno_salida = np.full(
        n,
        np.nan,
        dtype="float64",
    )

    for paso in range(
        1,
        horizonte + 1,
    ):
        limite = n - paso

        if limite <= 0:
            break

        activos = (
            codigo[:limite] == -1
        )

        toca_stop = (
            maximo[paso:]
            >= cierre[:limite]
            * (
                1.0
                + stop_loss
            )
        )

        toca_take = (
            minimo[paso:]
            <= cierre[:limite]
            * (
                1.0
                - take_profit
            )
        )

        # Conservador: cuando ambas barreras se tocan en la misma
        # vela de un minuto, se asigna primero el stop.
        perdedoras = (
            activos
            & toca_stop
        )

        ganadoras = (
            activos
            & ~toca_stop
            & toca_take
        )

        indices_perdedoras = (
            np.flatnonzero(
                perdedoras
            )
        )

        indices_ganadoras = (
            np.flatnonzero(
                ganadoras
            )
        )

        codigo[
            indices_perdedoras
        ] = 0

        minutos_salida[
            indices_perdedoras
        ] = paso

        retorno_salida[
            indices_perdedoras
        ] = -stop_loss

        codigo[
            indices_ganadoras
        ] = 1

        minutos_salida[
            indices_ganadoras
        ] = paso

        retorno_salida[
            indices_ganadoras
        ] = take_profit

    indices_completos = np.arange(
        0,
        max(
            n - horizonte,
            0,
        ),
    )

    sin_barrera = (
        codigo[
            indices_completos
        ]
        == -1
    )

    indices_sin_barrera = (
        indices_completos[
            sin_barrera
        ]
    )

    codigo[
        indices_sin_barrera
    ] = 2

    minutos_salida[
        indices_sin_barrera
    ] = horizonte

    retorno_salida[
        indices_sin_barrera
    ] = (
        cierre[
            indices_sin_barrera
        ]
        - cierre[
            indices_sin_barrera
            + horizonte
        ]
    ) / cierre[
        indices_sin_barrera
    ]

    etiquetas = np.full(
        n,
        None,
        dtype=object,
    )

    etiquetas[
        codigo == 0
    ] = "PERDEDORA"

    etiquetas[
        codigo == 1
    ] = "GANADORA"

    etiquetas[
        codigo == 2
    ] = "SIN_RESULTADO"

    datos[
        f"objetivo_{nombre}"
    ] = np.where(
        codigo == 1,
        1.0,
        np.where(
            codigo >= 0,
            0.0,
            np.nan,
        ),
    )

    datos[
        f"resultado_{nombre}"
    ] = etiquetas

    datos[
        f"retorno_salida_{nombre}"
    ] = retorno_salida.astype(
        "float32"
    )

    datos[
        f"minutos_salida_{nombre}"
    ] = minutos_salida

    datos[
        f"fecha_fin_horizonte_{nombre}"
    ] = (
        datos["fecha_apertura"]
        + pd.to_timedelta(
            horizonte,
            unit="m",
        )
    )

    return datos


def agregar_objetivos(
    datos: pd.DataFrame,
) -> pd.DataFrame:
    datos = (
        datos.sort_values(
            "fecha_apertura"
        )
        .reset_index(drop=True)
    )

    partes = []

    for _, segmento in datos.groupby(
        grupos_continuos(datos),
        sort=False,
    ):
        salida = segmento.copy()

        for nombre, configuracion in (
            OBJETIVOS.items()
        ):
            tipo = str(
                configuracion["tipo"]
            )

            if tipo == "cierre":
                salida = (
                    agregar_objetivo_cierre_segmento(
                        datos=salida,
                        nombre=nombre,
                        umbral_baja=float(
                            configuracion[
                                "umbral_baja"
                            ]
                        ),
                        horizonte_objetivo=int(
                            configuracion[
                                "horizonte_objetivo_minutos"
                            ]
                        ),
                        horizonte_salida=int(
                            configuracion[
                                "horizonte_salida_minutos"
                            ]
                        ),
                    )
                )
            elif tipo == "barreras_short":
                salida = (
                    agregar_objetivo_barreras_segmento(
                        datos=salida,
                        nombre=nombre,
                        take_profit=float(
                            configuracion[
                                "take_profit"
                            ]
                        ),
                        stop_loss=float(
                            configuracion[
                                "stop_loss"
                            ]
                        ),
                        horizonte=int(
                            configuracion[
                                "horizonte_minutos"
                            ]
                        ),
                    )
                )
            else:
                raise ValueError(
                    f"Tipo de objetivo desconocido: {tipo}"
                )

        partes.append(
            salida
        )

    return (
        pd.concat(
            partes,
            ignore_index=True,
        )
        .sort_values(
            "fecha_apertura"
        )
        .reset_index(drop=True)
    )


def procesar_periodo(
    indice: int,
    sobrescribir: bool,
) -> dict[str, object]:
    desde, hasta = (
        PERIODOS_DESARROLLO[
            indice
        ]
    )

    ruta_salida = ruta_salida_baja(
        SIMBOLO,
        desde,
        hasta,
    )

    if (
        ruta_salida.exists()
        and not sobrescribir
    ):
        datos = pd.read_parquet(
            ruta_salida,
            columns=[
                "fecha_apertura",
            ],
        )

        return {
            "simbolo": SIMBOLO,
            "desde": desde,
            "hasta": hasta,
            "estado": "conservado",
            "filas": len(datos),
            "ruta": str(ruta_salida),
        }

    actual = cargar_fuente(
        SIMBOLO,
        desde,
        hasta,
    )

    partes = [
        actual,
    ]

    siguiente_indice = (
        indice + 1
    )

    if siguiente_indice < len(
        PERIODOS_FUENTE
    ):
        (
            desde_siguiente,
            hasta_siguiente,
        ) = PERIODOS_FUENTE[
            siguiente_indice
        ]

        siguiente = cargar_fuente(
            SIMBOLO,
            desde_siguiente,
            hasta_siguiente,
        ).head(
            MAXIMO_CONTEXTO_FUTURO
        )

        partes.append(
            siguiente
        )

    combinado = (
        pd.concat(
            partes,
            ignore_index=True,
        )
        .drop_duplicates(
            subset=[
                "fecha_apertura",
            ],
            keep="last",
        )
        .sort_values(
            "fecha_apertura"
        )
        .reset_index(drop=True)
    )

    combinado = agregar_objetivos(
        combinado
    )

    mascara_periodo = (
        (
            combinado[
                "fecha_apertura"
            ]
            >= pd.Timestamp(
                desde,
                tz="UTC",
            )
        )
        & (
            combinado[
                "fecha_apertura"
            ]
            < pd.Timestamp(
                hasta,
                tz="UTC",
            )
        )
    )

    salida = (
        combinado.loc[
            mascara_periodo
        ]
        .copy()
        .reset_index(drop=True)
    )

    matriz = salida[
        list(
            COLUMNAS_COMPLETAS_101
        )
    ].to_numpy(
        dtype="float64"
    )

    mascara_finitas = np.isfinite(
        matriz
    ).all(axis=1)

    filas_no_finitas = int(
        (
            ~mascara_finitas
        ).sum()
    )

    salida = (
        salida.loc[
            mascara_finitas
        ]
        .reset_index(drop=True)
    )

    if salida.empty:
        raise RuntimeError(
            f"No quedaron filas válidas en {desde} a {hasta}."
        )

    RUTA_DATOS_BAJA.mkdir(
        parents=True,
        exist_ok=True,
    )

    salida.to_parquet(
        ruta_salida,
        index=False,
    )

    resumen = {
        "simbolo": SIMBOLO,
        "desde": desde,
        "hasta": hasta,
        "estado": "generado",
        "filas_fuente": len(actual),
        "filas_salida": len(salida),
        "filas_no_finitas": filas_no_finitas,
        "ruta": str(ruta_salida),
    }

    del actual
    del combinado
    del salida
    gc.collect()

    return resumen


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sobrescribir",
        action="store_true",
    )

    argumentos = parser.parse_args()

    print(
        "\nGENERACIÓN DE DATOS BAJA V1"
    )
    print("=" * 72)
    print(
        f"Fuente: {RUTA_DATOS_V5}"
    )
    print(
        f"Variables: {len(COLUMNAS_COMPLETAS_101)}"
    )
    print(
        f"Objetivos: {len(OBJETIVOS)}"
    )
    print(
        "2025 se usa solamente como contexto futuro de 2024."
    )

    resumenes = []

    for indice in range(
        len(
            PERIODOS_DESARROLLO
        )
    ):
        resumen = procesar_periodo(
            indice=indice,
            sobrescribir=(
                argumentos.sobrescribir
            ),
        )

        resumenes.append(
            resumen
        )

        print(
            f"- {resumen['desde']} → {resumen['hasta']}: "
            f"{int(resumen['filas'] if 'filas' in resumen else resumen['filas_salida']):,} "
            f"filas ({resumen['estado']})".replace(
                ",",
                ".",
            )
        )

    ruta_resumen = (
        RUTA_DATOS_BAJA
        / "resumen_generacion_baja_v1.json"
    )

    ruta_resumen.write_text(
        json.dumps(
            {
                "fecha_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "version": "baja_v1_short",
                "uso_2025_para_seleccion": False,
                "uso_2026_para_seleccion": False,
                "resumenes": resumenes,
            },
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\nDATOS BAJA V1 GENERADOS"
    )
    print(
        f"- {ruta_resumen}"
    )


if __name__ == "__main__":
    main()
