from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cripto.corto_plazo_v5.configuracion import (
    ALIAS_COLUMNAS,
    COLUMNAS_MODELO_V2A,
    COLUMNAS_TECNICAS_V5,
    OBJETIVOS_BARRERAS,
    PERIODOS,
    RUTA_DATOS_V2,
    RUTA_DATOS_V5,
)
from cripto.corto_plazo_v5.utilidades import (
    resolver_columna,
)


MAXIMO_HISTORICO = 1440
MAXIMO_FUTURO = max(
    configuracion["horizonte_minutos"]
    for configuracion in OBJETIVOS_BARRERAS.values()
)


def construir_ruta_v2(
    simbolo: str,
    desde: str,
    hasta: str,
) -> Path:
    """Construye una ruta V2A."""

    return (
        RUTA_DATOS_V2
        / (
            f"{simbolo}_1m_4h_"
            f"{desde}_{hasta}_variables.parquet"
        )
    )


def construir_ruta_v5(
    simbolo: str,
    desde: str,
    hasta: str,
) -> Path:
    """Construye una ruta V5."""

    return (
        RUTA_DATOS_V5
        / (
            f"{simbolo}_1m_v5_"
            f"{desde}_{hasta}.parquet"
        )
    )


def encontrar_archivo_precios(
    carpeta: Path,
    simbolo: str,
    desde: str,
    hasta: str,
) -> Path:
    """Busca un Parquet de precios por símbolo y periodo."""

    candidatos = list(
        carpeta.rglob(
            "*.parquet"
        )
    )

    puntuados = []

    for ruta in candidatos:
        nombre = ruta.name.lower()

        puntuacion = 0

        if simbolo.lower() in nombre:
            puntuacion += 4

        if desde in nombre:
            puntuacion += 2

        if hasta in nombre:
            puntuacion += 2

        if "1m" in nombre:
            puntuacion += 1

        if puntuacion >= 7:
            puntuados.append(
                (
                    puntuacion,
                    ruta,
                )
            )

    if not puntuados:
        raise FileNotFoundError(
            "No se encontró un Parquet de precios para "
            f"{simbolo} {desde} {hasta} dentro de {carpeta}."
        )

    puntuados.sort(
        key=lambda elemento: (
            -elemento[0],
            len(
                str(
                    elemento[1]
                )
            ),
        )
    )

    return puntuados[
        0
    ][
        1
    ]


def canonizar_ohlc(
    datos: pd.DataFrame,
) -> pd.DataFrame:
    """Renombra las columnas OHLC a nombres canónicos."""

    renombres = {}

    for canonica, alias in ALIAS_COLUMNAS.items():
        encontrada = resolver_columna(
            datos.columns,
            alias,
        )

        if encontrada is None:
            raise ValueError(
                f"No se encontró la columna {canonica}. "
                f"Alias admitidos: {alias}"
            )

        renombres[
            encontrada
        ] = canonica

    datos = datos.rename(
        columns=renombres
    )

    columnas = [
        "fecha_apertura",
        "apertura",
        "maximo",
        "minimo",
        "cierre",
    ]

    datos = datos[
        columnas
    ].copy()

    datos["fecha_apertura"] = pd.to_datetime(
        datos["fecha_apertura"],
        utc=True,
        errors="coerce",
    )

    for columna in (
        "apertura",
        "maximo",
        "minimo",
        "cierre",
    ):
        datos[
            columna
        ] = pd.to_numeric(
            datos[
                columna
            ],
            errors="coerce",
        )

    datos = datos.dropna(
        subset=columnas
    )

    datos = (
        datos
        .drop_duplicates(
            subset=[
                "fecha_apertura",
            ],
            keep="last",
        )
        .sort_values(
            "fecha_apertura"
        )
        .reset_index(
            drop=True
        )
    )

    return datos


def cargar_periodo(
    simbolo: str,
    desde: str,
    hasta: str,
    ruta_precios: Path | None,
) -> pd.DataFrame:
    """Carga variables V2A y agrega OHLC si fuera necesario."""

    ruta_v2 = construir_ruta_v2(
        simbolo=simbolo,
        desde=desde,
        hasta=hasta,
    )

    if not ruta_v2.exists():
        raise FileNotFoundError(
            f"No existe: {ruta_v2}"
        )

    datos = pd.read_parquet(
        ruta_v2
    )

    fecha_v2 = resolver_columna(
        datos.columns,
        ALIAS_COLUMNAS[
            "fecha_apertura"
        ],
    )

    if fecha_v2 is None:
        raise ValueError(
            f"{ruta_v2.name} no contiene fecha_apertura."
        )

    if fecha_v2 != "fecha_apertura":
        datos = datos.rename(
            columns={
                fecha_v2: "fecha_apertura",
            }
        )

    datos["fecha_apertura"] = pd.to_datetime(
        datos["fecha_apertura"],
        utc=True,
        errors="coerce",
    )

    columnas_ohlc_en_v2 = {
        canonica: resolver_columna(
            datos.columns,
            alias,
        )
        for canonica, alias in ALIAS_COLUMNAS.items()
        if canonica != "fecha_apertura"
    }

    if all(
        valor is not None
        for valor in columnas_ohlc_en_v2.values()
    ):
        renombres = {
            encontrada: canonica
            for canonica, encontrada in columnas_ohlc_en_v2.items()
            if encontrada != canonica
        }

        datos = datos.rename(
            columns=renombres
        )

    else:
        if ruta_precios is None:
            faltantes = [
                canonica
                for canonica, encontrada in columnas_ohlc_en_v2.items()
                if encontrada is None
            ]

            raise ValueError(
                "La V2A no contiene OHLC. Faltan: "
                + ", ".join(
                    faltantes
                )
                + ". Ejecuta con --ruta-precios."
            )

        archivo_precios = encontrar_archivo_precios(
            carpeta=ruta_precios,
            simbolo=simbolo,
            desde=desde,
            hasta=hasta,
        )

        precios = canonizar_ohlc(
            pd.read_parquet(
                archivo_precios
            )
        )

        datos = datos.merge(
            precios,
            on="fecha_apertura",
            how="inner",
            validate="one_to_one",
        )

    columnas_necesarias = [
        "fecha_apertura",
        "apertura",
        "maximo",
        "minimo",
        "cierre",
        *COLUMNAS_MODELO_V2A,
    ]

    faltantes = [
        columna
        for columna in columnas_necesarias
        if columna not in datos.columns
    ]

    if faltantes:
        raise ValueError(
            f"{ruta_v2.name}: faltan columnas: "
            + ", ".join(
                faltantes
            )
        )

    datos = (
        datos[
            columnas_necesarias
        ]
        .sort_values(
            "fecha_apertura"
        )
        .reset_index(
            drop=True
        )
    )

    return datos


def regresion_rodante(
    valores: np.ndarray,
    ventana: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula pendiente y R² con fórmulas de sumas acumuladas."""

    n = len(
        valores
    )

    pendiente = np.full(
        n,
        np.nan,
        dtype="float64",
    )

    r2 = np.full(
        n,
        np.nan,
        dtype="float64",
    )

    if n < ventana:
        return pendiente, r2

    indices_globales = np.arange(
        n,
        dtype="float64",
    )

    acumulada_y = np.concatenate(
        [
            np.array(
                [
                    0.0,
                ]
            ),
            np.cumsum(
                valores,
                dtype="float64",
            ),
        ]
    )

    acumulada_y2 = np.concatenate(
        [
            np.array(
                [
                    0.0,
                ]
            ),
            np.cumsum(
                valores * valores,
                dtype="float64",
            ),
        ]
    )

    acumulada_iy = np.concatenate(
        [
            np.array(
                [
                    0.0,
                ]
            ),
            np.cumsum(
                indices_globales * valores,
                dtype="float64",
            ),
        ]
    )

    finales = np.arange(
        ventana,
        n + 1,
    )

    inicios = finales - ventana

    suma_y = (
        acumulada_y[
            finales
        ]
        - acumulada_y[
            inicios
        ]
    )

    suma_y2 = (
        acumulada_y2[
            finales
        ]
        - acumulada_y2[
            inicios
        ]
    )

    suma_iy_global = (
        acumulada_iy[
            finales
        ]
        - acumulada_iy[
            inicios
        ]
    )

    suma_x = (
        ventana
        * (
            ventana
            - 1
        )
        / 2
    )

    suma_x2 = (
        ventana
        * (
            ventana
            - 1
        )
        * (
            2
            * ventana
            - 1
        )
        / 6
    )

    suma_xy = (
        suma_iy_global
        - inicios
        * suma_y
    )

    denominador_x = (
        ventana
        * suma_x2
        - suma_x
        * suma_x
    )

    numerador = (
        ventana
        * suma_xy
        - suma_x
        * suma_y
    )

    pendientes = (
        numerador
        / denominador_x
    )

    denominador_y = (
        ventana
        * suma_y2
        - suma_y
        * suma_y
    )

    denominador_r2 = (
        denominador_x
        * denominador_y
    )

    r2_validos = np.divide(
        numerador * numerador,
        denominador_r2,
        out=np.zeros_like(
            numerador
        ),
        where=denominador_r2 > 0,
    )

    pendiente[
        ventana - 1:
    ] = pendientes

    r2[
        ventana - 1:
    ] = np.clip(
        r2_validos,
        0.0,
        1.0,
    )

    return pendiente, r2


def contar_consecutivas(
    mascara: np.ndarray,
) -> np.ndarray:
    """Cuenta rachas consecutivas verdaderas."""

    salida = np.zeros(
        len(
            mascara
        ),
        dtype="int16",
    )

    contador = 0

    for indice, valor in enumerate(
        mascara
    ):
        if valor:
            contador += 1
        else:
            contador = 0

        salida[
            indice
        ] = contador

    return salida


def _agregar_variables_tecnicas_segmento(
    datos: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula variables causales usando solo presente y pasado."""

    cierre = datos[
        "cierre"
    ].to_numpy(
        dtype="float64"
    )

    apertura = datos[
        "apertura"
    ].to_numpy(
        dtype="float64"
    )

    log_cierre = np.log(
        cierre
    )

    retorno_1m = np.empty_like(
        log_cierre
    )

    retorno_1m[
        0
    ] = np.nan

    retorno_1m[
        1:
    ] = np.diff(
        log_cierre
    )

    for ventana in (
        15,
        60,
        240,
    ):
        pendiente, r2 = regresion_rodante(
            valores=log_cierre,
            ventana=ventana,
        )

        datos[
            f"pendiente_log_{ventana}m"
        ] = pendiente.astype(
            "float32"
        )

        datos[
            f"r2_tendencia_{ventana}m"
        ] = r2.astype(
            "float32"
        )

    series_cierre = pd.Series(
        cierre
    )

    velocidades = {}

    for ventana in (
        5,
        15,
        60,
        240,
    ):
        velocidades[
            ventana
        ] = (
            np.log(
                series_cierre
                / series_cierre.shift(
                    ventana
                )
            )
            / ventana
        )

    datos[
        "aceleracion_5m_15m"
    ] = (
        velocidades[
            5
        ]
        - velocidades[
            15
        ]
    ).astype(
        "float32"
    )

    datos[
        "aceleracion_15m_60m"
    ] = (
        velocidades[
            15
        ]
        - velocidades[
            60
        ]
    ).astype(
        "float32"
    )

    datos[
        "aceleracion_60m_240m"
    ] = (
        velocidades[
            60
        ]
        - velocidades[
            240
        ]
    ).astype(
        "float32"
    )

    cierre_serie = pd.Series(
        cierre
    )

    for ventana in (
        60,
        240,
        1440,
    ):
        maximo_previo = (
            cierre_serie
            .shift(
                1
            )
            .rolling(
                ventana,
                min_periods=ventana,
            )
            .max()
        )

        minimo_previo = (
            cierre_serie
            .shift(
                1
            )
            .rolling(
                ventana,
                min_periods=ventana,
            )
            .min()
        )

        rango = (
            maximo_previo
            - minimo_previo
        )

        datos[
            f"distancia_maximo_{ventana}m"
        ] = (
            cierre_serie
            / maximo_previo
            - 1.0
        ).astype(
            "float32"
        )

        datos[
            f"distancia_minimo_{ventana}m"
        ] = (
            cierre_serie
            / minimo_previo
            - 1.0
        ).astype(
            "float32"
        )

        datos[
            f"posicion_rango_{ventana}m"
        ] = (
            (
                cierre_serie
                - minimo_previo
            )
            / rango.replace(
                0.0,
                np.nan,
            )
        ).astype(
            "float32"
        )

        if ventana in (
            60,
            240,
        ):
            datos[
                f"ruptura_maximo_{ventana}m"
            ] = (
                cierre_serie
                > maximo_previo
            ).astype(
                "int8"
            )

            datos[
                f"ruptura_minimo_{ventana}m"
            ] = (
                cierre_serie
                < minimo_previo
            ).astype(
                "int8"
            )

    retorno_serie = pd.Series(
        retorno_1m
    )

    cuadrado_positivo = retorno_serie.clip(
        lower=0.0
    ) ** 2

    cuadrado_negativo = retorno_serie.clip(
        upper=0.0
    ) ** 2

    for ventana in (
        60,
        240,
    ):
        semi_positiva = np.sqrt(
            cuadrado_positivo.rolling(
                ventana,
                min_periods=ventana,
            ).mean()
        )

        semi_negativa = np.sqrt(
            cuadrado_negativo.rolling(
                ventana,
                min_periods=ventana,
            ).mean()
        )

        datos[
            f"semivolatilidad_positiva_{ventana}m"
        ] = semi_positiva.astype(
            "float32"
        )

        datos[
            f"semivolatilidad_negativa_{ventana}m"
        ] = semi_negativa.astype(
            "float32"
        )

        datos[
            f"ratio_semivolatilidad_{ventana}m"
        ] = (
            semi_positiva
            / semi_negativa.replace(
                0.0,
                np.nan,
            )
        ).astype(
            "float32"
        )

    vela_positiva = cierre > apertura
    vela_negativa = cierre < apertura

    cuerpo_relativo = (
        cierre
        - apertura
    ) / apertura

    cuerpos_positivos = np.clip(
        cuerpo_relativo,
        0.0,
        None,
    )

    cuerpos_negativos = np.clip(
        -cuerpo_relativo,
        0.0,
        None,
    )

    for ventana in (
        15,
        60,
    ):
        datos[
            f"porcentaje_velas_positivas_{ventana}m"
        ] = (
            pd.Series(
                vela_positiva.astype(
                    "float32"
                )
            )
            .rolling(
                ventana,
                min_periods=ventana,
            )
            .mean()
            .astype(
                "float32"
            )
        )

        datos[
            f"cuerpos_positivos_acumulados_{ventana}m"
        ] = (
            pd.Series(
                cuerpos_positivos
            )
            .rolling(
                ventana,
                min_periods=ventana,
            )
            .sum()
            .astype(
                "float32"
            )
        )

        datos[
            f"cuerpos_negativos_acumulados_{ventana}m"
        ] = (
            pd.Series(
                cuerpos_negativos
            )
            .rolling(
                ventana,
                min_periods=ventana,
            )
            .sum()
            .astype(
                "float32"
            )
        )

    datos[
        "velas_positivas_consecutivas"
    ] = contar_consecutivas(
        vela_positiva
    )

    datos[
        "velas_negativas_consecutivas"
    ] = contar_consecutivas(
        vela_negativa
    )

    volatilidad_15 = pd.Series(
        retorno_1m
    ).rolling(
        15,
        min_periods=15,
    ).std()

    volatilidad_60 = pd.Series(
        retorno_1m
    ).rolling(
        60,
        min_periods=60,
    ).std()

    volatilidad_240 = pd.Series(
        retorno_1m
    ).rolling(
        240,
        min_periods=240,
    ).std()

    datos[
        "expansion_volatilidad_15m_60m"
    ] = (
        volatilidad_15
        / volatilidad_60.replace(
            0.0,
            np.nan,
        )
    ).astype(
        "float32"
    )

    datos[
        "expansion_volatilidad_60m_240m"
    ] = (
        volatilidad_60
        / volatilidad_240.replace(
            0.0,
            np.nan,
        )
    ).astype(
        "float32"
    )

    return datos



def agregar_variables_tecnicas(
    datos: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula variables sin atravesar huecos temporales."""

    datos = (
        datos
        .sort_values(
            "fecha_apertura"
        )
        .reset_index(
            drop=True
        )
    )

    diferencia = datos[
        "fecha_apertura"
    ].diff()

    nuevo_segmento = (
        diferencia.isna()
        | (
            diferencia
            != pd.Timedelta(
                minutes=1
            )
        )
    )

    grupos = nuevo_segmento.cumsum()

    partes = []

    for _, segmento in datos.groupby(
        grupos,
        sort=False,
    ):
        partes.append(
            _agregar_variables_tecnicas_segmento(
                segmento.copy()
            )
        )

    return (
        pd.concat(
            partes,
            ignore_index=True,
        )
        .sort_values(
            "fecha_apertura"
        )
        .reset_index(
            drop=True
        )
    )


def _agregar_objetivo_barreras_segmento(
    datos: pd.DataFrame,
    nombre: str,
    take_profit: float,
    stop_loss: float,
    horizonte: int,
) -> pd.DataFrame:
    """Etiqueta qué barrera se toca primero. En empate intravela gana el stop."""

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

    n = len(
        datos
    )

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

        activos = codigo[
            :limite
        ] == -1

        toca_stop = (
            minimo[
                paso:
            ]
            <= cierre[
                :limite
            ]
            * (
                1.0
                - stop_loss
            )
        )

        toca_take = (
            maximo[
                paso:
            ]
            >= cierre[
                :limite
            ]
            * (
                1.0
                + take_profit
            )
        )

        perdedoras = (
            activos
            & toca_stop
        )

        ganadoras = (
            activos
            & ~toca_stop
            & toca_take
        )

        indices_perdedoras = np.flatnonzero(
            perdedoras
        )

        indices_ganadoras = np.flatnonzero(
            ganadoras
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

    indices_sin_barrera = indices_completos[
        sin_barrera
    ]

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
            + horizonte
        ]
        / cierre[
            indices_sin_barrera
        ]
        - 1.0
    )

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
        1,
        np.where(
            codigo >= 0,
            0,
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
        datos[
            "fecha_apertura"
        ]
        + pd.to_timedelta(
            horizonte,
            unit="m",
        )
    )

    return datos



def agregar_objetivo_barreras(
    datos: pd.DataFrame,
    nombre: str,
    take_profit: float,
    stop_loss: float,
    horizonte: int,
) -> pd.DataFrame:
    """Etiqueta barreras sin permitir que una operación atraviese un hueco."""

    datos = (
        datos
        .sort_values(
            "fecha_apertura"
        )
        .reset_index(
            drop=True
        )
    )

    diferencia = datos[
        "fecha_apertura"
    ].diff()

    nuevo_segmento = (
        diferencia.isna()
        | (
            diferencia
            != pd.Timedelta(
                minutes=1
            )
        )
    )

    grupos = nuevo_segmento.cumsum()

    partes = []

    for _, segmento in datos.groupby(
        grupos,
        sort=False,
    ):
        partes.append(
            _agregar_objetivo_barreras_segmento(
                datos=segmento.copy(),
                nombre=nombre,
                take_profit=take_profit,
                stop_loss=stop_loss,
                horizonte=horizonte,
            )
        )

    return (
        pd.concat(
            partes,
            ignore_index=True,
        )
        .sort_values(
            "fecha_apertura"
        )
        .reset_index(
            drop=True
        )
    )


def procesar_periodo(
    simbolo: str,
    indice_periodo: int,
    ruta_precios: Path | None,
    sobrescribir: bool,
) -> dict:
    """Procesa un año con contexto anterior y futuro."""

    desde, hasta = PERIODOS[
        indice_periodo
    ]

    ruta_salida = construir_ruta_v5(
        simbolo=simbolo,
        desde=desde,
        hasta=hasta,
    )

    if ruta_salida.exists() and not sobrescribir:
        raise FileExistsError(
            f"Ya existe: {ruta_salida}"
        )

    actual = cargar_periodo(
        simbolo=simbolo,
        desde=desde,
        hasta=hasta,
        ruta_precios=ruta_precios,
    )

    partes = []

    if indice_periodo > 0:
        desde_anterior, hasta_anterior = PERIODOS[
            indice_periodo - 1
        ]

        anterior = cargar_periodo(
            simbolo=simbolo,
            desde=desde_anterior,
            hasta=hasta_anterior,
            ruta_precios=ruta_precios,
        ).tail(
            MAXIMO_HISTORICO
        )

        partes.append(
            anterior
        )

    partes.append(
        actual
    )

    if indice_periodo + 1 < len(
        PERIODOS
    ):
        desde_siguiente, hasta_siguiente = PERIODOS[
            indice_periodo + 1
        ]

        siguiente = cargar_periodo(
            simbolo=simbolo,
            desde=desde_siguiente,
            hasta=hasta_siguiente,
            ruta_precios=ruta_precios,
        ).head(
            MAXIMO_FUTURO
        )

        partes.append(
            siguiente
        )

    datos = (
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
        .reset_index(
            drop=True
        )
    )

    datos = agregar_variables_tecnicas(
        datos
    )

    for nombre, configuracion in OBJETIVOS_BARRERAS.items():
        datos = agregar_objetivo_barreras(
            datos=datos,
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

    mascara_periodo = (
        (
            datos[
                "fecha_apertura"
            ]
            >= pd.Timestamp(
                desde,
                tz="UTC",
            )
        )
        & (
            datos[
                "fecha_apertura"
            ]
            < pd.Timestamp(
                hasta,
                tz="UTC",
            )
        )
    )

    salida = datos.loc[
        mascara_periodo
    ].copy()

    columnas_validacion = [
        *COLUMNAS_TECNICAS_V5,
    ]

    salida = salida.dropna(
        subset=columnas_validacion
    )

    RUTA_DATOS_V5.mkdir(
        parents=True,
        exist_ok=True,
    )

    salida.to_parquet(
        ruta_salida,
        index=False,
    )

    resumen = {
        "simbolo": simbolo,
        "desde": desde,
        "hasta": hasta,
        "filas_entrada_periodo": len(
            actual
        ),
        "filas_salida": len(
            salida
        ),
        "ruta": str(
            ruta_salida
        ),
    }

    del actual
    del datos
    del salida
    gc.collect()

    return resumen


def main() -> None:
    """Genera variables y objetivos V5."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--simbolo",
        default="BTCUSDT",
    )

    parser.add_argument(
        "--ruta-precios",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--sobrescribir",
        action="store_true",
    )

    argumentos = parser.parse_args()

    print(
        "\nGENERACIÓN DE DATOS V5"
    )
    print("=" * 72)
    print(
        f"Símbolo: {argumentos.simbolo}"
    )
    print(
        f"Variables V2A: {len(COLUMNAS_MODELO_V2A)}"
    )
    print(
        f"Variables técnicas nuevas: {len(COLUMNAS_TECNICAS_V5)}"
    )
    print(
        f"Objetivos de barreras: {len(OBJETIVOS_BARRERAS)}"
    )

    resumenes = []

    for indice in range(
        len(
            PERIODOS
        )
    ):
        resumen = procesar_periodo(
            simbolo=argumentos.simbolo,
            indice_periodo=indice,
            ruta_precios=argumentos.ruta_precios,
            sobrescribir=argumentos.sobrescribir,
        )

        resumenes.append(
            resumen
        )

        print(
            f"- {resumen['desde']} → {resumen['hasta']}: "
            f"{resumen['filas_salida']:,} filas".replace(
                ",",
                ".",
            )
        )

    ruta_resumen = (
        RUTA_DATOS_V5
        / "resumen_generacion_v5.json"
    )

    ruta_resumen.write_text(
        json.dumps(
            resumenes,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\nGeneración completada:"
    )
    print(
        f"- {ruta_resumen}"
    )


if __name__ == "__main__":
    main()
