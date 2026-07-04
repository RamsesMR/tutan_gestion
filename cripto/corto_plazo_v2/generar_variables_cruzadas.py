from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cripto.corto_plazo_v2.configuracion import (
    COLUMNAS_CRUZADAS,
    COLUMNAS_MODELO_V1,
    COLUMNAS_MODELO_V2A,
    FILAS_HISTORIAL,
    NOMBRE_HORIZONTE,
    PERIODOS_DESARROLLO,
    RUTA_DATOS_V1,
    RUTA_DATOS_V2,
    RUTA_RESUMEN_GENERACION,
    SIMBOLOS,
    VENTANAS_RELACION,
    VENTANAS_RENDIMIENTO_CRUZADO,
    VENTANAS_VOLATILIDAD_RELATIVA,
    VENTANAS_ZSCORE_DIVERGENCIA,
    VERSION_MODELO,
)


PREFIJOS = {
    "BTCUSDT": "btc",
    "ETHUSDT": "eth",
}

COLUMNAS_OBLIGATORIAS_V1 = {
    "simbolo",
    "fecha_apertura",
    "precio_cierre",
    "fecha_objetivo",
    "rendimiento_objetivo",
    *COLUMNAS_MODELO_V1,
}


def construir_ruta(
    carpeta: Path,
    simbolo: str,
    desde: str,
    hasta: str,
) -> Path:
    """Construye la ruta de un archivo anual de variables."""

    nombre = (
        f"{simbolo}_1m_{NOMBRE_HORIZONTE}_"
        f"{desde}_{hasta}_variables.parquet"
    )

    return carpeta / nombre


def validar_archivo_v1(
    datos: pd.DataFrame,
    ruta: Path,
    simbolo: str,
) -> pd.DataFrame:
    """Valida un archivo de la V1 antes de usarlo."""

    faltantes = COLUMNAS_OBLIGATORIAS_V1.difference(
        datos.columns
    )

    if faltantes:
        raise ValueError(
            f"{ruta.name}: faltan columnas: "
            + ", ".join(sorted(faltantes))
        )

    datos = datos.copy()

    datos["fecha_apertura"] = pd.to_datetime(
        datos["fecha_apertura"],
        utc=True,
        errors="coerce",
    )

    datos["fecha_objetivo"] = pd.to_datetime(
        datos["fecha_objetivo"],
        utc=True,
        errors="coerce",
    )

    if datos[
        [
            "fecha_apertura",
            "fecha_objetivo",
        ]
    ].isna().any().any():
        raise ValueError(
            f"{ruta.name}: contiene fechas inválidas."
        )

    if datos["fecha_apertura"].duplicated().any():
        raise ValueError(
            f"{ruta.name}: contiene fechas duplicadas."
        )

    if not datos["fecha_apertura"].is_monotonic_increasing:
        datos = (
            datos
            .sort_values("fecha_apertura")
            .reset_index(drop=True)
        )

    simbolos_encontrados = set(
        datos["simbolo"]
        .astype(str)
        .unique()
        .tolist()
    )

    if simbolos_encontrados != {simbolo}:
        raise ValueError(
            f"{ruta.name}: se esperaba únicamente {simbolo}, "
            f"pero contiene {sorted(simbolos_encontrados)}."
        )

    variables = datos[
        list(COLUMNAS_MODELO_V1)
    ].to_numpy(
        dtype="float64"
    )

    if not np.isfinite(variables).all():
        raise ValueError(
            f"{ruta.name}: las variables V1 contienen nulos o infinitos."
        )

    return datos


def cargar_periodo(
    simbolo: str,
    desde: str,
    hasta: str,
) -> tuple[pd.DataFrame, Path]:
    """Carga un periodo de la V1."""

    ruta = construir_ruta(
        carpeta=RUTA_DATOS_V1,
        simbolo=simbolo,
        desde=desde,
        hasta=hasta,
    )

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el archivo V1: {ruta}"
        )

    datos = pd.read_parquet(
        ruta
    )

    datos = validar_archivo_v1(
        datos=datos,
        ruta=ruta,
        simbolo=simbolo,
    )

    return datos, ruta


def validar_alineacion(
    btc: pd.DataFrame,
    eth: pd.DataFrame,
    desde: str,
    hasta: str,
) -> None:
    """Exige que BTC y ETH tengan exactamente los mismos minutos."""

    if len(btc) != len(eth):
        raise ValueError(
            f"{desde} a {hasta}: BTC y ETH tienen distinta cantidad "
            f"de filas ({len(btc)} frente a {len(eth)})."
        )

    fechas_btc = btc[
        "fecha_apertura"
    ].to_numpy()

    fechas_eth = eth[
        "fecha_apertura"
    ].to_numpy()

    if not np.array_equal(
        fechas_btc,
        fechas_eth,
    ):
        diferencias = int(
            np.count_nonzero(
                fechas_btc != fechas_eth
            )
        )

        raise ValueError(
            f"{desde} a {hasta}: BTC y ETH no están sincronizados. "
            f"Fechas diferentes: {diferencias}."
        )


def nombre_rendimiento(
    ventana: int,
) -> str:
    """Devuelve el nombre V1 del rendimiento de una ventana."""

    if ventana == 1:
        return "rendimiento_1m"

    return f"rendimiento_{ventana}m"


def construir_base_relaciones(
    btc: pd.DataFrame,
    eth: pd.DataFrame,
) -> pd.DataFrame:
    """Crea una tabla mínima sincronizada para calcular relaciones."""

    base = pd.DataFrame(
        {
            "fecha_apertura": btc[
                "fecha_apertura"
            ].to_numpy(),
            "btc_precio_cierre": btc[
                "precio_cierre"
            ].to_numpy(
                dtype="float64"
            ),
            "eth_precio_cierre": eth[
                "precio_cierre"
            ].to_numpy(
                dtype="float64"
            ),
        }
    )

    for simbolo, datos in (
        ("BTCUSDT", btc),
        ("ETHUSDT", eth),
    ):
        prefijo = PREFIJOS[
            simbolo
        ]

        for ventana in VENTANAS_RENDIMIENTO_CRUZADO:
            columna = nombre_rendimiento(
                ventana
            )

            base[
                f"{prefijo}_rendimiento_{ventana}m"
            ] = datos[
                columna
            ].to_numpy(
                dtype="float64"
            )

        for ventana in VENTANAS_VOLATILIDAD_RELATIVA:
            base[
                f"{prefijo}_volatilidad_{ventana}m"
            ] = datos[
                f"volatilidad_{ventana}m"
            ].to_numpy(
                dtype="float64"
            )

    return base


def division_segura(
    numerador: pd.Series,
    denominador: pd.Series,
) -> pd.Series:
    """Divide y devuelve NaN cuando el denominador no es válido."""

    valores_numerador = numerador.to_numpy(
        dtype="float64"
    )

    valores_denominador = denominador.to_numpy(
        dtype="float64"
    )

    resultado = np.full(
        len(numerador),
        np.nan,
        dtype="float64",
    )

    np.divide(
        valores_numerador,
        valores_denominador,
        out=resultado,
        where=(
            np.isfinite(valores_denominador)
            & (np.abs(valores_denominador) > 1e-15)
        ),
    )

    return pd.Series(
        resultado,
        index=numerador.index,
        dtype="float64",
    )


def crear_grupos_continuos(
    datos: pd.DataFrame,
) -> pd.DataFrame:
    """Separa la serie cuando existe un hueco superior a un minuto."""

    datos = datos.copy()

    diferencias = datos[
        "fecha_apertura"
    ].diff()

    nuevo_grupo = diferencias.ne(
        pd.Timedelta(minutes=1)
    )

    datos["grupo_continuo"] = (
        nuevo_grupo
        .cumsum()
        .astype("int64")
    )

    return datos


def calcular_variables_grupo(
    grupo: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula todas las variables cruzadas de un grupo continuo."""

    resultado = pd.DataFrame(
        index=grupo.index
    )

    btc_1m = grupo[
        "btc_rendimiento_1m"
    ]

    eth_1m = grupo[
        "eth_rendimiento_1m"
    ]

    divergencia_btc_eth_1m = (
        btc_1m - eth_1m
    )

    correlaciones: dict[int, pd.Series] = {}
    covarianzas: dict[int, pd.Series] = {}
    varianzas_btc: dict[int, pd.Series] = {}
    varianzas_eth: dict[int, pd.Series] = {}

    for ventana in VENTANAS_RELACION:
        correlaciones[ventana] = (
            btc_1m
            .rolling(
                window=ventana,
                min_periods=ventana,
            )
            .corr(eth_1m)
        )

        covarianzas[ventana] = (
            btc_1m
            .rolling(
                window=ventana,
                min_periods=ventana,
            )
            .cov(eth_1m)
        )

        varianzas_btc[ventana] = (
            btc_1m
            .rolling(
                window=ventana,
                min_periods=ventana,
            )
            .var()
        )

        varianzas_eth[ventana] = (
            eth_1m
            .rolling(
                window=ventana,
                min_periods=ventana,
            )
            .var()
        )

    rendimientos_1440 = {}

    for prefijo in (
        "btc",
        "eth",
    ):
        rendimientos_1440[prefijo] = (
            grupo[
                f"{prefijo}_precio_cierre"
            ]
            / grupo[
                f"{prefijo}_precio_cierre"
            ].shift(1440)
            - 1
        )

    for simbolo in SIMBOLOS:
        propio = PREFIJOS[
            simbolo
        ]

        otro = (
            "eth"
            if propio == "btc"
            else "btc"
        )

        for ventana in VENTANAS_RENDIMIENTO_CRUZADO:
            propio_rendimiento = grupo[
                f"{propio}_rendimiento_{ventana}m"
            ]

            otro_rendimiento = grupo[
                f"{otro}_rendimiento_{ventana}m"
            ]

            resultado[
                f"{simbolo}__otro_rendimiento_{ventana}m"
            ] = otro_rendimiento

            resultado[
                f"{simbolo}__divergencia_rendimiento_{ventana}m"
            ] = (
                propio_rendimiento
                - otro_rendimiento
            )

        for ventana in VENTANAS_VOLATILIDAD_RELATIVA:
            propia_volatilidad = grupo[
                f"{propio}_volatilidad_{ventana}m"
            ]

            otra_volatilidad = grupo[
                f"{otro}_volatilidad_{ventana}m"
            ]

            resultado[
                f"{simbolo}__otro_volatilidad_{ventana}m"
            ] = otra_volatilidad

            resultado[
                f"{simbolo}__ratio_volatilidad_{ventana}m"
            ] = division_segura(
                propia_volatilidad,
                otra_volatilidad,
            )

        for ventana in VENTANAS_RELACION:
            resultado[
                f"{simbolo}__correlacion_btc_eth_{ventana}m"
            ] = correlaciones[
                ventana
            ]

            varianza_otro = (
                varianzas_eth[
                    ventana
                ]
                if otro == "eth"
                else varianzas_btc[
                    ventana
                ]
            )

            beta = division_segura(
                covarianzas[
                    ventana
                ],
                varianza_otro,
            )

            resultado[
                f"{simbolo}__beta_propio_otro_{ventana}m"
            ] = beta

            if ventana == 1440:
                rendimiento_propio = rendimientos_1440[
                    propio
                ]

                rendimiento_otro = rendimientos_1440[
                    otro
                ]

            else:
                rendimiento_propio = grupo[
                    f"{propio}_rendimiento_{ventana}m"
                ]

                rendimiento_otro = grupo[
                    f"{otro}_rendimiento_{ventana}m"
                ]

            resultado[
                f"{simbolo}__residual_propio_otro_{ventana}m"
            ] = (
                rendimiento_propio
                - beta * rendimiento_otro
            )

        divergencia_1m = (
            divergencia_btc_eth_1m
            if propio == "btc"
            else -divergencia_btc_eth_1m
        )

        for ventana in VENTANAS_ZSCORE_DIVERGENCIA:
            media = (
                divergencia_1m
                .rolling(
                    window=ventana,
                    min_periods=ventana,
                )
                .mean()
            )

            desviacion = (
                divergencia_1m
                .rolling(
                    window=ventana,
                    min_periods=ventana,
                )
                .std()
            )

            resultado[
                f"{simbolo}__zscore_divergencia_{ventana}m"
            ] = division_segura(
                divergencia_1m - media,
                desviacion,
            )

    return resultado


def calcular_variables_cruzadas(
    base: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula las relaciones sin atravesar huecos temporales."""

    base = crear_grupos_continuos(
        base
    )

    partes: list[
        pd.DataFrame
    ] = []

    for _, grupo in base.groupby(
        "grupo_continuo",
        sort=False,
    ):
        partes.append(
            calcular_variables_grupo(
                grupo=grupo
            )
        )

    if not partes:
        raise ValueError(
            "No existen grupos continuos para calcular variables."
        )

    resultado = (
        pd.concat(
            partes
        )
        .sort_index()
    )

    return resultado


def extraer_variables_simbolo(
    variables_cruzadas: pd.DataFrame,
    simbolo: str,
) -> pd.DataFrame:
    """Elimina el prefijo técnico y devuelve las 27 columnas finales."""

    prefijo = f"{simbolo}__"

    columnas_origen = [
        f"{prefijo}{columna}"
        for columna in COLUMNAS_CRUZADAS
    ]

    faltantes = set(
        columnas_origen
    ).difference(
        variables_cruzadas.columns
    )

    if faltantes:
        raise ValueError(
            f"No se generaron todas las variables de {simbolo}: "
            + ", ".join(sorted(faltantes))
        )

    resultado = variables_cruzadas[
        columnas_origen
    ].copy()

    resultado.columns = list(
        COLUMNAS_CRUZADAS
    )

    return resultado


def validar_salida(
    datos: pd.DataFrame,
    simbolo: str,
    ruta: Path,
) -> None:
    """Valida que el Parquet V2A sea apto para entrenamiento."""

    faltantes = set(
        COLUMNAS_MODELO_V2A
    ).difference(
        datos.columns
    )

    if faltantes:
        raise ValueError(
            f"{ruta.name}: faltan variables V2A: "
            + ", ".join(sorted(faltantes))
        )

    if datos["fecha_apertura"].duplicated().any():
        raise ValueError(
            f"{ruta.name}: contiene fechas duplicadas."
        )

    if not datos["fecha_apertura"].is_monotonic_increasing:
        raise ValueError(
            f"{ruta.name}: las fechas no están ordenadas."
        )

    variables = datos[
        list(COLUMNAS_MODELO_V2A)
    ].to_numpy(
        dtype="float64"
    )

    if not np.isfinite(
        variables
    ).all():
        raise ValueError(
            f"{ruta.name}: existen nulos o infinitos en las 63 variables."
        )

    simbolos = set(
        datos["simbolo"]
        .astype(str)
        .unique()
        .tolist()
    )

    if simbolos != {simbolo}:
        raise ValueError(
            f"{ruta.name}: contiene símbolos inesperados."
        )


def guardar_periodo(
    datos_originales: pd.DataFrame,
    variables_cruzadas: pd.DataFrame,
    mascara_periodo_actual: pd.Series,
    simbolo: str,
    desde: str,
    hasta: str,
    sobrescribir: bool,
) -> dict[str, Any]:
    """Une V1 con las variables cruzadas y guarda el periodo V2A."""

    variables_simbolo = extraer_variables_simbolo(
        variables_cruzadas=variables_cruzadas,
        simbolo=simbolo,
    )

    variables_actuales = (
        variables_simbolo
        .loc[
            mascara_periodo_actual.to_numpy()
        ]
        .reset_index(drop=True)
    )

    if len(variables_actuales) != len(datos_originales):
        raise ValueError(
            f"{simbolo} {desde}-{hasta}: no coinciden las filas originales "
            "con las variables cruzadas del periodo actual."
        )

    validas = np.isfinite(
        variables_actuales.to_numpy(
            dtype="float64"
        )
    ).all(axis=1)

    salida = (
        datos_originales
        .loc[validas]
        .reset_index(drop=True)
        .copy()
    )

    variables_validas = (
        variables_actuales
        .loc[validas]
        .reset_index(drop=True)
        .astype("float32")
    )

    for columna in COLUMNAS_CRUZADAS:
        salida[
            columna
        ] = variables_validas[
            columna
        ]

    ruta_salida = construir_ruta(
        carpeta=RUTA_DATOS_V2,
        simbolo=simbolo,
        desde=desde,
        hasta=hasta,
    )

    if ruta_salida.exists() and not sobrescribir:
        raise FileExistsError(
            f"Ya existe: {ruta_salida}. "
            "Usa --sobrescribir para regenerarlo."
        )

    validar_salida(
        datos=salida,
        simbolo=simbolo,
        ruta=ruta_salida,
    )

    salida.to_parquet(
        ruta_salida,
        index=False,
        compression="snappy",
    )

    descartadas = (
        len(datos_originales)
        - len(salida)
    )

    registro = {
        "version": VERSION_MODELO,
        "simbolo": simbolo,
        "desde": desde,
        "hasta": hasta,
        "archivo_entrada_filas": len(datos_originales),
        "archivo_salida_filas": len(salida),
        "filas_descartadas_variables_cruzadas": descartadas,
        "variables_v1": len(COLUMNAS_MODELO_V1),
        "variables_cruzadas": len(COLUMNAS_CRUZADAS),
        "variables_totales": len(COLUMNAS_MODELO_V2A),
        "primera_fecha": (
            salida["fecha_apertura"]
            .min()
            .isoformat()
        ),
        "ultima_fecha": (
            salida["fecha_apertura"]
            .max()
            .isoformat()
        ),
        "archivo_salida": str(
            ruta_salida
        ),
        "tamano_mb": (
            ruta_salida.stat().st_size
            / (1024 * 1024)
        ),
    }

    print(
        f"{simbolo}: "
        f"{len(salida):,} filas válidas | "
        f"{descartadas:,} descartadas | "
        f"{ruta_salida.name}"
        .replace(",", ".")
    )

    return registro


def generar_variables(
    sobrescribir: bool,
) -> None:
    """Genera todos los archivos V2A de desarrollo."""

    RUTA_DATOS_V2.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nGENERACIÓN DE VARIABLES CRUZADAS V2A"
    )
    print("=" * 70)
    print(
        "La V1 no será modificada."
    )
    print(
        "Se procesarán únicamente 2021-2025."
    )
    print(
        f"Variables V1: {len(COLUMNAS_MODELO_V1)}"
    )
    print(
        f"Variables cruzadas: {len(COLUMNAS_CRUZADAS)}"
    )
    print(
        f"Variables totales V2A: {len(COLUMNAS_MODELO_V2A)}"
    )

    historial: pd.DataFrame | None = None

    resumenes: list[
        dict[str, Any]
    ] = []

    for desde, hasta in PERIODOS_DESARROLLO:
        print(
            f"\nPERIODO {desde} A {hasta}"
        )
        print("-" * 70)

        btc, ruta_btc = cargar_periodo(
            simbolo="BTCUSDT",
            desde=desde,
            hasta=hasta,
        )

        eth, ruta_eth = cargar_periodo(
            simbolo="ETHUSDT",
            desde=desde,
            hasta=hasta,
        )

        print(
            f"Entrada BTC: {ruta_btc.name}"
        )
        print(
            f"Entrada ETH: {ruta_eth.name}"
        )

        validar_alineacion(
            btc=btc,
            eth=eth,
            desde=desde,
            hasta=hasta,
        )

        base_actual = construir_base_relaciones(
            btc=btc,
            eth=eth,
        )

        base_actual[
            "es_periodo_actual"
        ] = True

        if historial is not None:
            historial_temporal = historial.copy()

            historial_temporal[
                "es_periodo_actual"
            ] = False

            base_completa = pd.concat(
                [
                    historial_temporal,
                    base_actual,
                ],
                ignore_index=True,
            )

        else:
            base_completa = base_actual.copy()

        variables_cruzadas = calcular_variables_cruzadas(
            base=base_completa.drop(
                columns=[
                    "es_periodo_actual",
                ]
            )
        )

        mascara_actual = base_completa[
            "es_periodo_actual"
        ]

        resumenes.append(
            guardar_periodo(
                datos_originales=btc,
                variables_cruzadas=variables_cruzadas,
                mascara_periodo_actual=mascara_actual,
                simbolo="BTCUSDT",
                desde=desde,
                hasta=hasta,
                sobrescribir=sobrescribir,
            )
        )

        resumenes.append(
            guardar_periodo(
                datos_originales=eth,
                variables_cruzadas=variables_cruzadas,
                mascara_periodo_actual=mascara_actual,
                simbolo="ETHUSDT",
                desde=desde,
                hasta=hasta,
                sobrescribir=sobrescribir,
            )
        )

        columnas_historial = [
            columna
            for columna in base_actual.columns
            if columna != "es_periodo_actual"
        ]

        historial = (
            pd.concat(
                [
                    (
                        historial[
                            columnas_historial
                        ]
                        if historial is not None
                        else pd.DataFrame(
                            columns=columnas_historial
                        )
                    ),
                    base_actual[
                        columnas_historial
                    ],
                ],
                ignore_index=True,
            )
            .tail(
                FILAS_HISTORIAL
            )
            .reset_index(drop=True)
        )

        del btc
        del eth
        del base_actual
        del base_completa
        del variables_cruzadas
        gc.collect()

    resumen = pd.DataFrame(
        resumenes
    )

    resumen.to_csv(
        RUTA_RESUMEN_GENERACION,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nVARIABLES V2A GENERADAS CORRECTAMENTE"
    )
    print("=" * 70)
    print(
        f"Resumen: {RUTA_RESUMEN_GENERACION}"
    )
    print(
        "\nNo se utilizó enero-mayo de 2026."
    )


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene los argumentos del comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Genera la V2A añadiendo relaciones BTC-ETH "
            "a las 36 variables de la V1."
        )
    )

    parser.add_argument(
        "--sobrescribir",
        action="store_true",
        help=(
            "Permite regenerar archivos V2A ya existentes."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Punto de entrada."""

    argumentos = obtener_argumentos()

    try:
        generar_variables(
            sobrescribir=argumentos.sobrescribir
        )

    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(
            "\nNo se pudieron generar las variables cruzadas V2A."
        )
        print(
            f"Detalle: {error}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
