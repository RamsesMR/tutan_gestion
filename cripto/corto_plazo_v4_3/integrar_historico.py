from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
import pyarrow.parquet as pq

from compartido.base_datos.conexion import obtener_conexion
from cripto.corto_plazo.configuracion import (
    COLUMNAS_BASE,
    HORIZONTE_MINUTOS,
    INTERVALO,
    NOMBRE_HORIZONTE,
    RUTA_DATOS_PREPARADOS,
    SIMBOLOS,
)
from cripto.corto_plazo.preparar_datos import (
    COLUMNAS_NUMERICAS,
    agregar_objetivo,
)
from cripto.corto_plazo.variables import (
    procesar_archivo,
)
from cripto.corto_plazo_v2.configuracion import (
    COLUMNAS_CRUZADAS,
    COLUMNAS_MODELO_V1,
    FILAS_HISTORIAL,
    RUTA_DATOS_V2 as RUTA_DATOS_MODELO,
)
from cripto.corto_plazo_v2.generar_variables_cruzadas import (
    calcular_variables_cruzadas,
    cargar_periodo,
    construir_base_relaciones,
    construir_ruta as construir_ruta_variables_completas,
    extraer_variables_simbolo,
    validar_alineacion,
    validar_salida,
)


PRIMERA_VELA_COMUN = datetime(
    2017,
    8,
    17,
    4,
    0,
    tzinfo=timezone.utc,
)

FIN_HISTORICO_EXCLUSIVO = datetime(
    2021,
    1,
    1,
    tzinfo=timezone.utc,
)

PERIODOS_HISTORICOS = (
    (
        PRIMERA_VELA_COMUN,
        datetime(
            2018,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    ),
    (
        datetime(
            2018,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        datetime(
            2019,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    ),
    (
        datetime(
            2019,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        datetime(
            2020,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    ),
    (
        datetime(
            2020,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        FIN_HISTORICO_EXCLUSIVO,
    ),
)

COLUMNAS_MODELO_COMPLETO = (
    *COLUMNAS_MODELO_V1,
    *COLUMNAS_CRUZADAS,
)

RUTA_RESUMEN_PREPARACION = (
    RUTA_DATOS_PREPARADOS.parent
    / "resumen_integracion_historica_20170817_20201231.csv"
)

RUTA_RESUMEN_VARIABLES = (
    RUTA_DATOS_MODELO
    / "resumen_variables_historicas_20170817_20201231.csv"
)

RUTA_VALIDACION_CSV = (
    RUTA_DATOS_MODELO
    / "validacion_integracion_historica_20170817_20201231.csv"
)

RUTA_VALIDACION_MD = (
    RUTA_DATOS_MODELO
    / "resumen_validacion_integracion_historica_20170817_20201231.md"
)

RUTA_MANIFIESTO = (
    RUTA_DATOS_MODELO
    / "manifiesto_integracion_historica_20170817_20201231.json"
)


def formatear_fecha_archivo(
    fecha: datetime,
) -> str:
    """Devuelve la fecha utilizada en los nombres de archivo."""

    return fecha.strftime(
        "%Y-%m-%d"
    )


def construir_ruta_preparada(
    simbolo: str,
    desde: datetime,
    hasta: datetime,
) -> Path:
    """Construye la ruta del Parquet preparado sin variables."""

    nombre = (
        f"{simbolo}_"
        f"{INTERVALO}_"
        f"{NOMBRE_HORIZONTE}_"
        f"{formatear_fecha_archivo(desde)}_"
        f"{formatear_fecha_archivo(hasta)}.parquet"
    )

    return (
        RUTA_DATOS_PREPARADOS
        / nombre
    )


def construir_ruta_variables_base(
    simbolo: str,
    desde: datetime,
    hasta: datetime,
) -> Path:
    """Construye la ruta del Parquet con las variables base."""

    ruta_preparada = construir_ruta_preparada(
        simbolo=simbolo,
        desde=desde,
        hasta=hasta,
    )

    return ruta_preparada.with_name(
        f"{ruta_preparada.stem}_variables.parquet"
    )


def consultar_velas(
    fecha_desde: datetime,
    fecha_hasta: datetime,
) -> pd.DataFrame:
    """
    Consulta simultáneamente BTC y ETH.

    Se incluyen cuatro horas anteriores y posteriores para reproducir
    el contexto que utiliza el pipeline actual al crear variables y objetivo.
    """

    fecha_desde_consulta = (
        fecha_desde
        - timedelta(
            minutes=HORIZONTE_MINUTOS
        )
    )

    fecha_hasta_consulta = (
        fecha_hasta
        + timedelta(
            minutes=HORIZONTE_MINUTOS
        )
    )

    consulta = """
        SELECT
            mercados.simbolo_proveedor AS simbolo,
            velas.fecha_apertura,
            velas.precio_apertura,
            velas.precio_maximo,
            velas.precio_minimo,
            velas.precio_cierre,
            velas.volumen,
            velas.volumen_activo_cotizacion,
            velas.numero_operaciones,
            velas.volumen_comprador_base,
            velas.volumen_comprador_cotizacion
        FROM velas
        INNER JOIN mercados
            ON mercados.id = velas.mercado_id
        INNER JOIN fuentes_datos
            ON fuentes_datos.id = mercados.fuente_datos_id
        WHERE mercados.simbolo_proveedor IN (%s, %s)
          AND mercados.tipo_mercado = 'spot'
          AND fuentes_datos.nombre = 'Binance'
          AND velas.intervalo = %s
          AND velas.cerrada = TRUE
          AND velas.fecha_apertura >= %s
          AND velas.fecha_apertura < %s
        ORDER BY
            velas.fecha_apertura,
            mercados.simbolo_proveedor;
    """

    with obtener_conexion() as conexion:
        with conexion.cursor() as cursor:
            cursor.execute(
                consulta,
                (
                    SIMBOLOS[0],
                    SIMBOLOS[1],
                    INTERVALO,
                    fecha_desde_consulta,
                    fecha_hasta_consulta,
                ),
            )

            filas = cursor.fetchall()

            if cursor.description is None:
                raise RuntimeError(
                    "La consulta no devolvió una descripción de columnas."
                )

            columnas = [
                columna.name
                for columna in cursor.description
            ]

    if not filas:
        raise ValueError(
            "No se encontraron velas para el periodo solicitado."
        )

    return pd.DataFrame(
        filas,
        columns=columnas,
    )


def normalizar_tipos(
    datos: pd.DataFrame,
) -> pd.DataFrame:
    """Convierte fechas y columnas numéricas al formato del pipeline actual."""

    datos = datos.copy()

    datos["simbolo"] = (
        datos["simbolo"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    datos["fecha_apertura"] = pd.to_datetime(
        datos["fecha_apertura"],
        utc=True,
        errors="coerce",
    )

    if datos["fecha_apertura"].isna().any():
        raise ValueError(
            "Existen fechas de apertura inválidas."
        )

    for columna in COLUMNAS_NUMERICAS:
        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce",
        )

    if datos[
        list(COLUMNAS_NUMERICAS)
    ].isna().any().any():
        raise ValueError(
            "Existen valores numéricos que no pudieron convertirse."
        )

    datos["numero_operaciones"] = (
        datos["numero_operaciones"]
        .astype("int64")
    )

    simbolos_encontrados = set(
        datos["simbolo"]
        .unique()
        .tolist()
    )

    if simbolos_encontrados != set(SIMBOLOS):
        raise ValueError(
            "La consulta no devolvió exactamente BTCUSDT y ETHUSDT. "
            f"Encontrados: {sorted(simbolos_encontrados)}"
        )

    return (
        datos
        .sort_values(
            [
                "fecha_apertura",
                "simbolo",
            ]
        )
        .reset_index(drop=True)
    )


def sincronizar_por_minuto(
    datos: pd.DataFrame,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, int],
]:
    """
    Normaliza al minuto únicamente en memoria.

    Solo conserva minutos con exactamente una vela de BTC y una de ETH.
    PostgreSQL y los archivos originales no se modifican.
    """

    datos = datos.copy()

    datos["fecha_minuto"] = (
        datos["fecha_apertura"]
        .dt.floor("min")
    )

    conteos = (
        datos
        .groupby(
            [
                "fecha_minuto",
                "simbolo",
            ],
            sort=True,
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            columns=list(SIMBOLOS),
            fill_value=0,
        )
    )

    minutos_validos = conteos.index[
        (conteos[SIMBOLOS[0]] == 1)
        & (conteos[SIMBOLOS[1]] == 1)
    ]

    minutos_colision = int(
        (
            (conteos[SIMBOLOS[0]] > 1)
            | (conteos[SIMBOLOS[1]] > 1)
        ).sum()
    )

    minutos_no_comunes = int(
        (
            (conteos[SIMBOLOS[0]] == 0)
            | (conteos[SIMBOLOS[1]] == 0)
        ).sum()
    )

    datos_validos = (
        datos
        .loc[
            datos["fecha_minuto"].isin(
                minutos_validos
            )
        ]
        .copy()
    )

    datos_validos["fecha_apertura"] = (
        datos_validos["fecha_minuto"]
    )

    datos_validos = datos_validos.drop(
        columns=[
            "fecha_minuto",
        ]
    )

    resultado: dict[
        str,
        pd.DataFrame,
    ] = {}

    columnas_salida = [
        "simbolo",
        *COLUMNAS_BASE,
    ]

    for simbolo in SIMBOLOS:
        datos_simbolo = (
            datos_validos
            .loc[
                datos_validos["simbolo"]
                == simbolo
            ]
            .sort_values(
                "fecha_apertura"
            )
            .reset_index(drop=True)
        )

        if datos_simbolo[
            "fecha_apertura"
        ].duplicated().any():
            raise ValueError(
                f"{simbolo}: quedaron minutos duplicados "
                "después de sincronizar."
            )

        resultado[simbolo] = (
            datos_simbolo[
                columnas_salida
            ]
            .copy()
        )

    fechas_btc = resultado[
        "BTCUSDT"
    ]["fecha_apertura"]

    fechas_eth = resultado[
        "ETHUSDT"
    ]["fecha_apertura"]

    if not fechas_btc.equals(
        fechas_eth
    ):
        raise ValueError(
            "BTC y ETH no quedaron perfectamente alineados por minuto."
        )

    metricas = {
        "registros_consultados": len(
            datos
        ),
        "minutos_union": len(
            conteos
        ),
        "minutos_validos_comunes": len(
            minutos_validos
        ),
        "minutos_colision": minutos_colision,
        "minutos_no_comunes": minutos_no_comunes,
    }

    return resultado, metricas


def guardar_datos_preparados(
    datos_por_simbolo: dict[
        str,
        pd.DataFrame,
    ],
    desde: datetime,
    hasta: datetime,
    sobrescribir: bool,
) -> list[dict[str, Any]]:
    """Crea los archivos preparados y sus variables base."""

    registros: list[
        dict[str, Any]
    ] = []

    RUTA_DATOS_PREPARADOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    for simbolo in SIMBOLOS:
        ruta_preparada = construir_ruta_preparada(
            simbolo=simbolo,
            desde=desde,
            hasta=hasta,
        )

        ruta_variables = construir_ruta_variables_base(
            simbolo=simbolo,
            desde=desde,
            hasta=hasta,
        )

        for ruta in (
            ruta_preparada,
            ruta_variables,
        ):
            if ruta.exists() and not sobrescribir:
                raise FileExistsError(
                    f"Ya existe: {ruta}. "
                    "Usa --sobrescribir para regenerarlo."
                )

        datos_preparados, muestras_descartadas = agregar_objetivo(
            datos=datos_por_simbolo[
                simbolo
            ],
            fecha_desde=desde,
            fecha_hasta=hasta,
        )

        datos_preparados.to_parquet(
            ruta_preparada,
            index=False,
            compression="snappy",
        )

        ruta_generada = procesar_archivo(
            ruta_entrada=ruta_preparada
        )

        if ruta_generada != ruta_variables:
            raise RuntimeError(
                "La ruta generada por variables.py no coincide "
                f"con la esperada: {ruta_generada}"
            )

        datos_variables = pd.read_parquet(
            ruta_variables,
            columns=[
                "fecha_apertura",
            ],
        )

        registros.append(
            {
                "simbolo": simbolo,
                "desde": desde.isoformat(),
                "hasta_exclusivo": hasta.isoformat(),
                "filas_sincronizadas_con_contexto": len(
                    datos_por_simbolo[
                        simbolo
                    ]
                ),
                "filas_preparadas": len(
                    datos_preparados
                ),
                "filas_variables_base": len(
                    datos_variables
                ),
                "muestras_objetivo_descartadas": muestras_descartadas,
                "archivo_preparado": str(
                    ruta_preparada
                ),
                "archivo_variables_base": str(
                    ruta_variables
                ),
            }
        )

        del datos_preparados
        del datos_variables
        gc.collect()

    return registros


def preparar_historico(
    sobrescribir: bool,
) -> None:
    """Prepara y sincroniza todos los periodos históricos."""

    print(
        "\nINTEGRACIÓN DEL HISTÓRICO 2017-2020"
    )
    print("=" * 72)
    print(
        "Se normalizará al minuto únicamente en archivos derivados."
    )
    print(
        "PostgreSQL, ZIP originales y modelos no serán modificados."
    )

    resumenes: list[
        dict[str, Any]
    ] = []

    for desde, hasta in PERIODOS_HISTORICOS:
        print(
            "\nPERIODO "
            f"{desde.isoformat()} "
            "A "
            f"{hasta.isoformat()}"
        )
        print("-" * 72)

        datos = consultar_velas(
            fecha_desde=desde,
            fecha_hasta=hasta,
        )

        datos = normalizar_tipos(
            datos
        )

        (
            datos_por_simbolo,
            metricas,
        ) = sincronizar_por_minuto(
            datos
        )

        print(
            "Registros consultados: "
            f"{metricas['registros_consultados']:,}"
            .replace(",", ".")
        )
        print(
            "Minutos comunes válidos con contexto: "
            f"{metricas['minutos_validos_comunes']:,}"
            .replace(",", ".")
        )
        print(
            "Minutos excluidos por colisión: "
            f"{metricas['minutos_colision']}"
        )
        print(
            "Minutos excluidos por faltar un símbolo: "
            f"{metricas['minutos_no_comunes']}"
        )

        registros_periodo = guardar_datos_preparados(
            datos_por_simbolo=datos_por_simbolo,
            desde=desde,
            hasta=hasta,
            sobrescribir=sobrescribir,
        )

        for registro in registros_periodo:
            registro.update(
                metricas
            )

            resumenes.append(
                registro
            )

        del datos
        del datos_por_simbolo
        gc.collect()

    RUTA_RESUMEN_PREPARACION.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        resumenes
    ).to_csv(
        RUTA_RESUMEN_PREPARACION,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nPREPARACIÓN HISTÓRICA COMPLETADA"
    )
    print("=" * 72)
    print(
        f"Resumen: {RUTA_RESUMEN_PREPARACION}"
    )



def guardar_periodo_sincronizado(
    btc: pd.DataFrame,
    eth: pd.DataFrame,
    variables_relacionadas: pd.DataFrame,
    mascara_periodo_actual: pd.Series,
    desde: str,
    hasta: str,
    sobrescribir: bool,
) -> list[dict[str, Any]]:
    """
    Guarda BTC y ETH usando exactamente la misma máscara de filas válidas.

    Algunas variables relacionadas pueden ser finitas para un símbolo y no
    para el otro durante tramos planos. Para conservar la alineación temporal,
    una fila solo se guarda cuando las 27 variables son válidas en ambos.
    """

    variables_por_simbolo: dict[
        str,
        pd.DataFrame,
    ] = {}

    datos_por_simbolo = {
        "BTCUSDT": btc,
        "ETHUSDT": eth,
    }

    mascara_actual = (
        mascara_periodo_actual
        .to_numpy(
            dtype=bool
        )
    )

    for simbolo in SIMBOLOS:
        variables_simbolo = extraer_variables_simbolo(
            variables_cruzadas=variables_relacionadas,
            simbolo=simbolo,
        )

        variables_actuales = (
            variables_simbolo
            .loc[
                mascara_actual
            ]
            .reset_index(
                drop=True
            )
        )

        if len(
            variables_actuales
        ) != len(
            datos_por_simbolo[
                simbolo
            ]
        ):
            raise ValueError(
                f"{simbolo} {desde}-{hasta}: no coinciden las filas "
                "originales con las variables relacionadas."
            )

        variables_por_simbolo[
            simbolo
        ] = variables_actuales

    validas_btc = np.isfinite(
        variables_por_simbolo[
            "BTCUSDT"
        ].to_numpy(
            dtype="float64"
        )
    ).all(
        axis=1
    )

    validas_eth = np.isfinite(
        variables_por_simbolo[
            "ETHUSDT"
        ].to_numpy(
            dtype="float64"
        )
    ).all(
        axis=1
    )

    validas_comunes = (
        validas_btc
        & validas_eth
    )

    descartadas_btc = int(
        (
            ~validas_btc
        ).sum()
    )

    descartadas_eth = int(
        (
            ~validas_eth
        ).sum()
    )

    descartadas_comunes = int(
        (
            ~validas_comunes
        ).sum()
    )

    registros: list[
        dict[str, Any]
    ] = []

    rutas_generadas: dict[
        str,
        Path,
    ] = {}

    for simbolo in SIMBOLOS:
        datos_originales = datos_por_simbolo[
            simbolo
        ]

        salida = (
            datos_originales
            .loc[
                validas_comunes
            ]
            .reset_index(
                drop=True
            )
            .copy()
        )

        variables_validas = (
            variables_por_simbolo[
                simbolo
            ]
            .loc[
                validas_comunes
            ]
            .reset_index(
                drop=True
            )
            .astype(
                "float32"
            )
        )

        for columna in COLUMNAS_CRUZADAS:
            salida[
                columna
            ] = variables_validas[
                columna
            ]

        ruta_salida = construir_ruta_variables_completas(
            carpeta=RUTA_DATOS_MODELO,
            simbolo=simbolo,
            desde=desde,
            hasta=hasta,
        )

        if (
            ruta_salida.exists()
            and not sobrescribir
        ):
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

        rutas_generadas[
            simbolo
        ] = ruta_salida

        registro = {
            "simbolo": simbolo,
            "desde": desde,
            "hasta": hasta,
            "archivo_entrada_filas": len(
                datos_originales
            ),
            "archivo_salida_filas": len(
                salida
            ),
            "filas_invalidas_btc": descartadas_btc,
            "filas_invalidas_eth": descartadas_eth,
            "filas_descartadas_mascara_comun": descartadas_comunes,
            "variables_base": len(
                COLUMNAS_MODELO_V1
            ),
            "variables_relacionadas": len(
                COLUMNAS_CRUZADAS
            ),
            "variables_totales": len(
                COLUMNAS_MODELO_COMPLETO
            ),
            "primera_fecha": (
                salida[
                    "fecha_apertura"
                ]
                .min()
                .isoformat()
            ),
            "ultima_fecha": (
                salida[
                    "fecha_apertura"
                ]
                .max()
                .isoformat()
            ),
            "archivo_salida": str(
                ruta_salida
            ),
            "tamano_mb": (
                ruta_salida.stat().st_size
                / (
                    1024
                    * 1024
                )
            ),
        }

        registros.append(
            registro
        )

        print(
            f"{simbolo}: "
            f"{len(salida):,} filas válidas comunes | "
            f"{descartadas_comunes:,} descartadas | "
            f"{ruta_salida.name}"
            .replace(
                ",",
                ".",
            )
        )

    fechas_btc = pd.to_datetime(
        pd.read_parquet(
            rutas_generadas[
                "BTCUSDT"
            ],
            columns=[
                "fecha_apertura",
            ],
        )[
            "fecha_apertura"
        ],
        utc=True,
    )

    fechas_eth = pd.to_datetime(
        pd.read_parquet(
            rutas_generadas[
                "ETHUSDT"
            ],
            columns=[
                "fecha_apertura",
            ],
        )[
            "fecha_apertura"
        ],
        utc=True,
    )

    if not fechas_btc.equals(
        fechas_eth
    ):
        raise ValueError(
            f"{desde} a {hasta}: BTC y ETH no quedaron alineados "
            "después de aplicar la máscara común."
        )

    return registros


def generar_variables_completas(
    sobrescribir: bool,
) -> None:
    """
    Genera las variables relacionadas entre BTC y ETH que consume V4.

    Reutiliza exactamente las funciones ya existentes en el proyecto.
    """

    print(
        "\nGENERACIÓN DE VARIABLES COMPLETAS PARA V4"
    )
    print("=" * 72)
    print(
        "Se procesarán únicamente los periodos históricos 2017-2020."
    )
    print(
        f"Variables base: {len(COLUMNAS_MODELO_V1)}"
    )
    print(
        f"Variables relacionadas: {len(COLUMNAS_CRUZADAS)}"
    )
    print(
        f"Variables totales: {len(COLUMNAS_MODELO_COMPLETO)}"
    )

    RUTA_DATOS_MODELO.mkdir(
        parents=True,
        exist_ok=True,
    )

    historial: pd.DataFrame | None = None

    resumenes: list[
        dict[str, Any]
    ] = []

    for desde_dt, hasta_dt in PERIODOS_HISTORICOS:
        desde = formatear_fecha_archivo(
            desde_dt
        )

        hasta = formatear_fecha_archivo(
            hasta_dt
        )

        print(
            f"\nPERIODO {desde} A {hasta}"
        )
        print("-" * 72)

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
            historial_temporal = (
                historial.copy()
            )

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
            base_completa = (
                base_actual.copy()
            )

        variables_relacionadas = calcular_variables_cruzadas(
            base=base_completa.drop(
                columns=[
                    "es_periodo_actual",
                ]
            )
        )

        mascara_actual = base_completa[
            "es_periodo_actual"
        ]

        registros_periodo = guardar_periodo_sincronizado(
            btc=btc,
            eth=eth,
            variables_relacionadas=variables_relacionadas,
            mascara_periodo_actual=mascara_actual,
            desde=desde,
            hasta=hasta,
            sobrescribir=sobrescribir,
        )

        resumenes.extend(
            registros_periodo
        )

        columnas_historial = [
            columna
            for columna in base_actual.columns
            if columna
            != "es_periodo_actual"
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
        del variables_relacionadas
        gc.collect()

    pd.DataFrame(
        resumenes
    ).to_csv(
        RUTA_RESUMEN_VARIABLES,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nVARIABLES HISTÓRICAS GENERADAS CORRECTAMENTE"
    )
    print("=" * 72)
    print(
        f"Resumen: {RUTA_RESUMEN_VARIABLES}"
    )


def obtener_columnas_esquema(
    ruta: Path,
) -> list[str]:
    """Obtiene los nombres de columnas sin cargar todo el Parquet."""

    return list(
        pq.read_schema(
            ruta
        ).names
    )


def obtener_tipos_esquema(
    ruta: Path,
) -> dict[str, str]:
    """Obtiene los tipos Arrow del Parquet."""

    esquema = pq.read_schema(
        ruta
    )

    return {
        campo.name: str(
            campo.type
        )
        for campo in esquema
    }


def validar_variables_finitas(
    ruta: Path,
    columnas: tuple[str, ...],
) -> None:
    """Comprueba nulos e infinitos por lotes."""

    archivo = pq.ParquetFile(
        ruta
    )

    for lote in archivo.iter_batches(
        batch_size=100000,
        columns=list(columnas),
    ):
        valores = (
            lote
            .to_pandas()
            .to_numpy(
                dtype="float64"
            )
        )

        if not np.isfinite(
            valores
        ).all():
            raise ValueError(
                f"{ruta.name}: contiene variables nulas o infinitas."
            )


def validar_archivo_completo(
    ruta: Path,
    simbolo: str,
    desde: datetime,
    hasta: datetime,
    ruta_referencia: Path | None,
) -> dict[str, Any]:
    """Valida estructura, fechas, objetivo y variables."""

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe: {ruta}"
        )

    columnas = obtener_columnas_esquema(
        ruta
    )

    columnas_requeridas = {
        "simbolo",
        "fecha_apertura",
        "fecha_objetivo",
        "rendimiento_objetivo",
        *COLUMNAS_MODELO_COMPLETO,
    }

    faltantes = columnas_requeridas.difference(
        columnas
    )

    if faltantes:
        raise ValueError(
            f"{ruta.name}: faltan columnas: "
            + ", ".join(
                sorted(faltantes)
            )
        )

    datos_fechas = pd.read_parquet(
        ruta,
        columns=[
            "simbolo",
            "fecha_apertura",
            "fecha_objetivo",
            "rendimiento_objetivo",
        ],
    )

    datos_fechas[
        "fecha_apertura"
    ] = pd.to_datetime(
        datos_fechas[
            "fecha_apertura"
        ],
        utc=True,
        errors="coerce",
    )

    datos_fechas[
        "fecha_objetivo"
    ] = pd.to_datetime(
        datos_fechas[
            "fecha_objetivo"
        ],
        utc=True,
        errors="coerce",
    )

    if (
        datos_fechas[
            [
                "fecha_apertura",
                "fecha_objetivo",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            f"{ruta.name}: contiene fechas inválidas."
        )

    if datos_fechas[
        "fecha_apertura"
    ].duplicated().any():
        raise ValueError(
            f"{ruta.name}: contiene fechas duplicadas."
        )

    if not datos_fechas[
        "fecha_apertura"
    ].is_monotonic_increasing:
        raise ValueError(
            f"{ruta.name}: las fechas no están ordenadas."
        )

    simbolos_archivo = set(
        datos_fechas[
            "simbolo"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    if simbolos_archivo != {
        simbolo
    }:
        raise ValueError(
            f"{ruta.name}: se esperaba únicamente {simbolo}, "
            f"pero contiene {sorted(simbolos_archivo)}."
        )

    if (
        datos_fechas[
            "fecha_apertura"
        ].min()
        < desde
    ):
        raise ValueError(
            f"{ruta.name}: contiene fechas anteriores al periodo."
        )

    if (
        datos_fechas[
            "fecha_apertura"
        ].max()
        >= hasta
    ):
        raise ValueError(
            f"{ruta.name}: contiene fechas fuera del límite exclusivo."
        )

    diferencia_objetivo = (
        datos_fechas[
            "fecha_objetivo"
        ]
        - datos_fechas[
            "fecha_apertura"
        ]
    )

    if not (
        diferencia_objetivo
        == pd.Timedelta(
            minutes=HORIZONTE_MINUTOS
        )
    ).all():
        raise ValueError(
            f"{ruta.name}: existen objetivos que no están "
            f"a {HORIZONTE_MINUTOS} minutos."
        )

    rendimientos = pd.to_numeric(
        datos_fechas[
            "rendimiento_objetivo"
        ],
        errors="coerce",
    ).to_numpy(
        dtype="float64"
    )

    if not np.isfinite(
        rendimientos
    ).all():
        raise ValueError(
            f"{ruta.name}: el objetivo contiene nulos o infinitos."
        )

    validar_variables_finitas(
        ruta=ruta,
        columnas=tuple(
            COLUMNAS_MODELO_COMPLETO
        ),
    )

    esquema_compatible = (
        "NO_COMPROBADO"
    )

    if (
        ruta_referencia is not None
        and ruta_referencia.exists()
    ):
        tipos_actuales = obtener_tipos_esquema(
            ruta
        )

        tipos_referencia = obtener_tipos_esquema(
            ruta_referencia
        )

        columnas_referencia = obtener_columnas_esquema(
            ruta_referencia
        )

        if columnas != columnas_referencia:
            raise ValueError(
                f"{ruta.name}: el orden o conjunto de columnas "
                f"no coincide con {ruta_referencia.name}."
            )

        diferencias_tipos = {
            columna: (
                tipos_actuales.get(
                    columna
                ),
                tipos_referencia.get(
                    columna
                ),
            )
            for columna in columnas
            if tipos_actuales.get(
                columna
            )
            != tipos_referencia.get(
                columna
            )
        }

        if diferencias_tipos:
            raise ValueError(
                f"{ruta.name}: los tipos no coinciden con "
                f"{ruta_referencia.name}: {diferencias_tipos}"
            )

        esquema_compatible = (
            "SI"
        )

    return {
        "simbolo": simbolo,
        "desde": desde.isoformat(),
        "hasta_exclusivo": hasta.isoformat(),
        "archivo": str(
            ruta
        ),
        "filas": len(
            datos_fechas
        ),
        "primera_fecha": (
            datos_fechas[
                "fecha_apertura"
            ]
            .min()
            .isoformat()
        ),
        "ultima_fecha": (
            datos_fechas[
                "fecha_apertura"
            ]
            .max()
            .isoformat()
        ),
        "columnas": len(
            columnas
        ),
        "variables_totales": len(
            COLUMNAS_MODELO_COMPLETO
        ),
        "esquema_compatible_referencia_2021": esquema_compatible,
        "valido": True,
    }


def validar_integracion() -> None:
    """Valida los archivos históricos finales y su alineación."""

    print(
        "\nVALIDACIÓN FINAL DE LA INTEGRACIÓN HISTÓRICA"
    )
    print("=" * 72)

    registros: list[
        dict[str, Any]
    ] = []

    referencia_por_simbolo = {
        simbolo: construir_ruta_variables_completas(
            carpeta=RUTA_DATOS_MODELO,
            simbolo=simbolo,
            desde="2021-01-01",
            hasta="2022-01-01",
        )
        for simbolo in SIMBOLOS
    }

    for desde, hasta in PERIODOS_HISTORICOS:
        desde_texto = formatear_fecha_archivo(
            desde
        )

        hasta_texto = formatear_fecha_archivo(
            hasta
        )

        rutas = {
            simbolo: construir_ruta_variables_completas(
                carpeta=RUTA_DATOS_MODELO,
                simbolo=simbolo,
                desde=desde_texto,
                hasta=hasta_texto,
            )
            for simbolo in SIMBOLOS
        }

        fechas: dict[
            str,
            pd.Series,
        ] = {}

        for simbolo in SIMBOLOS:
            registro = validar_archivo_completo(
                ruta=rutas[
                    simbolo
                ],
                simbolo=simbolo,
                desde=desde,
                hasta=hasta,
                ruta_referencia=referencia_por_simbolo[
                    simbolo
                ],
            )

            registros.append(
                registro
            )

            fechas[
                simbolo
            ] = pd.read_parquet(
                rutas[
                    simbolo
                ],
                columns=[
                    "fecha_apertura",
                ],
            )[
                "fecha_apertura"
            ]

        fechas_btc = pd.to_datetime(
            fechas[
                "BTCUSDT"
            ],
            utc=True,
        )

        fechas_eth = pd.to_datetime(
            fechas[
                "ETHUSDT"
            ],
            utc=True,
        )

        if not fechas_btc.equals(
            fechas_eth
        ):
            raise ValueError(
                f"Las fechas finales de BTC y ETH no coinciden "
                f"en {desde_texto} a {hasta_texto}."
            )

        print(
            f"{desde_texto} a {hasta_texto}: "
            f"{len(fechas_btc):,} filas alineadas y válidas."
            .replace(",", ".")
        )

    RUTA_VALIDACION_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        registros
    ).to_csv(
        RUTA_VALIDACION_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    total_filas = sum(
        int(
            registro[
                "filas"
            ]
        )
        for registro in registros
        if registro[
            "simbolo"
        ]
        == "BTCUSDT"
    )

    lineas = [
        "# Validación de integración histórica 2017-2020",
        "",
        "- **Estado:** APROBADA.",
        "- **Mercados:** BTCUSDT y ETHUSDT Spot de Binance.",
        "- **Intervalo:** 1m.",
        f"- **Primera vela común original:** {PRIMERA_VELA_COMUN.isoformat()}.",
        f"- **Hasta exclusivo:** {FIN_HISTORICO_EXCLUSIVO.isoformat()}.",
        f"- **Variables totales utilizadas por V4:** {len(COLUMNAS_MODELO_COMPLETO)}.",
        f"- **Filas finales alineadas por símbolo:** {total_filas:,}.".replace(
            ",",
            ".",
        ),
        "- **Duplicados:** 0.",
        "- **Nulos o infinitos en variables:** 0.",
        "- **Objetivos fuera del horizonte:** 0.",
        "- **PostgreSQL modificado:** no.",
        "- **Modelos modificados:** no.",
        "",
        "Los archivos históricos quedaron preparados con el mismo esquema "
        "que los archivos de referencia de 2021 cuando estos estaban "
        "disponibles localmente.",
        "",
    ]

    RUTA_VALIDACION_MD.write_text(
        "\n".join(
            lineas
        ),
        encoding="utf-8",
    )

    manifiesto = {
        "fecha_generacion_utc": datetime.now(
            timezone.utc
        ).isoformat(
            timespec="seconds"
        ),
        "estado": "historico_integrado_y_validado",
        "desde": PRIMERA_VELA_COMUN.isoformat(),
        "hasta_exclusivo": FIN_HISTORICO_EXCLUSIVO.isoformat(),
        "simbolos": list(
            SIMBOLOS
        ),
        "intervalo": INTERVALO,
        "horizonte_minutos": HORIZONTE_MINUTOS,
        "periodos": [
            {
                "desde": desde.isoformat(),
                "hasta_exclusivo": hasta.isoformat(),
            }
            for desde, hasta in PERIODOS_HISTORICOS
        ],
        "variables_base": len(
            COLUMNAS_MODELO_V1
        ),
        "variables_relacionadas": len(
            COLUMNAS_CRUZADAS
        ),
        "variables_totales": len(
            COLUMNAS_MODELO_COMPLETO
        ),
        "filas_finales_por_simbolo": total_filas,
        "postgresql_modificado": False,
        "modelos_modificados": False,
        "archivos_validacion": {
            "csv": str(
                RUTA_VALIDACION_CSV
            ),
            "markdown": str(
                RUTA_VALIDACION_MD
            ),
        },
    }

    with RUTA_MANIFIESTO.open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            manifiesto,
            archivo,
            ensure_ascii=False,
            indent=4,
        )

    print(
        "\nINTEGRACIÓN HISTÓRICA APROBADA"
    )
    print("=" * 72)
    print(
        f"Validación CSV: {RUTA_VALIDACION_CSV}"
    )
    print(
        f"Resumen: {RUTA_VALIDACION_MD}"
    )
    print(
        f"Manifiesto: {RUTA_MANIFIESTO}"
    )
    print(
        "\nNo se modificaron modelos ni resultados existentes."
    )


def ejecutar(
    fase: str,
    sobrescribir: bool,
) -> None:
    """Ejecuta una fase concreta o el proceso completo."""

    if fase in (
        "preparar",
        "todo",
    ):
        preparar_historico(
            sobrescribir=sobrescribir
        )

    if fase in (
        "cruzar",
        "todo",
    ):
        generar_variables_completas(
            sobrescribir=sobrescribir
        )

    if fase in (
        "validar",
        "todo",
    ):
        validar_integracion()


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene los argumentos del comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Integra el histórico sincronizado 2017-2020 "
            "en el pipeline de variables que consume V4."
        )
    )

    parser.add_argument(
        "--fase",
        choices=(
            "preparar",
            "cruzar",
            "validar",
            "todo",
        ),
        default="todo",
        help=(
            "Fase que se ejecutará. Por defecto realiza "
            "preparación, variables relacionadas y validación."
        ),
    )

    parser.add_argument(
        "--sobrescribir",
        action="store_true",
        help=(
            "Permite regenerar únicamente los archivos históricos "
            "con los mismos nombres."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Punto de entrada."""

    argumentos = obtener_argumentos()

    try:
        ejecutar(
            fase=argumentos.fase,
            sobrescribir=argumentos.sobrescribir,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        psycopg.Error,
    ) as error:
        print(
            "\nNo se pudo completar la integración histórica."
        )
        print(
            f"Detalle: {error}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
