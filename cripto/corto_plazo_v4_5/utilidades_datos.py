from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from compartido.base_datos.conexion import obtener_conexion
from cripto.corto_plazo_v4_5.configuracion import (
    COLUMNAS_FLUJO_SPOT,
    COLUMNAS_FUTUROS,
    COLUMNAS_META,
    FILAS_CONTEXTO,
    INTERVALO,
    RUTA_DATOS_BASE,
    RUTA_DATOS_COMPLETOS,
    RUTA_DATOS_SPOT,
    RUTA_FUNDING,
    RUTA_KLINES_FUTUROS,
    VENTANAS_FLUJO_FUTUROS,
    VENTANAS_FLUJO_SPOT,
    VENTANAS_FUTUROS_RENDIMIENTO,
)


COLUMNAS_KLINES = (
    "fecha_apertura",
    "precio_apertura",
    "precio_maximo",
    "precio_minimo",
    "precio_cierre",
    "volumen",
    "fecha_cierre",
    "volumen_cotizacion",
    "numero_operaciones",
    "volumen_comprador_base",
    "volumen_comprador_cotizacion",
    "ignorar",
)


def ruta_archivo_base(
    simbolo: str,
    desde: str,
    hasta: str,
) -> Path:
    return (
        RUTA_DATOS_BASE
        / f"{simbolo}_1m_4h_{desde}_{hasta}_variables.parquet"
    )


def ruta_archivo_etapa(
    simbolo: str,
    desde: str,
    hasta: str,
    etapa: str,
) -> Path:
    if etapa == "spot":
        carpeta = RUTA_DATOS_SPOT
    elif etapa == "completo":
        carpeta = RUTA_DATOS_COMPLETOS
    else:
        raise ValueError(f"Etapa desconocida: {etapa}")

    return (
        carpeta
        / f"{simbolo}_1m_4h_{desde}_{hasta}_variables.parquet"
    )


def convertir_fecha(valor: str) -> datetime:
    return datetime.strptime(
        valor,
        "%Y-%m-%d",
    ).replace(tzinfo=timezone.utc)


def division_segura(
    numerador: pd.Series,
    denominador: pd.Series,
    valor_si_cero: float = 0.0,
) -> pd.Series:
    valores_numerador = numerador.to_numpy(dtype="float64")
    valores_denominador = denominador.to_numpy(dtype="float64")
    salida = np.full(
        len(numerador),
        valor_si_cero,
        dtype="float64",
    )

    np.divide(
        valores_numerador,
        valores_denominador,
        out=salida,
        where=valores_denominador != 0,
    )

    return pd.Series(
        salida,
        index=numerador.index,
    )


def crear_grupos_continuos(
    datos: pd.DataFrame,
) -> pd.DataFrame:
    datos = datos.sort_values(
        "fecha_apertura"
    ).reset_index(drop=True).copy()

    diferencias = datos[
        "fecha_apertura"
    ].diff()

    datos["grupo_continuo_v4_5"] = (
        diferencias.ne(pd.Timedelta(minutes=1))
        .cumsum()
        .astype("int64")
    )

    return datos


def suma_rodante(
    datos: pd.DataFrame,
    columna: str,
    ventana: int,
) -> pd.Series:
    return (
        datos.groupby(
            "grupo_continuo_v4_5",
            sort=False,
        )[columna]
        .rolling(
            ventana,
            min_periods=ventana,
        )
        .sum()
        .reset_index(
            level=0,
            drop=True,
        )
        .sort_index()
    )


def media_rodante(
    datos: pd.DataFrame,
    columna: str,
    ventana: int,
) -> pd.Series:
    return (
        datos.groupby(
            "grupo_continuo_v4_5",
            sort=False,
        )[columna]
        .rolling(
            ventana,
            min_periods=ventana,
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
        )
        .sort_index()
    )


def desplazamiento_grupo(
    datos: pd.DataFrame,
    columna: str,
    periodos: int,
) -> pd.Series:
    return datos.groupby(
        "grupo_continuo_v4_5",
        sort=False,
    )[columna].shift(periodos)


def consultar_velas_spot(
    simbolo: str,
    desde: str,
    hasta: str,
) -> pd.DataFrame:
    fecha_desde = convertir_fecha(desde) - timedelta(
        minutes=FILAS_CONTEXTO
    )
    fecha_hasta = convertir_fecha(hasta)

    consulta = """
        SELECT
            velas.fecha_apertura,
            velas.precio_cierre,
            velas.volumen,
            velas.volumen_activo_cotizacion AS volumen_cotizacion,
            velas.numero_operaciones,
            velas.volumen_comprador_base,
            velas.volumen_comprador_cotizacion
        FROM velas
        INNER JOIN mercados
            ON mercados.id = velas.mercado_id
        INNER JOIN fuentes_datos
            ON fuentes_datos.id = mercados.fuente_datos_id
        WHERE mercados.simbolo_proveedor = %s
          AND LOWER(mercados.tipo_mercado) = 'spot'
          AND LOWER(fuentes_datos.nombre) = 'binance'
          AND velas.intervalo = %s
          AND velas.cerrada = TRUE
          AND velas.fecha_apertura >= %s
          AND velas.fecha_apertura < %s
        ORDER BY velas.fecha_apertura;
    """

    with obtener_conexion() as conexion:
        with conexion.cursor() as cursor:
            cursor.execute(
                consulta,
                (
                    simbolo,
                    INTERVALO,
                    fecha_desde,
                    fecha_hasta,
                ),
            )
            filas = cursor.fetchall()
            if cursor.description is None:
                raise RuntimeError(
                    "La consulta Spot no devolvió descripción de columnas."
                )
            columnas = [
                columna.name
                for columna in cursor.description
            ]

    if not filas:
        raise ValueError(
            f"No se encontraron velas Spot para {simbolo} {desde} a {hasta}."
        )

    datos = pd.DataFrame(
        filas,
        columns=columnas,
    )

    datos["fecha_apertura"] = pd.to_datetime(
        datos["fecha_apertura"],
        utc=True,
        errors="raise",
    )

    for columna in datos.columns:
        if columna != "fecha_apertura":
            datos[columna] = pd.to_numeric(
                datos[columna],
                errors="coerce",
            )

    if datos.isna().any().any():
        raise ValueError(
            f"Hay nulos en las velas Spot de {simbolo}."
        )

    return crear_grupos_continuos(datos)


def crear_variables_flujo_spot(
    datos: pd.DataFrame,
) -> pd.DataFrame:
    datos = datos.copy()

    datos["volumen_vendedor_base"] = (
        datos["volumen"]
        - datos["volumen_comprador_base"]
    ).clip(lower=0.0)

    datos["volumen_vendedor_cotizacion"] = (
        datos["volumen_cotizacion"]
        - datos["volumen_comprador_cotizacion"]
    ).clip(lower=0.0)

    datos["flujo_neto_base"] = (
        datos["volumen_comprador_base"]
        - datos["volumen_vendedor_base"]
    )

    datos["flujo_neto_cotizacion"] = (
        datos["volumen_comprador_cotizacion"]
        - datos["volumen_vendedor_cotizacion"]
    )

    for ventana in VENTANAS_FLUJO_SPOT:
        compra_base = suma_rodante(
            datos,
            "volumen_comprador_base",
            ventana,
        )
        venta_base = suma_rodante(
            datos,
            "volumen_vendedor_base",
            ventana,
        )
        total_base = compra_base + venta_base

        compra_cotizacion = suma_rodante(
            datos,
            "volumen_comprador_cotizacion",
            ventana,
        )
        venta_cotizacion = suma_rodante(
            datos,
            "volumen_vendedor_cotizacion",
            ventana,
        )
        total_cotizacion = compra_cotizacion + venta_cotizacion

        desequilibrio_base = division_segura(
            compra_base - venta_base,
            total_base,
        )
        desequilibrio_cotizacion = division_segura(
            compra_cotizacion - venta_cotizacion,
            total_cotizacion,
        )

        nombre_desequilibrio = (
            f"desequilibrio_spot_base_{ventana}m"
        )

        datos[nombre_desequilibrio] = desequilibrio_base
        datos[
            f"desequilibrio_spot_cotizacion_{ventana}m"
        ] = desequilibrio_cotizacion

        datos[
            f"log_ratio_compra_venta_spot_base_{ventana}m"
        ] = np.log(
            division_segura(
                compra_base + 1e-12,
                venta_base + 1e-12,
                valor_si_cero=1.0,
            ).clip(
                lower=1e-12,
            )
        )

        datos[
            f"aceleracion_desequilibrio_spot_base_{ventana}m"
        ] = (
            datos[nombre_desequilibrio]
            - desplazamiento_grupo(
                datos,
                nombre_desequilibrio,
                ventana,
            )
        )

        cierre_anterior = desplazamiento_grupo(
            datos,
            "precio_cierre",
            ventana,
        )
        rendimiento = division_segura(
            datos["precio_cierre"],
            cierre_anterior,
        ) - 1.0

        datos[
            f"confirmacion_precio_flujo_spot_{ventana}m"
        ] = rendimiento * desequilibrio_base

    operaciones_15 = suma_rodante(
        datos,
        "numero_operaciones",
        15,
    )
    volumen_15 = suma_rodante(
        datos,
        "volumen_cotizacion",
        15,
    )
    tamano_15 = division_segura(
        volumen_15,
        operaciones_15,
    )

    operaciones_60 = suma_rodante(
        datos,
        "numero_operaciones",
        60,
    )
    volumen_60 = suma_rodante(
        datos,
        "volumen_cotizacion",
        60,
    )
    tamano_60 = division_segura(
        volumen_60,
        operaciones_60,
    )

    operaciones_240 = suma_rodante(
        datos,
        "numero_operaciones",
        240,
    )
    volumen_240 = suma_rodante(
        datos,
        "volumen_cotizacion",
        240,
    )
    tamano_240 = division_segura(
        volumen_240,
        operaciones_240,
    )

    datos[
        "tamano_medio_operacion_spot_15m_relativo_240m"
    ] = division_segura(
        tamano_15,
        tamano_240,
    ) - 1.0

    datos[
        "tamano_medio_operacion_spot_60m_relativo_240m"
    ] = division_segura(
        tamano_60,
        tamano_240,
    ) - 1.0

    return datos[
        [
            "fecha_apertura",
            *COLUMNAS_FLUJO_SPOT,
        ]
    ].copy()


def normalizar_timestamp_binance(
    valores: pd.Series,
) -> pd.Series:
    numeros = pd.to_numeric(
        valores,
        errors="coerce",
    )

    if numeros.isna().any():
        raise ValueError(
            "Hay timestamps no numéricos en un archivo de Binance."
        )

    maximo = float(numeros.max())

    unidad = "us" if maximo >= 1_000_000_000_000_000 else "ms"

    return pd.to_datetime(
        numeros.astype("int64"),
        unit=unidad,
        utc=True,
        errors="raise",
    )


def meses_entre(
    desde: str,
    hasta: str,
) -> list[pd.Period]:
    inicio = pd.Period(
        pd.Timestamp(desde),
        freq="M",
    )
    final = pd.Period(
        pd.Timestamp(hasta) - pd.Timedelta(seconds=1),
        freq="M",
    )

    return list(
        pd.period_range(
            inicio,
            final,
            freq="M",
        )
    )


def ruta_zip_futuros(
    simbolo: str,
    periodo: pd.Period,
) -> Path:
    return (
        RUTA_KLINES_FUTUROS
        / simbolo
        / INTERVALO
        / f"{simbolo}-{INTERVALO}-{periodo.strftime('%Y-%m')}.zip"
    )


def leer_zip_klines(
    ruta_zip: Path,
) -> pd.DataFrame:
    if not ruta_zip.exists():
        raise FileNotFoundError(
            f"No existe el archivo de futuros: {ruta_zip}"
        )

    with ZipFile(
        ruta_zip,
        "r",
    ) as archivo_zip:
        csvs = [
            nombre
            for nombre in archivo_zip.namelist()
            if nombre.lower().endswith(".csv")
        ]

        if len(csvs) != 1:
            raise ValueError(
                f"{ruta_zip.name} debe contener exactamente un CSV."
            )

        with archivo_zip.open(csvs[0]) as archivo_binario:
            with TextIOWrapper(
                archivo_binario,
                encoding="utf-8",
                newline="",
            ) as archivo_texto:
                lector = csv.reader(archivo_texto)
                filas = [
                    fila
                    for fila in lector
                    if fila
                ]

    if not filas:
        raise ValueError(
            f"El archivo {ruta_zip.name} está vacío."
        )

    if not filas[0][0].strip().isdigit():
        filas = filas[1:]

    if any(len(fila) != 12 for fila in filas):
        raise ValueError(
            f"{ruta_zip.name} contiene filas con columnas inesperadas."
        )

    datos = pd.DataFrame(
        filas,
        columns=COLUMNAS_KLINES,
    )

    datos["fecha_apertura"] = normalizar_timestamp_binance(
        datos["fecha_apertura"]
    )

    columnas_numericas = [
        columna
        for columna in COLUMNAS_KLINES
        if columna not in {
            "fecha_apertura",
            "fecha_cierre",
            "ignorar",
        }
    ]

    for columna in columnas_numericas:
        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce",
        )

    if datos[columnas_numericas].isna().any().any():
        raise ValueError(
            f"{ruta_zip.name} contiene valores numéricos inválidos."
        )

    return datos


def cargar_klines_futuros(
    simbolo: str,
    desde: str,
    hasta: str,
) -> pd.DataFrame:
    fecha_desde = (
        pd.Timestamp(desde, tz="UTC")
        - pd.Timedelta(minutes=FILAS_CONTEXTO)
    )
    fecha_hasta = pd.Timestamp(
        hasta,
        tz="UTC",
    )

    periodos = meses_entre(
        fecha_desde.strftime("%Y-%m-%d"),
        hasta,
    )

    partes = [
        leer_zip_klines(
            ruta_zip_futuros(
                simbolo,
                periodo,
            )
        )
        for periodo in periodos
    ]

    datos = pd.concat(
        partes,
        ignore_index=True,
    )

    datos = datos.loc[
        (datos["fecha_apertura"] >= fecha_desde)
        & (datos["fecha_apertura"] < fecha_hasta)
    ].copy()

    datos = (
        datos.sort_values("fecha_apertura")
        .drop_duplicates(
            subset=["fecha_apertura"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return crear_grupos_continuos(datos)


def ruta_funding(
    simbolo: str,
) -> Path:
    return RUTA_FUNDING / f"{simbolo}_funding.parquet"


def cargar_funding(
    simbolo: str,
) -> pd.DataFrame:
    ruta = ruta_funding(simbolo)

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el histórico de funding: {ruta}"
        )

    datos = pd.read_parquet(ruta)

    requeridas = {
        "fecha_funding",
        "funding_rate",
    }

    faltantes = requeridas.difference(datos.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas en {ruta.name}: {sorted(faltantes)}"
        )

    datos = datos.copy()

    datos["fecha_funding"] = pd.to_datetime(
        datos["fecha_funding"],
        utc=True,
        errors="raise",
    )

    datos["funding_rate"] = pd.to_numeric(
        datos["funding_rate"],
        errors="coerce",
    )

    if datos["funding_rate"].isna().any():
        raise ValueError(
            f"Hay funding inválido en {ruta.name}."
        )

    return (
        datos.sort_values("fecha_funding")
        .drop_duplicates(
            subset=["fecha_funding"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def crear_variables_futuros(
    datos_futuros: pd.DataFrame,
    datos_spot: pd.DataFrame,
    datos_funding: pd.DataFrame,
) -> pd.DataFrame:
    futuros = datos_futuros.copy()

    futuros["volumen_vendedor_base"] = (
        futuros["volumen"]
        - futuros["volumen_comprador_base"]
    ).clip(lower=0.0)

    futuros["volumen_vendedor_cotizacion"] = (
        futuros["volumen_cotizacion"]
        - futuros["volumen_comprador_cotizacion"]
    ).clip(lower=0.0)

    for ventana in VENTANAS_FUTUROS_RENDIMIENTO:
        cierre_anterior = desplazamiento_grupo(
            futuros,
            "precio_cierre",
            ventana,
        )

        futuros[
            f"rendimiento_futuros_{ventana}m"
        ] = division_segura(
            futuros["precio_cierre"],
            cierre_anterior,
        ) - 1.0

    for ventana in VENTANAS_FLUJO_FUTUROS:
        compra_base = suma_rodante(
            futuros,
            "volumen_comprador_base",
            ventana,
        )
        venta_base = suma_rodante(
            futuros,
            "volumen_vendedor_base",
            ventana,
        )
        compra_cotizacion = suma_rodante(
            futuros,
            "volumen_comprador_cotizacion",
            ventana,
        )
        venta_cotizacion = suma_rodante(
            futuros,
            "volumen_vendedor_cotizacion",
            ventana,
        )

        futuros[
            f"desequilibrio_futuros_base_{ventana}m"
        ] = division_segura(
            compra_base - venta_base,
            compra_base + venta_base,
        )

        futuros[
            f"desequilibrio_futuros_cotizacion_{ventana}m"
        ] = division_segura(
            compra_cotizacion - venta_cotizacion,
            compra_cotizacion + venta_cotizacion,
        )

        futuros[
            f"confirmacion_precio_flujo_futuros_{ventana}m"
        ] = (
            futuros[
                f"rendimiento_futuros_{ventana}m"
            ]
            * futuros[
                f"desequilibrio_futuros_base_{ventana}m"
            ]
        )

    columnas_futuros = [
        "fecha_apertura",
        "precio_cierre",
        *[
            f"rendimiento_futuros_{ventana}m"
            for ventana in VENTANAS_FUTUROS_RENDIMIENTO
        ],
        *[
            f"desequilibrio_futuros_base_{ventana}m"
            for ventana in VENTANAS_FLUJO_FUTUROS
        ],
        *[
            f"desequilibrio_futuros_cotizacion_{ventana}m"
            for ventana in VENTANAS_FLUJO_FUTUROS
        ],
        *[
            f"confirmacion_precio_flujo_futuros_{ventana}m"
            for ventana in VENTANAS_FLUJO_FUTUROS
        ],
    ]

    futuros = futuros[
        columnas_futuros
    ].rename(
        columns={
            "precio_cierre": "precio_cierre_futuros",
        }
    )

    combinado = datos_spot.merge(
        futuros,
        on="fecha_apertura",
        how="left",
        validate="one_to_one",
    )

    combinado["basis_futuros_spot"] = division_segura(
        combinado["precio_cierre_futuros"],
        combinado["precio_cierre"],
    ) - 1.0

    combinado = crear_grupos_continuos(combinado)

    for ventana in VENTANAS_FUTUROS_RENDIMIENTO:
        combinado[
            f"diferencia_rendimiento_futuros_spot_{ventana}m"
        ] = (
            combinado[
                f"rendimiento_futuros_{ventana}m"
            ]
            - combinado[
                f"rendimiento_{ventana}m"
            ]
        )

        combinado[
            f"cambio_basis_{ventana}m"
        ] = (
            combinado["basis_futuros_spot"]
            - desplazamiento_grupo(
                combinado,
                "basis_futuros_spot",
                ventana,
            )
        )

    for ventana in VENTANAS_FLUJO_FUTUROS:
        combinado[
            f"divergencia_flujo_spot_futuros_{ventana}m"
        ] = (
            combinado[
                f"desequilibrio_spot_base_{ventana}m"
            ]
            - combinado[
                f"desequilibrio_futuros_base_{ventana}m"
            ]
        )

    funding = datos_funding.rename(
        columns={
            "fecha_funding": "fecha_apertura",
        }
    )[
        [
            "fecha_apertura",
            "funding_rate",
        ]
    ].sort_values("fecha_apertura")

    combinado = pd.merge_asof(
        combinado.sort_values("fecha_apertura"),
        funding,
        on="fecha_apertura",
        direction="backward",
        allow_exact_matches=True,
    )

    combinado["funding_ultimo"] = combinado[
        "funding_rate"
    ]

    combinado["funding_absoluto"] = combinado[
        "funding_ultimo"
    ].abs()

    combinado["cambio_funding_8h"] = (
        combinado["funding_ultimo"]
        - combinado.groupby(
            "grupo_continuo_v4_5",
            sort=False,
        )["funding_ultimo"].shift(480)
    )

    combinado["funding_media_24h"] = media_rodante(
        combinado,
        "funding_ultimo",
        1440,
    )

    return combinado[
        [
            "fecha_apertura",
            *COLUMNAS_FUTUROS,
        ]
    ].copy()


def cargar_base(
    simbolo: str,
    desde: str,
    hasta: str,
) -> pd.DataFrame:
    ruta = ruta_archivo_base(
        simbolo,
        desde,
        hasta,
    )

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el archivo base: {ruta}"
        )

    datos = pd.read_parquet(ruta)

    datos["fecha_apertura"] = pd.to_datetime(
        datos["fecha_apertura"],
        utc=True,
        errors="raise",
    )

    if "fecha_objetivo" in datos.columns:
        datos["fecha_objetivo"] = pd.to_datetime(
            datos["fecha_objetivo"],
            utc=True,
            errors="raise",
        )

    return (
        datos.sort_values("fecha_apertura")
        .reset_index(drop=True)
    )


def validar_columnas_finitas(
    datos: pd.DataFrame,
    columnas: tuple[str, ...] | list[str],
    contexto: str,
) -> None:
    faltantes = set(columnas).difference(datos.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas en {contexto}: {sorted(faltantes)}"
        )

    valores = datos[
        list(columnas)
    ].to_numpy(
        dtype="float64"
    )

    if not np.isfinite(valores).all():
        raise ValueError(
            f"Hay valores no finitos en {contexto}."
        )


def guardar_parquet_seguro(
    datos: pd.DataFrame,
    ruta: Path,
    sobrescribir: bool,
) -> None:
    if ruta.exists() and not sobrescribir:
        raise FileExistsError(
            f"Ya existe: {ruta}. Usa --sobrescribir."
        )

    ruta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporal = ruta.with_suffix(
        ruta.suffix + ".tmp"
    )

    datos.to_parquet(
        temporal,
        index=False,
    )

    temporal.replace(ruta)
