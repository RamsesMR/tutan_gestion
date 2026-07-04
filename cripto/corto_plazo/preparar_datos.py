from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import psycopg

from compartido.base_datos.conexion import obtener_conexion
from cripto.corto_plazo.configuracion import (
    COLUMNAS_BASE,
    HORIZONTE_MINUTOS,
    INTERVALO,
    NOMBRE_HORIZONTE,
    RUTA_DATOS_PREPARADOS,
    SIMBOLOS,
)


COLUMNAS_NUMERICAS = (
    "precio_apertura",
    "precio_maximo",
    "precio_minimo",
    "precio_cierre",
    "volumen",
    "volumen_activo_cotizacion",
    "numero_operaciones",
    "volumen_comprador_base",
    "volumen_comprador_cotizacion",
)


def convertir_fecha(valor: str) -> datetime:
    """Convierte una fecha YYYY-MM-DD en una fecha UTC."""

    try:
        return datetime.strptime(
            valor,
            "%Y-%m-%d",
        ).replace(tzinfo=timezone.utc)

    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "La fecha debe tener el formato YYYY-MM-DD."
        ) from error


def validar_argumentos(
    simbolo: str,
    fecha_desde: datetime,
    fecha_hasta: datetime,
) -> None:
    """Comprueba que el símbolo y las fechas sean válidos."""

    if simbolo not in SIMBOLOS:
        raise ValueError(
            f"Símbolo no permitido: {simbolo}. "
            f"Permitidos: {', '.join(SIMBOLOS)}"
        )

    if fecha_desde >= fecha_hasta:
        raise ValueError(
            "La fecha inicial debe ser anterior a la fecha final."
        )


def consultar_velas(
    simbolo: str,
    fecha_desde: datetime,
    fecha_hasta: datetime,
) -> pd.DataFrame:
    """
    Consulta las velas necesarias para preparar el conjunto.

    Se consultan cuatro horas anteriores para calcular las variables
    de las primeras muestras del periodo.

    También se consultan cuatro horas posteriores para calcular
    el objetivo de las últimas muestras del periodo.
    """

    fecha_desde_consulta = fecha_desde - timedelta(
        minutes=HORIZONTE_MINUTOS
    )

    fecha_hasta_consulta = fecha_hasta + timedelta(
        minutes=HORIZONTE_MINUTOS
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
        WHERE mercados.simbolo_proveedor = %s
          AND mercados.tipo_mercado = 'spot'
          AND fuentes_datos.nombre = 'Binance'
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


def normalizar_tipos(datos: pd.DataFrame) -> pd.DataFrame:
    """Convierte las columnas al formato adecuado."""

    datos = datos.copy()

    datos["fecha_apertura"] = pd.to_datetime(
        datos["fecha_apertura"],
        utc=True,
    )

    for columna in COLUMNAS_NUMERICAS:
        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce",
        )

    if datos[list(COLUMNAS_NUMERICAS)].isna().any().any():
        raise ValueError(
            "Se encontraron valores numéricos que no pudieron convertirse."
        )

    datos["numero_operaciones"] = datos[
        "numero_operaciones"
    ].astype("int64")

    return (
        datos
        .sort_values("fecha_apertura")
        .reset_index(drop=True)
    )


def agregar_objetivo(
    datos: pd.DataFrame,
    fecha_desde: datetime,
    fecha_hasta: datetime,
) -> tuple[pd.DataFrame, int]:
    """
    Crea el rendimiento futuro a cuatro horas.

    Las filas anteriores al periodo se conservan temporalmente
    para que variables.py pueda utilizarlas como contexto.

    Solo las filas pertenecientes al periodo solicitado se marcarán
    como muestras finales.
    """

    datos = datos.copy()

    diferencia = datos["fecha_apertura"].diff()

    datos["grupo_continuo"] = (
        diferencia.ne(pd.Timedelta(minutes=1))
    ).cumsum()

    datos["fecha_objetivo"] = (
        datos["fecha_apertura"]
        + pd.Timedelta(minutes=HORIZONTE_MINUTOS)
    )

    cierres_por_fecha = pd.Series(
        datos["precio_cierre"].to_numpy(),
        index=datos["fecha_apertura"],
    )

    grupos_por_fecha = pd.Series(
        datos["grupo_continuo"].to_numpy(),
        index=datos["fecha_apertura"],
    )

    datos["precio_cierre_futuro"] = (
        cierres_por_fecha
        .reindex(datos["fecha_objetivo"])
        .to_numpy()
    )

    grupo_objetivo = (
        grupos_por_fecha
        .reindex(datos["fecha_objetivo"])
        .to_numpy()
    )

    dentro_periodo = (
        (datos["fecha_apertura"] >= fecha_desde)
        & (datos["fecha_apertura"] < fecha_hasta)
    )

    objetivo_disponible = datos[
        "precio_cierre_futuro"
    ].notna()

    periodo_continuo = (
        grupo_objetivo
        == datos["grupo_continuo"].to_numpy()
    )

    muestras_validas = (
        objetivo_disponible
        & periodo_continuo
    )

    muestras_validas_periodo = (
        muestras_validas
        & dentro_periodo
    )

    total_periodo = int(dentro_periodo.sum())

    datos_preparados = datos.loc[
        muestras_validas
    ].copy()

    datos_preparados["es_muestra_periodo"] = (
        dentro_periodo
        .loc[muestras_validas]
        .to_numpy()
    )

    datos_preparados["rendimiento_objetivo"] = (
        datos_preparados["precio_cierre_futuro"]
        / datos_preparados["precio_cierre"]
        - 1
    )

    columnas_salida = [
        "simbolo",
        *COLUMNAS_BASE,
        "fecha_objetivo",
        "precio_cierre_futuro",
        "rendimiento_objetivo",
        "es_muestra_periodo",
    ]

    datos_preparados = (
        datos_preparados[columnas_salida]
        .reset_index(drop=True)
    )

    muestras_descartadas = (
        total_periodo
        - int(muestras_validas_periodo.sum())
    )

    return datos_preparados, muestras_descartadas


def construir_ruta_salida(
    simbolo: str,
    fecha_desde: datetime,
    fecha_hasta: datetime,
) -> Path:
    """Construye el nombre del archivo Parquet."""

    nombre_archivo = (
        f"{simbolo}_"
        f"{INTERVALO}_"
        f"{NOMBRE_HORIZONTE}_"
        f"{fecha_desde:%Y-%m-%d}_"
        f"{fecha_hasta:%Y-%m-%d}.parquet"
    )

    RUTA_DATOS_PREPARADOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    return RUTA_DATOS_PREPARADOS / nombre_archivo


def preparar_datos(
    simbolo: str,
    fecha_desde: datetime,
    fecha_hasta: datetime,
) -> Path:
    """Consulta, prepara y guarda un conjunto de datos."""

    simbolo = simbolo.upper().strip()

    validar_argumentos(
        simbolo=simbolo,
        fecha_desde=fecha_desde,
        fecha_hasta=fecha_hasta,
    )

    print("\nPREPARACIÓN DE DATOS DE CORTO PLAZO")
    print("=" * 60)
    print(f"Símbolo: {simbolo}")
    print(f"Intervalo: {INTERVALO}")
    print(f"Horizonte: {NOMBRE_HORIZONTE}")
    print(f"Desde: {fecha_desde}")
    print(f"Hasta: {fecha_hasta}")

    print("\nConsultando PostgreSQL...")

    datos = consultar_velas(
        simbolo=simbolo,
        fecha_desde=fecha_desde,
        fecha_hasta=fecha_hasta,
    )

    print(f"Velas consultadas: {len(datos)}")

    datos = normalizar_tipos(datos)

    datos_preparados, muestras_descartadas = agregar_objetivo(
        datos=datos,
        fecha_desde=fecha_desde,
        fecha_hasta=fecha_hasta,
    )

    ruta_salida = construir_ruta_salida(
        simbolo=simbolo,
        fecha_desde=fecha_desde,
        fecha_hasta=fecha_hasta,
    )

    datos_preparados.to_parquet(
        ruta_salida,
        index=False,
        compression="snappy",
    )

    muestras_periodo = int(
        datos_preparados["es_muestra_periodo"].sum()
    )

    muestras_contexto = (
        len(datos_preparados) - muestras_periodo
    )

    print("\nPreparación completada correctamente.")
    print(f"Filas guardadas: {len(datos_preparados)}")
    print(f"Muestras del periodo: {muestras_periodo}")
    print(f"Filas de contexto: {muestras_contexto}")
    print(f"Muestras descartadas: {muestras_descartadas}")
    print(f"Archivo: {ruta_salida}")
    print(
        "Tamaño: "
        f"{ruta_salida.stat().st_size / (1024 * 1024):.2f} MB"
    )

    return ruta_salida


def obtener_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara datos de velas de un minuto para "
            "el modelo cripto de corto plazo."
        )
    )

    parser.add_argument(
        "--simbolo",
        required=True,
        choices=SIMBOLOS,
    )

    parser.add_argument(
        "--desde",
        required=True,
        type=convertir_fecha,
        help="Fecha inicial incluida. Formato YYYY-MM-DD.",
    )

    parser.add_argument(
        "--hasta",
        required=True,
        type=convertir_fecha,
        help="Fecha final excluida. Formato YYYY-MM-DD.",
    )

    return parser.parse_args()


def main() -> None:
    argumentos = obtener_argumentos()

    try:
        preparar_datos(
            simbolo=argumentos.simbolo,
            fecha_desde=argumentos.desde,
            fecha_hasta=argumentos.hasta,
        )

    except (
        psycopg.Error,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print("\nNo se pudo completar la preparación.")
        print(f"Detalle: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()