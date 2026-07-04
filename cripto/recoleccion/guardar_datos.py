from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from io import TextIOWrapper
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import psycopg

from compartido.base_datos.conexion import obtener_conexion


RUTA_PROYECTO = Path(__file__).resolve().parents[2]

RUTA_DATOS_BRUTOS = (
    RUTA_PROYECTO
    / "datos"
    / "cripto"
    / "bruto"
    / "binance"
    / "spot"
    / "klines"
)

TAMANO_LOTE = 1000


def convertir_timestamp_utc(valor: str) -> datetime:
    """
    Convierte timestamps de Binance a datetime UTC.

    Antes de 2025 normalmente están expresados en milisegundos.
    Desde 2025, los archivos Spot utilizan microsegundos.
    """

    timestamp = int(valor)

    if timestamp >= 1_000_000_000_000_000:
        segundos = timestamp / 1_000_000
    else:
        segundos = timestamp / 1_000

    return datetime.fromtimestamp(
        segundos,
        tz=timezone.utc,
    )


def obtener_ids_mercado(simbolo: str) -> tuple[int, int]:
    """Obtiene los identificadores del mercado y de Binance."""

    with obtener_conexion() as conexion:
        with conexion.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    mercados.id,
                    fuentes_datos.id
                FROM mercados
                INNER JOIN fuentes_datos
                    ON fuentes_datos.id = mercados.fuente_datos_id
                WHERE UPPER(mercados.simbolo_proveedor) = UPPER(%s)
                  AND LOWER(mercados.tipo_mercado) = 'spot'
                  AND LOWER(fuentes_datos.nombre) = 'binance'
                  AND mercados.activo = TRUE
                  AND fuentes_datos.activo = TRUE;
                """,
                (simbolo,),
            )

            resultado = cursor.fetchone()

    if resultado is None:
        raise ValueError(
            f"No existe el mercado {simbolo} de Binance Spot "
            "en la base de datos."
        )

    mercado_id, fuente_datos_id = resultado

    return mercado_id, fuente_datos_id


def crear_ejecucion(
    fuente_datos_id: int,
    mercado_id: int,
    archivo: str,
) -> int:
    """Registra el inicio del proceso de importación."""

    with obtener_conexion() as conexion:
        with conexion.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO ejecuciones_recoleccion (
                    fuente_datos_id,
                    mercado_id,
                    tipo_proceso,
                    archivo,
                    estado
                )
                VALUES (%s, %s, %s, %s, 'iniciada')
                RETURNING id;
                """,
                (
                    fuente_datos_id,
                    mercado_id,
                    "historico_mensual",
                    archivo,
                ),
            )

            resultado = cursor.fetchone()

            if resultado is None:
                raise RuntimeError(
                    "No se pudo obtener el identificador de la ejecución."
                )

            ejecucion_id = resultado[0]

        conexion.commit()

    return ejecucion_id


def completar_ejecucion(
    ejecucion_id: int,
    registros_leidos: int,
    registros_insertados: int,
    registros_duplicados: int,
    registros_error: int,
) -> None:
    """
    Marca la ejecución como completada.

    Las filas inválidas se cuentan y se omiten, pero no se inventan
    ni se insertan en la tabla de velas.
    """

    mensaje = None

    if registros_error > 0:
        mensaje = (
            f"Se omitieron {registros_error} fila(s) inválida(s) "
            "del archivo de origen."
        )

    with obtener_conexion() as conexion:
        with conexion.cursor() as cursor:
            cursor.execute(
                """
                UPDATE ejecuciones_recoleccion
                SET
                    fecha_fin = CURRENT_TIMESTAMP,
                    registros_leidos = %s,
                    registros_insertados = %s,
                    registros_duplicados = %s,
                    registros_error = %s,
                    estado = 'completada',
                    mensaje_error = %s
                WHERE id = %s;
                """,
                (
                    registros_leidos,
                    registros_insertados,
                    registros_duplicados,
                    registros_error,
                    mensaje,
                    ejecucion_id,
                ),
            )

        conexion.commit()


def registrar_error_ejecucion(
    ejecucion_id: int,
    registros_leidos: int,
    mensaje_error: str,
) -> None:
    """Marca la ejecución como fallida."""

    with obtener_conexion() as conexion:
        with conexion.cursor() as cursor:
            cursor.execute(
                """
                UPDATE ejecuciones_recoleccion
                SET
                    fecha_fin = CURRENT_TIMESTAMP,
                    registros_leidos = %s,
                    registros_insertados = 0,
                    registros_duplicados = 0,
                    registros_error = 1,
                    estado = 'error',
                    mensaje_error = %s
                WHERE id = %s;
                """,
                (
                    registros_leidos,
                    mensaje_error,
                    ejecucion_id,
                ),
            )

        conexion.commit()


def validar_vela(
    fecha_apertura: datetime,
    fecha_cierre: datetime,
    precio_apertura: Decimal,
    precio_maximo: Decimal,
    precio_minimo: Decimal,
    precio_cierre: Decimal,
    volumen: Decimal,
    volumen_cotizacion: Decimal,
    numero_operaciones: int,
    volumen_comprador_base: Decimal,
    volumen_comprador_cotizacion: Decimal,
) -> None:
    """Comprueba que los valores principales sean coherentes."""

    if fecha_cierre <= fecha_apertura:
        raise ValueError(
            "La fecha de cierre debe ser posterior a la apertura."
        )

    precios = [
        precio_apertura,
        precio_maximo,
        precio_minimo,
        precio_cierre,
    ]

    if any(precio <= 0 for precio in precios):
        raise ValueError("Los precios deben ser mayores que cero.")

    if precio_maximo < max(
        precio_apertura,
        precio_cierre,
        precio_minimo,
    ):
        raise ValueError("El precio máximo de la vela es incoherente.")

    if precio_minimo > min(
        precio_apertura,
        precio_cierre,
        precio_maximo,
    ):
        raise ValueError("El precio mínimo de la vela es incoherente.")

    volumenes = [
        volumen,
        volumen_cotizacion,
        volumen_comprador_base,
        volumen_comprador_cotizacion,
    ]

    if any(valor < 0 for valor in volumenes):
        raise ValueError("Los volúmenes no pueden ser negativos.")

    if numero_operaciones < 0:
        raise ValueError(
            "El número de operaciones no puede ser negativo."
        )


def transformar_fila(
    fila: list[str],
    mercado_id: int,
    intervalo: str,
    archivo_origen: str,
) -> tuple:
    """Transforma una fila del CSV al formato de la tabla velas."""

    if len(fila) != 12:
        raise ValueError(
            f"La fila contiene {len(fila)} columnas; se esperaban 12."
        )

    try:
        fecha_apertura = convertir_timestamp_utc(fila[0])
        precio_apertura = Decimal(fila[1])
        precio_maximo = Decimal(fila[2])
        precio_minimo = Decimal(fila[3])
        precio_cierre = Decimal(fila[4])
        volumen = Decimal(fila[5])
        fecha_cierre = convertir_timestamp_utc(fila[6])
        volumen_cotizacion = Decimal(fila[7])
        numero_operaciones = int(fila[8])
        volumen_comprador_base = Decimal(fila[9])
        volumen_comprador_cotizacion = Decimal(fila[10])

    except (ValueError, InvalidOperation) as error:
        raise ValueError(
            "La fila contiene un valor numérico inválido."
        ) from error

    validar_vela(
        fecha_apertura=fecha_apertura,
        fecha_cierre=fecha_cierre,
        precio_apertura=precio_apertura,
        precio_maximo=precio_maximo,
        precio_minimo=precio_minimo,
        precio_cierre=precio_cierre,
        volumen=volumen,
        volumen_cotizacion=volumen_cotizacion,
        numero_operaciones=numero_operaciones,
        volumen_comprador_base=volumen_comprador_base,
        volumen_comprador_cotizacion=volumen_comprador_cotizacion,
    )

    return (
        mercado_id,
        intervalo,
        fecha_apertura,
        fecha_cierre,
        precio_apertura,
        precio_maximo,
        precio_minimo,
        precio_cierre,
        volumen,
        volumen_cotizacion,
        numero_operaciones,
        volumen_comprador_base,
        volumen_comprador_cotizacion,
        True,
        archivo_origen,
    )


def insertar_lote(
    cursor: psycopg.Cursor,
    registros: list[tuple],
) -> int:
    """Inserta un lote de velas ignorando duplicados."""

    if not registros:
        return 0

    valores_por_registro = 15

    plantilla_registro = (
        "("
        + ", ".join(["%s"] * valores_por_registro)
        + ")"
    )

    plantillas = ", ".join(
        [plantilla_registro] * len(registros)
    )

    parametros = [
        valor
        for registro in registros
        for valor in registro
    ]

    cursor.execute(
        f"""
        INSERT INTO velas (
            mercado_id,
            intervalo,
            fecha_apertura,
            fecha_cierre,
            precio_apertura,
            precio_maximo,
            precio_minimo,
            precio_cierre,
            volumen,
            volumen_activo_cotizacion,
            numero_operaciones,
            volumen_comprador_base,
            volumen_comprador_cotizacion,
            cerrada,
            archivo_origen
        )
        VALUES {plantillas}
        ON CONFLICT (
            mercado_id,
            intervalo,
            fecha_apertura
        )
        DO NOTHING;
        """,
        parametros,
    )

    return cursor.rowcount


def importar_archivo(
    ruta_zip: Path,
    simbolo: str,
    intervalo: str,
) -> None:
    """Lee el ZIP, valida sus velas y las guarda en PostgreSQL."""

    if not ruta_zip.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {ruta_zip}"
        )

    mercado_id, fuente_datos_id = obtener_ids_mercado(simbolo)

    ejecucion_id = crear_ejecucion(
        fuente_datos_id=fuente_datos_id,
        mercado_id=mercado_id,
        archivo=ruta_zip.name,
    )

    registros_leidos = 0
    registros_insertados = 0
    registros_error = 0

    try:
        with ZipFile(ruta_zip, "r") as archivo_zip:
            archivos_csv = [
                nombre
                for nombre in archivo_zip.namelist()
                if nombre.lower().endswith(".csv")
            ]

            if len(archivos_csv) != 1:
                raise ValueError(
                    "El ZIP debe contener exactamente un archivo CSV."
                )

            nombre_csv = archivos_csv[0]

            with obtener_conexion() as conexion:
                with conexion.cursor() as cursor:
                    with archivo_zip.open(nombre_csv) as archivo_binario:
                        with TextIOWrapper(
                            archivo_binario,
                            encoding="utf-8",
                            newline="",
                        ) as archivo_texto:

                            lector = csv.reader(archivo_texto)
                            lote: list[tuple] = []

                            for numero_linea, fila in enumerate(
                                lector,
                                start=1,
                            ):
                                if not fila:
                                    continue

                                if (
                                    numero_linea == 1
                                    and not fila[0].strip().isdigit()
                                ):
                                    continue

                                registros_leidos += 1

                                try:
                                    registro = transformar_fila(
                                        fila=fila,
                                        mercado_id=mercado_id,
                                        intervalo=intervalo,
                                        archivo_origen=ruta_zip.name,
                                    )

                                except ValueError as error:
                                    registros_error += 1

                                    print(
                                        "Advertencia: se omitió la línea "
                                        f"{numero_linea}: {error}"
                                    )

                                    continue

                                lote.append(registro)

                                if len(lote) >= TAMANO_LOTE:
                                    registros_insertados += insertar_lote(
                                        cursor,
                                        lote,
                                    )
                                    lote.clear()

                            if lote:
                                registros_insertados += insertar_lote(
                                    cursor,
                                    lote,
                                )

                conexion.commit()

        registros_duplicados = (
            registros_leidos
            - registros_insertados
            - registros_error
        )

        completar_ejecucion(
            ejecucion_id=ejecucion_id,
            registros_leidos=registros_leidos,
            registros_insertados=registros_insertados,
            registros_duplicados=registros_duplicados,
            registros_error=registros_error,
        )

        print("\nImportación completada correctamente.")
        print(f"Archivo: {ruta_zip.name}")
        print(f"Registros leídos: {registros_leidos}")
        print(f"Registros insertados: {registros_insertados}")
        print(f"Registros duplicados: {registros_duplicados}")
        print(f"Registros con error: {registros_error}")

    except Exception as error:
        registrar_error_ejecucion(
            ejecucion_id=ejecucion_id,
            registros_leidos=registros_leidos,
            mensaje_error=str(error),
        )

        raise


def obtener_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Importa velas históricas de Binance Spot "
            "desde un archivo ZIP."
        )
    )

    parser.add_argument(
        "--simbolo",
        default="BTCUSDT",
        help="Símbolo del mercado.",
    )

    parser.add_argument(
        "--intervalo",
        default="1m",
        help="Intervalo de las velas.",
    )

    parser.add_argument(
        "--anio",
        type=int,
        required=True,
        help="Año del archivo.",
    )

    parser.add_argument(
        "--mes",
        type=int,
        required=True,
        help="Mes del archivo.",
    )

    return parser.parse_args()


def main() -> None:
    argumentos = obtener_argumentos()

    simbolo = argumentos.simbolo.upper().strip()
    intervalo = argumentos.intervalo.strip()

    nombre_archivo = (
        f"{simbolo}-{intervalo}-"
        f"{argumentos.anio}-{argumentos.mes:02d}.zip"
    )

    ruta_zip = (
        RUTA_DATOS_BRUTOS
        / simbolo
        / intervalo
        / nombre_archivo
    )

    print(f"Importando: {ruta_zip}")

    try:
        importar_archivo(
            ruta_zip=ruta_zip,
            simbolo=simbolo,
            intervalo=intervalo,
        )

    except (
        FileNotFoundError,
        BadZipFile,
        psycopg.Error,
        ValueError,
        OSError,
    ) as error:
        print("\nNo se pudo completar la importación.")
        print(f"Detalle: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()