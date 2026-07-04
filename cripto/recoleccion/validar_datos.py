from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import psycopg

from compartido.base_datos.conexion import obtener_conexion


INTERVALOS = {
    "1m": timedelta(minutes=1),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}


def obtener_periodo(anio: int, mes: int) -> tuple[datetime, datetime]:
    """Devuelve el inicio del mes y el inicio del mes siguiente."""

    if mes < 1 or mes > 12:
        raise ValueError("El mes debe estar comprendido entre 1 y 12.")

    inicio = datetime(
        year=anio,
        month=mes,
        day=1,
        tzinfo=timezone.utc,
    )

    if mes == 12:
        fin = datetime(
            year=anio + 1,
            month=1,
            day=1,
            tzinfo=timezone.utc,
        )
    else:
        fin = datetime(
            year=anio,
            month=mes + 1,
            day=1,
            tzinfo=timezone.utc,
        )

    return inicio, fin


def obtener_mercado_id(
    cursor: psycopg.Cursor,
    simbolo: str,
) -> int:
    """Obtiene el mercado Binance Spot correspondiente."""

    cursor.execute(
        """
        SELECT mercados.id
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
            f"No existe el mercado {simbolo} de Binance Spot."
        )

    return resultado[0]


def obtener_resumen(
    cursor: psycopg.Cursor,
    mercado_id: int,
    intervalo: str,
    inicio: datetime,
    fin: datetime,
) -> tuple:
    """Obtiene el resumen principal del periodo."""

    cursor.execute(
        """
        SELECT
            COUNT(*) AS total,
            COUNT(DISTINCT fecha_apertura) AS fechas_distintas,
            MIN(fecha_apertura) AS primera_vela,
            MAX(fecha_apertura) AS ultima_vela
        FROM velas
        WHERE mercado_id = %s
          AND intervalo = %s
          AND fecha_apertura >= %s
          AND fecha_apertura < %s;
        """,
        (
            mercado_id,
            intervalo,
            inicio,
            fin,
        ),
    )

    return cursor.fetchone()


def contar_velas_incoherentes(
    cursor: psycopg.Cursor,
    mercado_id: int,
    intervalo: str,
    inicio: datetime,
    fin: datetime,
) -> int:
    """Cuenta velas cuyos valores no sean coherentes."""

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM velas
        WHERE mercado_id = %s
          AND intervalo = %s
          AND fecha_apertura >= %s
          AND fecha_apertura < %s
          AND (
                fecha_cierre <= fecha_apertura

                OR precio_apertura <= 0
                OR precio_maximo <= 0
                OR precio_minimo <= 0
                OR precio_cierre <= 0

                OR precio_maximo < GREATEST(
                    precio_apertura,
                    precio_cierre,
                    precio_minimo
                )

                OR precio_minimo > LEAST(
                    precio_apertura,
                    precio_cierre,
                    precio_maximo
                )

                OR volumen < 0
                OR volumen_activo_cotizacion < 0
                OR numero_operaciones < 0
                OR volumen_comprador_base < 0
                OR volumen_comprador_cotizacion < 0
          );
        """,
        (
            mercado_id,
            intervalo,
            inicio,
            fin,
        ),
    )

    return cursor.fetchone()[0]


def obtener_fechas_apertura(
    cursor: psycopg.Cursor,
    mercado_id: int,
    intervalo: str,
    inicio: datetime,
    fin: datetime,
) -> list[datetime]:
    """Obtiene las fechas de apertura ordenadas."""

    cursor.execute(
        """
        SELECT fecha_apertura
        FROM velas
        WHERE mercado_id = %s
          AND intervalo = %s
          AND fecha_apertura >= %s
          AND fecha_apertura < %s
        ORDER BY fecha_apertura;
        """,
        (
            mercado_id,
            intervalo,
            inicio,
            fin,
        ),
    )

    return [fila[0] for fila in cursor.fetchall()]


def detectar_huecos(
    fechas: list[datetime],
    duracion_intervalo: timedelta,
) -> list[tuple[datetime, datetime, int]]:
    """Detecta huecos entre velas consecutivas."""

    huecos = []

    for fecha_anterior, fecha_actual in zip(
        fechas,
        fechas[1:],
    ):
        diferencia = fecha_actual - fecha_anterior

        if diferencia > duracion_intervalo:
            velas_faltantes = (
                int(diferencia / duracion_intervalo) - 1
            )

            huecos.append(
                (
                    fecha_anterior,
                    fecha_actual,
                    velas_faltantes,
                )
            )

    return huecos


def validar_periodo(
    simbolo: str,
    intervalo: str,
    anio: int,
    mes: int,
) -> None:
    """Valida un mes concreto de velas."""

    simbolo = simbolo.upper().strip()
    intervalo = intervalo.strip()

    if intervalo not in INTERVALOS:
        raise ValueError(
            f"Intervalo no permitido: {intervalo}. "
            f"Permitidos: {', '.join(INTERVALOS)}"
        )

    inicio, fin = obtener_periodo(anio, mes)
    duracion_intervalo = INTERVALOS[intervalo]

    total_esperado = int(
        (fin - inicio) / duracion_intervalo
    )

    with obtener_conexion() as conexion:
        with conexion.cursor() as cursor:
            mercado_id = obtener_mercado_id(
                cursor,
                simbolo,
            )

            (
                total,
                fechas_distintas,
                primera_vela,
                ultima_vela,
            ) = obtener_resumen(
                cursor=cursor,
                mercado_id=mercado_id,
                intervalo=intervalo,
                inicio=inicio,
                fin=fin,
            )

            velas_incoherentes = contar_velas_incoherentes(
                cursor=cursor,
                mercado_id=mercado_id,
                intervalo=intervalo,
                inicio=inicio,
                fin=fin,
            )

            fechas = obtener_fechas_apertura(
                cursor=cursor,
                mercado_id=mercado_id,
                intervalo=intervalo,
                inicio=inicio,
                fin=fin,
            )

    duplicados = total - fechas_distintas
    huecos = detectar_huecos(
        fechas,
        duracion_intervalo,
    )

    velas_faltantes = max(
        total_esperado - fechas_distintas,
        0,
    )

    print("\nVALIDACIÓN DE DATOS")
    print("=" * 50)
    print(f"Mercado: {simbolo}")
    print(f"Intervalo: {intervalo}")
    print(f"Periodo: {anio}-{mes:02d}")
    print(f"Velas esperadas: {total_esperado}")
    print(f"Velas encontradas: {total}")
    print(f"Velas faltantes: {velas_faltantes}")
    print(f"Duplicados: {duplicados}")
    print(f"Velas incoherentes: {velas_incoherentes}")
    print(f"Huecos temporales: {len(huecos)}")
    print(f"Primera vela: {primera_vela}")
    print(f"Última vela: {ultima_vela}")

    if huecos:
        print("\nPrimeros huecos encontrados:")

        for anterior, actual, cantidad in huecos[:10]:
            print(
                f"- Entre {anterior} y {actual}: "
                f"{cantidad} vela(s) faltante(s)"
            )

    errores_criticos = (
        duplicados > 0
        or velas_incoherentes > 0
    )

    advertencias = (
        velas_faltantes > 0
        or len(huecos) > 0
    )

    if errores_criticos:
        print("\nResultado: se encontraron errores críticos.")

        raise ValueError(
            "Existen duplicados o velas con valores incoherentes."
        )

    if advertencias:
        print(
            "\nResultado: datos válidos con advertencias "
            "por huecos temporales."
        )
        print(
            "Los huecos no se completarán con datos inventados."
        )
        return

    print("\nResultado: datos completos y coherentes.")


def obtener_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida un mes de velas históricas "
            "guardadas en PostgreSQL."
        )
    )

    parser.add_argument(
        "--simbolo",
        default="BTCUSDT",
    )

    parser.add_argument(
        "--intervalo",
        default="1m",
        choices=sorted(INTERVALOS),
    )

    parser.add_argument(
        "--anio",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--mes",
        type=int,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    argumentos = obtener_argumentos()

    try:
        validar_periodo(
            simbolo=argumentos.simbolo,
            intervalo=argumentos.intervalo,
            anio=argumentos.anio,
            mes=argumentos.mes,
        )

    except (psycopg.Error, ValueError) as error:
        print("\nNo se pudo completar la validación.")
        print(f"Detalle: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()