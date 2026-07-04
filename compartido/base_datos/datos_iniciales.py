import psycopg

from compartido.base_datos.conexion import obtener_conexion


def insertar_fuente(cursor, nombre: str, tipo_fuente: str) -> int:
    cursor.execute(
        """
        INSERT INTO fuentes_datos (
            nombre,
            tipo_fuente
        )
        VALUES (%s, %s)
        ON CONFLICT (nombre)
        DO UPDATE SET
            tipo_fuente = EXCLUDED.tipo_fuente,
            activo = TRUE,
            actualizado_en = CURRENT_TIMESTAMP
        RETURNING id;
        """,
        (nombre, tipo_fuente),
    )

    return cursor.fetchone()[0]


def insertar_activo(
    cursor,
    simbolo: str,
    nombre: str,
    tipo_activo: str,
) -> int:
    cursor.execute(
        """
        INSERT INTO activos (
            simbolo,
            nombre,
            tipo_activo
        )
        VALUES (%s, %s, %s)
        ON CONFLICT (simbolo, tipo_activo)
        DO UPDATE SET
            nombre = EXCLUDED.nombre,
            activo = TRUE,
            actualizado_en = CURRENT_TIMESTAMP
        RETURNING id;
        """,
        (simbolo, nombre, tipo_activo),
    )

    return cursor.fetchone()[0]


def insertar_mercado(
    cursor,
    fuente_datos_id: int,
    activo_base_id: int,
    activo_cotizacion_id: int,
    simbolo_proveedor: str,
    tipo_mercado: str,
) -> int:
    cursor.execute(
        """
        INSERT INTO mercados (
            fuente_datos_id,
            activo_base_id,
            activo_cotizacion_id,
            simbolo_proveedor,
            tipo_mercado
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (
            fuente_datos_id,
            simbolo_proveedor,
            tipo_mercado
        )
        DO UPDATE SET
            activo_base_id = EXCLUDED.activo_base_id,
            activo_cotizacion_id = EXCLUDED.activo_cotizacion_id,
            activo = TRUE,
            actualizado_en = CURRENT_TIMESTAMP
        RETURNING id;
        """,
        (
            fuente_datos_id,
            activo_base_id,
            activo_cotizacion_id,
            simbolo_proveedor,
            tipo_mercado,
        ),
    )

    return cursor.fetchone()[0]


def cargar_datos_iniciales() -> None:
    try:
        with obtener_conexion() as conexion:
            with conexion.cursor() as cursor:
                binance_id = insertar_fuente(
                    cursor,
                    nombre="Binance",
                    tipo_fuente="exchange",
                )

                btc_id = insertar_activo(
                    cursor,
                    simbolo="BTC",
                    nombre="Bitcoin",
                    tipo_activo="criptomoneda",
                )

                eth_id = insertar_activo(
                    cursor,
                    simbolo="ETH",
                    nombre="Ethereum",
                    tipo_activo="criptomoneda",
                )

                usdt_id = insertar_activo(
                    cursor,
                    simbolo="USDT",
                    nombre="Tether",
                    tipo_activo="criptomoneda",
                )

                insertar_mercado(
                    cursor,
                    fuente_datos_id=binance_id,
                    activo_base_id=btc_id,
                    activo_cotizacion_id=usdt_id,
                    simbolo_proveedor="BTCUSDT",
                    tipo_mercado="spot",
                )

                insertar_mercado(
                    cursor,
                    fuente_datos_id=binance_id,
                    activo_base_id=eth_id,
                    activo_cotizacion_id=usdt_id,
                    simbolo_proveedor="ETHUSDT",
                    tipo_mercado="spot",
                )

            conexion.commit()

        print("Datos iniciales cargados correctamente.")
        print("- Fuente: Binance")
        print("- Activos: BTC, ETH y USDT")
        print("- Mercados: BTCUSDT y ETHUSDT")

    except psycopg.Error as error:
        print("No se pudieron cargar los datos iniciales.")
        print(f"Detalle: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    cargar_datos_iniciales()