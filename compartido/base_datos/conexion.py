import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


# Raíz del proyecto: tutan_gestion/
RUTA_PROYECTO = Path(__file__).resolve().parents[2]

# Carga las variables privadas del archivo .env
load_dotenv(RUTA_PROYECTO / ".env")


def obtener_conexion() -> psycopg.Connection:
    """Abre y devuelve una conexión con PostgreSQL."""

    variables_requeridas = [
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ]

    variables_faltantes = [
        variable
        for variable in variables_requeridas
        if not os.getenv(variable)
    ]

    if variables_faltantes:
        raise RuntimeError(
            "Faltan variables en el archivo .env: "
            + ", ".join(variables_faltantes)
        )

    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=5,
    )


def probar_conexion() -> None:
    """Comprueba la conexión y muestra los datos principales."""

    try:
        with obtener_conexion() as conexion:
            with conexion.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        current_database(),
                        current_user,
                        current_setting('TimeZone');
                    """
                )

                base_datos, usuario, zona_horaria = cursor.fetchone()

                print("Conexión realizada correctamente")
                print(f"Base de datos: {base_datos}")
                print(f"Usuario: {usuario}")
                print(f"Zona horaria: {zona_horaria}")

    except psycopg.Error as error:
        print("No se pudo conectar con PostgreSQL.")
        print(f"Detalle: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    probar_conexion()