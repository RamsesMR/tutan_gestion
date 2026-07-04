import psycopg

from compartido.base_datos.conexion import obtener_conexion


TABLAS = [
    """
    CREATE TABLE IF NOT EXISTS fuentes_datos (
        id BIGSERIAL PRIMARY KEY,
        nombre VARCHAR(100) NOT NULL UNIQUE,
        url_base TEXT,
        tipo_fuente VARCHAR(50) NOT NULL,
        activo BOOLEAN NOT NULL DEFAULT TRUE,
        creado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        actualizado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """,

    """
    CREATE TABLE IF NOT EXISTS activos (
        id BIGSERIAL PRIMARY KEY,
        simbolo VARCHAR(30) NOT NULL,
        nombre VARCHAR(150) NOT NULL,
        tipo_activo VARCHAR(50) NOT NULL,
        activo BOOLEAN NOT NULL DEFAULT TRUE,
        creado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        actualizado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

        CONSTRAINT uq_activos_simbolo_tipo
            UNIQUE (simbolo, tipo_activo)
    );
    """,

    """
    CREATE TABLE IF NOT EXISTS mercados (
        id BIGSERIAL PRIMARY KEY,
        fuente_datos_id BIGINT NOT NULL,
        activo_base_id BIGINT NOT NULL,
        activo_cotizacion_id BIGINT NOT NULL,
        simbolo_proveedor VARCHAR(50) NOT NULL,
        tipo_mercado VARCHAR(30) NOT NULL,
        activo BOOLEAN NOT NULL DEFAULT TRUE,
        creado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        actualizado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

        CONSTRAINT fk_mercados_fuente
            FOREIGN KEY (fuente_datos_id)
            REFERENCES fuentes_datos(id),

        CONSTRAINT fk_mercados_activo_base
            FOREIGN KEY (activo_base_id)
            REFERENCES activos(id),

        CONSTRAINT fk_mercados_activo_cotizacion
            FOREIGN KEY (activo_cotizacion_id)
            REFERENCES activos(id),

        CONSTRAINT ck_mercados_activos_distintos
            CHECK (activo_base_id <> activo_cotizacion_id),

        CONSTRAINT uq_mercados_fuente_simbolo_tipo
            UNIQUE (
                fuente_datos_id,
                simbolo_proveedor,
                tipo_mercado
            )
    );
    """,

    """
    CREATE TABLE IF NOT EXISTS velas (
        id BIGSERIAL PRIMARY KEY,
        mercado_id BIGINT NOT NULL,
        intervalo VARCHAR(10) NOT NULL,
        fecha_apertura TIMESTAMPTZ NOT NULL,
        fecha_cierre TIMESTAMPTZ NOT NULL,

        precio_apertura NUMERIC(38, 18) NOT NULL,
        precio_maximo NUMERIC(38, 18) NOT NULL,
        precio_minimo NUMERIC(38, 18) NOT NULL,
        precio_cierre NUMERIC(38, 18) NOT NULL,

        volumen NUMERIC(38, 18) NOT NULL,
        volumen_activo_cotizacion NUMERIC(38, 18),
        numero_operaciones BIGINT NOT NULL,

        volumen_comprador_base NUMERIC(38, 18),
        volumen_comprador_cotizacion NUMERIC(38, 18),

        cerrada BOOLEAN NOT NULL DEFAULT TRUE,
        archivo_origen TEXT,
        recolectado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

        CONSTRAINT fk_velas_mercado
            FOREIGN KEY (mercado_id)
            REFERENCES mercados(id),

        CONSTRAINT uq_velas_mercado_intervalo_fecha
            UNIQUE (
                mercado_id,
                intervalo,
                fecha_apertura
            ),

        CONSTRAINT ck_velas_fechas
            CHECK (fecha_cierre > fecha_apertura),

        CONSTRAINT ck_velas_precios_positivos
            CHECK (
                precio_apertura > 0
                AND precio_maximo > 0
                AND precio_minimo > 0
                AND precio_cierre > 0
            ),

        CONSTRAINT ck_velas_precio_maximo
            CHECK (
                precio_maximo >= precio_apertura
                AND precio_maximo >= precio_cierre
                AND precio_maximo >= precio_minimo
            ),

        CONSTRAINT ck_velas_precio_minimo
            CHECK (
                precio_minimo <= precio_apertura
                AND precio_minimo <= precio_cierre
                AND precio_minimo <= precio_maximo
            ),

        CONSTRAINT ck_velas_volumen
            CHECK (volumen >= 0),

        CONSTRAINT ck_velas_numero_operaciones
            CHECK (numero_operaciones >= 0)
    );
    """,

    """
    CREATE TABLE IF NOT EXISTS ejecuciones_recoleccion (
        id BIGSERIAL PRIMARY KEY,
        fuente_datos_id BIGINT NOT NULL,
        mercado_id BIGINT,
        tipo_proceso VARCHAR(50) NOT NULL,
        archivo TEXT,

        fecha_inicio TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        fecha_fin TIMESTAMPTZ,

        registros_leidos BIGINT NOT NULL DEFAULT 0,
        registros_insertados BIGINT NOT NULL DEFAULT 0,
        registros_duplicados BIGINT NOT NULL DEFAULT 0,
        registros_error BIGINT NOT NULL DEFAULT 0,

        estado VARCHAR(20) NOT NULL DEFAULT 'iniciada',
        mensaje_error TEXT,

        CONSTRAINT fk_ejecuciones_fuente
            FOREIGN KEY (fuente_datos_id)
            REFERENCES fuentes_datos(id),

        CONSTRAINT fk_ejecuciones_mercado
            FOREIGN KEY (mercado_id)
            REFERENCES mercados(id),

        CONSTRAINT ck_ejecuciones_estado
            CHECK (
                estado IN (
                    'iniciada',
                    'completada',
                    'error'
                )
            ),

        CONSTRAINT ck_ejecuciones_fecha_fin
            CHECK (
                fecha_fin IS NULL
                OR fecha_fin >= fecha_inicio
            )
    );
    """,
    
        """
    CREATE TABLE IF NOT EXISTS incidencias_datos (
        id BIGSERIAL PRIMARY KEY,
        ejecucion_recoleccion_id BIGINT NOT NULL,
        mercado_id BIGINT NOT NULL,
        intervalo VARCHAR(10) NOT NULL,
        archivo TEXT NOT NULL,
        numero_linea BIGINT NOT NULL,
        fecha_apertura_original TEXT,
        tipo_incidencia VARCHAR(50) NOT NULL,
        detalle TEXT NOT NULL,
        fila_original TEXT NOT NULL,
        creado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

        CONSTRAINT fk_incidencias_ejecucion
            FOREIGN KEY (ejecucion_recoleccion_id)
            REFERENCES ejecuciones_recoleccion(id),

        CONSTRAINT fk_incidencias_mercado
            FOREIGN KEY (mercado_id)
            REFERENCES mercados(id)
    );
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_incidencias_ejecucion
    ON incidencias_datos (ejecucion_recoleccion_id);
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_velas_fecha_apertura
    ON velas (fecha_apertura);
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_ejecuciones_fecha_inicio
    ON ejecuciones_recoleccion (fecha_inicio);
    """
]


def crear_tablas() -> None:
    """Crea las tablas iniciales del proyecto."""

    try:
        with obtener_conexion() as conexion:
            with conexion.cursor() as cursor:
                for sentencia in TABLAS:
                    cursor.execute(sentencia)

            conexion.commit()

        print("Tablas creadas correctamente:")
        print("- fuentes_datos")
        print("- activos")
        print("- mercados")
        print("- velas")
        print("- ejecuciones_recoleccion")
        print("- incidencias_datos")

    except psycopg.Error as error:
        print("No se pudieron crear las tablas.")
        print(f"Detalle: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    crear_tablas()