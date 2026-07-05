from __future__ import annotations

import json

import pandas as pd
import pyarrow.parquet as pq

from compartido.base_datos.conexion import obtener_conexion
from cripto.corto_plazo_v4_5.configuracion import (
    COLUMNAS_CONTROL,
    PERIODOS_ARCHIVOS,
    RUTA_AUDITORIA,
    SIMBOLOS,
)
from cripto.corto_plazo_v4_5.utilidades_datos import (
    ruta_archivo_base,
)


COLUMNAS_BD_REQUERIDAS = (
    "volumen",
    "volumen_activo_cotizacion",
    "numero_operaciones",
    "volumen_comprador_base",
    "volumen_comprador_cotizacion",
)


def auditar_base_datos() -> list[dict]:
    resultados = []

    consulta = """
        SELECT
            mercados.simbolo_proveedor,
            MIN(velas.fecha_apertura) AS primera_fecha,
            MAX(velas.fecha_apertura) AS ultima_fecha,
            COUNT(*) AS filas,
            COUNT(*) FILTER (
                WHERE velas.volumen IS NULL
                   OR velas.volumen_activo_cotizacion IS NULL
                   OR velas.numero_operaciones IS NULL
                   OR velas.volumen_comprador_base IS NULL
                   OR velas.volumen_comprador_cotizacion IS NULL
            ) AS filas_incompletas
        FROM velas
        INNER JOIN mercados
            ON mercados.id = velas.mercado_id
        INNER JOIN fuentes_datos
            ON fuentes_datos.id = mercados.fuente_datos_id
        WHERE mercados.simbolo_proveedor = %s
          AND LOWER(mercados.tipo_mercado) = 'spot'
          AND LOWER(fuentes_datos.nombre) = 'binance'
          AND velas.intervalo = '1m'
          AND velas.fecha_apertura >= %s
          AND velas.fecha_apertura < %s
        GROUP BY mercados.simbolo_proveedor;
    """

    with obtener_conexion() as conexion:
        with conexion.cursor() as cursor:
            for simbolo in SIMBOLOS:
                for desde, hasta in PERIODOS_ARCHIVOS:
                    cursor.execute(
                        consulta,
                        (
                            simbolo,
                            desde,
                            hasta,
                        ),
                    )
                    fila = cursor.fetchone()

                    if fila is None:
                        raise ValueError(
                            f"No hay datos Spot para {simbolo} {desde} a {hasta}."
                        )

                    resultados.append(
                        {
                            "simbolo": simbolo,
                            "desde": desde,
                            "hasta": hasta,
                            "primera_fecha": fila[1].isoformat(),
                            "ultima_fecha": fila[2].isoformat(),
                            "filas": int(fila[3]),
                            "filas_incompletas": int(fila[4]),
                        }
                    )

    return resultados


def auditar_parquets() -> list[dict]:
    resultados = []

    for simbolo in SIMBOLOS:
        for desde, hasta in PERIODOS_ARCHIVOS:
            ruta = ruta_archivo_base(
                simbolo,
                desde,
                hasta,
            )

            if not ruta.exists():
                raise FileNotFoundError(
                    f"No existe: {ruta}"
                )

            esquema = pq.read_schema(ruta)
            columnas = set(esquema.names)
            faltantes = set(COLUMNAS_CONTROL).difference(columnas)

            if faltantes:
                raise ValueError(
                    f"{ruta.name} no contiene las 63 variables: "
                    f"{sorted(faltantes)}"
                )

            metadata = pq.ParquetFile(ruta).metadata

            resultados.append(
                {
                    "simbolo": simbolo,
                    "desde": desde,
                    "hasta": hasta,
                    "archivo": ruta.name,
                    "filas": int(metadata.num_rows),
                    "columnas": int(metadata.num_columns),
                    "faltantes_control": 0,
                }
            )

    return resultados


def main() -> None:
    print("\nAUDITORÍA DE FUENTES V4.5")
    print("=" * 72)

    RUTA_AUDITORIA.mkdir(
        parents=True,
        exist_ok=True,
    )

    resultados_bd = auditar_base_datos()
    resultados_parquet = auditar_parquets()

    pd.DataFrame(
        resultados_bd
    ).to_csv(
        RUTA_AUDITORIA / "auditoria_spot_postgresql.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        resultados_parquet
    ).to_csv(
        RUTA_AUDITORIA / "auditoria_parquets_base.csv",
        index=False,
        encoding="utf-8-sig",
    )

    resumen = {
        "base_datos_aprobada": all(
            registro["filas_incompletas"] == 0
            for registro in resultados_bd
        ),
        "parquets_aprobados": True,
        "columnas_bd_requeridas": list(COLUMNAS_BD_REQUERIDAS),
        "periodos": len(PERIODOS_ARCHIVOS),
        "simbolos": list(SIMBOLOS),
    }

    (
        RUTA_AUDITORIA / "resumen_auditoria.json"
    ).write_text(
        json.dumps(
            resumen,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    if not resumen["base_datos_aprobada"]:
        raise ValueError(
            "La auditoría encontró filas incompletas en PostgreSQL."
        )

    print("PostgreSQL: columnas y periodos correctos.")
    print("Parquets base: 63 variables disponibles.")
    print("\nAUDITORÍA APROBADA")
    print(f"- {RUTA_AUDITORIA}")


if __name__ == "__main__":
    main()
