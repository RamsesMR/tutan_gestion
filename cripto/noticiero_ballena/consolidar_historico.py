"""Consolida las partes mensuales del histórico de noticiero_ballena sin cargar todo en RAM.

Usa DuckDB para deduplicar y ordenar con derrame a disco. Este script es independiente
 del descargador de BigQuery: no vuelve a consultar ni consumir cuota.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_BASE = RAIZ_PROYECTO / "datos" / "cripto" / "crudos" / "noticiero_ballena" / "historico_bigquery"
RUTA_PARTES = RUTA_BASE / "partes"
RUTA_SALIDA = RUTA_BASE / "blockchain_bigquery_historico.parquet"
RUTA_MANIFIESTO = RUTA_BASE / "manifiesto_historico.json"
RUTA_TEMPORAL = RUTA_BASE / "tmp_consolidacion_duckdb"

CANDIDATAS_FECHA = (
    "fecha_disponible",
    "fecha_disponibilidad_modelo",
    "fecha_detectada_utc",
    "block_timestamp",
    "fecha_evento",
    "fecha",
)


def _sql_literal(valor: str | Path) -> str:
    return str(valor).replace("\\", "/").replace("'", "''")


def _tamano_humano(bytes_: int) -> str:
    unidades = ("B", "KiB", "MiB", "GiB", "TiB")
    valor = float(bytes_)
    for unidad in unidades:
        if valor < 1024 or unidad == unidades[-1]:
            return f"{valor:,.2f} {unidad}"
        valor /= 1024
    return f"{bytes_:,} B"


def _partes_validas(carpeta: Path) -> list[Path]:
    partes = sorted(p for p in carpeta.glob("*.parquet") if p.is_file() and p.stat().st_size > 0)
    if not partes:
        raise FileNotFoundError(f"No se encontraron Parquet mensuales en: {carpeta}")
    return partes


def _columnas(conexion: duckdb.DuckDBPyConnection, patron: str) -> list[str]:
    filas = conexion.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{patron}', union_by_name=true)"
    ).fetchall()
    return [str(fila[0]) for fila in filas]


def _elegir_fecha(columnas: Iterable[str]) -> str | None:
    disponibles = set(columnas)
    return next((nombre for nombre in CANDIDATAS_FECHA if nombre in disponibles), None)


def _configurar_duckdb(
    conexion: duckdb.DuckDBPyConnection,
    temporal: Path,
    memoria: str,
    hilos: int,
) -> None:
    temporal.mkdir(parents=True, exist_ok=True)
    conexion.execute(f"SET temp_directory='{_sql_literal(temporal)}'")
    conexion.execute(f"SET memory_limit='{memoria}'")
    conexion.execute(f"SET threads={max(1, hilos)}")
    conexion.execute("SET preserve_insertion_order=false")


def _espacio_suficiente(base: Path, partes: list[Path]) -> tuple[bool, int, int]:
    tamano_partes = sum(p.stat().st_size for p in partes)
    libre = shutil.disk_usage(base).free
    # DuckDB puede necesitar archivos temporales durante el hash/sort. Se exige un margen prudente.
    requerido = max(2 * tamano_partes + 1_073_741_824, 3_221_225_472)
    return libre >= requerido, libre, requerido


def consolidar(
    carpeta_partes: Path = RUTA_PARTES,
    salida: Path = RUTA_SALIDA,
    manifiesto: Path = RUTA_MANIFIESTO,
    temporal: Path = RUTA_TEMPORAL,
    memoria: str = "1GB",
    hilos: int = 4,
    sobrescribir: bool = False,
) -> dict:
    partes = _partes_validas(carpeta_partes)
    carpeta_partes.parent.mkdir(parents=True, exist_ok=True)
    salida.parent.mkdir(parents=True, exist_ok=True)

    suficiente, libre, requerido = _espacio_suficiente(salida.parent, partes)
    if not suficiente:
        raise RuntimeError(
            "Espacio libre insuficiente para consolidar con seguridad. "
            f"Libre: {_tamano_humano(libre)}; recomendado: {_tamano_humano(requerido)}."
        )

    if salida.exists():
        if not sobrescribir:
            raise FileExistsError(
                f"El consolidado ya existe: {salida}. Usa --sobrescribir para regenerarlo."
            )
        salida.unlink()

    patron = _sql_literal(carpeta_partes / "*.parquet")
    salida_sql = _sql_literal(salida)

    conexion = duckdb.connect(database=":memory:")
    try:
        _configurar_duckdb(conexion, temporal, memoria, hilos)
        columnas = _columnas(conexion, patron)
        if "evento_id" not in columnas:
            raise RuntimeError(
                "Las partes mensuales no contienen la columna obligatoria 'evento_id'. "
                f"Columnas encontradas: {columnas}"
            )

        columna_fecha = _elegir_fecha(columnas)
        orden_dedupe = (
            f'"{columna_fecha}" DESC NULLS LAST, filename DESC'
            if columna_fecha
            else "filename DESC"
        )
        orden_final = f'ORDER BY "{columna_fecha}" ASC NULLS LAST' if columna_fecha else ""

        total_crudo = int(
            conexion.execute(
                f"SELECT COUNT(*) FROM read_parquet('{patron}', union_by_name=true)"
            ).fetchone()[0]
        )

        consulta = f"""
            SELECT * EXCLUDE (__rn, filename)
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY evento_id
                        ORDER BY {orden_dedupe}
                    ) AS __rn
                FROM read_parquet(
                    '{patron}',
                    union_by_name=true,
                    filename=true
                )
            ) AS deduplicado
            WHERE __rn = 1
            {orden_final}
        """

        print("Consolidando con DuckDB (usa disco temporal y no carga todo en RAM)...")
        print(f"Partes mensuales: {len(partes)}")
        print(f"Filas antes de deduplicar: {total_crudo:,}")
        print(f"Memoria máxima DuckDB: {memoria}; hilos: {max(1, hilos)}")
        print(f"Temporal: {temporal}")

        conexion.execute(
            f"COPY ({consulta}) TO '{salida_sql}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )

        total_final = int(
            conexion.execute(
                f"SELECT COUNT(*) FROM read_parquet('{salida_sql}')"
            ).fetchone()[0]
        )
        duplicados = total_crudo - total_final

        rango = None
        if columna_fecha:
            minimo, maximo = conexion.execute(
                f'SELECT MIN("{columna_fecha}"), MAX("{columna_fecha}") '
                f"FROM read_parquet('{salida_sql}')"
            ).fetchone()
            rango = {
                "columna": columna_fecha,
                "minimo": minimo.isoformat() if hasattr(minimo, "isoformat") else str(minimo),
                "maximo": maximo.isoformat() if hasattr(maximo, "isoformat") else str(maximo),
            }

        try:
            archivo_resultado = str(salida.relative_to(RAIZ_PROYECTO))
        except ValueError:
            archivo_resultado = str(salida)

        resultado = {
            "version": "noticiero_ballena_v1",
            "generado_utc": datetime.now(timezone.utc).isoformat(),
            "metodo_consolidacion": "duckdb_externo_dedupe_evento_id",
            "partes_mensuales": len(partes),
            "filas_crudas": total_crudo,
            "filas_finales": total_final,
            "duplicados_eliminados": duplicados,
            "archivo": archivo_resultado,
            "tamano_bytes": salida.stat().st_size,
            "tamano_humano": _tamano_humano(salida.stat().st_size),
            "rango_temporal": rango,
            "memoria_duckdb": memoria,
            "hilos": max(1, hilos),
        }

        contenido_anterior: dict = {}
        if manifiesto.exists():
            try:
                cargado = json.loads(manifiesto.read_text(encoding="utf-8"))
                if isinstance(cargado, dict):
                    contenido_anterior = cargado
            except (OSError, json.JSONDecodeError):
                contenido_anterior = {}
        contenido_anterior["consolidacion"] = resultado
        manifiesto.write_text(
            json.dumps(contenido_anterior, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return resultado
    finally:
        conexion.close()
        # Solo se elimina el temporal si DuckDB ya cerró y la salida fue creada correctamente.
        if salida.exists() and temporal.exists():
            shutil.rmtree(temporal, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolida el histórico mensual de noticiero_ballena sin agotar la RAM."
    )
    parser.add_argument("--partes", type=Path, default=RUTA_PARTES)
    parser.add_argument("--salida", type=Path, default=RUTA_SALIDA)
    parser.add_argument("--manifiesto", type=Path, default=RUTA_MANIFIESTO)
    parser.add_argument("--temporal", type=Path, default=RUTA_TEMPORAL)
    parser.add_argument("--memoria", default="1GB", help="Ej.: 512MB, 1GB, 2GB")
    parser.add_argument("--hilos", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--sobrescribir", action="store_true")
    args = parser.parse_args()

    resultado = consolidar(
        carpeta_partes=args.partes,
        salida=args.salida,
        manifiesto=args.manifiesto,
        temporal=args.temporal,
        memoria=args.memoria,
        hilos=args.hilos,
        sobrescribir=args.sobrescribir,
    )

    print("\nCONSOLIDACIÓN COMPLETADA")
    print(f"Filas finales: {resultado['filas_finales']:,}")
    print(f"Duplicados eliminados: {resultado['duplicados_eliminados']:,}")
    print(f"Peso final: {resultado['tamano_humano']}")
    archivo = Path(resultado["archivo"])
    if not archivo.is_absolute():
        archivo = RAIZ_PROYECTO / archivo
    print(f"Archivo: {archivo}")


if __name__ == "__main__":
    main()
