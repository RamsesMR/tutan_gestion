from __future__ import annotations

from pathlib import Path

import pandas as pd

from cripto.corto_plazo.configuracion import (
    RUTA_DATOS_PREPARADOS,
)


SIMBOLOS = (
    "BTCUSDT",
    "ETHUSDT",
)

PERIODOS = (
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-01-01"),
    ("2026-01-01", "2026-06-01"),
)

RUTA_RESUMEN = (
    RUTA_DATOS_PREPARADOS
    / "resumen_objetivo_4h.csv"
)


def construir_ruta(
    simbolo: str,
    desde: str,
    hasta: str,
) -> Path:
    """Construye la ruta del archivo de variables."""

    nombre_archivo = (
        f"{simbolo}_1m_4h_"
        f"{desde}_{hasta}_variables.parquet"
    )

    return RUTA_DATOS_PREPARADOS / nombre_archivo


def calcular_estadisticas(
    rendimientos: pd.Series,
    simbolo: str,
    periodo: str,
) -> dict[str, object]:
    """Calcula estadísticas descriptivas del objetivo."""

    rendimientos = pd.to_numeric(
        rendimientos,
        errors="coerce",
    ).dropna()

    movimientos_absolutos = rendimientos.abs()

    return {
        "simbolo": simbolo,
        "periodo": periodo,
        "muestras": len(rendimientos),
        "media": float(rendimientos.mean()),
        "mediana": float(rendimientos.median()),
        "desviacion_estandar": float(rendimientos.std()),
        "minimo": float(rendimientos.min()),
        "percentil_01": float(rendimientos.quantile(0.01)),
        "percentil_05": float(rendimientos.quantile(0.05)),
        "percentil_25": float(rendimientos.quantile(0.25)),
        "percentil_75": float(rendimientos.quantile(0.75)),
        "percentil_95": float(rendimientos.quantile(0.95)),
        "percentil_99": float(rendimientos.quantile(0.99)),
        "maximo": float(rendimientos.max()),
        "porcentaje_positivo": float(
            (rendimientos > 0).mean() * 100
        ),
        "porcentaje_negativo": float(
            (rendimientos < 0).mean() * 100
        ),
        "porcentaje_cero": float(
            (rendimientos == 0).mean() * 100
        ),
        "porcentaje_movimiento_mayor_0_1": float(
            (movimientos_absolutos >= 0.001).mean() * 100
        ),
        "porcentaje_movimiento_mayor_0_25": float(
            (movimientos_absolutos >= 0.0025).mean() * 100
        ),
        "porcentaje_movimiento_mayor_0_5": float(
            (movimientos_absolutos >= 0.005).mean() * 100
        ),
        "porcentaje_movimiento_mayor_1": float(
            (movimientos_absolutos >= 0.01).mean() * 100
        ),
    }


def formatear_numero(valor: object) -> str:
    """Formatea números para mostrarlos en consola."""

    if isinstance(valor, int):
        return f"{valor:,}".replace(",", ".")

    if isinstance(valor, float):
        return f"{valor:.6f}"

    return str(valor)


def mostrar_resumen(
    estadisticas: dict[str, object],
) -> None:
    """Muestra las estadísticas principales."""

    print(
        f"\n{estadisticas['simbolo']} "
        f"- {estadisticas['periodo']}"
    )
    print("-" * 60)

    campos = (
        ("Muestras", "muestras"),
        ("Media", "media"),
        ("Mediana", "mediana"),
        ("Desviación estándar", "desviacion_estandar"),
        ("Percentil 5", "percentil_05"),
        ("Percentil 95", "percentil_95"),
        ("% positivo", "porcentaje_positivo"),
        ("% negativo", "porcentaje_negativo"),
        (
            "% movimiento absoluto >= 0,10%",
            "porcentaje_movimiento_mayor_0_1",
        ),
        (
            "% movimiento absoluto >= 0,25%",
            "porcentaje_movimiento_mayor_0_25",
        ),
        (
            "% movimiento absoluto >= 0,50%",
            "porcentaje_movimiento_mayor_0_5",
        ),
        (
            "% movimiento absoluto >= 1,00%",
            "porcentaje_movimiento_mayor_1",
        ),
    )

    for etiqueta, campo in campos:
        print(
            f"{etiqueta}: "
            f"{formatear_numero(estadisticas[campo])}"
        )


def main() -> None:
    """Analiza el objetivo de todos los conjuntos."""

    print("\nANÁLISIS DEL OBJETIVO A CUATRO HORAS")
    print("=" * 60)

    resumenes: list[dict[str, object]] = []
    rendimientos_por_simbolo: dict[str, list[pd.Series]] = {
        simbolo: []
        for simbolo in SIMBOLOS
    }

    todos_los_rendimientos: list[pd.Series] = []

    for simbolo in SIMBOLOS:
        for desde, hasta in PERIODOS:
            ruta = construir_ruta(
                simbolo=simbolo,
                desde=desde,
                hasta=hasta,
            )

            if not ruta.exists():
                raise FileNotFoundError(
                    f"No existe el archivo: {ruta}"
                )

            datos = pd.read_parquet(
                ruta,
                columns=["rendimiento_objetivo"],
            )

            rendimientos = pd.to_numeric(
                datos["rendimiento_objetivo"],
                errors="coerce",
            )

            if rendimientos.isna().any():
                raise ValueError(
                    f"El archivo contiene objetivos nulos: {ruta}"
                )

            periodo = (
                f"{desde[:4]}"
                if hasta.endswith("-01-01")
                else "2026-01 a 2026-05"
            )

            estadisticas = calcular_estadisticas(
                rendimientos=rendimientos,
                simbolo=simbolo,
                periodo=periodo,
            )

            resumenes.append(estadisticas)

            rendimientos_por_simbolo[
                simbolo
            ].append(rendimientos)

            todos_los_rendimientos.append(rendimientos)

            mostrar_resumen(estadisticas)

    for simbolo in SIMBOLOS:
        rendimientos_simbolo = pd.concat(
            rendimientos_por_simbolo[simbolo],
            ignore_index=True,
        )

        estadisticas = calcular_estadisticas(
            rendimientos=rendimientos_simbolo,
            simbolo=simbolo,
            periodo="TOTAL",
        )

        resumenes.append(estadisticas)
        mostrar_resumen(estadisticas)

    rendimientos_globales = pd.concat(
        todos_los_rendimientos,
        ignore_index=True,
    )

    estadisticas_globales = calcular_estadisticas(
        rendimientos=rendimientos_globales,
        simbolo="BTCUSDT + ETHUSDT",
        periodo="TOTAL GLOBAL",
    )

    resumenes.append(estadisticas_globales)
    mostrar_resumen(estadisticas_globales)

    resumen = pd.DataFrame(resumenes)

    RUTA_RESUMEN.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resumen.to_csv(
        RUTA_RESUMEN,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print("ANÁLISIS COMPLETADO")
    print("=" * 60)
    print(f"Archivo generado: {RUTA_RESUMEN}")


if __name__ == "__main__":
    main()