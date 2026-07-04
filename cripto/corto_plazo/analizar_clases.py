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

UMBRALES = (
    0.001,
    0.0025,
    0.005,
    0.01,
)

CANTIDAD_EXTREMOS = 20

RUTA_RESUMEN = (
    RUTA_DATOS_PREPARADOS
    / "resumen_clases_objetivo_4h.csv"
)

RUTA_EXTREMOS = (
    RUTA_DATOS_PREPARADOS
    / "extremos_objetivo_4h.csv"
)


def construir_ruta(
    simbolo: str,
    desde: str,
    hasta: str,
) -> Path:
    """Construye la ruta de un conjunto con variables."""

    nombre_archivo = (
        f"{simbolo}_1m_4h_"
        f"{desde}_{hasta}_variables.parquet"
    )

    return RUTA_DATOS_PREPARADOS / nombre_archivo


def obtener_nombre_periodo(
    desde: str,
    hasta: str,
) -> str:
    """Devuelve un nombre legible para el periodo."""

    if desde == "2026-01-01":
        return "2026-01 a 2026-05"

    return desde[:4]


def contar_clases(
    rendimientos: pd.Series,
    umbral: float,
) -> dict[str, int]:
    """Cuenta las clases baja, neutral y sube."""

    cantidad_baja = int(
        (rendimientos <= -umbral).sum()
    )

    cantidad_sube = int(
        (rendimientos >= umbral).sum()
    )

    cantidad_neutral = (
        len(rendimientos)
        - cantidad_baja
        - cantidad_sube
    )

    return {
        "baja": cantidad_baja,
        "neutral": cantidad_neutral,
        "sube": cantidad_sube,
    }


def crear_resumen(
    simbolo: str,
    periodo: str,
    umbral: float,
    conteos: dict[str, int],
) -> dict[str, object]:
    """Construye una fila del resumen de clases."""

    total = sum(conteos.values())

    if total == 0:
        raise ValueError(
            "No existen muestras para calcular las clases."
        )

    return {
        "simbolo": simbolo,
        "periodo": periodo,
        "umbral": umbral,
        "umbral_porcentaje": umbral * 100,
        "muestras": total,
        "cantidad_baja": conteos["baja"],
        "cantidad_neutral": conteos["neutral"],
        "cantidad_sube": conteos["sube"],
        "porcentaje_baja": (
            conteos["baja"] / total * 100
        ),
        "porcentaje_neutral": (
            conteos["neutral"] / total * 100
        ),
        "porcentaje_sube": (
            conteos["sube"] / total * 100
        ),
    }


def sumar_conteos(
    destino: dict[str, int],
    origen: dict[str, int],
) -> None:
    """Suma conteos de clases."""

    for clase in (
        "baja",
        "neutral",
        "sube",
    ):
        destino[clase] += origen[clase]


def mostrar_resumen(
    resumen: dict[str, object],
) -> None:
    """Muestra un resumen de clases en consola."""

    print(
        f"\n{resumen['simbolo']} "
        f"- {resumen['periodo']} "
        f"- umbral {resumen['umbral_porcentaje']:.2f} %"
    )

    print("-" * 60)

    print(
        "Muestras: "
        f"{int(resumen['muestras']):,}"
        .replace(",", ".")
    )

    print(
        "BAJA: "
        f"{int(resumen['cantidad_baja']):,}"
        .replace(",", ".")
        + f" ({resumen['porcentaje_baja']:.2f} %)"
    )

    print(
        "NEUTRAL: "
        f"{int(resumen['cantidad_neutral']):,}"
        .replace(",", ".")
        + f" ({resumen['porcentaje_neutral']:.2f} %)"
    )

    print(
        "SUBE: "
        f"{int(resumen['cantidad_sube']):,}"
        .replace(",", ".")
        + f" ({resumen['porcentaje_sube']:.2f} %)"
    )


def obtener_extremos(
    datos: pd.DataFrame,
    simbolo: str,
    periodo: str,
) -> pd.DataFrame:
    """Obtiene los movimientos absolutos más grandes."""

    extremos = datos.copy()

    extremos["simbolo"] = simbolo
    extremos["periodo"] = periodo

    extremos["movimiento_absoluto"] = (
        extremos["rendimiento_objetivo"].abs()
    )

    extremos = extremos.nlargest(
        CANTIDAD_EXTREMOS,
        "movimiento_absoluto",
    )

    extremos["rendimiento_porcentaje"] = (
        extremos["rendimiento_objetivo"] * 100
    )

    extremos["movimiento_absoluto_porcentaje"] = (
        extremos["movimiento_absoluto"] * 100
    )

    return extremos[
        [
            "simbolo",
            "periodo",
            "fecha_apertura",
            "fecha_objetivo",
            "precio_cierre",
            "precio_cierre_futuro",
            "rendimiento_objetivo",
            "rendimiento_porcentaje",
            "movimiento_absoluto_porcentaje",
        ]
    ]


def main() -> None:
    """Analiza las posibles clases y los movimientos extremos."""

    print("\nANÁLISIS DE CLASES DEL OBJETIVO")
    print("=" * 70)

    resumenes: list[dict[str, object]] = []
    extremos_parciales: list[pd.DataFrame] = []

    totales_simbolo = {
        simbolo: {
            umbral: {
                "baja": 0,
                "neutral": 0,
                "sube": 0,
            }
            for umbral in UMBRALES
        }
        for simbolo in SIMBOLOS
    }

    totales_globales = {
        umbral: {
            "baja": 0,
            "neutral": 0,
            "sube": 0,
        }
        for umbral in UMBRALES
    }

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

            periodo = obtener_nombre_periodo(
                desde=desde,
                hasta=hasta,
            )

            datos = pd.read_parquet(
                ruta,
                columns=[
                    "fecha_apertura",
                    "fecha_objetivo",
                    "precio_cierre",
                    "precio_cierre_futuro",
                    "rendimiento_objetivo",
                ],
            )

            datos["rendimiento_objetivo"] = pd.to_numeric(
                datos["rendimiento_objetivo"],
                errors="coerce",
            )

            if datos["rendimiento_objetivo"].isna().any():
                raise ValueError(
                    f"Existen objetivos nulos en: {ruta}"
                )

            extremos_parciales.append(
                obtener_extremos(
                    datos=datos,
                    simbolo=simbolo,
                    periodo=periodo,
                )
            )

            for umbral in UMBRALES:
                conteos = contar_clases(
                    rendimientos=datos[
                        "rendimiento_objetivo"
                    ],
                    umbral=umbral,
                )

                sumar_conteos(
                    totales_simbolo[simbolo][umbral],
                    conteos,
                )

                sumar_conteos(
                    totales_globales[umbral],
                    conteos,
                )

                resumenes.append(
                    crear_resumen(
                        simbolo=simbolo,
                        periodo=periodo,
                        umbral=umbral,
                        conteos=conteos,
                    )
                )

    for simbolo in SIMBOLOS:
        for umbral in UMBRALES:
            resumen = crear_resumen(
                simbolo=simbolo,
                periodo="TOTAL",
                umbral=umbral,
                conteos=totales_simbolo[
                    simbolo
                ][umbral],
            )

            resumenes.append(resumen)

            if umbral == 0.005:
                mostrar_resumen(resumen)

    for umbral in UMBRALES:
        resumen_global = crear_resumen(
            simbolo="BTCUSDT + ETHUSDT",
            periodo="TOTAL GLOBAL",
            umbral=umbral,
            conteos=totales_globales[umbral],
        )

        resumenes.append(resumen_global)

        if umbral == 0.005:
            mostrar_resumen(resumen_global)

    resumen_clases = pd.DataFrame(
        resumenes
    )

    resumen_clases.to_csv(
        RUTA_RESUMEN,
        index=False,
        encoding="utf-8-sig",
    )

    extremos = pd.concat(
        extremos_parciales,
        ignore_index=True,
    )

    extremos = extremos.nlargest(
        CANTIDAD_EXTREMOS,
        "movimiento_absoluto_porcentaje",
    ).reset_index(drop=True)

    extremos.to_csv(
        RUTA_EXTREMOS,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO")
    print("=" * 70)
    print(f"Resumen de clases: {RUTA_RESUMEN}")
    print(f"Movimientos extremos: {RUTA_EXTREMOS}")


if __name__ == "__main__":
    main()