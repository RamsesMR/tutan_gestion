from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from cripto.corto_plazo.configuracion import (
    HORIZONTE_MINUTOS,
    RUTA_DATOS_PREPARADOS,
)


SIMBOLOS = (
    "BTCUSDT",
    "ETHUSDT",
)

UMBRAL_CLASE = 0.005

DIVISIONES = {
    "entrenamiento": {
        "desde": "2021-01-01",
        "hasta": "2025-01-01",
        "periodos": (
            ("2021-01-01", "2022-01-01"),
            ("2022-01-01", "2023-01-01"),
            ("2023-01-01", "2024-01-01"),
            ("2024-01-01", "2025-01-01"),
        ),
    },
    "validacion": {
        "desde": "2025-01-01",
        "hasta": "2026-01-01",
        "periodos": (
            ("2025-01-01", "2026-01-01"),
        ),
    },
    "prueba": {
        "desde": "2026-01-01",
        "hasta": "2026-06-01",
        "periodos": (
            ("2026-01-01", "2026-06-01"),
        ),
    },
}

RUTA_MANIFIESTO = (
    RUTA_DATOS_PREPARADOS
    / "manifiesto_divisiones_4h.csv"
)

RUTA_RESUMEN = (
    RUTA_DATOS_PREPARADOS
    / "resumen_divisiones_4h.csv"
)


def construir_ruta(
    simbolo: str,
    desde: str,
    hasta: str,
) -> Path:
    """Construye la ruta de un archivo anual con variables."""

    nombre_archivo = (
        f"{simbolo}_1m_4h_"
        f"{desde}_{hasta}_variables.parquet"
    )

    return RUTA_DATOS_PREPARADOS / nombre_archivo


def clasificar_objetivo(
    rendimientos: pd.Series,
) -> pd.Series:
    """Convierte el rendimiento futuro en tres clases."""

    condiciones = (
        rendimientos <= -UMBRAL_CLASE,
        rendimientos >= UMBRAL_CLASE,
    )

    clases = (
        "BAJA",
        "SUBE",
    )

    resultado = np.select(
        condiciones,
        clases,
        default="NEUTRAL",
    )

    return pd.Series(
        resultado,
        index=rendimientos.index,
        dtype="string",
    )


def crear_conteos_vacios() -> dict[str, int]:
    """Crea el contador de clases."""

    return {
        "BAJA": 0,
        "NEUTRAL": 0,
        "SUBE": 0,
    }


def sumar_clases(
    destino: dict[str, int],
    clases: pd.Series,
) -> None:
    """Suma los conteos de clases al total de una división."""

    conteos = clases.value_counts()

    for clase in destino:
        destino[clase] += int(
            conteos.get(clase, 0)
        )


def validar_horizonte(
    datos: pd.DataFrame,
    ruta: Path,
) -> None:
    """Comprueba que todos los objetivos estén a cuatro horas."""

    horizonte_esperado = pd.Timedelta(
        minutes=HORIZONTE_MINUTOS
    )

    horizontes = (
        datos["fecha_objetivo"]
        - datos["fecha_apertura"]
    )

    incorrectos = int(
        horizontes.ne(horizonte_esperado).sum()
    )

    if incorrectos > 0:
        raise ValueError(
            f"{ruta.name} contiene {incorrectos} "
            "muestras con horizonte incorrecto."
        )


def procesar_archivo(
    simbolo: str,
    division: str,
    desde_archivo: str,
    hasta_archivo: str,
    desde_division: pd.Timestamp,
    hasta_division: pd.Timestamp,
) -> tuple[dict[str, object], pd.Series]:
    """Analiza un archivo y aplica la separación temporal."""

    ruta = construir_ruta(
        simbolo=simbolo,
        desde=desde_archivo,
        hasta=hasta_archivo,
    )

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {ruta}"
        )

    datos = pd.read_parquet(
        ruta,
        columns=[
            "fecha_apertura",
            "fecha_objetivo",
            "rendimiento_objetivo",
        ],
    )

    datos["fecha_apertura"] = pd.to_datetime(
        datos["fecha_apertura"],
        utc=True,
        errors="coerce",
    )

    datos["fecha_objetivo"] = pd.to_datetime(
        datos["fecha_objetivo"],
        utc=True,
        errors="coerce",
    )

    datos["rendimiento_objetivo"] = pd.to_numeric(
        datos["rendimiento_objetivo"],
        errors="coerce",
    )

    if datos.isna().any().any():
        raise ValueError(
            f"Existen valores nulos en: {ruta}"
        )

    validar_horizonte(
        datos=datos,
        ruta=ruta,
    )

    total_archivo = len(datos)

    dentro_division = (
        (datos["fecha_apertura"] >= desde_division)
        & (datos["fecha_apertura"] < hasta_division)
    )

    objetivo_dentro_division = (
        datos["fecha_objetivo"] < hasta_division
    )

    muestras_utilizables = (
        dentro_division
        & objetivo_dentro_division
    )

    datos_utilizables = datos.loc[
        muestras_utilizables
    ].copy()

    if datos_utilizables.empty:
        raise ValueError(
            f"No quedaron muestras utilizables en: {ruta}"
        )

    clases = clasificar_objetivo(
        datos_utilizables["rendimiento_objetivo"]
    )

    filas_utilizables = len(datos_utilizables)

    filas_purgadas = (
        total_archivo - filas_utilizables
    )

    registro = {
        "simbolo": simbolo,
        "division": division,
        "archivo": str(ruta),
        "desde_archivo": desde_archivo,
        "hasta_archivo": hasta_archivo,
        "desde_division": desde_division.isoformat(),
        "hasta_division": hasta_division.isoformat(),
        "filas_archivo": total_archivo,
        "filas_utilizables": filas_utilizables,
        "filas_purgadas": filas_purgadas,
        "primera_fecha_utilizable": (
            datos_utilizables["fecha_apertura"]
            .min()
            .isoformat()
        ),
        "ultima_fecha_utilizable": (
            datos_utilizables["fecha_apertura"]
            .max()
            .isoformat()
        ),
        "primer_objetivo": (
            datos_utilizables["fecha_objetivo"]
            .min()
            .isoformat()
        ),
        "ultimo_objetivo": (
            datos_utilizables["fecha_objetivo"]
            .max()
            .isoformat()
        ),
    }

    del datos
    del datos_utilizables
    gc.collect()

    return registro, clases


def crear_resumen(
    simbolo: str,
    division: str,
    conteos: dict[str, int],
    filas_purgadas: int,
) -> dict[str, object]:
    """Crea el resumen de una división temporal."""

    total = sum(conteos.values())

    if total == 0:
        raise ValueError(
            f"No existen muestras para {simbolo} - {division}."
        )

    return {
        "simbolo": simbolo,
        "division": division,
        "umbral": UMBRAL_CLASE,
        "umbral_porcentaje": UMBRAL_CLASE * 100,
        "muestras": total,
        "filas_purgadas": filas_purgadas,
        "cantidad_baja": conteos["BAJA"],
        "cantidad_neutral": conteos["NEUTRAL"],
        "cantidad_sube": conteos["SUBE"],
        "porcentaje_baja": (
            conteos["BAJA"] / total * 100
        ),
        "porcentaje_neutral": (
            conteos["NEUTRAL"] / total * 100
        ),
        "porcentaje_sube": (
            conteos["SUBE"] / total * 100
        ),
    }


def validar_separaciones(
    manifiesto: pd.DataFrame,
) -> None:
    """Comprueba que los objetivos no crucen las divisiones."""

    orden_divisiones = (
        "entrenamiento",
        "validacion",
        "prueba",
    )

    for simbolo in SIMBOLOS:
        resumen_simbolo = {}

        for division in orden_divisiones:
            registros = manifiesto.loc[
                (
                    manifiesto["simbolo"] == simbolo
                )
                & (
                    manifiesto["division"] == division
                )
            ]

            if registros.empty:
                raise ValueError(
                    f"Falta la división {division} "
                    f"para {simbolo}."
                )

            resumen_simbolo[division] = {
                "primera_fecha": pd.to_datetime(
                    registros[
                        "primera_fecha_utilizable"
                    ]
                ).min(),
                "ultima_fecha": pd.to_datetime(
                    registros[
                        "ultima_fecha_utilizable"
                    ]
                ).max(),
                "ultimo_objetivo": pd.to_datetime(
                    registros[
                        "ultimo_objetivo"
                    ]
                ).max(),
            }

        entrenamiento = resumen_simbolo[
            "entrenamiento"
        ]

        validacion = resumen_simbolo[
            "validacion"
        ]

        prueba = resumen_simbolo[
            "prueba"
        ]

        if (
            entrenamiento["ultimo_objetivo"]
            >= validacion["primera_fecha"]
        ):
            raise ValueError(
                f"{simbolo}: el objetivo de entrenamiento "
                "cruza hacia validación."
            )

        if (
            validacion["ultimo_objetivo"]
            >= prueba["primera_fecha"]
        ):
            raise ValueError(
                f"{simbolo}: el objetivo de validación "
                "cruza hacia prueba."
            )


def mostrar_resumen(
    resumen: dict[str, object],
) -> None:
    """Muestra una división en consola."""

    print(
        f"\n{resumen['simbolo']} "
        f"- {resumen['division'].upper()}"
    )

    print("-" * 60)

    print(
        "Muestras: "
        f"{int(resumen['muestras']):,}"
        .replace(",", ".")
    )

    print(
        "Filas purgadas: "
        f"{int(resumen['filas_purgadas']):,}"
        .replace(",", ".")
    )

    print(
        "BAJA: "
        f"{resumen['porcentaje_baja']:.2f} %"
    )

    print(
        "NEUTRAL: "
        f"{resumen['porcentaje_neutral']:.2f} %"
    )

    print(
        "SUBE: "
        f"{resumen['porcentaje_sube']:.2f} %"
    )


def main() -> None:
    """Prepara y valida las divisiones temporales."""

    print("\nPREPARACIÓN DE DIVISIONES TEMPORALES")
    print("=" * 70)

    registros_manifiesto: list[
        dict[str, object]
    ] = []

    resumenes: list[
        dict[str, object]
    ] = []

    for simbolo in SIMBOLOS:
        for division, configuracion in DIVISIONES.items():
            desde_division = pd.Timestamp(
                configuracion["desde"],
                tz="UTC",
            )

            hasta_division = pd.Timestamp(
                configuracion["hasta"],
                tz="UTC",
            )

            conteos = crear_conteos_vacios()
            filas_purgadas = 0

            for (
                desde_archivo,
                hasta_archivo,
            ) in configuracion["periodos"]:
                print(
                    f"\nProcesando {simbolo}: "
                    f"{desde_archivo} a {hasta_archivo}"
                )

                registro, clases = procesar_archivo(
                    simbolo=simbolo,
                    division=division,
                    desde_archivo=desde_archivo,
                    hasta_archivo=hasta_archivo,
                    desde_division=desde_division,
                    hasta_division=hasta_division,
                )

                registros_manifiesto.append(
                    registro
                )

                sumar_clases(
                    destino=conteos,
                    clases=clases,
                )

                filas_purgadas += int(
                    registro["filas_purgadas"]
                )

            resumen = crear_resumen(
                simbolo=simbolo,
                division=division,
                conteos=conteos,
                filas_purgadas=filas_purgadas,
            )

            resumenes.append(resumen)
            mostrar_resumen(resumen)

    manifiesto = pd.DataFrame(
        registros_manifiesto
    )

    resumen_divisiones = pd.DataFrame(
        resumenes
    )

    validar_separaciones(
        manifiesto=manifiesto
    )

    manifiesto.to_csv(
        RUTA_MANIFIESTO,
        index=False,
        encoding="utf-8-sig",
    )

    resumen_divisiones.to_csv(
        RUTA_RESUMEN,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("DIVISIONES PREPARADAS CORRECTAMENTE")
    print("=" * 70)

    print(
        "Manifiesto: "
        f"{RUTA_MANIFIESTO}"
    )

    print(
        "Resumen: "
        f"{RUTA_RESUMEN}"
    )

    print(
        "\nLos objetivos de entrenamiento y validación "
        "no cruzan hacia la siguiente división."
    )


if __name__ == "__main__":
    main()