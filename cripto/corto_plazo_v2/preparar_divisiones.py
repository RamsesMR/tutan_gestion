from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from cripto.corto_plazo_v2.configuracion import (
    DIVISIONES_DESARROLLO,
    HORIZONTE_MINUTOS,
    NOMBRE_HORIZONTE,
    RUTA_DATOS_V2,
    RUTA_MANIFIESTO_V2,
    RUTA_RESUMEN_DIVISIONES_V2,
    SIMBOLOS,
    UMBRAL_CLASE,
)


def construir_ruta(
    simbolo: str,
    desde: str,
    hasta: str,
) -> Path:
    """Construye la ruta de un Parquet V2A."""

    nombre = (
        f"{simbolo}_1m_{NOMBRE_HORIZONTE}_"
        f"{desde}_{hasta}_variables.parquet"
    )

    return RUTA_DATOS_V2 / nombre


def clasificar_objetivo(
    rendimientos: pd.Series,
) -> pd.Series:
    """Convierte el rendimiento futuro en tres clases."""

    valores = pd.to_numeric(
        rendimientos,
        errors="coerce",
    )

    if valores.isna().any():
        raise ValueError(
            "El objetivo contiene valores inválidos."
        )

    resultado = np.select(
        [
            valores <= -UMBRAL_CLASE,
            valores >= UMBRAL_CLASE,
        ],
        [
            "BAJA",
            "SUBE",
        ],
        default="NEUTRAL",
    )

    return pd.Series(
        resultado,
        index=rendimientos.index,
        dtype="string",
    )


def crear_conteos_vacios() -> dict[str, int]:
    """Crea un contador vacío de clases."""

    return {
        "BAJA": 0,
        "NEUTRAL": 0,
        "SUBE": 0,
    }


def sumar_clases(
    destino: dict[str, int],
    clases: pd.Series,
) -> None:
    """Suma los conteos de clases al acumulado."""

    conteos = clases.value_counts()

    for clase in destino:
        destino[clase] += int(
            conteos.get(clase, 0)
        )


def validar_horizonte(
    datos: pd.DataFrame,
    ruta: Path,
) -> None:
    """Comprueba que todos los objetivos estén exactamente a cuatro horas."""

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

    if incorrectos:
        raise ValueError(
            f"{ruta.name}: contiene {incorrectos} "
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
    """Valida un archivo V2A y determina sus filas utilizables."""

    ruta = construir_ruta(
        simbolo=simbolo,
        desde=desde_archivo,
        hasta=hasta_archivo,
    )

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el archivo V2A: {ruta}"
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
            f"{ruta.name}: contiene nulos o valores inválidos."
        )

    if datos["fecha_apertura"].duplicated().any():
        raise ValueError(
            f"{ruta.name}: contiene fechas de apertura duplicadas."
        )

    if not datos["fecha_apertura"].is_monotonic_increasing:
        raise ValueError(
            f"{ruta.name}: las fechas no están ordenadas."
        )

    validar_horizonte(
        datos=datos,
        ruta=ruta,
    )

    total_archivo = len(datos)

    mascara_utilizable = (
        (datos["fecha_apertura"] >= desde_division)
        & (datos["fecha_apertura"] < hasta_division)
        & (datos["fecha_objetivo"] > datos["fecha_apertura"])
        & (datos["fecha_objetivo"] < hasta_division)
    )

    utilizables = (
        datos.loc[
            mascara_utilizable
        ]
        .copy()
    )

    if utilizables.empty:
        raise ValueError(
            f"{ruta.name}: no quedaron muestras utilizables."
        )

    clases = clasificar_objetivo(
        utilizables["rendimiento_objetivo"]
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
        "filas_utilizables": len(utilizables),
        "filas_purgadas": total_archivo - len(utilizables),
        "primera_fecha_utilizable": (
            utilizables["fecha_apertura"]
            .min()
            .isoformat()
        ),
        "ultima_fecha_utilizable": (
            utilizables["fecha_apertura"]
            .max()
            .isoformat()
        ),
        "primer_objetivo": (
            utilizables["fecha_objetivo"]
            .min()
            .isoformat()
        ),
        "ultimo_objetivo": (
            utilizables["fecha_objetivo"]
            .max()
            .isoformat()
        ),
    }

    del datos
    del utilizables
    gc.collect()

    return registro, clases


def crear_resumen(
    simbolo: str,
    division: str,
    conteos: dict[str, int],
    filas_archivo: int,
    filas_purgadas: int,
) -> dict[str, object]:
    """Crea el resumen agregado de una división."""

    total = sum(
        conteos.values()
    )

    if total == 0:
        raise ValueError(
            f"No existen muestras para {simbolo} - {division}."
        )

    return {
        "simbolo": simbolo,
        "division": division,
        "umbral": UMBRAL_CLASE,
        "umbral_porcentaje": UMBRAL_CLASE * 100,
        "filas_archivo": filas_archivo,
        "muestras": total,
        "filas_purgadas": filas_purgadas,
        "cantidad_baja": conteos["BAJA"],
        "cantidad_neutral": conteos["NEUTRAL"],
        "cantidad_sube": conteos["SUBE"],
        "porcentaje_baja": conteos["BAJA"] / total * 100,
        "porcentaje_neutral": conteos["NEUTRAL"] / total * 100,
        "porcentaje_sube": conteos["SUBE"] / total * 100,
    }


def validar_separaciones(
    manifiesto: pd.DataFrame,
) -> None:
    """Comprueba que entrenamiento no cruce hacia validación."""

    for simbolo in SIMBOLOS:
        entrenamiento = manifiesto.loc[
            (
                manifiesto["simbolo"] == simbolo
            )
            & (
                manifiesto["division"] == "entrenamiento"
            )
        ]

        validacion = manifiesto.loc[
            (
                manifiesto["simbolo"] == simbolo
            )
            & (
                manifiesto["division"] == "validacion"
            )
        ]

        if entrenamiento.empty or validacion.empty:
            raise ValueError(
                f"{simbolo}: faltan divisiones obligatorias."
            )

        ultimo_objetivo_entrenamiento = pd.to_datetime(
            entrenamiento["ultimo_objetivo"],
            utc=True,
        ).max()

        primera_fecha_validacion = pd.to_datetime(
            validacion["primera_fecha_utilizable"],
            utc=True,
        ).min()

        if (
            ultimo_objetivo_entrenamiento
            >= primera_fecha_validacion
        ):
            raise ValueError(
                f"{simbolo}: entrenamiento cruza hacia validación."
            )


def mostrar_resumen(
    resumen: dict[str, object],
) -> None:
    """Muestra una división en consola."""

    print(
        f"\n{resumen['simbolo']} "
        f"- {str(resumen['division']).upper()}"
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
        f"BAJA: {float(resumen['porcentaje_baja']):.2f} %"
    )
    print(
        f"NEUTRAL: {float(resumen['porcentaje_neutral']):.2f} %"
    )
    print(
        f"SUBE: {float(resumen['porcentaje_sube']):.2f} %"
    )


def main() -> None:
    """Genera el manifiesto temporal propio de la V2A."""

    print(
        "\nPREPARACIÓN DE DIVISIONES TEMPORALES V2A"
    )
    print("=" * 70)
    print(
        "Se utilizarán únicamente entrenamiento 2021-2024 "
        "y validación 2025."
    )
    print(
        "Enero-mayo de 2026 no forma parte de este manifiesto."
    )

    registros: list[
        dict[str, object]
    ] = []

    resumenes: list[
        dict[str, object]
    ] = []

    for simbolo in SIMBOLOS:
        for division, configuracion in DIVISIONES_DESARROLLO.items():
            desde_division = pd.Timestamp(
                configuracion["desde"],
                tz="UTC",
            )

            hasta_division = pd.Timestamp(
                configuracion["hasta"],
                tz="UTC",
            )

            conteos = crear_conteos_vacios()
            filas_archivo = 0
            filas_purgadas = 0

            for desde_archivo, hasta_archivo in configuracion["periodos"]:
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

                registros.append(
                    registro
                )

                sumar_clases(
                    destino=conteos,
                    clases=clases,
                )

                filas_archivo += int(
                    registro["filas_archivo"]
                )

                filas_purgadas += int(
                    registro["filas_purgadas"]
                )

            resumen = crear_resumen(
                simbolo=simbolo,
                division=division,
                conteos=conteos,
                filas_archivo=filas_archivo,
                filas_purgadas=filas_purgadas,
            )

            resumenes.append(
                resumen
            )

            mostrar_resumen(
                resumen
            )

    manifiesto = pd.DataFrame(
        registros
    )

    resumen_divisiones = pd.DataFrame(
        resumenes
    )

    validar_separaciones(
        manifiesto=manifiesto
    )

    RUTA_DATOS_V2.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifiesto.to_csv(
        RUTA_MANIFIESTO_V2,
        index=False,
        encoding="utf-8-sig",
    )

    resumen_divisiones.to_csv(
        RUTA_RESUMEN_DIVISIONES_V2,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print(
        "DIVISIONES V2A PREPARADAS CORRECTAMENTE"
    )
    print("=" * 70)
    print(
        f"Manifiesto: {RUTA_MANIFIESTO_V2}"
    )
    print(
        f"Resumen: {RUTA_RESUMEN_DIVISIONES_V2}"
    )
    print(
        "\nLos objetivos de entrenamiento no cruzan "
        "hacia validación."
    )


if __name__ == "__main__":
    main()
