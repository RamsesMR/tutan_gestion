from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from cripto.corto_plazo.configuracion import (
    HORIZONTE_MINUTOS,
    RUTA_DATOS_PREPARADOS,
)


PERIODOS = (
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-01-01"),
    ("2026-01-01", "2026-06-01"),
)

SIMBOLOS = (
    "BTCUSDT",
    "ETHUSDT",
)

COLUMNAS_OBLIGATORIAS = {
    "simbolo",
    "fecha_apertura",
    "precio_apertura",
    "precio_maximo",
    "precio_minimo",
    "precio_cierre",
    "volumen",
    "volumen_activo_cotizacion",
    "numero_operaciones",
    "volumen_comprador_base",
    "volumen_comprador_cotizacion",
    "fecha_objetivo",
    "precio_cierre_futuro",
    "rendimiento_objetivo",
}


def construir_ruta(
    simbolo: str,
    desde: str,
    hasta: str,
) -> Path:
    """Construye la ruta de un conjunto final de variables."""

    nombre = (
        f"{simbolo}_1m_4h_"
        f"{desde}_{hasta}_variables.parquet"
    )

    return RUTA_DATOS_PREPARADOS / nombre


def validar_archivo(
    ruta: Path,
    simbolo: str,
    desde: str,
    hasta: str,
    columnas_referencia: list[str] | None,
) -> tuple[dict[str, object], list[str], list[str]]:
    """Valida un archivo Parquet de variables."""

    errores: list[str] = []

    if not ruta.exists():
        return (
            {},
            [f"No existe el archivo: {ruta}"],
            columnas_referencia or [],
        )

    datos = pd.read_parquet(ruta)

    columnas_faltantes = COLUMNAS_OBLIGATORIAS.difference(
        datos.columns
    )

    if columnas_faltantes:
        errores.append(
            "Faltan columnas: "
            + ", ".join(sorted(columnas_faltantes))
        )

    columnas_actuales = list(datos.columns)

    if columnas_referencia is None:
        columnas_referencia = columnas_actuales

    elif columnas_actuales != columnas_referencia:
        errores.append(
            "Las columnas o su orden no coinciden "
            "con el primer conjunto validado."
        )

    if datos.empty:
        errores.append("El archivo no contiene muestras.")

        return (
            {},
            errores,
            columnas_referencia,
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

    total_nulos = int(
        datos.isna().sum().sum()
    )

    if total_nulos > 0:
        errores.append(
            f"Se encontraron {total_nulos} valores nulos."
        )

    simbolos_encontrados = (
        datos["simbolo"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if simbolos_encontrados != [simbolo]:
        errores.append(
            "Símbolo incorrecto. "
            f"Encontrados: {simbolos_encontrados}"
        )

    fechas_duplicadas = int(
        datos["fecha_apertura"]
        .duplicated()
        .sum()
    )

    if fechas_duplicadas > 0:
        errores.append(
            f"Se encontraron {fechas_duplicadas} fechas duplicadas."
        )

    if not datos["fecha_apertura"].is_monotonic_increasing:
        errores.append(
            "Las fechas no están ordenadas de forma ascendente."
        )

    horizonte_esperado = pd.Timedelta(
        minutes=HORIZONTE_MINUTOS
    )

    horizontes = (
        datos["fecha_objetivo"]
        - datos["fecha_apertura"]
    )

    horizontes_incorrectos = int(
        horizontes.ne(horizonte_esperado).sum()
    )

    if horizontes_incorrectos > 0:
        errores.append(
            f"Hay {horizontes_incorrectos} muestras "
            "con horizonte incorrecto."
        )

    total_infinitos = 0

    columnas_numericas = datos.select_dtypes(
        include="number"
    ).columns

    for columna in columnas_numericas:
        valores = datos[columna].to_numpy(
            dtype="float64",
            na_value=np.nan,
        )

        total_infinitos += int(
            np.isinf(valores).sum()
        )

    if total_infinitos > 0:
        errores.append(
            f"Se encontraron {total_infinitos} valores infinitos."
        )

    fecha_desde = pd.Timestamp(
        desde,
        tz="UTC",
    )

    fecha_hasta = pd.Timestamp(
        hasta,
        tz="UTC",
    )

    primera_fecha = datos["fecha_apertura"].min()
    ultima_fecha = datos["fecha_apertura"].max()

    fechas_fuera_periodo = int(
        (
            (datos["fecha_apertura"] < fecha_desde)
            | (datos["fecha_apertura"] >= fecha_hasta)
        ).sum()
    )

    if fechas_fuera_periodo > 0:
        errores.append(
            f"Hay {fechas_fuera_periodo} muestras "
            "fuera del periodo correspondiente."
        )

    resumen = {
        "simbolo": simbolo,
        "desde": desde,
        "hasta": hasta,
        "muestras": len(datos),
        "columnas": len(datos.columns),
        "primera_fecha": primera_fecha,
        "ultima_fecha": ultima_fecha,
        "errores": len(errores),
    }

    del datos
    gc.collect()

    return (
        resumen,
        errores,
        columnas_referencia,
    )


def validar_solapamientos(
    resumenes: list[dict[str, object]],
) -> list[str]:
    """Comprueba que los bloques de un símbolo no se solapen."""

    errores: list[str] = []

    for simbolo in SIMBOLOS:
        bloques = [
            resumen
            for resumen in resumenes
            if resumen["simbolo"] == simbolo
        ]

        bloques.sort(
            key=lambda bloque: str(bloque["desde"])
        )

        for anterior, actual in zip(
            bloques,
            bloques[1:],
        ):
            if (
                anterior["ultima_fecha"]
                >= actual["primera_fecha"]
            ):
                errores.append(
                    f"{simbolo}: existe solapamiento entre "
                    f"{anterior['desde']}–{anterior['hasta']} y "
                    f"{actual['desde']}–{actual['hasta']}."
                )

    return errores


def main() -> None:
    """Valida todos los conjuntos definitivos."""

    print("\nVALIDACIÓN GLOBAL DE CONJUNTOS")
    print("=" * 80)

    columnas_referencia: list[str] | None = None
    resumenes: list[dict[str, object]] = []
    errores_globales: list[str] = []

    for simbolo in SIMBOLOS:
        for desde, hasta in PERIODOS:
            ruta = construir_ruta(
                simbolo=simbolo,
                desde=desde,
                hasta=hasta,
            )

            print(f"\nValidando: {ruta.name}")

            (
                resumen,
                errores,
                columnas_referencia,
            ) = validar_archivo(
                ruta=ruta,
                simbolo=simbolo,
                desde=desde,
                hasta=hasta,
                columnas_referencia=columnas_referencia,
            )

            if resumen:
                resumenes.append(resumen)

                print(
                    f"Muestras: {resumen['muestras']:,}"
                    .replace(",", ".")
                )
                print(
                    f"Columnas: {resumen['columnas']}"
                )
                print(
                    f"Primera fecha: {resumen['primera_fecha']}"
                )
                print(
                    f"Última fecha: {resumen['ultima_fecha']}"
                )

            if errores:
                for error in errores:
                    mensaje = f"{ruta.name}: {error}"
                    errores_globales.append(mensaje)
                    print(f"ERROR: {error}")

            else:
                print("Resultado: correcto.")

    errores_globales.extend(
        validar_solapamientos(resumenes)
    )

    print("\n" + "=" * 80)
    print("RESUMEN GLOBAL")
    print("=" * 80)

    total_general = 0

    for simbolo in SIMBOLOS:
        total_simbolo = sum(
            int(resumen["muestras"])
            for resumen in resumenes
            if resumen["simbolo"] == simbolo
        )

        total_general += total_simbolo

        print(
            f"{simbolo}: "
            f"{total_simbolo:,} muestras"
            .replace(",", ".")
        )

    print(
        "TOTAL: "
        f"{total_general:,} muestras"
        .replace(",", ".")
    )

    if errores_globales:
        print("\nVALIDACIÓN FINALIZADA CON ERRORES:")

        for error in errores_globales:
            print(f"- {error}")

        raise SystemExit(1)

    print("\nTodos los conjuntos son coherentes.")
    print("No se detectaron nulos, infinitos, duplicados ni solapamientos.")


if __name__ == "__main__":
    main()