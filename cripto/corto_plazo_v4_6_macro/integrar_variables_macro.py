from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_6_macro.configuracion import (
    BASES,
    COLUMNAS_MACRO,
    RUTA_DATOS_INTEGRADOS,
    RUTA_VARIABLES_MACRO,
    SERIES_MACRO,
    SIMBOLOS,
)
from cripto.corto_plazo_v4_5.configuracion import (
    PERIODOS_ARCHIVOS,
)


def ruta_referencia(
    base: str,
    simbolo: str,
    desde: str,
    hasta: str,
):
    carpeta = BASES[
        base
    ]["ruta_referencia"]

    return (
        carpeta
        / f"{simbolo}_1m_4h_{desde}_{hasta}_variables.parquet"
    )


def normalizar_fecha(
    serie: pd.Series,
) -> pd.Series:
    indice = pd.DatetimeIndex(
        pd.to_datetime(
            serie,
            utc=True,
            errors="raise",
        )
    ).as_unit("ns")

    return pd.Series(
        indice,
        index=serie.index,
        dtype="datetime64[ns, UTC]",
    )


def cargar_eventos_macro(
) -> dict[str, pd.DataFrame]:
    eventos = {}

    for configuracion in SERIES_MACRO.values():
        columna = str(
            configuracion["columna"]
        )

        ruta = (
            RUTA_VARIABLES_MACRO
            / f"{columna}.parquet"
        )

        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe: {ruta}"
            )

        datos = pd.read_parquet(
            ruta,
            columns=[
                "fecha_disponibilidad",
                columna,
            ],
        )

        datos["fecha_disponibilidad"] = (
            normalizar_fecha(
                datos["fecha_disponibilidad"]
            )
        )

        datos = (
            datos.sort_values(
                "fecha_disponibilidad"
            )
            .drop_duplicates(
                subset=[
                    "fecha_disponibilidad"
                ],
                keep="last",
            )
            .reset_index(drop=True)
        )

        eventos[columna] = datos

    return eventos


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sobrescribir",
        action="store_true",
    )

    argumentos = parser.parse_args()

    eventos = cargar_eventos_macro()

    print(
        "\nINTEGRACIÓN DE VARIABLES MACRO V4.6"
    )
    print("=" * 72)
    print(
        "Regla: último valor publicado antes de cada vela."
    )
    print(
        "No se interpola y no se usa información futura."
    )

    for base, configuracion_base in BASES.items():
        columnas_base = tuple(
            configuracion_base["columnas"]
        )

        carpeta_salida = (
            RUTA_DATOS_INTEGRADOS
            / base
        )

        carpeta_salida.mkdir(
            parents=True,
            exist_ok=True,
        )

        for simbolo in SIMBOLOS:
            for desde, hasta in PERIODOS_ARCHIVOS:
                ruta_entrada = ruta_referencia(
                    base,
                    simbolo,
                    desde,
                    hasta,
                )

                if not ruta_entrada.exists():
                    raise FileNotFoundError(
                        f"No existe: {ruta_entrada}"
                    )

                ruta_salida = (
                    carpeta_salida
                    / f"{simbolo}_1m_4h_{desde}_{hasta}_variables.parquet"
                )

                if (
                    ruta_salida.exists()
                    and not argumentos.sobrescribir
                ):
                    raise FileExistsError(
                        f"Ya existe: {ruta_salida}. "
                        "Usa --sobrescribir."
                    )

                print(
                    f"\n{base} | {simbolo}: {desde} a {hasta}"
                )
                print("-" * 72)

                datos = pd.read_parquet(
                    ruta_entrada
                )

                datos["fecha_apertura"] = (
                    normalizar_fecha(
                        datos["fecha_apertura"]
                    )
                )

                datos["fecha_objetivo"] = (
                    normalizar_fecha(
                        datos["fecha_objetivo"]
                    )
                )

                datos = (
                    datos.sort_values(
                        "fecha_apertura"
                    )
                    .reset_index(drop=True)
                )

                filas_referencia = len(datos)

                for columna, tabla_eventos in eventos.items():
                    datos = pd.merge_asof(
                        datos,
                        tabla_eventos,
                        left_on="fecha_apertura",
                        right_on="fecha_disponibilidad",
                        direction="backward",
                        allow_exact_matches=True,
                    )

                    datos = datos.drop(
                        columns=[
                            "fecha_disponibilidad"
                        ]
                    )

                matriz_macro = datos[
                    list(COLUMNAS_MACRO)
                ].to_numpy(dtype="float64")

                mascara_macro = np.isfinite(
                    matriz_macro
                ).all(axis=1)

                filas_validas = int(
                    mascara_macro.sum()
                )

                datos = (
                    datos.loc[
                        mascara_macro
                    ]
                    .reset_index(drop=True)
                )

                matriz_base = datos[
                    list(columnas_base)
                ].to_numpy(dtype="float64")

                if not np.isfinite(
                    matriz_base
                ).all():
                    raise ValueError(
                        f"Hay variables base no finitas en {ruta_entrada.name}."
                    )

                datos.to_parquet(
                    ruta_salida,
                    index=False,
                )

                print(
                    f"Filas referencia: {filas_referencia:,}".replace(
                        ",",
                        ".",
                    )
                )
                print(
                    f"Filas válidas: {filas_validas:,}".replace(
                        ",",
                        ".",
                    )
                )
                print(
                    "Filas descartadas por falta de macro: "
                    f"{filas_referencia - filas_validas:,}".replace(
                        ",",
                        ".",
                    )
                )
                print(f"- {ruta_salida}")

                del datos
                del matriz_macro
                del matriz_base
                gc.collect()

    print(
        "\nINTEGRACIÓN MACRO COMPLETADA"
    )


if __name__ == "__main__":
    main()
