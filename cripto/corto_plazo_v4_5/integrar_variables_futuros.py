from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_5.configuracion import (
    COLUMNAS_COMPLETAS,
    COLUMNAS_FUTUROS,
    PERIODOS_ARCHIVOS,
    RUTA_DATOS_COMPLETOS,
    SIMBOLOS,
)
from cripto.corto_plazo_v4_5.utilidades_datos import (
    cargar_funding,
    cargar_klines_futuros,
    crear_variables_futuros,
    guardar_parquet_seguro,
    ruta_archivo_etapa,
    validar_columnas_finitas,
)


def aplicar_mascara_validez(
    datos: pd.DataFrame,
) -> tuple[pd.DataFrame, int, dict[str, int]]:
    """
    Excluye filas no finitas después de integrar futuros.

    No rellena ni interpola valores ausentes.
    """

    valores = datos[
        list(COLUMNAS_COMPLETAS)
    ].to_numpy(
        dtype="float64"
    )

    matriz_no_finitos = ~np.isfinite(
        valores
    )

    mascara_validas = ~matriz_no_finitos.any(
        axis=1
    )

    columnas_afectadas = {
        columna: int(cantidad)
        for columna, cantidad in zip(
            COLUMNAS_COMPLETAS,
            matriz_no_finitos.sum(axis=0),
        )
        if int(cantidad) > 0
    }

    filas_descartadas = int(
        (~mascara_validas).sum()
    )

    resultado = (
        datos.loc[
            mascara_validas
        ]
        .reset_index(drop=True)
    )

    if resultado.empty:
        raise ValueError(
            "La máscara común de futuros descartó todas las filas."
        )

    return (
        resultado,
        filas_descartadas,
        columnas_afectadas,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sobrescribir",
        action="store_true",
    )
    argumentos = parser.parse_args()

    RUTA_DATOS_COMPLETOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    resumenes = []

    print("\nINTEGRACIÓN DE FUTUROS, BASIS Y FUNDING V4.5")
    print("=" * 72)
    print(f"Variables nuevas de futuros: {len(COLUMNAS_FUTUROS)}")
    print(
        "Regla de validez: se excluyen filas sin contexto suficiente; "
        "no se rellena ni interpola."
    )

    for simbolo in SIMBOLOS:
        funding = cargar_funding(
            simbolo
        )

        for desde, hasta in PERIODOS_ARCHIVOS:
            print(
                f"\n{simbolo}: {desde} a {hasta}"
            )
            print("-" * 72)

            ruta_spot = ruta_archivo_etapa(
                simbolo,
                desde,
                hasta,
                "spot",
            )

            if not ruta_spot.exists():
                raise FileNotFoundError(
                    f"Primero genera el flujo Spot: {ruta_spot}"
                )

            spot = pd.read_parquet(
                ruta_spot
            )

            spot["fecha_apertura"] = pd.to_datetime(
                spot["fecha_apertura"],
                utc=True,
                errors="raise",
            )

            futuros = cargar_klines_futuros(
                simbolo,
                desde,
                hasta,
            )

            variables_futuros = crear_variables_futuros(
                datos_futuros=futuros,
                datos_spot=spot,
                datos_funding=funding,
            )

            combinado = spot.merge(
                variables_futuros,
                on="fecha_apertura",
                how="left",
                validate="one_to_one",
            )

            if len(combinado) != len(spot):
                raise RuntimeError(
                    "La integración de futuros modificó el número de filas "
                    "antes de aplicar la máscara común de validez."
                )

            filas_spot = len(
                combinado
            )

            (
                combinado,
                filas_descartadas,
                columnas_afectadas,
            ) = aplicar_mascara_validez(
                combinado
            )

            validar_columnas_finitas(
                combinado,
                COLUMNAS_COMPLETAS,
                f"{simbolo} {desde} a {hasta}",
            )

            ruta_salida = ruta_archivo_etapa(
                simbolo,
                desde,
                hasta,
                "completo",
            )

            guardar_parquet_seguro(
                combinado,
                ruta_salida,
                argumentos.sobrescribir,
            )

            resumenes.append(
                {
                    "simbolo": simbolo,
                    "desde": desde,
                    "hasta": hasta,
                    "filas_spot": filas_spot,
                    "filas_validas": len(combinado),
                    "filas_descartadas": filas_descartadas,
                    "variables_futuros": len(COLUMNAS_FUTUROS),
                    "columnas_con_no_finitos": json.dumps(
                        columnas_afectadas,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "archivo": str(ruta_salida),
                }
            )

            print(
                f"Filas Spot: {filas_spot:,}".replace(",", ".")
            )
            print(
                f"Filas válidas: {len(combinado):,}".replace(",", ".")
            )
            print(
                f"Filas descartadas: {filas_descartadas:,}".replace(",", ".")
            )

            if columnas_afectadas:
                print(
                    "Columnas que necesitaron más contexto:"
                )

                for columna, cantidad in columnas_afectadas.items():
                    print(
                        f"  - {columna}: {cantidad:,}".replace(",", ".")
                    )

            print(f"- {ruta_salida}")

    pd.DataFrame(
        resumenes
    ).to_csv(
        RUTA_DATOS_COMPLETOS / "resumen_integracion_futuros.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (
        RUTA_DATOS_COMPLETOS / "manifiesto_integracion_futuros.json"
    ).write_text(
        json.dumps(
            {
                "variables_futuros": list(
                    COLUMNAS_FUTUROS
                ),
                "regla_validez": (
                    "Máscara común finita; sin relleno ni interpolación."
                ),
                "uso_2025_para_seleccion": False,
                "uso_2026_para_seleccion": False,
            },
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print("\nINTEGRACIÓN DE FUTUROS APROBADA")


if __name__ == "__main__":
    main()
