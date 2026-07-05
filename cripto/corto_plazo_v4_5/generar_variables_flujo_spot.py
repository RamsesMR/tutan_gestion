from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_5.configuracion import (
    COLUMNAS_CONTROL,
    COLUMNAS_FLUJO_SPOT,
    PERIODOS_ARCHIVOS,
    RUTA_DATOS_SPOT,
    SIMBOLOS,
)
from cripto.corto_plazo_v4_5.utilidades_datos import (
    cargar_base,
    consultar_velas_spot,
    crear_variables_flujo_spot,
    guardar_parquet_seguro,
    ruta_archivo_etapa,
    validar_columnas_finitas,
)


def aplicar_mascara_validez(
    datos: pd.DataFrame,
    columnas_modelo: tuple[str, ...],
) -> tuple[pd.DataFrame, int, dict[str, int]]:
    """
    Conserva solo las filas finitas para todas las variables.

    No rellena ni interpola. Las filas posteriores a un hueco se descartan
    hasta recuperar todo el contexto necesario para las ventanas.
    """

    valores = datos[
        list(columnas_modelo)
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
            columnas_modelo,
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
            "La máscara común de validez descartó todas las filas."
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

    RUTA_DATOS_SPOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    resumenes = []

    columnas_modelo = (
        *COLUMNAS_CONTROL,
        *COLUMNAS_FLUJO_SPOT,
    )

    print("\nGENERACIÓN DE VARIABLES DE FLUJO SPOT V4.5")
    print("=" * 72)
    print(f"Variables nuevas: {len(COLUMNAS_FLUJO_SPOT)}")
    print(
        "Regla de validez: se excluyen filas sin contexto suficiente; "
        "no se rellena ni interpola."
    )

    for simbolo in SIMBOLOS:
        for desde, hasta in PERIODOS_ARCHIVOS:
            print(
                f"\n{simbolo}: {desde} a {hasta}"
            )
            print("-" * 72)

            base = cargar_base(
                simbolo,
                desde,
                hasta,
            )

            velas = consultar_velas_spot(
                simbolo,
                desde,
                hasta,
            )

            variables = crear_variables_flujo_spot(
                velas
            )

            combinado = base.merge(
                variables,
                on="fecha_apertura",
                how="left",
                validate="one_to_one",
            )

            if len(combinado) != len(base):
                raise RuntimeError(
                    "El cruce Spot modificó el número de filas antes "
                    "de aplicar la máscara común de validez."
                )

            filas_base = len(
                combinado
            )

            (
                combinado,
                filas_descartadas,
                columnas_afectadas,
            ) = aplicar_mascara_validez(
                combinado,
                columnas_modelo,
            )

            validar_columnas_finitas(
                combinado,
                columnas_modelo,
                f"{simbolo} {desde} a {hasta}",
            )

            ruta_salida = ruta_archivo_etapa(
                simbolo,
                desde,
                hasta,
                "spot",
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
                    "filas_base": filas_base,
                    "filas_validas": len(combinado),
                    "filas_descartadas": filas_descartadas,
                    "variables_control": len(COLUMNAS_CONTROL),
                    "variables_flujo_spot": len(COLUMNAS_FLUJO_SPOT),
                    "columnas_con_no_finitos": json.dumps(
                        columnas_afectadas,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "archivo": str(ruta_salida),
                }
            )

            print(
                f"Filas base: {filas_base:,}".replace(",", ".")
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
        RUTA_DATOS_SPOT / "resumen_variables_flujo_spot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (
        RUTA_DATOS_SPOT / "manifiesto_flujo_spot.json"
    ).write_text(
        json.dumps(
            {
                "variables_nuevas": list(
                    COLUMNAS_FLUJO_SPOT
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

    print("\nVARIABLES DE FLUJO SPOT GENERADAS CORRECTAMENTE")


if __name__ == "__main__":
    main()
