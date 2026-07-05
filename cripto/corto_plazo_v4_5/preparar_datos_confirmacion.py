from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_5.configuracion import (
    COLUMNAS_CONTROL,
    COLUMNAS_FLUJO_SPOT,
    RUTA_DATOS_SPOT,
    SIMBOLO_OPERATIVO,
)
from cripto.corto_plazo_v4_5.configuracion_confirmacion import (
    PERIODOS_DATOS_CONFIRMACION,
    VERSION_CONFIRMACION,
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
) -> tuple[
    pd.DataFrame,
    int,
    dict[str, int],
]:
    valores = datos[
        list(
            columnas_modelo
        )
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
        columna: int(
            cantidad
        )
        for columna, cantidad
        in zip(
            columnas_modelo,
            matriz_no_finitos.sum(
                axis=0
            ),
        )
        if int(
            cantidad
        )
        > 0
    }

    filas_descartadas = int(
        (
            ~mascara_validas
        ).sum()
    )

    resultado = (
        datos.loc[
            mascara_validas
        ]
        .reset_index(
            drop=True
        )
    )

    if resultado.empty:
        raise ValueError(
            "La máscara de validez descartó todas las filas."
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

    columnas_modelo = (
        *COLUMNAS_CONTROL,
        *COLUMNAS_FLUJO_SPOT,
    )

    resumenes = []

    print(
        "\nPREPARACIÓN SPOT PARA CONFIRMAR V4.5"
    )
    print("=" * 72)
    print(
        "Periodos: 2025 y enero-mayo de 2026."
    )
    print(
        "No se rellena ni interpola ninguna variable."
    )

    for (
        desde,
        hasta,
    ) in PERIODOS_DATOS_CONFIRMACION:
        ruta_salida = ruta_archivo_etapa(
            SIMBOLO_OPERATIVO,
            desde,
            hasta,
            "spot",
        )

        if (
            ruta_salida.exists()
            and not argumentos.sobrescribir
        ):
            datos_existentes = pd.read_parquet(
                ruta_salida
            )

            validar_columnas_finitas(
                datos_existentes,
                columnas_modelo,
                (
                    f"{SIMBOLO_OPERATIVO} "
                    f"{desde} a {hasta}"
                ),
            )

            resumenes.append(
                {
                    "simbolo": SIMBOLO_OPERATIVO,
                    "desde": desde,
                    "hasta": hasta,
                    "estado": "conservado",
                    "filas_validas": len(
                        datos_existentes
                    ),
                    "filas_descartadas": None,
                    "archivo": str(
                        ruta_salida
                    ),
                }
            )

            print(
                f"\nConservado: {ruta_salida}"
            )

            continue

        print(
            f"\n{SIMBOLO_OPERATIVO}: "
            f"{desde} a {hasta}"
        )
        print("-" * 72)

        base = cargar_base(
            SIMBOLO_OPERATIVO,
            desde,
            hasta,
        )

        velas = consultar_velas_spot(
            SIMBOLO_OPERATIVO,
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

        if len(
            combinado
        ) != len(
            base
        ):
            raise RuntimeError(
                "El cruce Spot modificó el número de filas "
                "antes de aplicar la máscara de validez."
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
            (
                f"{SIMBOLO_OPERATIVO} "
                f"{desde} a {hasta}"
            ),
        )

        guardar_parquet_seguro(
            combinado,
            ruta_salida,
            argumentos.sobrescribir,
        )

        resumenes.append(
            {
                "simbolo": SIMBOLO_OPERATIVO,
                "desde": desde,
                "hasta": hasta,
                "estado": "generado",
                "filas_base": filas_base,
                "filas_validas": len(
                    combinado
                ),
                "filas_descartadas": (
                    filas_descartadas
                ),
                "columnas_con_no_finitos": (
                    columnas_afectadas
                ),
                "archivo": str(
                    ruta_salida
                ),
            }
        )

        print(
            "Filas base: "
            f"{filas_base:,}".replace(
                ",",
                ".",
            )
        )
        print(
            "Filas válidas: "
            f"{len(combinado):,}".replace(
                ",",
                ".",
            )
        )
        print(
            "Filas descartadas: "
            f"{filas_descartadas:,}".replace(
                ",",
                ".",
            )
        )
        print(
            f"- {ruta_salida}"
        )

    ruta_resumen = (
        RUTA_DATOS_SPOT
        / "resumen_confirmacion_2025_2026.csv"
    )

    pd.DataFrame(
        resumenes
    ).to_csv(
        ruta_resumen,
        index=False,
        encoding="utf-8-sig",
    )

    ruta_manifiesto = (
        RUTA_DATOS_SPOT
        / "manifiesto_confirmacion_2025_2026.json"
    )

    ruta_manifiesto.write_text(
        json.dumps(
            {
                "fecha_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "version": VERSION_CONFIRMACION,
                "simbolo": SIMBOLO_OPERATIVO,
                "periodos": [
                    {
                        "desde": desde,
                        "hasta": hasta,
                    }
                    for desde, hasta
                    in PERIODOS_DATOS_CONFIRMACION
                ],
                "variables_control": len(
                    COLUMNAS_CONTROL
                ),
                "variables_spot": len(
                    COLUMNAS_FLUJO_SPOT
                ),
                "variables_totales": len(
                    columnas_modelo
                ),
                "uso_2025_para_seleccion_v4_5": False,
                "uso_2026_para_seleccion_v4_5": False,
                "regla_validez": (
                    "Máscara común finita; sin relleno ni interpolación."
                ),
            },
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\nDATOS DE CONFIRMACIÓN PREPARADOS"
    )
    print(
        f"- {ruta_resumen}"
    )
    print(
        f"- {ruta_manifiesto}"
    )


if __name__ == "__main__":
    main()
