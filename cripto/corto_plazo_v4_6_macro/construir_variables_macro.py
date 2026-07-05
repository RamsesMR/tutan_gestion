from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from cripto.corto_plazo_v4_6_macro.configuracion import (
    RUTA_MACRO_BRUTO,
    RUTA_VARIABLES_MACRO,
    SERIES_MACRO,
)
from cripto.corto_plazo_v4_6_macro.utilidades_macro import (
    construir_variable,
    leer_serie_fred,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sobrescribir",
        action="store_true",
    )

    argumentos = parser.parse_args()

    RUTA_VARIABLES_MACRO.mkdir(
        parents=True,
        exist_ok=True,
    )

    detalle_salida = {}

    print(
        "\nCONSTRUCCIÓN DE VARIABLES MACRO V4.6"
    )
    print("=" * 72)
    print(
        "Las transformaciones se calculan en la frecuencia original."
    )
    print(
        "Después se asigna la fecha real o conservadora de disponibilidad."
    )

    for clave, configuracion in SERIES_MACRO.items():
        serie = str(
            configuracion["serie"]
        )
        columna = str(
            configuracion["columna"]
        )

        ruta_salida = (
            RUTA_VARIABLES_MACRO
            / f"{columna}.parquet"
        )

        if (
            ruta_salida.exists()
            and not argumentos.sobrescribir
        ):
            raise FileExistsError(
                f"Ya existe: {ruta_salida}. "
                "Usa --sobrescribir."
            )

        datos = leer_serie_fred(
            RUTA_MACRO_BRUTO / f"{serie}.csv",
            serie,
        )

        variable = construir_variable(
            datos=datos,
            columna_salida=columna,
            transformacion=str(
                configuracion["transformacion"]
            ),
            periodos=int(
                configuracion["periodos"]
            ),
            regla_disponibilidad=str(
                configuracion["disponibilidad"]
            ),
        )

        variable.to_parquet(
            ruta_salida,
            index=False,
        )

        detalle_salida[clave] = {
            "serie": serie,
            "columna": columna,
            "filas": len(variable),
            "primera_observacion": str(
                variable[
                    "fecha_observacion"
                ].min()
            ),
            "primera_disponibilidad": str(
                variable[
                    "fecha_disponibilidad"
                ].min()
            ),
            "ultima_disponibilidad": str(
                variable[
                    "fecha_disponibilidad"
                ].max()
            ),
            "archivo": str(ruta_salida),
        }

        print(
            f"\n{clave}: {columna}"
        )
        print("-" * 72)
        print(
            f"Serie: {serie}"
        )
        print(
            f"Registros transformados: {len(variable):,}".replace(
                ",",
                ".",
            )
        )
        print(
            "Primera disponibilidad: "
            f"{variable['fecha_disponibilidad'].min()}"
        )
        print(
            "Última disponibilidad: "
            f"{variable['fecha_disponibilidad'].max()}"
        )
        print(f"- {ruta_salida}")

    ruta_detalle = (
        RUTA_VARIABLES_MACRO
        / "detalle_variables_macro.json"
    )

    ruta_detalle.write_text(
        json.dumps(
            {
                "fecha_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "variables": detalle_salida,
            },
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\nVARIABLES MACRO CONSTRUIDAS CORRECTAMENTE"
    )
    print(f"- {ruta_detalle}")


if __name__ == "__main__":
    main()
