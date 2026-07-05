from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_6_macro.configuracion import (
    RUTA_AUDITORIA,
    RUTA_MACRO_BRUTO,
    RUTA_VARIABLES_MACRO,
    SERIES_MACRO,
)
from cripto.corto_plazo_v4_6_macro.utilidades_macro import (
    leer_serie_fred,
)


FECHA_MAXIMA_PRIMER_DATO = pd.Timestamp(
    "2020-12-31",
    tz="UTC",
)

FECHA_MINIMA_ULTIMO_DATO = pd.Timestamp(
    "2024-12-15",
    tz="UTC",
)


def main() -> None:
    RUTA_AUDITORIA.mkdir(
        parents=True,
        exist_ok=True,
    )

    registros = []

    print(
        "\nAUDITORÍA DE DATOS MACRO V4.6"
    )
    print("=" * 72)

    for clave, configuracion in SERIES_MACRO.items():
        serie = str(
            configuracion["serie"]
        )
        columna = str(
            configuracion["columna"]
        )

        datos_fuente = leer_serie_fred(
            RUTA_MACRO_BRUTO / f"{serie}.csv",
            serie,
        )

        ruta_variable = (
            RUTA_VARIABLES_MACRO
            / f"{columna}.parquet"
        )

        if not ruta_variable.exists():
            raise FileNotFoundError(
                f"No existe: {ruta_variable}"
            )

        variable = pd.read_parquet(
            ruta_variable
        )

        variable["fecha_disponibilidad"] = (
            pd.to_datetime(
                variable["fecha_disponibilidad"],
                utc=True,
                errors="raise",
            )
        )

        valores = variable[
            columna
        ].to_numpy(dtype="float64")

        aprobado = bool(
            not variable.empty
            and variable[
                "fecha_disponibilidad"
            ].is_monotonic_increasing
            and not variable[
                "fecha_disponibilidad"
            ].duplicated().any()
            and np.isfinite(valores).all()
            and variable[
                "fecha_disponibilidad"
            ].min()
            <= FECHA_MAXIMA_PRIMER_DATO
            and variable[
                "fecha_disponibilidad"
            ].max()
            >= FECHA_MINIMA_ULTIMO_DATO
        )

        registro = {
            "clave": clave,
            "serie": serie,
            "columna": columna,
            "filas_fuente": len(datos_fuente),
            "filas_variable": len(variable),
            "primera_observacion": str(
                datos_fuente[
                    "fecha_observacion"
                ].min()
            ),
            "ultima_observacion": str(
                datos_fuente[
                    "fecha_observacion"
                ].max()
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
            "aprobado": aprobado,
        }

        registros.append(registro)

        print(
            f"{serie:<12} "
            f"{len(variable):>6} transformados | "
            f"{registro['primera_disponibilidad']} | "
            f"{registro['ultima_disponibilidad']} | "
            f"aprobado={aprobado}"
        )

    tabla = pd.DataFrame(registros)

    ruta_csv = (
        RUTA_AUDITORIA
        / "auditoria_fuentes_macro.csv"
    )

    tabla.to_csv(
        ruta_csv,
        index=False,
        encoding="utf-8-sig",
    )

    ruta_json = (
        RUTA_AUDITORIA
        / "auditoria_fuentes_macro.json"
    )

    ruta_json.write_text(
        json.dumps(
            {
                "aprobado": bool(
                    tabla["aprobado"].all()
                ),
                "registros": registros,
            },
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    if not tabla["aprobado"].all():
        raise ValueError(
            "AUDITORÍA MACRO NO APROBADA"
        )

    print(
        "\nAUDITORÍA MACRO APROBADA"
    )
    print(
        "ADVERTENCIA: FRED entrega el histórico vigente. "
        "Toda variable superviviente deberá confirmarse con "
        "vintages de ALFRED antes de ser promovida."
    )
    print(f"- {ruta_csv}")
    print(f"- {ruta_json}")


if __name__ == "__main__":
    main()
