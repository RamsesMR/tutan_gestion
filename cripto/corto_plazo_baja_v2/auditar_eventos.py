from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cripto.corto_plazo_baja_v2.configuracion import (
    COLUMNAS_BASE_63,
    COLUMNAS_MACRO,
    PERIODOS_FUENTE,
    REJILLAS_EVENTOS_MINUTOS,
    RUTA_AUDITORIAS,
)
from cripto.corto_plazo_baja_v2.utilidades import (
    asegurar_carpetas,
    guardar_json,
    ruta_eventos,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rejillas",
        default=",".join(str(v) for v in REJILLAS_EVENTOS_MINUTOS),
    )
    args = parser.parse_args()
    rejillas = tuple(int(v.strip()) for v in args.rejillas.split(",") if v.strip())

    asegurar_carpetas(RUTA_AUDITORIAS)
    errores: list[str] = []
    resumen: list[dict] = []

    for desde, hasta in PERIODOS_FUENTE:
        for rejilla in rejillas:
            ruta = ruta_eventos(desde, hasta, rejilla)
            if not ruta.exists():
                errores.append(f"No existe {ruta}")
                continue
            datos = pd.read_parquet(ruta)
            datos["fecha_apertura"] = pd.to_datetime(
                datos["fecha_apertura"], utc=True, errors="raise"
            )
            if not datos["fecha_apertura"].is_monotonic_increasing:
                errores.append(f"Fechas desordenadas: {ruta.name}")
            if datos["event_id"].duplicated().any():
                errores.append(f"event_id duplicado: {ruta.name}")
            faltantes = sorted(
                {"clase_evento", "retorno_neto", "duracion", *COLUMNAS_BASE_63}
                - set(datos.columns)
            )
            if faltantes:
                errores.append(f"{ruta.name}: faltan {faltantes}")
            if not set(datos["clase_evento"].dropna().unique()).issubset({0, 1, 2}):
                errores.append(f"Clases inválidas: {ruta.name}")
            x = datos[list(COLUMNAS_BASE_63)].to_numpy(dtype="float64")
            if not np.isfinite(x).all():
                errores.append(f"Variables base no finitas: {ruta.name}")

            cobertura = {
                c: float(datos[c].notna().mean()) if c in datos.columns else 0.0
                for c in COLUMNAS_MACRO
            }
            resumen.append(
                {
                    "archivo": ruta.name,
                    "filas": int(len(datos)),
                    "desde": str(datos["fecha_apertura"].min()),
                    "hasta": str(datos["fecha_apertura"].max()),
                    "clases": {
                        str(k): int(v)
                        for k, v in datos["clase_evento"]
                        .value_counts()
                        .sort_index()
                        .items()
                    },
                    "cobertura_macro": cobertura,
                }
            )

    guardar_json(
        RUTA_AUDITORIAS / "auditoria_eventos.json",
        {"aprobada": not errores, "errores": errores, "archivos": resumen},
    )
    for fila in resumen:
        print(
            f"{fila['archivo']} | filas={fila['filas']} | "
            f"clases={fila['clases']}"
        )
    if errores:
        print("\nAUDITORÍA BAJA V2 RECHAZADA")
        for error in errores:
            print(f"- {error}")
        raise SystemExit(1)
    print("\nAUDITORÍA BAJA V2 APROBADA")


if __name__ == "__main__":
    main()
