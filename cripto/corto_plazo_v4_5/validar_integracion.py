from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_5.configuracion import (
    COLUMNAS_COMPLETAS,
    COLUMNAS_SPOT,
    PERIODOS_ARCHIVOS,
    RUTA_AUDITORIA,
    SIMBOLOS,
)
from cripto.corto_plazo_v4_5.utilidades_datos import (
    cargar_base,
    ruta_archivo_etapa,
)


def cargar_referencia(
    simbolo: str,
    desde: str,
    hasta: str,
    etapa: str,
) -> pd.DataFrame:
    if etapa == "spot":
        referencia = cargar_base(
            simbolo,
            desde,
            hasta,
        )[
            [
                "fecha_apertura",
            ]
        ].copy()
    elif etapa == "completo":
        ruta_spot = ruta_archivo_etapa(
            simbolo,
            desde,
            hasta,
            "spot",
        )

        if not ruta_spot.exists():
            raise FileNotFoundError(
                f"No existe la referencia Spot: {ruta_spot}"
            )

        referencia = pd.read_parquet(
            ruta_spot,
            columns=[
                "fecha_apertura",
            ],
        )
    else:
        raise ValueError(
            f"Etapa desconocida: {etapa}"
        )

    referencia["fecha_apertura"] = pd.to_datetime(
        referencia["fecha_apertura"],
        utc=True,
        errors="raise",
    )

    return (
        referencia.sort_values(
            "fecha_apertura"
        )
        .reset_index(drop=True)
    )


def validar_archivo(
    simbolo: str,
    desde: str,
    hasta: str,
    etapa: str,
    columnas: tuple[str, ...],
) -> dict:
    referencia = cargar_referencia(
        simbolo,
        desde,
        hasta,
        etapa,
    )

    ruta = ruta_archivo_etapa(
        simbolo,
        desde,
        hasta,
        etapa,
    )

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe: {ruta}"
        )

    datos = pd.read_parquet(
        ruta
    )

    datos["fecha_apertura"] = pd.to_datetime(
        datos["fecha_apertura"],
        utc=True,
        errors="raise",
    )

    if datos.empty:
        raise ValueError(
            f"{ruta.name}: no contiene filas."
        )

    if datos["fecha_apertura"].duplicated().any():
        raise ValueError(
            f"{ruta.name}: contiene fechas duplicadas."
        )

    if not datos["fecha_apertura"].is_monotonic_increasing:
        raise ValueError(
            f"{ruta.name}: las fechas no están ordenadas."
        )

    fechas_referencia = pd.Index(
        referencia["fecha_apertura"]
    )

    fechas_datos = pd.Index(
        datos["fecha_apertura"]
    )

    if not fechas_datos.isin(
        fechas_referencia
    ).all():
        raise ValueError(
            f"{ruta.name}: contiene fechas ajenas a su archivo de referencia."
        )

    faltantes = set(
        columnas
    ).difference(
        datos.columns
    )

    if faltantes:
        raise ValueError(
            f"{ruta.name}: faltan columnas {sorted(faltantes)}"
        )

    valores = datos[
        list(columnas)
    ].to_numpy(
        dtype="float64"
    )

    if not np.isfinite(
        valores
    ).all():
        raise ValueError(
            f"{ruta.name}: contiene valores no finitos."
        )

    return {
        "simbolo": simbolo,
        "desde": desde,
        "hasta": hasta,
        "etapa": etapa,
        "filas_referencia": len(
            referencia
        ),
        "filas_validas": len(
            datos
        ),
        "filas_descartadas": (
            len(referencia)
            - len(datos)
        ),
        "columnas_modelo": len(
            columnas
        ),
        "aprobado": True,
    }


def main() -> None:
    print("\nVALIDACIÓN FINAL DE DATOS V4.5")
    print("=" * 72)

    RUTA_AUDITORIA.mkdir(
        parents=True,
        exist_ok=True,
    )

    resultados = []

    for simbolo in SIMBOLOS:
        for desde, hasta in PERIODOS_ARCHIVOS:
            resultados.append(
                validar_archivo(
                    simbolo,
                    desde,
                    hasta,
                    "spot",
                    COLUMNAS_SPOT,
                )
            )

            ruta_completa = ruta_archivo_etapa(
                simbolo,
                desde,
                hasta,
                "completo",
            )

            if ruta_completa.exists():
                resultados.append(
                    validar_archivo(
                        simbolo,
                        desde,
                        hasta,
                        "completo",
                        COLUMNAS_COMPLETAS,
                    )
                )

    tabla = pd.DataFrame(
        resultados
    )

    tabla.to_csv(
        RUTA_AUDITORIA / "validacion_integracion_v4_5.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (
        RUTA_AUDITORIA / "resumen_validacion_v4_5.json"
    ).write_text(
        json.dumps(
            {
                "aprobado": bool(
                    tabla["aprobado"].all()
                ),
                "archivos_validados": len(
                    tabla
                ),
                "etapas": sorted(
                    tabla["etapa"].unique().tolist()
                ),
                "regla_validez": (
                    "Los archivos derivados pueden ser subconjuntos de las "
                    "fechas de referencia; todas sus variables deben ser finitas."
                ),
            },
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        tabla.to_string(
            index=False
        )
    )
    print("\nVALIDACIÓN V4.5 APROBADA")


if __name__ == "__main__":
    main()
