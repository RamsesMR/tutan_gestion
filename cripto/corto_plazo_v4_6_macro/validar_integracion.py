from __future__ import annotations

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_5.configuracion import (
    PERIODOS_ARCHIVOS,
)
from cripto.corto_plazo_v4_6_macro.configuracion import (
    BASES,
    COLUMNAS_MACRO,
    RUTA_DATOS_INTEGRADOS,
    SIMBOLOS,
)


def normalizar_fecha(
    serie: pd.Series,
) -> pd.Series:
    """Convierte cualquier precisión temporal a nanosegundos UTC."""

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


def fechas_en_nanosegundos(
    serie: pd.Series,
) -> set[int]:
    """Devuelve fechas comparables aunque el Parquet use us, ms o ns."""

    indice = pd.DatetimeIndex(
        normalizar_fecha(
            serie
        )
    )

    return set(
        indice.asi8.tolist()
    )


def main() -> None:
    registros = []

    print(
        "\nVALIDACIÓN FINAL DE DATOS V4.6-MACRO"
    )
    print("=" * 72)

    for base, configuracion_base in BASES.items():
        columnas_base = tuple(
            configuracion_base["columnas"]
        )

        for simbolo in SIMBOLOS:
            for desde, hasta in PERIODOS_ARCHIVOS:
                ruta_referencia = (
                    configuracion_base[
                        "ruta_referencia"
                    ]
                    / f"{simbolo}_1m_4h_{desde}_{hasta}_variables.parquet"
                )

                ruta_integrada = (
                    RUTA_DATOS_INTEGRADOS
                    / base
                    / f"{simbolo}_1m_4h_{desde}_{hasta}_variables.parquet"
                )

                if not ruta_referencia.exists():
                    raise FileNotFoundError(
                        f"No existe: {ruta_referencia}"
                    )

                if not ruta_integrada.exists():
                    raise FileNotFoundError(
                        f"No existe: {ruta_integrada}"
                    )

                referencia = pd.read_parquet(
                    ruta_referencia,
                    columns=[
                        "fecha_apertura",
                    ],
                )

                columnas_revision = [
                    "fecha_apertura",
                    "fecha_objetivo",
                    *columnas_base,
                    *COLUMNAS_MACRO,
                ]

                integrada = pd.read_parquet(
                    ruta_integrada,
                    columns=list(
                        dict.fromkeys(
                            columnas_revision
                        )
                    ),
                )

                referencia[
                    "fecha_apertura"
                ] = normalizar_fecha(
                    referencia[
                        "fecha_apertura"
                    ]
                )

                integrada[
                    "fecha_apertura"
                ] = normalizar_fecha(
                    integrada[
                        "fecha_apertura"
                    ]
                )

                integrada[
                    "fecha_objetivo"
                ] = normalizar_fecha(
                    integrada[
                        "fecha_objetivo"
                    ]
                )

                matriz = integrada[
                    [
                        *columnas_base,
                        *COLUMNAS_MACRO,
                    ]
                ].to_numpy(dtype="float64")

                fechas_referencia = (
                    fechas_en_nanosegundos(
                        referencia[
                            "fecha_apertura"
                        ]
                    )
                )

                fechas_integradas = (
                    fechas_en_nanosegundos(
                        integrada[
                            "fecha_apertura"
                        ]
                    )
                )

                aprobado = bool(
                    not integrada.empty
                    and len(integrada)
                    <= len(referencia)
                    and fechas_integradas.issubset(
                        fechas_referencia
                    )
                    and not integrada[
                        "fecha_apertura"
                    ].duplicated().any()
                    and np.isfinite(matriz).all()
                    and (
                        integrada[
                            "fecha_objetivo"
                        ]
                        > integrada[
                            "fecha_apertura"
                        ]
                    ).all()
                )

                registros.append(
                    {
                        "base": base,
                        "simbolo": simbolo,
                        "desde": desde,
                        "hasta": hasta,
                        "filas_referencia": len(
                            referencia
                        ),
                        "filas_validas": len(
                            integrada
                        ),
                        "filas_descartadas": (
                            len(referencia)
                            - len(integrada)
                        ),
                        "columnas_modelo_control": len(
                            columnas_base
                        ),
                        "columnas_modelo_completo": (
                            len(columnas_base)
                            + len(COLUMNAS_MACRO)
                        ),
                        "aprobado": aprobado,
                    }
                )

    tabla = pd.DataFrame(registros)

    print(
        tabla.to_string(
            index=False
        )
    )

    if not tabla["aprobado"].all():
        raise ValueError(
            "VALIDACIÓN V4.6-MACRO NO APROBADA"
        )

    print(
        "\nVALIDACIÓN V4.6-MACRO APROBADA"
    )


if __name__ == "__main__":
    main()
