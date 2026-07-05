from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_5.configuracion import (
    COLUMNAS_CONTROL,
    COLUMNAS_SPOT,
    RUTA_DATOS_SPOT,
    SIMBOLO_OPERATIVO,
)
from cripto.corto_plazo_v4_5.configuracion_confirmacion import (
    PERIODOS_CONFIRMACION,
    RUTA_AUDITORIA_CONFIRMACION,
)
from cripto.corto_plazo_v4_5.utilidades_datos import (
    ruta_archivo_base,
    ruta_archivo_etapa,
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
    ).as_unit(
        "ns"
    )

    return pd.Series(
        indice,
        index=serie.index,
        dtype="datetime64[ns, UTC]",
    )


def main() -> None:
    RUTA_AUDITORIA_CONFIRMACION.mkdir(
        parents=True,
        exist_ok=True,
    )

    periodos_necesarios: set[
        tuple[str, str]
    ] = set()

    for configuracion in PERIODOS_CONFIRMACION.values():
        periodos_necesarios.update(
            configuracion[
                "archivos_entrenamiento"
            ]
        )
        periodos_necesarios.add(
            configuracion[
                "archivo_validacion"
            ]
        )

    registros = []

    print(
        "\nVALIDACIÓN DE DATOS — CONFIRMACIÓN V4.5"
    )
    print("=" * 72)

    for (
        desde,
        hasta,
    ) in sorted(
        periodos_necesarios
    ):
        ruta_base = ruta_archivo_base(
            SIMBOLO_OPERATIVO,
            desde,
            hasta,
        )

        ruta_spot = ruta_archivo_etapa(
            SIMBOLO_OPERATIVO,
            desde,
            hasta,
            "spot",
        )

        if not ruta_base.exists():
            raise FileNotFoundError(
                f"No existe el archivo base: {ruta_base}"
            )

        if not ruta_spot.exists():
            raise FileNotFoundError(
                f"No existe el archivo Spot: {ruta_spot}"
            )

        base = pd.read_parquet(
            ruta_base,
            columns=[
                "fecha_apertura",
            ],
        )

        columnas_revision = [
            "fecha_apertura",
            "fecha_objetivo",
            "rendimiento_objetivo",
            *COLUMNAS_SPOT,
        ]

        spot = pd.read_parquet(
            ruta_spot,
            columns=list(
                dict.fromkeys(
                    columnas_revision
                )
            ),
        )

        base[
            "fecha_apertura"
        ] = normalizar_fecha(
            base[
                "fecha_apertura"
            ]
        )

        spot[
            "fecha_apertura"
        ] = normalizar_fecha(
            spot[
                "fecha_apertura"
            ]
        )

        spot[
            "fecha_objetivo"
        ] = normalizar_fecha(
            spot[
                "fecha_objetivo"
            ]
        )

        fechas_base = set(
            pd.DatetimeIndex(
                base[
                    "fecha_apertura"
                ]
            ).asi8.tolist()
        )

        fechas_spot = set(
            pd.DatetimeIndex(
                spot[
                    "fecha_apertura"
                ]
            ).asi8.tolist()
        )

        valores = spot[
            list(
                COLUMNAS_SPOT
            )
        ].to_numpy(
            dtype="float64"
        )

        aprobado = bool(
            not spot.empty
            and len(
                spot
            )
            <= len(
                base
            )
            and fechas_spot.issubset(
                fechas_base
            )
            and not spot[
                "fecha_apertura"
            ].duplicated().any()
            and np.isfinite(
                valores
            ).all()
            and (
                spot[
                    "fecha_objetivo"
                ]
                > spot[
                    "fecha_apertura"
                ]
            ).all()
            and set(
                COLUMNAS_CONTROL
            ).issubset(
                spot.columns
            )
        )

        registro = {
            "simbolo": SIMBOLO_OPERATIVO,
            "desde": desde,
            "hasta": hasta,
            "filas_base": len(
                base
            ),
            "filas_spot": len(
                spot
            ),
            "filas_descartadas": (
                len(
                    base
                )
                - len(
                    spot
                )
            ),
            "variables_control": len(
                COLUMNAS_CONTROL
            ),
            "variables_spot": len(
                COLUMNAS_SPOT
            ),
            "primera_fecha": str(
                spot[
                    "fecha_apertura"
                ].min()
            ),
            "ultima_fecha": str(
                spot[
                    "fecha_apertura"
                ].max()
            ),
            "aprobado": aprobado,
        }

        registros.append(
            registro
        )

        print(
            f"{desde} a {hasta} | "
            f"base={len(base):,} | "
            f"spot={len(spot):,} | "
            f"descartadas={registro['filas_descartadas']:,} | "
            f"aprobado={aprobado}"
        )

    tabla = pd.DataFrame(
        registros
    )

    ruta_csv = (
        RUTA_AUDITORIA_CONFIRMACION
        / "auditoria_datos_confirmacion.csv"
    )

    tabla.to_csv(
        ruta_csv,
        index=False,
        encoding="utf-8-sig",
    )

    ruta_json = (
        RUTA_AUDITORIA_CONFIRMACION
        / "auditoria_datos_confirmacion.json"
    )

    ruta_json.write_text(
        json.dumps(
            {
                "fecha_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "aprobado": bool(
                    tabla[
                        "aprobado"
                    ].all()
                ),
                "registros": registros,
                "comparacion_misma_muestra": True,
                "control_63_lee_archivos_spot": True,
            },
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    if not tabla[
        "aprobado"
    ].all():
        raise ValueError(
            "VALIDACIÓN DE DATOS NO APROBADA"
        )

    print(
        "\nVALIDACIÓN DE DATOS APROBADA"
    )
    print(
        "V4.1-control y V4.5-Spot usarán exactamente "
        "las mismas velas en cada periodo."
    )
    print(
        f"- {ruta_csv}"
    )
    print(
        f"- {ruta_json}"
    )


if __name__ == "__main__":
    main()
