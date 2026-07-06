from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cripto.corto_plazo_baja_v1.configuracion import (
    COLUMNAS_COMPLETAS_101,
    OBJETIVOS,
    PERIODOS_DESARROLLO,
    RUTA_DATOS_BAJA,
    SIMBOLO,
)
from cripto.corto_plazo_baja_v1.generar_datos_baja import (
    ruta_salida_baja,
)


def main() -> None:
    registros = []

    print(
        "\nAUDITORÍA DE DATOS BAJA V1"
    )
    print("=" * 72)

    for desde, hasta in PERIODOS_DESARROLLO:
        ruta = ruta_salida_baja(
            SIMBOLO,
            desde,
            hasta,
        )

        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe: {ruta}"
            )

        datos = pd.read_parquet(
            ruta
        )

        datos[
            "fecha_apertura"
        ] = pd.to_datetime(
            datos[
                "fecha_apertura"
            ],
            utc=True,
            errors="raise",
        )

        matriz = datos[
            list(
                COLUMNAS_COMPLETAS_101
            )
        ].to_numpy(
            dtype="float64"
        )

        variables_finitas = bool(
            np.isfinite(
                matriz
            ).all()
        )

        fechas_validas = bool(
            datos[
                "fecha_apertura"
            ].is_monotonic_increasing
            and not datos[
                "fecha_apertura"
            ].duplicated().any()
        )

        print(
            f"\n{desde} a {hasta}"
        )
        print("-" * 72)
        print(
            f"Filas: {len(datos):,}".replace(
                ",",
                ".",
            )
        )
        print(
            f"Variables finitas: {variables_finitas}"
        )
        print(
            f"Fechas válidas: {fechas_validas}"
        )

        for nombre in OBJETIVOS:
            columna_objetivo = (
                f"objetivo_{nombre}"
            )

            columna_retorno = (
                f"retorno_salida_{nombre}"
            )

            columna_duracion = (
                f"minutos_salida_{nombre}"
            )

            columna_resultado = (
                f"resultado_{nombre}"
            )

            faltantes = [
                columna
                for columna in (
                    columna_objetivo,
                    columna_retorno,
                    columna_duracion,
                    columna_resultado,
                    f"fecha_fin_horizonte_{nombre}",
                )
                if columna not in datos.columns
            ]

            if faltantes:
                raise ValueError(
                    f"{ruta.name}: faltan columnas para {nombre}: "
                    + ", ".join(
                        faltantes
                    )
                )

            objetivo = pd.to_numeric(
                datos[
                    columna_objetivo
                ],
                errors="coerce",
            )

            retorno = pd.to_numeric(
                datos[
                    columna_retorno
                ],
                errors="coerce",
            )

            duracion = pd.to_numeric(
                datos[
                    columna_duracion
                ],
                errors="coerce",
            )

            mascara = (
                objetivo.notna()
                & retorno.notna()
                & np.isfinite(
                    retorno
                )
                & (
                    duracion > 0
                )
            )

            filas_validas = int(
                mascara.sum()
            )

            proporcion_valida = (
                filas_validas
                / len(datos)
                if len(datos)
                else 0.0
            )

            prevalencia = float(
                objetivo.loc[
                    mascara
                ].mean()
                if filas_validas
                else 0.0
            )

            conteos = (
                datos.loc[
                    mascara,
                    columna_resultado,
                ]
                .value_counts(
                    dropna=False
                )
                .to_dict()
            )

            aprobado = bool(
                variables_finitas
                and fechas_validas
                and filas_validas > 0
                and proporcion_valida
                >= 0.98
                and 0.0
                < prevalencia
                < 1.0
            )

            registros.append(
                {
                    "desde": desde,
                    "hasta": hasta,
                    "objetivo": nombre,
                    "filas_totales": len(
                        datos
                    ),
                    "filas_validas": filas_validas,
                    "proporcion_valida": proporcion_valida,
                    "prevalencia": prevalencia,
                    "porcentaje_positivos": (
                        prevalencia * 100.0
                    ),
                    "retorno_medio_eventos": float(
                        retorno.loc[
                            mascara
                        ].mean()
                    ),
                    "duracion_media_minutos": float(
                        duracion.loc[
                            mascara
                        ].mean()
                    ),
                    "conteos_resultado": json.dumps(
                        {
                            str(clave): int(valor)
                            for clave, valor
                            in conteos.items()
                        },
                        ensure_ascii=False,
                    ),
                    "aprobado": aprobado,
                }
            )

            print(
                f"{nombre}: "
                f"válidas={filas_validas:,} "
                f"({proporcion_valida:.2%}), "
                f"prevalencia={prevalencia:.4f} "
                f"({prevalencia:.2%}), "
                f"aprobado={aprobado}".replace(
                    ",",
                    ".",
                )
            )

    tabla = pd.DataFrame(
        registros
    )

    ruta_csv = (
        RUTA_DATOS_BAJA
        / "auditoria_datos_baja_v1.csv"
    )

    tabla.to_csv(
        ruta_csv,
        index=False,
        encoding="utf-8-sig",
    )

    ruta_json = (
        RUTA_DATOS_BAJA
        / "auditoria_datos_baja_v1.json"
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
                "uso_2025_para_seleccion": False,
                "uso_2026_para_seleccion": False,
                "registros": registros,
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
            "AUDITORÍA DE DATOS BAJA V1 NO APROBADA"
        )

    print(
        "\nAUDITORÍA BAJA V1 APROBADA"
    )
    print(
        f"- {ruta_csv}"
    )
    print(
        f"- {ruta_json}"
    )


if __name__ == "__main__":
    main()
