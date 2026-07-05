from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cripto.corto_plazo_v4_3.configuracion import (
    RUTA_COMPARACION,
    RUTA_MODELOS_V4_3,
    RUTA_PROYECTO,
    RUTA_RESULTADO_2025,
    RUTA_RESULTADO_2026,
    RUTA_RESULTADOS_DESARROLLO,
)


METRICAS = (
    "operaciones",
    "precision_clasificacion",
    "porcentaje_acierto_clasificacion",
    "porcentaje_operaciones_positivas",
    "retorno_neto_medio",
    "retorno_neto_mediano",
    "factor_beneficio",
    "maximo_drawdown",
)


def cargar_json(
    ruta: Path,
) -> dict[str, Any]:
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe: {ruta}"
        )

    with ruta.open(
        "r",
        encoding="utf-8",
    ) as archivo:
        return json.load(
            archivo
        )


def buscar_ultimo_detalle(
    carpeta: Path,
) -> Path | None:
    candidatos = sorted(
        carpeta.glob(
            "*/detalle.json"
        )
    )

    if not candidatos:
        return None

    return candidatos[-1]


def registrar_comparacion(
    registros: list[dict[str, Any]],
    periodo: str,
    metricas_v4_1: dict[str, Any],
    metricas_v4_3: dict[str, Any],
) -> None:
    for metrica in METRICAS:
        valor_v4_1 = float(
            metricas_v4_1[
                metrica
            ]
        )

        valor_v4_3 = float(
            metricas_v4_3[
                metrica
            ]
        )

        registros.append(
            {
                "periodo": periodo,
                "metrica": metrica,
                "v4_1": valor_v4_1,
                "v4_3": valor_v4_3,
                "diferencia_v4_3_menos_v4_1": (
                    valor_v4_3
                    - valor_v4_1
                ),
            }
        )


def main() -> None:
    """Compara V4.3 con V4.1 sin promover automáticamente la candidata."""

    ruta_seleccion_v4_1 = (
        RUTA_PROYECTO
        / "modelos_entrenados"
        / "cripto"
        / "corto_plazo_v4_1"
        / "laboratorio"
        / "estrategia_seleccionada.csv"
    )

    ruta_desarrollo_v4_3 = (
        RUTA_RESULTADOS_DESARROLLO
        / "resultados_desarrollo.csv"
    )

    if not ruta_seleccion_v4_1.exists():
        raise FileNotFoundError(
            f"No existe: {ruta_seleccion_v4_1}"
        )

    if not ruta_desarrollo_v4_3.exists():
        raise FileNotFoundError(
            f"No existe: {ruta_desarrollo_v4_3}"
        )

    seleccion_v4_1 = pd.read_csv(
        ruta_seleccion_v4_1
    ).iloc[0]

    desarrollo_v4_3 = pd.read_csv(
        ruta_desarrollo_v4_3
    )

    registros: list[
        dict[str, Any]
    ] = []

    for anio in (
        "2023",
        "2024",
    ):
        fila_v4_3 = desarrollo_v4_3.loc[
            desarrollo_v4_3[
                "pliegue"
            ]
            == f"validacion_{anio}"
        ]

        if fila_v4_3.empty:
            raise ValueError(
                f"Falta validacion_{anio} en V4.3."
            )

        metricas_v4_1 = {
            metrica: seleccion_v4_1[
                f"{metrica}_{anio}"
            ]
            for metrica in METRICAS
        }

        metricas_v4_3 = {
            metrica: fila_v4_3.iloc[0][
                metrica
            ]
            for metrica in METRICAS
        }

        registrar_comparacion(
            registros=registros,
            periodo=anio,
            metricas_v4_1=metricas_v4_1,
            metricas_v4_3=metricas_v4_3,
        )

    ruta_v4_1 = (
        RUTA_PROYECTO
        / "modelos_entrenados"
        / "cripto"
        / "corto_plazo_v4_1"
    )

    comparaciones_conocidas = (
        (
            "2025_conocido",
            ruta_v4_1
            / "confirmacion_2025",
            RUTA_RESULTADO_2025
            / "detalle.json",
            "metricas_2025",
        ),
        (
            "2026_conocido",
            ruta_v4_1
            / "confirmacion_2026",
            RUTA_RESULTADO_2026
            / "detalle.json",
            "metricas_2026",
        ),
    )

    for (
        periodo,
        carpeta_v4_1,
        ruta_v4_3,
        clave_v4_1,
    ) in comparaciones_conocidas:
        ultimo_v4_1 = buscar_ultimo_detalle(
            carpeta=carpeta_v4_1
        )

        if (
            ultimo_v4_1 is None
            or not ruta_v4_3.exists()
        ):
            continue

        detalle_v4_1 = cargar_json(
            ultimo_v4_1
        )

        detalle_v4_3 = cargar_json(
            ruta_v4_3
        )

        registrar_comparacion(
            registros=registros,
            periodo=periodo,
            metricas_v4_1=detalle_v4_1[
                clave_v4_1
            ],
            metricas_v4_3=detalle_v4_3[
                "metricas"
            ],
        )

    RUTA_COMPARACION.mkdir(
        parents=True,
        exist_ok=True,
    )

    tabla = pd.DataFrame(
        registros
    )

    ruta_csv = (
        RUTA_COMPARACION
        / "comparacion_v4_1_vs_v4_3.csv"
    )

    ruta_md = (
        RUTA_COMPARACION
        / "resumen_comparacion_v4_1_vs_v4_3.md"
    )

    tabla.to_csv(
        ruta_csv,
        index=False,
        encoding="utf-8-sig",
    )

    lineas = [
        "# Comparación V4.1 frente a V4.3",
        "",
        "V4.3 conserva la estrategia de V4.1 y cambia únicamente el histórico de entrenamiento.",
        "",
        "Los resultados de 2025 y 2026 se consideran periodos ya conocidos, no pruebas vírgenes.",
        "",
        "Esta comparación no promueve automáticamente V4.3. V4.1 sigue siendo la campeona hasta una revisión explícita.",
        "",
    ]

    for periodo in tabla[
        "periodo"
    ].drop_duplicates():
        lineas.extend(
            [
                f"## {periodo}",
                "",
                "| Métrica | V4.1 | V4.3 | Diferencia |",
                "|---|---:|---:|---:|",
            ]
        )

        for _, fila in tabla.loc[
            tabla[
                "periodo"
            ]
            == periodo
        ].iterrows():
            lineas.append(
                "| "
                f"{fila['metrica']} | "
                f"{fila['v4_1']:.8f} | "
                f"{fila['v4_3']:.8f} | "
                f"{fila['diferencia_v4_3_menos_v4_1']:+.8f} |"
            )

        lineas.append("")

    ruta_md.write_text(
        "\n".join(
            lineas
        ),
        encoding="utf-8",
    )

    print(
        "\nCOMPARACIÓN GENERADA"
    )
    print("=" * 72)
    print(f"- {ruta_csv}")
    print(f"- {ruta_md}")
    print(
        "\nV4.1 continúa siendo la campeona hasta revisar los resultados."
    )


if __name__ == "__main__":
    main()
