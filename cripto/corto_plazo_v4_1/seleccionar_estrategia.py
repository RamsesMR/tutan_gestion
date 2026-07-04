from __future__ import annotations

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_1.configuracion import (
    COSTE_SELECCION,
    OPERACIONES_MINIMAS_POR_PLIEGUE,
    RUTA_RESULTADOS_LABORATORIO,
    RUTA_RESUMEN_LABORATORIO,
    RUTA_SELECCION,
)


CLAVES = [
    "umbral_sube",
    "modo_entrada",
    "umbral_veto_baja",
    "usa_veto_baja",
    "tipo_salida",
    "nombre_salida",
    "horizonte_salida",
    "coste",
]


def main() -> None:
    """Combina 2023/2024 y selecciona una zona robusta."""

    if not RUTA_RESULTADOS_LABORATORIO.exists():
        raise FileNotFoundError(
            f"No existe: {RUTA_RESULTADOS_LABORATORIO}"
        )

    resultados = pd.read_csv(
        RUTA_RESULTADOS_LABORATORIO
    )

    resultados[
        "umbral_veto_baja_clave"
    ] = resultados[
        "umbral_veto_baja"
    ].fillna(
        -1.0
    )

    claves = [
        "umbral_sube",
        "modo_entrada",
        "umbral_veto_baja_clave",
        "usa_veto_baja",
        "tipo_salida",
        "nombre_salida",
        "horizonte_salida",
        "coste",
    ]

    metricas = [
        "operaciones",
        "precision_clasificacion",
        "porcentaje_acierto_clasificacion",
        "porcentaje_operaciones_positivas",
        "retorno_bruto_medio",
        "retorno_neto_medio",
        "retorno_neto_mediano",
        "factor_beneficio",
        "maximo_drawdown",
    ]

    pliegues = (
        "validacion_2023",
        "validacion_2024",
    )

    primera = resultados.loc[
        resultados[
            "pliegue"
        ] == pliegues[
            0
        ],
        [
            *claves,
            *metricas,
        ],
    ].rename(
        columns={
            metrica: f"{metrica}_2023"
            for metrica in metricas
        }
    )

    segunda = resultados.loc[
        resultados[
            "pliegue"
        ] == pliegues[
            1
        ],
        [
            *claves,
            *metricas,
        ],
    ].rename(
        columns={
            metrica: f"{metrica}_2024"
            for metrica in metricas
        }
    )

    resumen = primera.merge(
        segunda,
        on=claves,
        validate="one_to_one",
    )

    for metrica in metricas:
        resumen[
            f"{metrica}_minimo"
        ] = resumen[
            [
                f"{metrica}_2023",
                f"{metrica}_2024",
            ]
        ].min(
            axis=1
        )

        resumen[
            f"{metrica}_medio"
        ] = resumen[
            [
                f"{metrica}_2023",
                f"{metrica}_2024",
            ]
        ].mean(
            axis=1
        )

        resumen[
            f"{metrica}_diferencia"
        ] = (
            resumen[
                f"{metrica}_2023"
            ]
            - resumen[
                f"{metrica}_2024"
            ]
        ).abs()

    resumen[
        "umbral_veto_baja"
    ] = resumen[
        "umbral_veto_baja_clave"
    ].replace(
        -1.0,
        np.nan,
    )

    resumen[
        "cumple_filtros"
    ] = (
        (
            resumen[
                "coste"
            ].round(
                6
            )
            == round(
                COSTE_SELECCION,
                6,
            )
        )
        & (
            resumen[
                "operaciones_minimo"
            ]
            >= OPERACIONES_MINIMAS_POR_PLIEGUE
        )
        & (
            resumen[
                "retorno_neto_medio_minimo"
            ]
            > 0.0
        )
        & (
            resumen[
                "retorno_neto_mediano_minimo"
            ]
            >= 0.0
        )
        & (
            resumen[
                "factor_beneficio_minimo"
            ]
            > 1.0
        )
    )

    resumen[
        "puntuacion"
    ] = (
        8.0
        * resumen[
            "retorno_neto_medio_minimo"
        ]
        + 2.0
        * resumen[
            "retorno_neto_mediano_minimo"
        ]
        + 0.08
        * (
            resumen[
                "factor_beneficio_minimo"
            ]
            - 1.0
        )
        + 0.001
        * resumen[
            "porcentaje_operaciones_positivas_minimo"
        ]
        + 0.0005
        * resumen[
            "porcentaje_acierto_clasificacion_minimo"
        ]
        + 0.03
        * resumen[
            "maximo_drawdown_minimo"
        ]
        - 0.50
        * resumen[
            "retorno_neto_medio_diferencia"
        ]
    )

    candidatos_coste = resumen.loc[
        resumen[
            "coste"
        ].round(
            6
        )
        == round(
            COSTE_SELECCION,
            6,
        )
    ].copy()

    elegibles = candidatos_coste.loc[
        candidatos_coste[
            "cumple_filtros"
        ]
    ].copy()

    fuente = (
        elegibles
        if not elegibles.empty
        else candidatos_coste
    )

    seleccion = fuente.sort_values(
        [
            "puntuacion",
            "operaciones_minimo",
        ],
        ascending=[
            False,
            False,
        ],
    ).head(
        1
    ).copy()

    RUTA_RESUMEN_LABORATORIO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resumen.sort_values(
        "puntuacion",
        ascending=False,
    ).to_csv(
        RUTA_RESUMEN_LABORATORIO,
        index=False,
        encoding="utf-8-sig",
    )

    seleccion.to_csv(
        RUTA_SELECCION,
        index=False,
        encoding="utf-8-sig",
    )

    fila = seleccion.iloc[
        0
    ]

    print(
        "\nSELECCIÓN V4.1"
    )
    print("=" * 72)
    print(
        f"Cumple filtros: {bool(fila['cumple_filtros'])}"
    )
    print(
        f"Entrada: {fila['modo_entrada']}"
    )
    print(
        f"Umbral SUBE: {fila['umbral_sube']:.2f}"
    )
    print(
        "Veto BAJA: "
        + (
            "sin veto"
            if not bool(
                fila[
                    "usa_veto_baja"
                ]
            )
            else f"{fila['umbral_veto_baja']:.2f}"
        )
    )
    print(
        f"Salida: {fila['nombre_salida']}"
    )
    print(
        f"Coste: {fila['coste']:.2%}"
    )
    print(
        f"Precisión 2023: {fila['precision_clasificacion_2023']:.2%}"
    )
    print(
        f"Precisión 2024: {fila['precision_clasificacion_2024']:.2%}"
    )
    print(
        "Porcentaje de acierto mínimo: "
        f"{fila['porcentaje_acierto_clasificacion_minimo']:.2f} %"
    )
    print(
        "Operaciones positivas mínimas: "
        f"{fila['porcentaje_operaciones_positivas_minimo']:.2f} %"
    )
    print(
        f"Operaciones mínimas: {int(fila['operaciones_minimo'])}"
    )
    print(
        f"Retorno neto mínimo: {fila['retorno_neto_medio_minimo']:.4%}"
    )
    print(
        f"Retorno mediano mínimo: {fila['retorno_neto_mediano_minimo']:.4%}"
    )
    print(
        f"Factor beneficio mínimo: {fila['factor_beneficio_minimo']:.3f}"
    )
    print(
        f"Drawdown peor: {fila['maximo_drawdown_minimo']:.2%}"
    )
    print(
        f"\n- {RUTA_SELECCION}"
    )


if __name__ == "__main__":
    main()
