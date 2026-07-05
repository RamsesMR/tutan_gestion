from __future__ import annotations

import itertools
import json

import pandas as pd

from cripto.corto_plazo_v4_4.configuracion import (
    CONDICIONES_SALIDA,
    MINUTOS_MINIMOS_EN_POSICION,
    MODOS_EVALUACION,
    PLIEGUES,
    RUTA_LABORATORIO,
    UMBRALES_BAJA,
    VERSION_MODELO,
)
from cripto.corto_plazo_v4_4.utilidades import (
    cargar_pliegue,
    convertir_para_json,
    simular_estrategia,
)


def construir_estrategia_id(
    umbral_baja: float | None,
    minutos_minimos: int,
    condicion_salida: str,
) -> str:
    """Construye un identificador reproducible."""

    if umbral_baja is None:
        return "control_v4_1_fija_480m"

    umbral = str(
        umbral_baja
    ).replace(
        ".",
        ""
    )

    return (
        f"baja_{umbral}_"
        f"min{minutos_minimos}_"
        f"{condicion_salida}"
    )


def main() -> None:
    """Evalúa un laboratorio pequeño y predefinido de salidas BAJA."""

    RUTA_LABORATORIO.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidatos = [
        (
            None,
            0,
            "siempre",
        )
    ]

    candidatos.extend(
        itertools.product(
            UMBRALES_BAJA,
            MINUTOS_MINIMOS_EN_POSICION,
            CONDICIONES_SALIDA,
        )
    )

    registros = []

    print(
        "\nLABORATORIO DE SALIDAS BAJA V4.4"
    )
    print("=" * 72)
    print(
        "Entrada congelada: cruce SUBE 0.44, sin veto."
    )
    print(
        "Salida de control: 480 minutos, coste 0.10 %."
    )
    print(
        "2025 y 2026 no se utilizan."
    )

    for pliegue in PLIEGUES:
        print(
            f"\n{pliegue}"
        )
        print("-" * 72)

        datos = cargar_pliegue(
            pliegue
        )

        for modo_evaluacion in MODOS_EVALUACION:
            for (
                umbral_baja,
                minutos_minimos,
                condicion_salida,
            ) in candidatos:
                estrategia_id = construir_estrategia_id(
                    umbral_baja=umbral_baja,
                    minutos_minimos=int(
                        minutos_minimos
                    ),
                    condicion_salida=str(
                        condicion_salida
                    ),
                )

                metricas, _ = simular_estrategia(
                    datos=datos,
                    modo_evaluacion=modo_evaluacion,
                    umbral_baja=(
                        None
                        if umbral_baja is None
                        else float(
                            umbral_baja
                        )
                    ),
                    minutos_minimos=int(
                        minutos_minimos
                    ),
                    condicion_salida=str(
                        condicion_salida
                    ),
                )

                registros.append(
                    {
                        "pliegue": pliegue,
                        "modo_evaluacion": modo_evaluacion,
                        "estrategia_id": estrategia_id,
                        "es_control_v4_1": (
                            umbral_baja is None
                        ),
                        "umbral_baja": umbral_baja,
                        "minutos_minimos": int(
                            minutos_minimos
                        ),
                        "condicion_salida": str(
                            condicion_salida
                        ),
                        **metricas,
                    }
                )

        print(
            "Combinaciones evaluadas correctamente."
        )

    resultados = pd.DataFrame(
        registros
    )

    ruta_csv = (
        RUTA_LABORATORIO
        / "resultados_laboratorio.csv"
    )

    resultados.to_csv(
        ruta_csv,
        index=False,
        encoding="utf-8-sig",
    )

    detalle = {
        "version": VERSION_MODELO,
        "uso_2025": False,
        "uso_2026": False,
        "entrada_v4_1_modificada": False,
        "combinaciones": len(
            resultados
        ),
        "resultados": [
            {
                clave: convertir_para_json(
                    valor
                )
                for clave, valor in registro.items()
            }
            for registro in resultados.to_dict(
                orient="records"
            )
        ],
    }

    (
        RUTA_LABORATORIO
        / "detalle_laboratorio.json"
    ).write_text(
        json.dumps(
            detalle,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\nLABORATORIO FINALIZADO"
    )
    print(
        f"- {ruta_csv}"
    )


if __name__ == "__main__":
    main()
