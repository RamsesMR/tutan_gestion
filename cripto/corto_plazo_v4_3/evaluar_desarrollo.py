from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from cripto.corto_plazo_v4_3.configuracion import (
    ESTRATEGIA_CONGELADA,
    PLIEGUES_TEMPORALES,
    RUTA_PREDICCIONES_DESARROLLO,
    RUTA_RESULTADOS_DESARROLLO,
    SIMBOLO,
    VERSION_MODELO,
)
from cripto.corto_plazo_v4_3.evaluacion import (
    evaluar_datos,
)


def main() -> None:
    """Evalúa V4.3 con la estrategia V4.1 sin volver a seleccionarla."""

    registros = []

    print(
        "\nEVALUACIÓN DE DESARROLLO V4.3"
    )
    print("=" * 72)
    print(
        "Estrategia congelada: cruce 0,44, salida 480m, coste 0,10 %."
    )

    for nombre_pliegue in PLIEGUES_TEMPORALES:
        ruta = (
            RUTA_PREDICCIONES_DESARROLLO
            / f"{SIMBOLO}_{nombre_pliegue}.parquet"
        )

        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe: {ruta}"
            )

        datos = pd.read_parquet(
            ruta
        )

        metricas = evaluar_datos(
            datos=datos
        )

        registro = {
            "pliegue": nombre_pliegue,
            **ESTRATEGIA_CONGELADA,
            **metricas,
        }

        registros.append(
            registro
        )

        print("\n" + nombre_pliegue)
        print("-" * 72)
        print(
            f"Operaciones: {metricas['operaciones']}"
        )
        print(
            "Precisión: "
            f"{metricas['precision_clasificacion']:.4f}"
        )
        print(
            "Porcentaje de acierto: "
            f"{metricas['porcentaje_acierto_clasificacion']:.2f} %"
        )
        print(
            "Operaciones positivas: "
            f"{metricas['porcentaje_operaciones_positivas']:.2f} %"
        )
        print(
            "Retorno neto medio: "
            f"{metricas['retorno_neto_medio']:.4%}"
        )
        print(
            "Factor de beneficio: "
            f"{metricas['factor_beneficio']:.4f}"
        )
        print(
            "Drawdown máximo: "
            f"{metricas['maximo_drawdown']:.2%}"
        )

    RUTA_RESULTADOS_DESARROLLO.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_csv = (
        RUTA_RESULTADOS_DESARROLLO
        / "resultados_desarrollo.csv"
    )

    ruta_json = (
        RUTA_RESULTADOS_DESARROLLO
        / "detalle_desarrollo.json"
    )

    pd.DataFrame(
        registros
    ).to_csv(
        ruta_csv,
        index=False,
        encoding="utf-8-sig",
    )

    detalle = {
        "fecha_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "version": VERSION_MODELO,
        "estrategia_congelada": ESTRATEGIA_CONGELADA,
        "historico_ampliado_desde": "2017-08-17",
        "estrategia_modificada": False,
        "resultados": registros,
    }

    ruta_json.write_text(
        json.dumps(
            detalle,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print("\nArchivos generados:")
    print(f"- {ruta_csv}")
    print(f"- {ruta_json}")


if __name__ == "__main__":
    main()
