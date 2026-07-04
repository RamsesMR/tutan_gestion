from __future__ import annotations

import json
import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_2.configuracion import (
    OPERACIONES_MINIMAS_ANIO,
    RUTA_POLITICA,
    RUTA_RESULTADOS_FILTROS,
)


CLAVES = [
    "horizonte",
    "umbral_rearme",
    "enfriamiento",
    "distancia_minima",
    "diferencia_minima",
    "exige_pendiente_positiva_5m",
]


def main() -> None:
    resultados = pd.read_csv(RUTA_RESULTADOS_FILTROS)

    for columna in ("umbral_rearme", "diferencia_minima"):
        resultados[f"{columna}_clave"] = (
            resultados[columna].fillna(-999.0)
        )

    claves = [
        "horizonte",
        "umbral_rearme_clave",
        "enfriamiento",
        "distancia_minima",
        "diferencia_minima_clave",
        "exige_pendiente_positiva_5m",
    ]

    metricas = [
        "operaciones",
        "precision",
        "porcentaje_acierto",
        "porcentaje_positivas",
        "retorno_neto_medio",
        "retorno_neto_mediano",
        "factor_beneficio",
        "maximo_drawdown",
    ]

    a = resultados[
        resultados["periodo"] == "validacion_2023"
    ][claves + metricas].rename(
        columns={m: f"{m}_2023" for m in metricas}
    )

    b = resultados[
        resultados["periodo"] == "validacion_2024"
    ][claves + metricas].rename(
        columns={m: f"{m}_2024" for m in metricas}
    )

    resumen = a.merge(b, on=claves, validate="one_to_one")

    resumen["operaciones_minimo"] = resumen[
        ["operaciones_2023", "operaciones_2024"]
    ].min(axis=1)

    resumen["retorno_minimo"] = resumen[
        ["retorno_neto_medio_2023", "retorno_neto_medio_2024"]
    ].min(axis=1)

    resumen["mediana_minima"] = resumen[
        ["retorno_neto_mediano_2023", "retorno_neto_mediano_2024"]
    ].min(axis=1)

    resumen["pf_minimo"] = resumen[
        ["factor_beneficio_2023", "factor_beneficio_2024"]
    ].min(axis=1)

    resumen["positivas_minimo"] = resumen[
        ["porcentaje_positivas_2023", "porcentaje_positivas_2024"]
    ].min(axis=1)

    resumen["drawdown_peor"] = resumen[
        ["maximo_drawdown_2023", "maximo_drawdown_2024"]
    ].min(axis=1)

    resumen["estabilidad_retorno"] = (
        resumen["retorno_neto_medio_2023"]
        - resumen["retorno_neto_medio_2024"]
    ).abs()

    resumen["cumple"] = (
        (resumen["operaciones_minimo"] >= OPERACIONES_MINIMAS_ANIO)
        & (resumen["retorno_minimo"] > 0)
        & (resumen["mediana_minima"] >= 0)
        & (resumen["pf_minimo"] > 1)
    )

    resumen["puntuacion"] = (
        10.0 * resumen["retorno_minimo"]
        + 0.10 * (resumen["pf_minimo"] - 1.0)
        + 0.001 * resumen["positivas_minimo"]
        + 0.04 * resumen["drawdown_peor"]
        - 1.0 * resumen["estabilidad_retorno"]
    )

    elegibles = resumen[resumen["cumple"]].copy()

    if elegibles.empty:
        raise RuntimeError(
            "Ninguna política V4.2 cumple los filtros."
        )

    fila = elegibles.sort_values(
        ["puntuacion", "operaciones_minimo"],
        ascending=[False, False],
    ).iloc[0]

    politica = {
        "horizonte": int(fila["horizonte"]),
        "umbral_rearme": (
            None
            if float(fila["umbral_rearme_clave"]) == -999.0
            else float(fila["umbral_rearme_clave"])
        ),
        "enfriamiento": int(fila["enfriamiento"]),
        "distancia_minima": float(fila["distancia_minima"]),
        "diferencia_minima": (
            None
            if float(fila["diferencia_minima_clave"]) == -999.0
            else float(fila["diferencia_minima_clave"])
        ),
        "exige_pendiente_positiva_5m": bool(
            fila["exige_pendiente_positiva_5m"]
        ),
        "metricas": {
            "operaciones_2023": int(fila["operaciones_2023"]),
            "operaciones_2024": int(fila["operaciones_2024"]),
            "precision_2023": float(fila["precision_2023"]),
            "precision_2024": float(fila["precision_2024"]),
            "retorno_neto_medio_2023": float(
                fila["retorno_neto_medio_2023"]
            ),
            "retorno_neto_medio_2024": float(
                fila["retorno_neto_medio_2024"]
            ),
            "factor_beneficio_2023": float(
                fila["factor_beneficio_2023"]
            ),
            "factor_beneficio_2024": float(
                fila["factor_beneficio_2024"]
            ),
            "drawdown_2023": float(fila["maximo_drawdown_2023"]),
            "drawdown_2024": float(fila["maximo_drawdown_2024"]),
        },
    }

    RUTA_POLITICA.parent.mkdir(parents=True, exist_ok=True)

    RUTA_POLITICA.write_text(
        json.dumps(politica, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    print("\nPOLÍTICA V4.2 SELECCIONADA")
    print("=" * 72)
    print(json.dumps(politica, ensure_ascii=False, indent=4))
    print(f"\n- {RUTA_POLITICA}")


if __name__ == "__main__":
    main()
