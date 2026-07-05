from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cripto.corto_plazo_baja_v2.configuracion import (
    CONCENTRACION_MENSUAL_MAXIMA,
    DRAWDOWN_MINIMO_ADMITIDO,
    FACTOR_BENEFICIO_MINIMO,
    OPERACIONES_MINIMAS_POR_PLIEGUE,
    RUTA_HISTORIAL,
    RUTA_SELECCION,
)
from cripto.corto_plazo_baja_v2.utilidades import asegurar_carpetas, guardar_json


CLAVES = ["rejilla_minutos", "variante", "semilla", "margen_ev"]


def main() -> None:
    if not RUTA_HISTORIAL.exists():
        raise FileNotFoundError(
            f"No existe {RUTA_HISTORIAL}. Ejecuta primero entrenar_modelos."
        )
    asegurar_carpetas(RUTA_SELECCION)
    datos = pd.read_csv(RUTA_HISTORIAL)

    filas = []
    for claves, grupo in datos.groupby(CLAVES, dropna=False):
        por_pliegue = {
            fila["pliegue"]: fila
            for _, fila in grupo.iterrows()
        }
        if not {"validacion_2023", "validacion_2024"}.issubset(por_pliegue):
            continue
        a = por_pliegue["validacion_2023"]
        b = por_pliegue["validacion_2024"]
        aprobada = all([
            a["operaciones"] >= OPERACIONES_MINIMAS_POR_PLIEGUE,
            b["operaciones"] >= OPERACIONES_MINIMAS_POR_PLIEGUE,
            a["retorno_neto_medio"] > 0,
            b["retorno_neto_medio"] > 0,
            a["retorno_neto_mediano"] >= 0,
            b["retorno_neto_mediano"] >= 0,
            a["factor_beneficio"] > FACTOR_BENEFICIO_MINIMO,
            b["factor_beneficio"] > FACTOR_BENEFICIO_MINIMO,
            a["drawdown"] >= DRAWDOWN_MINIMO_ADMITIDO,
            b["drawdown"] >= DRAWDOWN_MINIMO_ADMITIDO,
            a["concentracion_mensual"] <= CONCENTRACION_MENSUAL_MAXIMA,
            b["concentracion_mensual"] <= CONCENTRACION_MENSUAL_MAXIMA,
        ])
        filas.append({
            **dict(zip(CLAVES, claves)),
            "aprobada_dos_pliegues": bool(aprobada),
            "operaciones_2023": int(a["operaciones"]),
            "operaciones_2024": int(b["operaciones"]),
            "precision_tp_2023": float(a["precision_tp"]),
            "porcentaje_acierto_tp_2023": float(a["porcentaje_acierto_tp"]),
            "precision_tp_2024": float(b["precision_tp"]),
            "porcentaje_acierto_tp_2024": float(b["porcentaje_acierto_tp"]),
            "retorno_neto_medio_2023": float(a["retorno_neto_medio"]),
            "retorno_neto_medio_2024": float(b["retorno_neto_medio"]),
            "retorno_neto_mediano_2023": float(a["retorno_neto_mediano"]),
            "retorno_neto_mediano_2024": float(b["retorno_neto_mediano"]),
            "factor_beneficio_2023": float(a["factor_beneficio"]),
            "factor_beneficio_2024": float(b["factor_beneficio"]),
            "drawdown_2023": float(a["drawdown"]),
            "drawdown_2024": float(b["drawdown"]),
            "concentracion_2023": float(a["concentracion_mensual"]),
            "concentracion_2024": float(b["concentracion_mensual"]),
            "retorno_medio_promedio": float(
                (a["retorno_neto_medio"] + b["retorno_neto_medio"]) / 2
            ),
            "peor_retorno_medio": float(
                min(a["retorno_neto_medio"], b["retorno_neto_medio"])
            ),
            "peor_pf": float(min(a["factor_beneficio"], b["factor_beneficio"])),
        })

    seleccion = pd.DataFrame(filas)
    if not seleccion.empty:
        seleccion = seleccion.sort_values(
            [
                "aprobada_dos_pliegues",
                "peor_retorno_medio",
                "peor_pf",
                "retorno_medio_promedio",
            ],
            ascending=[False, False, False, False],
        )
    ruta_csv = RUTA_SELECCION / "seleccion_baja_v2.csv"
    seleccion.to_csv(ruta_csv, index=False)

    aprobadas = seleccion[seleccion["aprobada_dos_pliegues"]] if not seleccion.empty else seleccion
    decision = {
        "version": "baja_v2_eventos_macro",
        "decision": (
            "existen_configuraciones_provisionales_para_robustez"
            if len(aprobadas)
            else "ninguna_configuracion_baja_v2_aprobada"
        ),
        "candidatas_aprobadas": int(len(aprobadas)),
        "usa_2025_seleccion": False,
        "usa_2026_seleccion": False,
        "promocion_automatica": False,
        "siguiente_paso": (
            "robustez_semillas_sobre_candidatas"
            if len(aprobadas)
            else "analizar_ablaciones_y_no_avanzar_a_2025"
        ),
        "mejor_fila": (
            seleccion.iloc[0].to_dict() if not seleccion.empty else None
        ),
    }
    guardar_json(RUTA_SELECCION / "decision_baja_v2.json", decision)

    print("=== SELECCIÓN BAJA V2 ===")
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=str))
    print(f"CSV: {ruta_csv}")


if __name__ == "__main__":
    main()
