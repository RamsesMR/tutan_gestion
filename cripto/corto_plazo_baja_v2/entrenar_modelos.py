from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from cripto.corto_plazo_baja_v2.configuracion import (
    COLUMNAS_BASE_63,
    COSTE_BASE,
    MARGENES_EV,
    PARAMETROS_HISTGB,
    PLIEGUES_TEMPORALES,
    RUTA_HISTORIAL,
    RUTA_MODELOS,
    RUTA_RESULTADOS,
    SEMILLAS,
    VARIANTES,
)
from cripto.corto_plazo_baja_v2.utilidades import (
    asegurar_carpetas,
    construir_interacciones,
    metricas_operaciones,
    ruta_eventos,
    seleccionar_no_solapadas,
)


def cargar_periodos(
    periodos: tuple[tuple[str, str], ...],
    rejilla: int,
) -> pd.DataFrame:
    partes = []
    for desde, hasta in periodos:
        ruta = ruta_eventos(desde, hasta, rejilla)
        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe {ruta}. Ejecuta generar_eventos y auditar_eventos."
            )
        partes.append(pd.read_parquet(ruta))
    datos = pd.concat(partes, ignore_index=True)
    datos["fecha_apertura"] = pd.to_datetime(
        datos["fecha_apertura"], utc=True, errors="raise"
    )
    return datos.sort_values("fecha_apertura").reset_index(drop=True)


def columnas_variante(datos: pd.DataFrame, variante: str) -> list[str]:
    cfg = VARIANTES[variante]
    columnas = list(COLUMNAS_BASE_63)
    columnas.extend(cfg["columnas_macro"])
    if cfg["usar_interacciones"]:
        datos_inter = construir_interacciones(datos)
        extras = [
            c for c in datos_inter.columns
            if c not in datos.columns
        ]
        datos[extras] = datos_inter[extras]
        columnas.extend(extras)
    faltantes = [c for c in columnas if c not in datos.columns]
    if faltantes:
        raise KeyError(f"{variante}: faltan columnas {faltantes}")
    return list(dict.fromkeys(columnas))


def preparar_x(
    entrenamiento: pd.DataFrame,
    validacion: pd.DataFrame,
    columnas: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    utilizables = []
    for columna in columnas:
        tr = pd.to_numeric(entrenamiento[columna], errors="coerce")
        va = pd.to_numeric(validacion[columna], errors="coerce")
        if tr.notna().mean() < 0.80 or va.notna().mean() < 0.80:
            continue
        utilizables.append(columna)
    if len(utilizables) < 63:
        raise ValueError(
            f"Solo hay {len(utilizables)} columnas utilizables; "
            "las 63 variables base deben conservarse."
        )
    medianas = entrenamiento[utilizables].median(numeric_only=True)
    x_train = (
        entrenamiento[utilizables]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(medianas)
        .to_numpy(dtype="float64")
    )
    x_val = (
        validacion[utilizables]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(medianas)
        .to_numpy(dtype="float64")
    )
    if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
        raise ValueError("Persisten valores no finitos después de imputar.")
    return x_train, x_val, utilizables


def evaluar_clasificacion(
    y: np.ndarray,
    probabilidades: np.ndarray,
) -> dict:
    pred = np.argmax(probabilidades, axis=1)
    tp_real = (y == 1).astype("int8")
    p_tp = probabilidades[:, 1]
    salida = {
        "accuracy_multiclase": float((pred == y).mean()),
        "precision_tp": float(
            precision_score(tp_real, pred == 1, zero_division=0)
        ),
        "porcentaje_acierto_tp": float(
            precision_score(tp_real, pred == 1, zero_division=0) * 100.0
        ),
        "recall_tp": float(recall_score(tp_real, pred == 1, zero_division=0)),
        "f1_tp": float(f1_score(tp_real, pred == 1, zero_division=0)),
        "pr_auc_tp": float(average_precision_score(tp_real, p_tp)),
        "prevalencia_tp": float(tp_real.mean()),
        "matriz_confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }
    try:
        salida["roc_auc_tp"] = float(roc_auc_score(tp_real, p_tp))
    except ValueError:
        salida["roc_auc_tp"] = None
    salida["pr_auc_lift_tp"] = (
        salida["pr_auc_tp"] / salida["prevalencia_tp"]
        if salida["prevalencia_tp"] > 0
        else None
    )
    return salida


def entrenar_combinacion(
    rejilla: int,
    variante: str,
    semilla: int,
    pliegue: str,
    cfg_pliegue: dict,
) -> tuple[list[dict], dict]:
    train = cargar_periodos(tuple(cfg_pliegue["entrenamiento"]), rejilla)
    val = cargar_periodos((tuple(cfg_pliegue["validacion"]),), rejilla)

    limite_train = pd.Timestamp(cfg_pliegue["hasta_entrenamiento"], tz="UTC")
    train = train[train["fecha_fin_evento"] < limite_train].copy()

    columnas = columnas_variante(train, variante)
    if VARIANTES[variante]["usar_interacciones"]:
        val_inter = construir_interacciones(val)
        extras = [c for c in val_inter.columns if c not in val.columns]
        val[extras] = val_inter[extras]

    x_train, x_val, columnas_usadas = preparar_x(train, val, columnas)
    y_train = train["clase_evento"].to_numpy(dtype="int8")
    y_val = val["clase_evento"].to_numpy(dtype="int8")

    clasificador = HistGradientBoostingClassifier(
        **PARAMETROS_HISTGB,
        random_state=semilla,
    )
    clasificador.fit(x_train, y_train)
    probabilidades = clasificador.predict_proba(x_val)

    # HistGB devuelve columnas según classes_. Reordenamos siempre a SL, TP, TIMEOUT.
    probs = np.zeros((len(val), 3), dtype="float64")
    for posicion, clase in enumerate(clasificador.classes_):
        probs[:, int(clase)] = probabilidades[:, posicion]

    timeout_train = train["clase_evento"].to_numpy() == 2
    if timeout_train.sum() >= 100:
        regresor = HistGradientBoostingRegressor(
            **PARAMETROS_HISTGB,
            random_state=semilla,
        )
        regresor.fit(
            x_train[timeout_train],
            train.loc[timeout_train, "retorno_bruto"].to_numpy(dtype="float64"),
        )
        retorno_timeout = regresor.predict(x_val)
    else:
        regresor = None
        media_timeout = float(
            train.loc[timeout_train, "retorno_bruto"].mean()
        )
        retorno_timeout = np.full(len(val), media_timeout)

    ev = (
        probs[:, 1] * (0.0100 - COSTE_BASE)
        + probs[:, 0] * (-0.0050 - COSTE_BASE)
        + probs[:, 2] * (retorno_timeout - COSTE_BASE)
    )

    clasificacion = evaluar_clasificacion(y_val, probs)
    filas = []
    for margen in MARGENES_EV:
        mascara = ev > margen
        val_eval = val.copy()
        val_eval["ev_estimado"] = ev
        operaciones = seleccionar_no_solapadas(val_eval, mascara)
        met = metricas_operaciones(operaciones)
        filas.append(
            {
                "version": "baja_v2_eventos_macro",
                "pliegue": pliegue,
                "rejilla_minutos": rejilla,
                "variante": variante,
                "semilla": semilla,
                "margen_ev": margen,
                "columnas": len(columnas_usadas),
                "coste": COSTE_BASE,
                "usa_2025_seleccion": False,
                "usa_2026_seleccion": False,
                "promocion_automatica": False,
                **clasificacion,
                **met,
            }
        )

    artefacto = {
        "clasificador": clasificador,
        "regresor_timeout": regresor,
        "columnas": columnas_usadas,
        "medianas": train[columnas_usadas].median(numeric_only=True).to_dict(),
        "rejilla_minutos": rejilla,
        "variante": variante,
        "semilla": semilla,
        "pliegue": pliegue,
        "coste": COSTE_BASE,
    }
    return filas, artefacto


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rejillas", default="5,10,15,30")
    parser.add_argument("--variantes", default="control_63_eventos")
    parser.add_argument("--semillas", default="42")
    parser.add_argument("--sobrescribir", action="store_true")
    args = parser.parse_args()

    rejillas = [int(v) for v in args.rejillas.split(",") if v.strip()]
    variantes = (
        list(VARIANTES)
        if args.variantes == "todas"
        else [v.strip() for v in args.variantes.split(",") if v.strip()]
    )
    semillas = (
        list(SEMILLAS)
        if args.semillas == "todas"
        else [int(v) for v in args.semillas.split(",") if v.strip()]
    )

    desconocidas = sorted(set(variantes) - set(VARIANTES))
    if desconocidas:
        raise ValueError(f"Variantes desconocidas: {desconocidas}")

    asegurar_carpetas(RUTA_MODELOS, RUTA_RESULTADOS)
    historial_nuevo = []

    total = len(rejillas) * len(variantes) * len(semillas) * len(PLIEGUES_TEMPORALES)
    actual = 0
    for rejilla in rejillas:
        for variante in variantes:
            for semilla in semillas:
                for pliegue, cfg in PLIEGUES_TEMPORALES.items():
                    actual += 1
                    nombre = f"{variante}__{rejilla}m__s{semilla}__{pliegue}.joblib"
                    ruta_modelo = RUTA_MODELOS / nombre
                    if ruta_modelo.exists() and not args.sobrescribir:
                        print(f"[{actual}/{total}] CONSERVADO {nombre}")
                        continue
                    print(
                        f"[{actual}/{total}] {variante} | {rejilla}m | "
                        f"semilla {semilla} | {pliegue}"
                    )
                    filas, artefacto = entrenar_combinacion(
                        rejilla, variante, semilla, pliegue, cfg
                    )
                    joblib.dump(artefacto, ruta_modelo)
                    historial_nuevo.extend(filas)
                    gc.collect()

    if historial_nuevo:
        nuevo = pd.DataFrame(historial_nuevo)
        if RUTA_HISTORIAL.exists():
            anterior = pd.read_csv(RUTA_HISTORIAL)
            claves = [
                "pliegue", "rejilla_minutos", "variante", "semilla", "margen_ev"
            ]
            combinado = pd.concat([anterior, nuevo], ignore_index=True)
            combinado = combinado.drop_duplicates(claves, keep="last")
        else:
            combinado = nuevo
        combinado.to_csv(RUTA_HISTORIAL, index=False)
    print(f"Historial: {RUTA_HISTORIAL}")
    print("ENTRENAMIENTO BAJA V2 COMPLETADO")


if __name__ == "__main__":
    main()
