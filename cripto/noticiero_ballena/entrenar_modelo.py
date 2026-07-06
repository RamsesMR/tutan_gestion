from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from .configuracion import (
    ANIOS_BLOQUEADOS,
    ANIOS_DESARROLLO,
    ARCHIVO_ETIQUETADO,
    ARCHIVO_MANIFIESTO_MODELO,
    ARCHIVO_METRICAS,
    ARCHIVO_MODELO,
    ARCHIVO_OOF,
    CLASES_IMPACTO,
    DIAS_BLOQUE_VALIDACION_OOF,
    HORIZONTES_IMPACTO_MINUTOS,
    MIN_DIAS_ENTRENAMIENTO_OOF,
    PARAMETROS_CLASIFICADOR,
    PARAMETROS_REGRESOR,
    RATIO_INACTIVOS_ENTRENAMIENTO,
    SEMILLA,
    VERSION_MODELO,
)
from .utilidades import (
    fecha_utc,
    guardar_json,
    guardar_tabla,
    leer_tabla,
    probabilidades_por_clase,
    seleccionar_inactivos,
)


@dataclass(frozen=True)
class PliegueTemporal:
    nombre: str
    inicio_entrenamiento: pd.Timestamp
    fin_entrenamiento: pd.Timestamp
    inicio_validacion: pd.Timestamp
    fin_validacion: pd.Timestamp


def columnas_modelo(df: pd.DataFrame) -> list[str]:
    columnas = [
        c
        for c in df.columns
        if (c.startswith("nb_raw_") or c.startswith("nb_ctx_"))
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not columnas:
        raise ValueError("No se encontraron variables numéricas nb_raw_/nb_ctx_.")
    return sorted(columnas)


def crear_clasificador() -> Pipeline:
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
        ("modelo", HistGradientBoostingClassifier(**PARAMETROS_CLASIFICADOR)),
    ])


def crear_regresor() -> Pipeline:
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
        ("modelo", HistGradientBoostingRegressor(**PARAMETROS_REGRESOR)),
    ])


def pliegues_rolling(
    fechas: pd.Series,
    min_dias_entrenamiento: int,
    dias_validacion: int,
) -> Iterator[PliegueTemporal]:
    minimo = fechas.min().floor("D")
    maximo = fechas.max().ceil("D")
    inicio_validacion = minimo + pd.Timedelta(days=min_dias_entrenamiento)
    numero = 1
    while inicio_validacion < maximo:
        fin_validacion = min(inicio_validacion + pd.Timedelta(days=dias_validacion), maximo)
        yield PliegueTemporal(
            nombre=f"oof_{numero:02d}_{inicio_validacion:%Y%m%d}_{fin_validacion:%Y%m%d}",
            inicio_entrenamiento=minimo,
            fin_entrenamiento=inicio_validacion,
            inicio_validacion=inicio_validacion,
            fin_validacion=fin_validacion,
        )
        inicio_validacion = fin_validacion
        numero += 1


def _preparar_entrenamiento(df: pd.DataFrame, horizonte: int) -> pd.DataFrame:
    objetivo_clase = f"nb_clase_impacto_{horizonte}m"
    objetivo_retorno = f"nb_retorno_futuro_{horizonte}m"
    salida = df.dropna(subset=[objetivo_clase, objetivo_retorno]).copy()
    salida = salida.loc[~salida[objetivo_clase].eq("AMBIGUO")]
    salida = salida.loc[salida[objetivo_clase].isin(CLASES_IMPACTO)]
    return salida


def _metricas_clasificacion(y_true: pd.Series, pred: np.ndarray, proba: dict[str, np.ndarray]) -> dict:
    matriz_proba = np.column_stack([proba[c] for c in CLASES_IMPACTO])
    informe = classification_report(
        y_true,
        pred,
        labels=list(CLASES_IMPACTO),
        output_dict=True,
        zero_division=0,
    )
    precision_bajista = float(precision_score(y_true, pred, labels=["BAJISTA"], average="macro", zero_division=0))
    recall_bajista = float(recall_score(y_true, pred, labels=["BAJISTA"], average="macro", zero_division=0))
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_bajista": precision_bajista,
        "porcentaje_acierto_bajista": precision_bajista * 100.0,
        "recall_bajista": recall_bajista,
        "log_loss": float(log_loss(y_true, matriz_proba, labels=list(CLASES_IMPACTO))),
        "matriz_confusion": confusion_matrix(y_true, pred, labels=list(CLASES_IMPACTO)).tolist(),
        "classification_report": informe,
    }


def entrenar_y_generar_oof(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict]:
    datos = df.copy()
    datos["fecha_disponible"] = fecha_utc(datos["fecha_disponible"])
    datos = datos.loc[datos["fecha_disponible"].dt.year.isin(ANIOS_DESARROLLO)].copy()
    datos = datos.loc[~datos["fecha_disponible"].dt.year.isin(ANIOS_BLOQUEADOS)].copy()
    datos = datos.sort_values("fecha_disponible").reset_index(drop=True)
    datos["_fila_id"] = np.arange(len(datos), dtype="int64")
    columnas = columnas_modelo(datos)

    oof = datos[["_fila_id", "fecha_disponible", *columnas]].copy()
    metricas: dict[str, dict] = {}
    artefactos_horizonte: dict[str, dict] = {}

    for horizonte in HORIZONTES_IMPACTO_MINUTOS:
        objetivo_clase = f"nb_clase_impacto_{horizonte}m"
        objetivo_retorno = f"nb_retorno_futuro_{horizonte}m"
        disponibles = _preparar_entrenamiento(datos, horizonte)
        metricas_horizonte: dict[str, dict] = {}

        for pliegue in pliegues_rolling(
            datos["fecha_disponible"],
            MIN_DIAS_ENTRENAMIENTO_OOF,
            DIAS_BLOQUE_VALIDACION_OOF,
        ):
            train_total = disponibles.loc[
                (disponibles["fecha_disponible"] >= pliegue.inicio_entrenamiento)
                & (disponibles["fecha_disponible"] < pliegue.fin_entrenamiento)
            ]
            valid = disponibles.loc[
                (disponibles["fecha_disponible"] >= pliegue.inicio_validacion)
                & (disponibles["fecha_disponible"] < pliegue.fin_validacion)
            ]
            if len(train_total) < 500 or len(valid) < 50 or train_total[objetivo_clase].nunique() < 2:
                continue

            activos_train = train_total["nb_raw_activo_240m"].astype(bool)
            train = seleccionar_inactivos(
                train_total,
                activos_train,
                RATIO_INACTIVOS_ENTRENAMIENTO,
                SEMILLA,
            )
            clf = crear_clasificador()
            reg = crear_regresor()
            clf.fit(train[columnas], train[objetivo_clase])
            reg.fit(train[columnas], train[objetivo_retorno])

            pred = clf.predict(valid[columnas])
            proba = probabilidades_por_clase(clf, valid[columnas], CLASES_IMPACTO)
            pred_retorno = reg.predict(valid[columnas])
            m = _metricas_clasificacion(valid[objetivo_clase], pred, proba)
            m.update({
                "filas_entrenamiento_total": int(len(train_total)),
                "filas_entrenamiento_usadas": int(len(train)),
                "filas_validacion": int(len(valid)),
                "mae_retorno": float(mean_absolute_error(valid[objetivo_retorno], pred_retorno)),
                "inicio_validacion": str(pliegue.inicio_validacion),
                "fin_validacion": str(pliegue.fin_validacion),
            })
            metricas_horizonte[pliegue.nombre] = m

            ids = valid["_fila_id"].to_numpy()
            for clase in CLASES_IMPACTO:
                columna = f"nb_p_{clase.lower()}_{horizonte}m"
                oof.loc[oof["_fila_id"].isin(ids), columna] = proba[clase]
            oof.loc[oof["_fila_id"].isin(ids), f"nb_retorno_estimado_{horizonte}m"] = pred_retorno
            oof.loc[oof["_fila_id"].isin(ids), f"nb_pliegue_{horizonte}m"] = pliegue.nombre

        final = disponibles.copy()
        activos_final = final["nb_raw_activo_240m"].astype(bool)
        final_entrenamiento = seleccionar_inactivos(
            final,
            activos_final,
            RATIO_INACTIVOS_ENTRENAMIENTO,
            SEMILLA,
        )
        if len(final_entrenamiento) < 500 or final_entrenamiento[objetivo_clase].nunique() < 2:
            raise ValueError(f"Datos insuficientes para entrenar el modelo final de {horizonte}m")
        clf_final = crear_clasificador()
        reg_final = crear_regresor()
        clf_final.fit(final_entrenamiento[columnas], final_entrenamiento[objetivo_clase])
        reg_final.fit(final_entrenamiento[columnas], final_entrenamiento[objetivo_retorno])
        artefactos_horizonte[str(horizonte)] = {
            "clasificador": clf_final,
            "regresor": reg_final,
            "clases": [str(x) for x in clf_final.named_steps["modelo"].classes_],
        }
        metricas[str(horizonte)] = metricas_horizonte

    oof = oof.drop(columns=["_fila_id"])
    for horizonte in HORIZONTES_IMPACTO_MINUTOS:
        probabilidades = [oof.get(f"nb_p_{c.lower()}_{horizonte}m") for c in CLASES_IMPACTO]
        if any(p is None for p in probabilidades):
            continue
        matriz = pd.concat(probabilidades, axis=1)
        oof[f"nb_confianza_{horizonte}m"] = matriz.max(axis=1) * oof["nb_raw_calidad_datos"].fillna(0.0).clip(0, 1)

    artefacto = {
        "version": VERSION_MODELO,
        "columnas_modelo": columnas,
        "horizontes": artefactos_horizonte,
        "anios_desarrollo": ANIOS_DESARROLLO,
        "anios_bloqueados": ANIOS_BLOQUEADOS,
        "entrenado_hasta": str(datos["fecha_disponible"].max()),
        "objetivo": "Interpretar impacto de actividad de ballenas; no decidir operaciones.",
    }
    return artefacto, oof, metricas


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena Noticiero Ballena V1 y genera predicciones OOF.")
    parser.add_argument("--entrada", type=Path, default=ARCHIVO_ETIQUETADO)
    parser.add_argument("--modelo", type=Path, default=ARCHIVO_MODELO)
    parser.add_argument("--oof", type=Path, default=ARCHIVO_OOF)
    parser.add_argument("--metricas", type=Path, default=ARCHIVO_METRICAS)
    parser.add_argument("--manifiesto", type=Path, default=ARCHIVO_MANIFIESTO_MODELO)
    args = parser.parse_args()

    artefacto, oof, metricas = entrenar_y_generar_oof(leer_tabla(args.entrada))
    args.modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefacto, args.modelo)
    guardar_tabla(oof, args.oof)
    guardar_json(metricas, args.metricas)
    guardar_json({k: v for k, v in artefacto.items() if k != "horizontes"}, args.manifiesto)
    print(f"Modelo: {args.modelo}")
    print(f"Predicciones OOF: {args.oof}")
    print(f"Filas OOF con señal 15m: {int(oof.get('nb_p_bajista_15m', pd.Series(dtype=float)).notna().sum()):,}")
    for horizonte, pliegues in metricas.items():
        print(f"{horizonte}m -> pliegues evaluados: {len(pliegues)}")


if __name__ == "__main__":
    main()
