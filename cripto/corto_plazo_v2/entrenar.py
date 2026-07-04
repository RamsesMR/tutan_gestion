from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

from cripto.corto_plazo_v2.configuracion import (
    CLASES,
    COLUMNAS_CRUZADAS,
    COLUMNAS_MODELO_V1,
    COLUMNAS_MODELO_V2A,
    HORIZONTE_MINUTOS,
    NOMBRE_HORIZONTE,
    RUTA_MANIFIESTO_V2,
    RUTA_MODELOS_V2,
    SIMBOLOS,
    UMBRAL_CLASE,
    VERSION_MODELO,
)


CLASES_NP = np.array(
    CLASES,
    dtype=object,
)


def normalizar_etiqueta(
    etiqueta: str,
) -> str:
    """Convierte una etiqueta en un nombre seguro para carpetas."""

    etiqueta_limpia = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        etiqueta.strip(),
    )

    etiqueta_limpia = etiqueta_limpia.strip(
        "_"
    )

    if not etiqueta_limpia:
        raise ValueError(
            "La etiqueta no contiene caracteres válidos."
        )

    return etiqueta_limpia.lower()


def leer_columnas_parquet(
    ruta: Path,
) -> set[str]:
    """Lee únicamente el esquema de columnas de un Parquet."""

    archivo = pq.ParquetFile(
        ruta
    )

    return set(
        str(nombre)
        for nombre in archivo.schema_arrow.names
    )


def cargar_manifiesto(
    simbolo: str,
) -> pd.DataFrame:
    """Carga el manifiesto temporal propio de la V2A."""

    if not RUTA_MANIFIESTO_V2.exists():
        raise FileNotFoundError(
            "No existe el manifiesto V2A. Ejecuta primero: "
            "python -m cripto.corto_plazo_v2.preparar_divisiones"
        )

    manifiesto = pd.read_csv(
        RUTA_MANIFIESTO_V2
    )

    columnas_requeridas = {
        "simbolo",
        "division",
        "archivo",
        "desde_archivo",
        "desde_division",
        "hasta_division",
        "filas_utilizables",
    }

    faltantes = columnas_requeridas.difference(
        manifiesto.columns
    )

    if faltantes:
        raise ValueError(
            "Faltan columnas en el manifiesto V2A: "
            + ", ".join(
                sorted(faltantes)
            )
        )

    manifiesto = (
        manifiesto.loc[
            manifiesto["simbolo"] == simbolo
        ]
        .copy()
    )

    if manifiesto.empty:
        raise ValueError(
            f"No existen registros V2A para {simbolo}."
        )

    if "prueba" in set(
        manifiesto["division"]
        .astype(str)
        .tolist()
    ):
        raise ValueError(
            "El manifiesto V2A contiene una división de prueba. "
            "La V2A no debe utilizar 2026 durante su desarrollo."
        )

    return manifiesto


def obtener_registros_division(
    manifiesto: pd.DataFrame,
    division: str,
) -> list[dict[str, Any]]:
    """Obtiene los registros ordenados de una división."""

    registros = (
        manifiesto.loc[
            manifiesto["division"] == division
        ]
        .sort_values("desde_archivo")
    )

    if registros.empty:
        raise ValueError(
            f"No existen registros para la división {division}."
        )

    resultado: list[
        dict[str, Any]
    ] = []

    for _, fila in registros.iterrows():
        ruta = Path(
            str(fila["archivo"])
        )

        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe el archivo V2A: {ruta}"
            )

        columnas = leer_columnas_parquet(
            ruta
        )

        columnas_requeridas = {
            *COLUMNAS_MODELO_V2A,
            "fecha_apertura",
            "fecha_objetivo",
            "rendimiento_objetivo",
        }

        faltantes = columnas_requeridas.difference(
            columnas
        )

        if faltantes:
            raise ValueError(
                f"{ruta.name}: faltan columnas requeridas: "
                + ", ".join(
                    sorted(faltantes)
                )
            )

        resultado.append(
            {
                "ruta": ruta,
                "division": division,
                "desde_division": pd.Timestamp(
                    fila["desde_division"]
                ),
                "hasta_division": pd.Timestamp(
                    fila["hasta_division"]
                ),
                "filas_utilizables": int(
                    fila["filas_utilizables"]
                ),
            }
        )

    return resultado


def clasificar_objetivo(
    rendimientos: pd.Series,
) -> np.ndarray:
    """Convierte el rendimiento futuro en tres clases."""

    valores = pd.to_numeric(
        rendimientos,
        errors="coerce",
    ).to_numpy(
        dtype="float64"
    )

    if not np.isfinite(
        valores
    ).all():
        raise ValueError(
            "El objetivo contiene valores nulos o infinitos."
        )

    return np.select(
        [
            valores <= -UMBRAL_CLASE,
            valores >= UMBRAL_CLASE,
        ],
        [
            "BAJA",
            "SUBE",
        ],
        default="NEUTRAL",
    )


def cargar_datos(
    registro: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga únicamente las filas utilizables de un archivo V2A.

    La purga se vuelve a aplicar durante cada lectura. Así el
    entrenamiento no depende solamente de que el manifiesto sea correcto.
    """

    ruta = Path(
        registro["ruta"]
    )

    columnas = [
        *COLUMNAS_MODELO_V2A,
        "fecha_apertura",
        "fecha_objetivo",
        "rendimiento_objetivo",
    ]

    datos = pd.read_parquet(
        ruta,
        columns=columnas,
    )

    datos["fecha_apertura"] = pd.to_datetime(
        datos["fecha_apertura"],
        utc=True,
        errors="coerce",
    )

    datos["fecha_objetivo"] = pd.to_datetime(
        datos["fecha_objetivo"],
        utc=True,
        errors="coerce",
    )

    if datos[
        [
            "fecha_apertura",
            "fecha_objetivo",
        ]
    ].isna().any().any():
        raise ValueError(
            f"{ruta.name}: contiene fechas inválidas."
        )

    horizonte = (
        datos["fecha_objetivo"]
        - datos["fecha_apertura"]
    )

    horizonte_esperado = pd.Timedelta(
        minutes=HORIZONTE_MINUTOS
    )

    if horizonte.ne(
        horizonte_esperado
    ).any():
        raise ValueError(
            f"{ruta.name}: contiene horizontes distintos de "
            f"{HORIZONTE_MINUTOS} minutos."
        )

    desde_division = pd.Timestamp(
        registro["desde_division"]
    )

    hasta_division = pd.Timestamp(
        registro["hasta_division"]
    )

    mascara = (
        (datos["fecha_apertura"] >= desde_division)
        & (datos["fecha_apertura"] < hasta_division)
        & (datos["fecha_objetivo"] > datos["fecha_apertura"])
        & (datos["fecha_objetivo"] < hasta_division)
    )

    datos = (
        datos.loc[
            mascara
        ]
        .reset_index(drop=True)
    )

    esperado = int(
        registro["filas_utilizables"]
    )

    if len(datos) != esperado:
        raise ValueError(
            f"{ruta.name}: el manifiesto indica {esperado} filas "
            f"utilizables, pero el entrenamiento encontró {len(datos)}."
        )

    variables = (
        datos[
            list(COLUMNAS_MODELO_V2A)
        ]
        .to_numpy(
            dtype="float32"
        )
    )

    if not np.isfinite(
        variables
    ).all():
        raise ValueError(
            f"{ruta.name}: contiene variables nulas o infinitas."
        )

    clases = clasificar_objetivo(
        datos["rendimiento_objetivo"]
    )

    del datos
    gc.collect()

    return variables, clases


def recorrer_lotes(
    total: int,
    tamano_lote: int,
):
    """Genera los límites de cada lote."""

    for inicio in range(
        0,
        total,
        tamano_lote,
    ):
        final = min(
            inicio + tamano_lote,
            total,
        )

        yield inicio, final


def ajustar_escalador(
    registros: list[dict[str, Any]],
    tamano_lote: int,
) -> tuple[StandardScaler, Counter]:
    """Ajusta el escalador usando únicamente entrenamiento."""

    escalador = StandardScaler()
    conteos: Counter = Counter()

    print("\nAJUSTANDO ESCALADOR V2A")
    print("=" * 70)

    for registro in registros:
        ruta = Path(
            registro["ruta"]
        )

        print(
            f"Procesando: {ruta.name}"
        )

        variables, clases = cargar_datos(
            registro=registro
        )

        conteos.update(
            clases.tolist()
        )

        for inicio, final in recorrer_lotes(
            total=len(variables),
            tamano_lote=tamano_lote,
        ):
            escalador.partial_fit(
                variables[inicio:final]
            )

        del variables
        del clases
        gc.collect()

    return escalador, conteos


def calcular_pesos_clases(
    conteos: Counter,
) -> dict[str, float]:
    """Calcula pesos inversamente proporcionales."""

    total = sum(
        conteos[clase]
        for clase in CLASES
    )

    pesos: dict[
        str,
        float,
    ] = {}

    for clase in CLASES:
        cantidad = conteos[
            clase
        ]

        if cantidad == 0:
            raise ValueError(
                f"La clase {clase} no tiene muestras."
            )

        pesos[clase] = (
            total
            / (
                len(CLASES)
                * cantidad
            )
        )

    return pesos


def entrenar_modelo(
    registros: list[dict[str, Any]],
    escalador: StandardScaler,
    pesos_clases: dict[str, float],
    tamano_lote: int,
    epocas: int,
) -> SGDClassifier:
    """Entrena el mismo SGDClassifier de la V1 con 63 variables."""

    modelo = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        learning_rate="optimal",
        class_weight=pesos_clases,
        average=True,
        random_state=42,
    )

    primera_actualizacion = True

    print("\nENTRENAMIENTO DEL MODELO BASE V2A")
    print("=" * 70)

    for epoca in range(
        1,
        epocas + 1,
    ):
        print(
            f"\nÉpoca {epoca} de {epocas}"
        )

        for indice_archivo, registro in enumerate(
            registros
        ):
            ruta = Path(
                registro["ruta"]
            )

            print(
                f"Entrenando con: {ruta.name}"
            )

            variables, clases = cargar_datos(
                registro=registro
            )

            generador = np.random.default_rng(
                42
                + epoca * 100
                + indice_archivo
            )

            indices = generador.permutation(
                len(variables)
            )

            for inicio, final in recorrer_lotes(
                total=len(indices),
                tamano_lote=tamano_lote,
            ):
                indices_lote = indices[
                    inicio:final
                ]

                variables_lote = escalador.transform(
                    variables[
                        indices_lote
                    ]
                )

                clases_lote = clases[
                    indices_lote
                ]

                if primera_actualizacion:
                    modelo.partial_fit(
                        variables_lote,
                        clases_lote,
                        classes=CLASES_NP,
                    )

                    primera_actualizacion = False

                else:
                    modelo.partial_fit(
                        variables_lote,
                        clases_lote,
                    )

            del variables
            del clases
            del indices
            gc.collect()

    return modelo


def predecir_division(
    registros: list[dict[str, Any]],
    modelo: SGDClassifier,
    escalador: StandardScaler,
    tamano_lote: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predice una división sin cargar todos sus años simultáneamente."""

    reales: list[
        np.ndarray
    ] = []

    predicciones: list[
        np.ndarray
    ] = []

    for registro in registros:
        ruta = Path(
            registro["ruta"]
        )

        print(
            f"Evaluando: {ruta.name}"
        )

        variables, clases = cargar_datos(
            registro=registro
        )

        for inicio, final in recorrer_lotes(
            total=len(variables),
            tamano_lote=tamano_lote,
        ):
            variables_lote = escalador.transform(
                variables[
                    inicio:final
                ]
            )

            prediccion_lote = modelo.predict(
                variables_lote
            )

            reales.append(
                clases[
                    inicio:final
                ]
            )

            predicciones.append(
                prediccion_lote
            )

        del variables
        del clases
        gc.collect()

    if not reales:
        raise ValueError(
            "La división no produjo muestras para evaluar."
        )

    return (
        np.concatenate(
            reales
        ),
        np.concatenate(
            predicciones
        ),
    )


def calcular_metricas(
    reales: np.ndarray,
    predicciones: np.ndarray,
    nombre_modelo: str,
) -> dict[str, Any]:
    """Calcula las métricas principales."""

    return {
        "modelo": nombre_modelo,
        "muestras": len(reales),
        "accuracy": accuracy_score(
            reales,
            predicciones,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            reales,
            predicciones,
        ),
        "f1_macro": f1_score(
            reales,
            predicciones,
            labels=CLASES_NP,
            average="macro",
            zero_division=0,
        ),
    }


def crear_matriz(
    reales: np.ndarray,
    predicciones: np.ndarray,
) -> pd.DataFrame:
    """Devuelve una matriz de confusión etiquetada."""

    matriz = confusion_matrix(
        reales,
        predicciones,
        labels=CLASES_NP,
    )

    return pd.DataFrame(
        matriz,
        index=[
            f"real_{clase}"
            for clase in CLASES
        ],
        columns=[
            f"predicha_{clase}"
            for clase in CLASES
        ],
    )


def guardar_json(
    contenido: dict[str, Any],
    ruta: Path,
) -> None:
    """Guarda un diccionario en JSON."""

    with ruta.open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            contenido,
            archivo,
            ensure_ascii=False,
            indent=4,
        )


def guardar_resultados(
    simbolo: str,
    etiqueta: str,
    modelo: SGDClassifier,
    escalador: StandardScaler,
    conteos: Counter,
    pesos_clases: dict[str, float],
    reales: np.ndarray,
    predicciones: np.ndarray,
    epocas: int,
    tamano_lote: int,
    notas: str,
) -> None:
    """Guarda el último modelo y una copia histórica del experimento."""

    RUTA_MODELOS_V2.mkdir(
        parents=True,
        exist_ok=True,
    )

    fecha_utc = datetime.now(
        timezone.utc
    )

    marca_tiempo = fecha_utc.strftime(
        "%Y%m%dT%H%M%SZ"
    )

    etiqueta_segura = normalizar_etiqueta(
        etiqueta
    )

    identificador = (
        f"{marca_tiempo}_{etiqueta_segura}"
    )

    ruta_experimento = (
        RUTA_MODELOS_V2
        / "experimentos"
        / simbolo
        / identificador
    )

    ruta_experimento.mkdir(
        parents=True,
        exist_ok=False,
    )

    clase_mayoritaria = max(
        CLASES,
        key=lambda clase: conteos[
            clase
        ],
    )

    prediccion_mayoritaria = np.full(
        shape=len(reales),
        fill_value=clase_mayoritaria,
        dtype=object,
    )

    metricas_modelo = calcular_metricas(
        reales=reales,
        predicciones=predicciones,
        nombre_modelo="SGDClassifier V2A",
    )

    metricas_referencia = calcular_metricas(
        reales=reales,
        predicciones=prediccion_mayoritaria,
        nombre_modelo="DummyClassifier (most_frequent)",
    )

    informe = classification_report(
        reales,
        predicciones,
        labels=CLASES_NP,
        output_dict=True,
        zero_division=0,
    )

    paquete_modelo = {
        "version_modelo": VERSION_MODELO,
        "simbolo": simbolo,
        "horizonte": NOMBRE_HORIZONTE,
        "modelo": modelo,
        "escalador": escalador,
        "columnas_modelo": list(
            COLUMNAS_MODELO_V2A
        ),
        "columnas_v1": list(
            COLUMNAS_MODELO_V1
        ),
        "columnas_cruzadas": list(
            COLUMNAS_CRUZADAS
        ),
        "clases": list(
            CLASES
        ),
        "umbral_clase": UMBRAL_CLASE,
        "conteos_entrenamiento": dict(
            conteos
        ),
        "pesos_clases": pesos_clases,
        "periodo_entrenamiento": "2021-01-01/2025-01-01",
        "periodo_validacion": "2025-01-01/2026-01-01",
        "division_prueba_utilizada": False,
        "fecha_entrenamiento_utc": fecha_utc.isoformat(),
        "etiqueta": etiqueta,
    }

    rutas_experimento = {
        "modelo": (
            ruta_experimento
            / f"modelo_{simbolo}_{NOMBRE_HORIZONTE}.joblib"
        ),
        "metricas": (
            ruta_experimento
            / f"metricas_{simbolo}_{NOMBRE_HORIZONTE}.csv"
        ),
        "matriz": (
            ruta_experimento
            / f"matriz_confusion_{simbolo}_{NOMBRE_HORIZONTE}.csv"
        ),
        "matriz_referencia": (
            ruta_experimento
            / f"matriz_confusion_referencia_{simbolo}_{NOMBRE_HORIZONTE}.csv"
        ),
        "informe": (
            ruta_experimento
            / f"informe_clasificacion_{simbolo}_{NOMBRE_HORIZONTE}.json"
        ),
        "detalle": (
            ruta_experimento
            / "detalle_entrenamiento.json"
        ),
    }

    joblib.dump(
        paquete_modelo,
        rutas_experimento["modelo"],
    )

    pd.DataFrame(
        [
            metricas_modelo,
            metricas_referencia,
        ]
    ).to_csv(
        rutas_experimento["metricas"],
        index=False,
        encoding="utf-8-sig",
    )

    crear_matriz(
        reales=reales,
        predicciones=predicciones,
    ).to_csv(
        rutas_experimento["matriz"],
        encoding="utf-8-sig",
    )

    crear_matriz(
        reales=reales,
        predicciones=prediccion_mayoritaria,
    ).to_csv(
        rutas_experimento["matriz_referencia"],
        encoding="utf-8-sig",
    )

    guardar_json(
        contenido=informe,
        ruta=rutas_experimento["informe"],
    )

    detalle = {
        "fecha_utc": fecha_utc.isoformat(),
        "version_modelo": VERSION_MODELO,
        "simbolo": simbolo,
        "horizonte": NOMBRE_HORIZONTE,
        "etiqueta": etiqueta,
        "epocas": epocas,
        "tamano_lote": tamano_lote,
        "variables_v1": len(
            COLUMNAS_MODELO_V1
        ),
        "variables_cruzadas": len(
            COLUMNAS_CRUZADAS
        ),
        "variables_totales": len(
            COLUMNAS_MODELO_V2A
        ),
        "umbral_clase": UMBRAL_CLASE,
        "conteos_entrenamiento": dict(
            conteos
        ),
        "pesos_clases": pesos_clases,
        "metricas_modelo": metricas_modelo,
        "metricas_referencia": metricas_referencia,
        "notas": notas,
        "division_prueba_utilizada": False,
        "ruta_experimento": str(
            ruta_experimento
        ),
    }

    guardar_json(
        contenido=detalle,
        ruta=rutas_experimento["detalle"],
    )

    rutas_ultimo = {
        "modelo": (
            RUTA_MODELOS_V2
            / f"modelo_base_{simbolo}_{NOMBRE_HORIZONTE}.joblib"
        ),
        "metricas": (
            RUTA_MODELOS_V2
            / f"metricas_base_{simbolo}_{NOMBRE_HORIZONTE}.csv"
        ),
        "matriz": (
            RUTA_MODELOS_V2
            / f"matriz_confusion_base_{simbolo}_{NOMBRE_HORIZONTE}.csv"
        ),
        "matriz_referencia": (
            RUTA_MODELOS_V2
            / f"matriz_confusion_referencia_{simbolo}_{NOMBRE_HORIZONTE}.csv"
        ),
        "informe": (
            RUTA_MODELOS_V2
            / f"informe_clasificacion_base_{simbolo}_{NOMBRE_HORIZONTE}.json"
        ),
        "detalle": (
            RUTA_MODELOS_V2
            / f"ultimo_entrenamiento_{simbolo}.json"
        ),
    }

    for clave in (
        "modelo",
        "metricas",
        "matriz",
        "matriz_referencia",
        "informe",
    ):
        shutil.copy2(
            rutas_experimento[
                clave
            ],
            rutas_ultimo[
                clave
            ],
        )

    guardar_json(
        contenido=detalle,
        ruta=rutas_ultimo["detalle"],
    )

    ruta_historial = (
        RUTA_MODELOS_V2
        / "historial_entrenamientos.csv"
    )

    registro_historial = {
        "fecha_utc": fecha_utc.isoformat(),
        "version_modelo": VERSION_MODELO,
        "simbolo": simbolo,
        "horizonte": NOMBRE_HORIZONTE,
        "etiqueta": etiqueta,
        "epocas": epocas,
        "tamano_lote": tamano_lote,
        "variables_v1": len(
            COLUMNAS_MODELO_V1
        ),
        "variables_cruzadas": len(
            COLUMNAS_CRUZADAS
        ),
        "variables_totales": len(
            COLUMNAS_MODELO_V2A
        ),
        "accuracy": metricas_modelo[
            "accuracy"
        ],
        "balanced_accuracy": metricas_modelo[
            "balanced_accuracy"
        ],
        "f1_macro": metricas_modelo[
            "f1_macro"
        ],
        "accuracy_referencia": metricas_referencia[
            "accuracy"
        ],
        "division_prueba_utilizada": False,
        "ruta_experimento": str(
            ruta_experimento
        ),
        "notas": notas,
    }

    pd.DataFrame(
        [
            registro_historial
        ]
    ).to_csv(
        ruta_historial,
        mode="a",
        header=not ruta_historial.exists(),
        index=False,
        encoding="utf-8-sig",
    )

    print("\nRESULTADOS DE VALIDACIÓN V2A")
    print("=" * 70)

    for metricas in (
        metricas_modelo,
        metricas_referencia,
    ):
        print(
            f"\n{metricas['modelo']}"
        )
        print(
            f"Accuracy: {metricas['accuracy']:.4f}"
        )
        print(
            "Balanced Accuracy: "
            f"{metricas['balanced_accuracy']:.4f}"
        )
        print(
            f"F1-score macro: {metricas['f1_macro']:.4f}"
        )

    print("\nArchivos del último modelo:")
    for ruta in rutas_ultimo.values():
        print(
            f"- {ruta}"
        )

    print("\nExperimento histórico:")
    print(
        f"- {ruta_experimento}"
    )


def entrenar(
    simbolo: str,
    tamano_lote: int,
    epocas: int,
    etiqueta: str,
    notas: str,
) -> None:
    """Ejecuta el entrenamiento completo de la V2A."""

    manifiesto = cargar_manifiesto(
        simbolo=simbolo
    )

    registros_entrenamiento = obtener_registros_division(
        manifiesto=manifiesto,
        division="entrenamiento",
    )

    registros_validacion = obtener_registros_division(
        manifiesto=manifiesto,
        division="validacion",
    )

    print("\nMODELO BASE DE CORTO PLAZO V2A")
    print("=" * 70)
    print(
        f"Símbolo: {simbolo}"
    )
    print(
        f"Variables V1: {len(COLUMNAS_MODELO_V1)}"
    )
    print(
        f"Variables cruzadas: {len(COLUMNAS_CRUZADAS)}"
    )
    print(
        f"Variables totales: {len(COLUMNAS_MODELO_V2A)}"
    )
    print(
        f"Umbral: ±{UMBRAL_CLASE * 100:.2f} %"
    )
    print(
        f"Épocas: {epocas}"
    )
    print(
        "Tamaño de lote: "
        f"{tamano_lote:,}"
        .replace(",", ".")
    )
    print(
        f"Etiqueta: {etiqueta}"
    )

    if notas:
        print(
            f"Notas: {notas}"
        )

    print("\nVariables cruzadas:")

    for columna in COLUMNAS_CRUZADAS:
        print(
            f"- {columna}"
        )

    escalador, conteos = ajustar_escalador(
        registros=registros_entrenamiento,
        tamano_lote=tamano_lote,
    )

    pesos_clases = calcular_pesos_clases(
        conteos=conteos
    )

    print("\nDistribución de entrenamiento:")

    for clase in CLASES:
        print(
            f"{clase}: "
            f"{conteos[clase]:,}"
            .replace(",", ".")
        )

    print("\nPesos de clases:")

    for clase, peso in pesos_clases.items():
        print(
            f"{clase}: {peso:.6f}"
        )

    modelo = entrenar_modelo(
        registros=registros_entrenamiento,
        escalador=escalador,
        pesos_clases=pesos_clases,
        tamano_lote=tamano_lote,
        epocas=epocas,
    )

    print("\nVALIDACIÓN TEMPORAL V2A")
    print("=" * 70)

    reales, predicciones = predecir_division(
        registros=registros_validacion,
        modelo=modelo,
        escalador=escalador,
        tamano_lote=tamano_lote,
    )

    guardar_resultados(
        simbolo=simbolo,
        etiqueta=etiqueta,
        modelo=modelo,
        escalador=escalador,
        conteos=conteos,
        pesos_clases=pesos_clases,
        reales=reales,
        predicciones=predicciones,
        epocas=epocas,
        tamano_lote=tamano_lote,
        notas=notas,
    )

    print(
        "\nLa V2A utilizó entrenamiento 2021-2024 "
        "y validación 2025."
    )
    print(
        "Enero-mayo de 2026 no fue cargado ni evaluado."
    )


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene los argumentos del comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Entrena la V2A con las 36 variables originales "
            "y las 27 variables cruzadas BTC-ETH."
        )
    )

    parser.add_argument(
        "--simbolo",
        required=True,
        choices=SIMBOLOS,
    )

    parser.add_argument(
        "--epocas",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--tamano-lote",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--etiqueta",
        default="v2a_variables_cruzadas",
    )

    parser.add_argument(
        "--notas",
        default="",
    )

    argumentos = parser.parse_args()

    if argumentos.epocas <= 0:
        parser.error(
            "--epocas debe ser mayor que cero."
        )

    if argumentos.tamano_lote <= 0:
        parser.error(
            "--tamano-lote debe ser mayor que cero."
        )

    if not argumentos.etiqueta.strip():
        parser.error(
            "--etiqueta no puede estar vacía."
        )

    return argumentos


def main() -> None:
    """Punto de entrada."""

    argumentos = obtener_argumentos()

    try:
        entrenar(
            simbolo=argumentos.simbolo,
            tamano_lote=argumentos.tamano_lote,
            epocas=argumentos.epocas,
            etiqueta=argumentos.etiqueta.strip(),
            notas=argumentos.notas.strip(),
        )

    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(
            "\nNo se pudo entrenar el modelo V2A."
        )
        print(
            f"Detalle: {error}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
