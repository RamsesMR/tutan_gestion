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
    COLUMNAS_MODELO_V1,
    HORIZONTE_MINUTOS,
    NOMBRE_HORIZONTE,
    RUTA_MANIFIESTO_V2,
    RUTA_MODELOS_V2,
    SIMBOLOS,
    UMBRAL_CLASE,
)


NOMBRE_CONTROL = "control_v1_mismas_filas"
CLASES_NP = np.array(
    CLASES,
    dtype=object,
)

RUTA_CONTROL = (
    RUTA_MODELOS_V2
    / NOMBRE_CONTROL
)


def normalizar_etiqueta(
    etiqueta: str,
) -> str:
    """Convierte una etiqueta en un nombre seguro."""

    resultado = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        etiqueta.strip(),
    ).strip("_")

    if not resultado:
        raise ValueError(
            "La etiqueta no contiene caracteres válidos."
        )

    return resultado.lower()


def leer_columnas_parquet(
    ruta: Path,
) -> set[str]:
    """Lee únicamente el esquema del Parquet."""

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
    """Carga el mismo manifiesto V2A usado por el modelo de 63 variables."""

    if not RUTA_MANIFIESTO_V2.exists():
        raise FileNotFoundError(
            "No existe el manifiesto V2A. Ejecuta primero: "
            "python -m cripto.corto_plazo_v2.preparar_divisiones"
        )

    manifiesto = pd.read_csv(
        RUTA_MANIFIESTO_V2
    )

    requeridas = {
        "simbolo",
        "division",
        "archivo",
        "desde_archivo",
        "desde_division",
        "hasta_division",
        "filas_utilizables",
    }

    faltantes = requeridas.difference(
        manifiesto.columns
    )

    if faltantes:
        raise ValueError(
            "Faltan columnas en el manifiesto V2A: "
            + ", ".join(sorted(faltantes))
        )

    manifiesto = (
        manifiesto.loc[
            manifiesto["simbolo"] == simbolo
        ]
        .copy()
    )

    if manifiesto.empty:
        raise ValueError(
            f"No existen registros para {simbolo}."
        )

    if "prueba" in set(
        manifiesto["division"]
        .astype(str)
        .tolist()
    ):
        raise ValueError(
            "El manifiesto V2A contiene una división de prueba."
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
            f"No existen registros para {division}."
        )

    resultado: list[dict[str, Any]] = []

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

        requeridas = {
            *COLUMNAS_MODELO_V1,
            "fecha_apertura",
            "fecha_objetivo",
            "rendimiento_objetivo",
        }

        faltantes = requeridas.difference(
            columnas
        )

        if faltantes:
            raise ValueError(
                f"{ruta.name}: faltan columnas: "
                + ", ".join(sorted(faltantes))
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
    """Convierte el rendimiento futuro en BAJA, NEUTRAL o SUBE."""

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
            "El objetivo contiene nulos o infinitos."
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
    Carga exactamente las mismas filas que la V2A, pero solo 36 variables.
    """

    ruta = Path(
        registro["ruta"]
    )

    columnas = [
        *COLUMNAS_MODELO_V1,
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

    if horizonte.ne(
        pd.Timedelta(
            minutes=HORIZONTE_MINUTOS
        )
    ).any():
        raise ValueError(
            f"{ruta.name}: contiene horizontes incorrectos."
        )

    desde = pd.Timestamp(
        registro["desde_division"]
    )

    hasta = pd.Timestamp(
        registro["hasta_division"]
    )

    mascara = (
        (datos["fecha_apertura"] >= desde)
        & (datos["fecha_apertura"] < hasta)
        & (datos["fecha_objetivo"] > datos["fecha_apertura"])
        & (datos["fecha_objetivo"] < hasta)
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
            f"{ruta.name}: se esperaban {esperado} filas "
            f"y se obtuvieron {len(datos)}."
        )

    variables = (
        datos[
            list(COLUMNAS_MODELO_V1)
        ]
        .to_numpy(
            dtype="float32"
        )
    )

    if not np.isfinite(
        variables
    ).all():
        raise ValueError(
            f"{ruta.name}: contiene variables no finitas."
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
    """Genera los límites de los lotes."""

    for inicio in range(
        0,
        total,
        tamano_lote,
    ):
        yield (
            inicio,
            min(
                inicio + tamano_lote,
                total,
            ),
        )


def ajustar_escalador(
    registros: list[dict[str, Any]],
    tamano_lote: int,
) -> tuple[StandardScaler, Counter]:
    """Ajusta el escalador solo con entrenamiento."""

    escalador = StandardScaler()
    conteos: Counter = Counter()

    print(
        "\nAJUSTANDO ESCALADOR DEL CONTROL"
    )
    print("=" * 70)

    for registro in registros:
        ruta = Path(
            registro["ruta"]
        )

        print(
            f"Procesando: {ruta.name}"
        )

        variables, clases = cargar_datos(
            registro
        )

        conteos.update(
            clases.tolist()
        )

        for inicio, final in recorrer_lotes(
            len(variables),
            tamano_lote,
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
    """Calcula los mismos pesos inversos usados por la V2A."""

    total = sum(
        conteos[clase]
        for clase in CLASES
    )

    pesos: dict[str, float] = {}

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
    """Entrena el mismo SGDClassifier con solo las 36 variables V1."""

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

    print(
        "\nENTRENAMIENTO DEL CONTROL V1 CON FILAS V2A"
    )
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
                registro
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
                len(indices),
                tamano_lote,
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
    """Predice validación con las mismas filas de la V2A."""

    reales: list[np.ndarray] = []
    predicciones: list[np.ndarray] = []

    for registro in registros:
        ruta = Path(
            registro["ruta"]
        )

        print(
            f"Evaluando: {ruta.name}"
        )

        variables, clases = cargar_datos(
            registro
        )

        for inicio, final in recorrer_lotes(
            len(variables),
            tamano_lote,
        ):
            variables_lote = escalador.transform(
                variables[
                    inicio:final
                ]
            )

            predicciones_lote = modelo.predict(
                variables_lote
            )

            reales.append(
                clases[
                    inicio:final
                ]
            )

            predicciones.append(
                predicciones_lote
            )

        del variables
        del clases
        gc.collect()

    return (
        np.concatenate(reales),
        np.concatenate(predicciones),
    )


def calcular_metricas(
    reales: np.ndarray,
    predicciones: np.ndarray,
    nombre: str,
) -> dict[str, Any]:
    """Calcula las métricas globales."""

    return {
        "modelo": nombre,
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
    """Crea la matriz de confusión etiquetada."""

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
    """Guarda un JSON legible."""

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
    epocas: int,
    tamano_lote: int,
    notas: str,
    modelo: SGDClassifier,
    escalador: StandardScaler,
    conteos: Counter,
    pesos_clases: dict[str, float],
    reales: np.ndarray,
    predicciones: np.ndarray,
) -> None:
    """Guarda el control sin sobrescribir los modelos V2A."""

    RUTA_CONTROL.mkdir(
        parents=True,
        exist_ok=True,
    )

    fecha = datetime.now(
        timezone.utc
    )

    identificador = (
        fecha.strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        + "_"
        + normalizar_etiqueta(
            etiqueta
        )
    )

    ruta_experimento = (
        RUTA_CONTROL
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
        len(reales),
        clase_mayoritaria,
        dtype=object,
    )

    metricas_modelo = calcular_metricas(
        reales,
        predicciones,
        "SGDClassifier control V1 mismas filas",
    )

    metricas_referencia = calcular_metricas(
        reales,
        prediccion_mayoritaria,
        "DummyClassifier (most_frequent)",
    )

    informe = classification_report(
        reales,
        predicciones,
        labels=CLASES_NP,
        output_dict=True,
        zero_division=0,
    )

    paquete = {
        "version_modelo": NOMBRE_CONTROL,
        "simbolo": simbolo,
        "modelo": modelo,
        "escalador": escalador,
        "columnas_modelo": list(
            COLUMNAS_MODELO_V1
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
        "usa_mismas_filas_v2a": True,
        "division_prueba_utilizada": False,
        "fecha_entrenamiento_utc": fecha.isoformat(),
        "etiqueta": etiqueta,
    }

    nombres = {
        "modelo": f"modelo_control_{simbolo}_{NOMBRE_HORIZONTE}.joblib",
        "metricas": f"metricas_control_{simbolo}_{NOMBRE_HORIZONTE}.csv",
        "matriz": f"matriz_confusion_control_{simbolo}_{NOMBRE_HORIZONTE}.csv",
        "informe": f"informe_clasificacion_control_{simbolo}_{NOMBRE_HORIZONTE}.json",
        "detalle": "detalle_entrenamiento.json",
    }

    rutas_experimento = {
        clave: ruta_experimento / nombre
        for clave, nombre in nombres.items()
    }

    joblib.dump(
        paquete,
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
        reales,
        predicciones,
    ).to_csv(
        rutas_experimento["matriz"],
        encoding="utf-8-sig",
    )

    guardar_json(
        informe,
        rutas_experimento["informe"],
    )

    detalle = {
        "fecha_utc": fecha.isoformat(),
        "simbolo": simbolo,
        "etiqueta": etiqueta,
        "epocas": epocas,
        "tamano_lote": tamano_lote,
        "variables": len(
            COLUMNAS_MODELO_V1
        ),
        "mismas_filas_v2a": True,
        "conteos_entrenamiento": dict(
            conteos
        ),
        "pesos_clases": pesos_clases,
        "metricas_modelo": metricas_modelo,
        "metricas_referencia": metricas_referencia,
        "division_prueba_utilizada": False,
        "notas": notas,
        "ruta_experimento": str(
            ruta_experimento
        ),
    }

    guardar_json(
        detalle,
        rutas_experimento["detalle"],
    )

    rutas_ultimo = {
        clave: (
            RUTA_CONTROL
            / (
                f"ultimo_entrenamiento_control_{simbolo}.json"
                if clave == "detalle"
                else nombre
            )
        )
        for clave, nombre in nombres.items()
    }

    for clave in (
        "modelo",
        "metricas",
        "matriz",
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
        detalle,
        rutas_ultimo["detalle"],
    )

    ruta_historial = (
        RUTA_CONTROL
        / "historial_entrenamientos_control.csv"
    )

    fila_historial = {
        "fecha_utc": fecha.isoformat(),
        "simbolo": simbolo,
        "etiqueta": etiqueta,
        "epocas": epocas,
        "tamano_lote": tamano_lote,
        "variables": len(
            COLUMNAS_MODELO_V1
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
        "mismas_filas_v2a": True,
        "division_prueba_utilizada": False,
        "ruta_experimento": str(
            ruta_experimento
        ),
        "notas": notas,
    }

    pd.DataFrame(
        [
            fila_historial
        ]
    ).to_csv(
        ruta_historial,
        mode="a",
        header=not ruta_historial.exists(),
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nRESULTADOS DEL CONTROL V1 CON FILAS V2A"
    )
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

    print(
        "\nArchivos de control:"
    )

    for ruta in rutas_ultimo.values():
        print(
            f"- {ruta}"
        )


def ejecutar(
    simbolo: str,
    epocas: int,
    tamano_lote: int,
    etiqueta: str,
    notas: str,
) -> None:
    """Ejecuta el experimento de control completo."""

    manifiesto = cargar_manifiesto(
        simbolo
    )

    entrenamiento = obtener_registros_division(
        manifiesto,
        "entrenamiento",
    )

    validacion = obtener_registros_division(
        manifiesto,
        "validacion",
    )

    print(
        "\nCONTROL V1 CON LAS MISMAS FILAS DE LA V2A"
    )
    print("=" * 70)
    print(
        f"Símbolo: {simbolo}"
    )
    print(
        f"Variables: {len(COLUMNAS_MODELO_V1)}"
    )
    print(
        "Filas: exactamente las mismas del manifiesto V2A"
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

    escalador, conteos = ajustar_escalador(
        entrenamiento,
        tamano_lote,
    )

    pesos = calcular_pesos_clases(
        conteos
    )

    print(
        "\nDistribución de entrenamiento:"
    )

    for clase in CLASES:
        print(
            f"{clase}: "
            f"{conteos[clase]:,}"
            .replace(",", ".")
        )

    print(
        "\nPesos de clases:"
    )

    for clase, peso in pesos.items():
        print(
            f"{clase}: {peso:.6f}"
        )

    modelo = entrenar_modelo(
        entrenamiento,
        escalador,
        pesos,
        tamano_lote,
        epocas,
    )

    print(
        "\nVALIDACIÓN TEMPORAL DEL CONTROL"
    )
    print("=" * 70)

    reales, predicciones = predecir_division(
        validacion,
        modelo,
        escalador,
        tamano_lote,
    )

    guardar_resultados(
        simbolo=simbolo,
        etiqueta=etiqueta,
        epocas=epocas,
        tamano_lote=tamano_lote,
        notas=notas,
        modelo=modelo,
        escalador=escalador,
        conteos=conteos,
        pesos_clases=pesos,
        reales=reales,
        predicciones=predicciones,
    )

    print(
        "\nEl control no utilizó enero-mayo de 2026."
    )


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene argumentos de consola."""

    parser = argparse.ArgumentParser(
        description=(
            "Entrena un control con las 36 variables V1 "
            "y exactamente las mismas filas de la V2A."
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
        default="control_v1_mismas_filas",
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
        ejecutar(
            simbolo=argumentos.simbolo,
            epocas=argumentos.epocas,
            tamano_lote=argumentos.tamano_lote,
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
            "\nNo se pudo ejecutar el control."
        )
        print(
            f"Detalle: {error}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
