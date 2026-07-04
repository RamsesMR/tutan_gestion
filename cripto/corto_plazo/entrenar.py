from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

from cripto.corto_plazo.configuracion import (
    RUTA_DATOS_PREPARADOS,
    RUTA_PROYECTO,
)
from cripto.corto_plazo.variables import (
    VENTANAS_MINUTOS,
)
from cripto.corto_plazo.registro_entrenamientos import (
    registrar_entrenamiento,
)


SIMBOLOS = (
    "BTCUSDT",
    "ETHUSDT",
)

CLASES = np.array(
    [
        "BAJA",
        "NEUTRAL",
        "SUBE",
    ],
    dtype=object,
)

UMBRAL_CLASE = 0.005

RUTA_MANIFIESTO = (
    RUTA_DATOS_PREPARADOS
    / "manifiesto_divisiones_4h.csv"
)

RUTA_MODELOS = (
    RUTA_PROYECTO
    / "modelos_entrenados"
    / "cripto"
    / "corto_plazo"
)


def construir_columnas_modelo() -> list[str]:
    """Devuelve las 36 variables utilizadas por el modelo."""

    columnas = [
        "rango_relativo",
        "cuerpo_relativo",
        "mecha_superior_relativa",
        "mecha_inferior_relativa",
        "proporcion_compradora_base",
        "proporcion_compradora_cotizacion",
        "rendimiento_1m",
    ]

    for ventana in VENTANAS_MINUTOS:
        columnas.extend(
            [
                f"rendimiento_{ventana}m",
                f"volatilidad_{ventana}m",
                f"desviacion_media_cierre_{ventana}m",
                f"volumen_relativo_{ventana}m",
                f"operaciones_relativas_{ventana}m",
            ]
        )

    columnas.extend(
        [
            "minuto_dia_seno",
            "minuto_dia_coseno",
            "dia_semana_seno",
            "dia_semana_coseno",
        ]
    )

    return columnas


COLUMNAS_MODELO = construir_columnas_modelo()


def clasificar_objetivo(
    rendimientos: pd.Series,
) -> np.ndarray:
    """Convierte el rendimiento futuro en tres clases."""

    valores = pd.to_numeric(
        rendimientos,
        errors="coerce",
    ).to_numpy(dtype="float64")

    if not np.isfinite(valores).all():
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


def cargar_manifiesto(
    simbolo: str,
) -> pd.DataFrame:
    """Carga las divisiones correspondientes a un símbolo."""

    if not RUTA_MANIFIESTO.exists():
        raise FileNotFoundError(
            f"No existe el manifiesto: {RUTA_MANIFIESTO}"
        )

    manifiesto = pd.read_csv(
        RUTA_MANIFIESTO
    )

    columnas_requeridas = {
        "simbolo",
        "division",
        "archivo",
        "desde_archivo",
    }

    columnas_faltantes = (
        columnas_requeridas
        .difference(manifiesto.columns)
    )

    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas en el manifiesto: "
            + ", ".join(sorted(columnas_faltantes))
        )

    manifiesto = manifiesto.loc[
        manifiesto["simbolo"] == simbolo
    ].copy()

    if manifiesto.empty:
        raise ValueError(
            f"No existen registros para {simbolo}."
        )

    return manifiesto


def obtener_rutas_division(
    manifiesto: pd.DataFrame,
    division: str,
) -> list[Path]:
    """Obtiene las rutas ordenadas de una división."""

    registros = (
        manifiesto
        .loc[manifiesto["division"] == division]
        .sort_values("desde_archivo")
    )

    if registros.empty:
        raise ValueError(
            f"No existen archivos para la división {division}."
        )

    rutas: list[Path] = []

    for archivo in registros["archivo"]:
        nombre_archivo = (
            str(archivo)
            .replace("\\", "/")
            .rsplit("/", 1)[-1]
        )

        ruta = (
            RUTA_DATOS_PREPARADOS
            / nombre_archivo
        )

        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe el archivo: {ruta}"
            )

        rutas.append(ruta)

    return rutas


def cargar_datos(
    ruta: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Carga las variables y el objetivo de un Parquet."""

    columnas = [
        *COLUMNAS_MODELO,
        "rendimiento_objetivo",
    ]

    datos = pd.read_parquet(
        ruta,
        columns=columnas,
    )

    variables = (
        datos[COLUMNAS_MODELO]
        .to_numpy(dtype="float32")
    )

    if not np.isfinite(variables).all():
        raise ValueError(
            f"Existen variables no finitas en: {ruta}"
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
    rutas: list[Path],
    tamano_lote: int,
) -> tuple[StandardScaler, Counter]:
    """Ajusta el escalador y cuenta las clases de entrenamiento."""

    escalador = StandardScaler()
    conteos: Counter = Counter()

    print("\nAJUSTANDO ESCALADOR")
    print("=" * 70)

    for ruta in rutas:
        print(f"Procesando: {ruta.name}")

        variables, clases = cargar_datos(
            ruta
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
    """Calcula pesos inversamente proporcionales a cada clase."""

    total = sum(
        conteos[clase]
        for clase in CLASES
    )

    pesos: dict[str, float] = {}

    for clase in CLASES:
        cantidad = conteos[clase]

        if cantidad == 0:
            raise ValueError(
                f"La clase {clase} no tiene muestras."
            )

        pesos[str(clase)] = (
            total
            / (
                len(CLASES)
                * cantidad
            )
        )

    return pesos


def entrenar_modelo(
    rutas: list[Path],
    escalador: StandardScaler,
    pesos_clases: dict[str, float],
    tamano_lote: int,
    epocas: int,
) -> SGDClassifier:
    """Entrena el modelo lineal mediante lotes."""

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

    print("\nENTRENAMIENTO DEL MODELO BASE")
    print("=" * 70)

    for epoca in range(
        1,
        epocas + 1,
    ):
        print(
            f"\nÉpoca {epoca} de {epocas}"
        )

        for indice_archivo, ruta in enumerate(
            rutas
        ):
            print(f"Entrenando con: {ruta.name}")

            variables, clases = cargar_datos(
                ruta
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
                    variables[indices_lote]
                )

                clases_lote = clases[
                    indices_lote
                ]

                if primera_actualizacion:
                    modelo.partial_fit(
                        variables_lote,
                        clases_lote,
                        classes=CLASES,
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
    rutas: list[Path],
    modelo: SGDClassifier,
    escalador: StandardScaler,
    tamano_lote: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Obtiene predicciones de una división sin cargarla completa."""

    reales: list[np.ndarray] = []
    predicciones: list[np.ndarray] = []

    for ruta in rutas:
        print(f"Evaluando: {ruta.name}")

        variables, clases = cargar_datos(
            ruta
        )

        for inicio, final in recorrer_lotes(
            total=len(variables),
            tamano_lote=tamano_lote,
        ):
            variables_lote = escalador.transform(
                variables[inicio:final]
            )

            prediccion_lote = modelo.predict(
                variables_lote
            )

            reales.append(
                clases[inicio:final]
            )

            predicciones.append(
                prediccion_lote
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
    nombre_modelo: str,
) -> dict[str, object]:
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
            labels=CLASES,
            average="macro",
            zero_division=0,
        ),
    }


def guardar_resultados(
    simbolo: str,
    modelo: SGDClassifier,
    escalador: StandardScaler,
    conteos: Counter,
    pesos_clases: dict[str, float],
    reales: np.ndarray,
    predicciones: np.ndarray,
) -> None:
    """Guarda el modelo, las métricas y las matrices."""

    RUTA_MODELOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    clase_mayoritaria = max(
        CLASES,
        key=lambda clase: conteos[str(clase)],
    )

    prediccion_mayoritaria = np.full(
        shape=len(reales),
        fill_value=clase_mayoritaria,
        dtype=object,
    )

    metricas_modelo = calcular_metricas(
        reales=reales,
        predicciones=predicciones,
        nombre_modelo="SGDClassifier",
    )

    metricas_referencia = calcular_metricas(
        reales=reales,
        predicciones=prediccion_mayoritaria,
        nombre_modelo="DummyClassifier (most_frequent)",
    )

    ruta_modelo = (
        RUTA_MODELOS
        / f"modelo_base_{simbolo}_4h.joblib"
    )

    ruta_metricas = (
        RUTA_MODELOS
        / f"metricas_base_{simbolo}_4h.csv"
    )

    ruta_matriz = (
        RUTA_MODELOS
        / f"matriz_confusion_base_{simbolo}_4h.csv"
    )

    ruta_matriz_referencia = (
        RUTA_MODELOS
        / f"matriz_confusion_referencia_{simbolo}_4h.csv"
    )

    ruta_informe = (
        RUTA_MODELOS
        / f"informe_clasificacion_base_{simbolo}_4h.json"
    )

    paquete_modelo = {
        "simbolo": simbolo,
        "modelo": modelo,
        "escalador": escalador,
        "columnas_modelo": COLUMNAS_MODELO,
        "clases": CLASES.tolist(),
        "umbral_clase": UMBRAL_CLASE,
        "conteos_entrenamiento": dict(conteos),
        "pesos_clases": pesos_clases,
    }

    joblib.dump(
        paquete_modelo,
        ruta_modelo,
    )

    pd.DataFrame(
        [
            metricas_modelo,
            metricas_referencia,
        ]
    ).to_csv(
        ruta_metricas,
        index=False,
        encoding="utf-8-sig",
    )

    matriz = confusion_matrix(
        reales,
        predicciones,
        labels=CLASES,
    )

    pd.DataFrame(
        matriz,
        index=[
            f"real_{clase}"
            for clase in CLASES
        ],
        columns=[
            f"predicha_{clase}"
            for clase in CLASES
        ],
    ).to_csv(
        ruta_matriz,
        encoding="utf-8-sig",
    )

    matriz_referencia = confusion_matrix(
        reales,
        prediccion_mayoritaria,
        labels=CLASES,
    )

    pd.DataFrame(
        matriz_referencia,
        index=[
            f"real_{clase}"
            for clase in CLASES
        ],
        columns=[
            f"predicha_{clase}"
            for clase in CLASES
        ],
    ).to_csv(
        ruta_matriz_referencia,
        encoding="utf-8-sig",
    )

    informe = classification_report(
        reales,
        predicciones,
        labels=CLASES,
        output_dict=True,
        zero_division=0,
    )

    with ruta_informe.open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            informe,
            archivo,
            ensure_ascii=False,
            indent=4,
        )

    print("\nRESULTADOS DE VALIDACIÓN")
    print("=" * 70)

    for metricas in (
        metricas_modelo,
        metricas_referencia,
    ):
        print(
            f"\n{metricas['modelo']}"
        )

        print(
            f"Accuracy: "
            f"{metricas['accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{metricas['balanced_accuracy']:.4f}"
        )

        print(
            f"F1-score macro: "
            f"{metricas['f1_macro']:.4f}"
        )

    print("\nArchivos generados:")
    print(f"- {ruta_modelo}")
    print(f"- {ruta_metricas}")
    print(f"- {ruta_matriz}")
    print(f"- {ruta_matriz_referencia}")
    print(f"- {ruta_informe}")


def entrenar(
    simbolo: str,
    tamano_lote: int,
    epocas: int,
    etiqueta: str,
    notas: str,
) -> None:
    """Ejecuta el entrenamiento completo de un símbolo."""

    manifiesto = cargar_manifiesto(
        simbolo=simbolo
    )

    rutas_entrenamiento = obtener_rutas_division(
        manifiesto=manifiesto,
        division="entrenamiento",
    )

    rutas_validacion = obtener_rutas_division(
        manifiesto=manifiesto,
        division="validacion",
    )

    print("\nMODELO BASE DE CORTO PLAZO")
    print("=" * 70)
    print(f"Símbolo: {simbolo}")
    print(f"Variables: {len(COLUMNAS_MODELO)}")
    print(f"Umbral: ±{UMBRAL_CLASE * 100:.2f} %")
    print(f"Épocas: {epocas}")
    print(
        f"Tamaño de lote: "
        f"{tamano_lote:,}".replace(",", ".")
    )
    print(f"Etiqueta: {etiqueta}")

    if notas:
        print(f"Notas: {notas}")

    escalador, conteos = ajustar_escalador(
        rutas=rutas_entrenamiento,
        tamano_lote=tamano_lote,
    )

    pesos_clases = calcular_pesos_clases(
        conteos=conteos
    )

    print("\nDistribución de entrenamiento:")

    for clase in CLASES:
        print(
            f"{clase}: "
            f"{conteos[str(clase)]:,}"
            .replace(",", ".")
        )

    print("\nPesos de clases:")

    for clase, peso in pesos_clases.items():
        print(
            f"{clase}: {peso:.6f}"
        )

    modelo = entrenar_modelo(
        rutas=rutas_entrenamiento,
        escalador=escalador,
        pesos_clases=pesos_clases,
        tamano_lote=tamano_lote,
        epocas=epocas,
    )

    print("\nVALIDACIÓN TEMPORAL")
    print("=" * 70)

    reales, predicciones = predecir_division(
        rutas=rutas_validacion,
        modelo=modelo,
        escalador=escalador,
        tamano_lote=tamano_lote,
    )

    guardar_resultados(
        simbolo=simbolo,
        modelo=modelo,
        escalador=escalador,
        conteos=conteos,
        pesos_clases=pesos_clases,
        reales=reales,
        predicciones=predicciones,
    )

    registrar_entrenamiento(
        simbolo=simbolo,
        etiqueta=etiqueta,
        epocas=epocas,
        tamano_lote=tamano_lote,
        notas=notas,
    )

    print(
        "\nLa división de prueba de 2026 "
        "no fue utilizada."
    )


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene los argumentos del comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Entrena un modelo lineal base para "
            "clasificar movimientos a cuatro horas."
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
        default="modelo_base_sgd",
        help=(
            "Nombre corto para identificar esta versión "
            "en el historial."
        ),
    )

    parser.add_argument(
        "--notas",
        default="",
        help="Observaciones sobre el entrenamiento.",
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
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(
            "\nNo se pudo entrenar el modelo base."
        )

        print(
            f"Detalle: {error}"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()