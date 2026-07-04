from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from cripto.corto_plazo.configuracion import (
    RUTA_DATOS_PREPARADOS,
    RUTA_PROYECTO,
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

CODIGOS_CLASES = np.array(
    [
        0,
        1,
        2,
    ],
    dtype=np.int8,
)

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


def cargar_manifiesto(
    simbolo: str,
) -> pd.DataFrame:
    """Carga el manifiesto correspondiente a un símbolo."""

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
            + ", ".join(
                sorted(columnas_faltantes)
            )
        )

    manifiesto = manifiesto.loc[
        manifiesto["simbolo"] == simbolo
    ].copy()

    if manifiesto.empty:
        raise ValueError(
            f"No existen registros para {simbolo}."
        )

    return manifiesto


def obtener_rutas_validacion(
    manifiesto: pd.DataFrame,
) -> list[Path]:
    """Obtiene los archivos de la división de validación."""

    registros = (
        manifiesto
        .loc[
            manifiesto["division"]
            == "validacion"
        ]
        .sort_values("desde_archivo")
    )

    if registros.empty:
        raise ValueError(
            "El manifiesto no contiene archivos "
            "para validación."
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


def cargar_paquete_modelo(
    simbolo: str,
) -> dict[str, object]:
    """Carga el modelo, el escalador y sus metadatos."""

    ruta_modelo = (
        RUTA_MODELOS
        / f"modelo_base_{simbolo}_4h.joblib"
    )

    if not ruta_modelo.exists():
        raise FileNotFoundError(
            f"No existe el modelo: {ruta_modelo}"
        )

    paquete = joblib.load(
        ruta_modelo
    )

    claves_requeridas = {
        "modelo",
        "escalador",
        "columnas_modelo",
        "clases",
        "umbral_clase",
    }

    claves_faltantes = claves_requeridas.difference(
        paquete
    )

    if claves_faltantes:
        raise ValueError(
            "Faltan elementos en el paquete del modelo: "
            + ", ".join(
                sorted(claves_faltantes)
            )
        )

    modelo = paquete["modelo"]

    if not hasattr(
        modelo,
        "predict_proba",
    ):
        raise ValueError(
            "El modelo no permite obtener probabilidades."
        )

    return paquete


def clasificar_objetivo(
    rendimientos: pd.Series,
    umbral: float,
) -> np.ndarray:
    """Convierte el rendimiento futuro en códigos de clase."""

    valores = pd.to_numeric(
        rendimientos,
        errors="coerce",
    ).to_numpy(
        dtype="float64"
    )

    if not np.isfinite(valores).all():
        raise ValueError(
            "El objetivo contiene valores nulos "
            "o infinitos."
        )

    return np.select(
        [
            valores <= -umbral,
            valores >= umbral,
        ],
        [
            0,
            2,
        ],
        default=1,
    ).astype(
        np.int8
    )


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


def cargar_validacion(
    rutas: list[Path],
    paquete: dict[str, object],
    tamano_lote: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga las clases reales y las probabilidades del modelo.

    Solo utiliza la división de validación.
    """

    modelo = paquete["modelo"]
    escalador = paquete["escalador"]

    columnas_modelo = list(
        paquete["columnas_modelo"]
    )

    umbral_clase = float(
        paquete["umbral_clase"]
    )

    clases_modelo = [
        str(clase)
        for clase in modelo.classes_
    ]

    if set(clases_modelo) != set(CLASES):
        raise ValueError(
            "Las clases del modelo no coinciden con "
            "BAJA, NEUTRAL y SUBE."
        )

    orden_columnas = [
        clases_modelo.index(
            str(clase)
        )
        for clase in CLASES
    ]

    clases_reales: list[np.ndarray] = []
    probabilidades: list[np.ndarray] = []

    print("\nGENERANDO PROBABILIDADES DE VALIDACIÓN")
    print("=" * 70)

    for ruta in rutas:
        print(f"Procesando: {ruta.name}")

        columnas = [
            *columnas_modelo,
            "rendimiento_objetivo",
        ]

        datos = pd.read_parquet(
            ruta,
            columns=columnas,
        )

        variables = (
            datos[columnas_modelo]
            .to_numpy(
                dtype="float32"
            )
        )

        if not np.isfinite(
            variables
        ).all():
            raise ValueError(
                f"Existen variables no finitas en: {ruta}"
            )

        reales = clasificar_objetivo(
            rendimientos=datos[
                "rendimiento_objetivo"
            ],
            umbral=umbral_clase,
        )

        clases_reales.append(
            reales
        )

        for inicio, final in recorrer_lotes(
            total=len(variables),
            tamano_lote=tamano_lote,
        ):
            variables_lote = escalador.transform(
                variables[inicio:final]
            )

            probabilidades_lote = (
                modelo.predict_proba(
                    variables_lote
                )
            )

            probabilidades_lote = (
                probabilidades_lote[
                    :,
                    orden_columnas,
                ]
            )

            probabilidades.append(
                probabilidades_lote.astype(
                    "float32"
                )
            )

        del datos
        del variables
        del reales
        gc.collect()

    return (
        np.concatenate(
            clases_reales
        ),
        np.concatenate(
            probabilidades
        ),
    )


def crear_valores_rango(
    minimo: float,
    maximo: float,
    paso: float,
) -> np.ndarray:
    """Crea un rango numérico evitando errores decimales."""

    cantidad = int(
        round(
            (maximo - minimo)
            / paso
        )
    )

    valores = [
        round(
            minimo + indice * paso,
            6,
        )
        for indice in range(
            cantidad + 1
        )
    ]

    return np.array(
        valores,
        dtype="float64",
    )


def aplicar_multiplicadores(
    probabilidades: np.ndarray,
    multiplicador_baja: float,
    multiplicador_sube: float,
) -> np.ndarray:
    """
    Ajusta las puntuaciones utilizadas para elegir la clase.

    NEUTRAL conserva un multiplicador de 1.
    """

    puntuaciones = probabilidades.copy()

    puntuaciones[:, 0] *= (
        multiplicador_baja
    )

    puntuaciones[:, 2] *= (
        multiplicador_sube
    )

    return np.argmax(
        puntuaciones,
        axis=1,
    ).astype(
        np.int8
    )


def calcular_metricas(
    reales: np.ndarray,
    predicciones: np.ndarray,
    multiplicador_baja: float,
    multiplicador_sube: float,
) -> dict[str, float]:
    """Calcula las métricas de una combinación."""

    cantidad = len(
        predicciones
    )

    cantidad_baja = int(
        (predicciones == 0).sum()
    )

    cantidad_neutral = int(
        (predicciones == 1).sum()
    )

    cantidad_sube = int(
        (predicciones == 2).sum()
    )

    return {
        "multiplicador_baja": (
            multiplicador_baja
        ),
        "multiplicador_neutral": 1.0,
        "multiplicador_sube": (
            multiplicador_sube
        ),
        "accuracy": accuracy_score(
            reales,
            predicciones,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                reales,
                predicciones,
            )
        ),
        "f1_macro": f1_score(
            reales,
            predicciones,
            labels=CODIGOS_CLASES,
            average="macro",
            zero_division=0,
        ),
        "predicciones_baja": cantidad_baja,
        "predicciones_neutral": (
            cantidad_neutral
        ),
        "predicciones_sube": cantidad_sube,
        "porcentaje_predicho_baja": (
            cantidad_baja
            / cantidad
            * 100
        ),
        "porcentaje_predicho_neutral": (
            cantidad_neutral
            / cantidad
            * 100
        ),
        "porcentaje_predicho_sube": (
            cantidad_sube
            / cantidad
            * 100
        ),
    }


def evaluar_combinaciones(
    reales: np.ndarray,
    probabilidades: np.ndarray,
    valores_baja: np.ndarray,
    valores_sube: np.ndarray,
    combinaciones_evaluadas: set[
        tuple[float, float]
    ],
) -> list[dict[str, float]]:
    """Evalúa todas las combinaciones indicadas."""

    resultados: list[
        dict[str, float]
    ] = []

    total_combinaciones = (
        len(valores_baja)
        * len(valores_sube)
    )

    contador = 0

    for multiplicador_baja in valores_baja:
        for multiplicador_sube in valores_sube:
            contador += 1

            clave = (
                round(
                    float(
                        multiplicador_baja
                    ),
                    6,
                ),
                round(
                    float(
                        multiplicador_sube
                    ),
                    6,
                ),
            )

            if clave in combinaciones_evaluadas:
                continue

            combinaciones_evaluadas.add(
                clave
            )

            predicciones = aplicar_multiplicadores(
                probabilidades=probabilidades,
                multiplicador_baja=clave[0],
                multiplicador_sube=clave[1],
            )

            resultados.append(
                calcular_metricas(
                    reales=reales,
                    predicciones=predicciones,
                    multiplicador_baja=clave[0],
                    multiplicador_sube=clave[1],
                )
            )

            if contador % 25 == 0:
                print(
                    f"Combinaciones revisadas: "
                    f"{contador} de "
                    f"{total_combinaciones}"
                )

    return resultados


def seleccionar_mejor(
    resultados: pd.DataFrame,
    accuracy_minima: float,
) -> pd.Series:
    """Selecciona la mejor combinación dentro del límite de Accuracy."""

    elegibles = resultados.loc[
        resultados["accuracy"]
        >= accuracy_minima
    ].copy()

    if elegibles.empty:
        raise ValueError(
            "Ninguna combinación respetó la pérdida "
            "máxima permitida de Accuracy."
        )

    elegibles = elegibles.sort_values(
        [
            "f1_macro",
            "balanced_accuracy",
            "accuracy",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    return elegibles.iloc[0]


def guardar_resultados(
    simbolo: str,
    reales: np.ndarray,
    probabilidades: np.ndarray,
    resultados: pd.DataFrame,
    resultado_base: dict[str, float],
    mejor: pd.Series,
    perdida_maxima_accuracy: float,
) -> None:
    """Guarda el ajuste seleccionado y sus métricas."""

    RUTA_MODELOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    multiplicador_baja = float(
        mejor["multiplicador_baja"]
    )

    multiplicador_sube = float(
        mejor["multiplicador_sube"]
    )

    predicciones_ajustadas = (
        aplicar_multiplicadores(
            probabilidades=probabilidades,
            multiplicador_baja=multiplicador_baja,
            multiplicador_sube=multiplicador_sube,
        )
    )

    matriz = confusion_matrix(
        reales,
        predicciones_ajustadas,
        labels=CODIGOS_CLASES,
    )

    informe = classification_report(
        reales,
        predicciones_ajustadas,
        labels=CODIGOS_CLASES,
        target_names=CLASES,
        output_dict=True,
        zero_division=0,
    )

    ruta_candidatos = (
        RUTA_MODELOS
        / f"candidatos_ajuste_decision_{simbolo}_4h.csv"
    )

    ruta_configuracion = (
        RUTA_MODELOS
        / f"ajuste_decision_{simbolo}_4h.json"
    )

    ruta_metricas = (
        RUTA_MODELOS
        / f"metricas_ajuste_decision_{simbolo}_4h.csv"
    )

    ruta_matriz = (
        RUTA_MODELOS
        / f"matriz_confusion_ajuste_decision_{simbolo}_4h.csv"
    )

    ruta_informe = (
        RUTA_MODELOS
        / f"informe_ajuste_decision_{simbolo}_4h.json"
    )

    resultados = resultados.sort_values(
        [
            "f1_macro",
            "balanced_accuracy",
            "accuracy",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    resultados.to_csv(
        ruta_candidatos,
        index=False,
        encoding="utf-8-sig",
    )

    metricas_base = {
        "tipo": "decision_original",
        **resultado_base,
    }

    metricas_ajustadas = {
        "tipo": "decision_ajustada",
        **{
            columna: (
                float(mejor[columna])
                if isinstance(
                    mejor[columna],
                    (
                        np.floating,
                        float,
                    ),
                )
                else int(mejor[columna])
                if isinstance(
                    mejor[columna],
                    (
                        np.integer,
                        int,
                    ),
                )
                else mejor[columna]
            )
            for columna in mejor.index
        },
    }

    pd.DataFrame(
        [
            metricas_base,
            metricas_ajustadas,
        ]
    ).to_csv(
        ruta_metricas,
        index=False,
        encoding="utf-8-sig",
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

    configuracion = {
        "simbolo": simbolo,
        "horizonte": "4h",
        "division_utilizada": "validacion_2025",
        "division_prueba_utilizada": False,
        "multiplicadores": {
            "BAJA": multiplicador_baja,
            "NEUTRAL": 1.0,
            "SUBE": multiplicador_sube,
        },
        "criterio_seleccion": (
            "Mayor F1-score macro, después "
            "Balanced Accuracy y Accuracy."
        ),
        "perdida_maxima_accuracy_permitida": (
            perdida_maxima_accuracy
        ),
        "metricas_decision_original": (
            resultado_base
        ),
        "metricas_decision_ajustada": (
            metricas_ajustadas
        ),
    }

    with ruta_configuracion.open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            configuracion,
            archivo,
            ensure_ascii=False,
            indent=4,
        )

    print("\nRESULTADO DEL AJUSTE")
    print("=" * 70)

    print("\nDecisión original")
    print(
        f"Accuracy: "
        f"{resultado_base['accuracy']:.4f}"
    )
    print(
        f"Balanced Accuracy: "
        f"{resultado_base['balanced_accuracy']:.4f}"
    )
    print(
        f"F1-score macro: "
        f"{resultado_base['f1_macro']:.4f}"
    )

    print("\nDecisión ajustada")
    print(
        f"Multiplicador BAJA: "
        f"{multiplicador_baja:.2f}"
    )
    print(
        "Multiplicador NEUTRAL: 1.00"
    )
    print(
        f"Multiplicador SUBE: "
        f"{multiplicador_sube:.2f}"
    )
    print(
        f"Accuracy: "
        f"{float(mejor['accuracy']):.4f}"
    )
    print(
        f"Balanced Accuracy: "
        f"{float(mejor['balanced_accuracy']):.4f}"
    )
    print(
        f"F1-score macro: "
        f"{float(mejor['f1_macro']):.4f}"
    )

    print("\nDistribución predicha ajustada:")
    print(
        f"BAJA: "
        f"{float(mejor['porcentaje_predicho_baja']):.2f} %"
    )
    print(
        f"NEUTRAL: "
        f"{float(mejor['porcentaje_predicho_neutral']):.2f} %"
    )
    print(
        f"SUBE: "
        f"{float(mejor['porcentaje_predicho_sube']):.2f} %"
    )

    print("\nArchivos generados:")
    print(f"- {ruta_candidatos}")
    print(f"- {ruta_configuracion}")
    print(f"- {ruta_metricas}")
    print(f"- {ruta_matriz}")
    print(f"- {ruta_informe}")

    print(
        "\nLa división de prueba de 2026 "
        "no fue utilizada."
    )


def ajustar_decision(
    simbolo: str,
    tamano_lote: int,
    multiplicador_minimo: float,
    multiplicador_maximo: float,
    paso_grueso: float,
    paso_fino: float,
    perdida_maxima_accuracy: float,
) -> None:
    """Ejecuta el ajuste completo de la regla de decisión."""

    print("\nAJUSTE DE LA REGLA DE DECISIÓN")
    print("=" * 70)
    print(f"Símbolo: {simbolo}")
    print("División utilizada: validación de 2025")
    print("División de prueba de 2026: no utilizada")

    manifiesto = cargar_manifiesto(
        simbolo=simbolo
    )

    rutas_validacion = (
        obtener_rutas_validacion(
            manifiesto=manifiesto
        )
    )

    paquete = cargar_paquete_modelo(
        simbolo=simbolo
    )

    reales, probabilidades = cargar_validacion(
        rutas=rutas_validacion,
        paquete=paquete,
        tamano_lote=tamano_lote,
    )

    print(
        f"\nMuestras de validación: "
        f"{len(reales):,}".replace(",", ".")
    )

    predicciones_base = np.argmax(
        probabilidades,
        axis=1,
    ).astype(
        np.int8
    )

    resultado_base = calcular_metricas(
        reales=reales,
        predicciones=predicciones_base,
        multiplicador_baja=1.0,
        multiplicador_sube=1.0,
    )

    accuracy_minima = (
        resultado_base["accuracy"]
        - perdida_maxima_accuracy
    )

    print("\nBÚSQUEDA GRUESA")
    print("=" * 70)

    valores_gruesos = crear_valores_rango(
        minimo=multiplicador_minimo,
        maximo=multiplicador_maximo,
        paso=paso_grueso,
    )

    combinaciones_evaluadas: set[
        tuple[float, float]
    ] = set()

    resultados = evaluar_combinaciones(
        reales=reales,
        probabilidades=probabilidades,
        valores_baja=valores_gruesos,
        valores_sube=valores_gruesos,
        combinaciones_evaluadas=combinaciones_evaluadas,
    )

    resultados_gruesos = pd.DataFrame(
        resultados
    )

    mejor_grueso = seleccionar_mejor(
        resultados=resultados_gruesos,
        accuracy_minima=accuracy_minima,
    )

    print("\nMejor combinación gruesa:")
    print(
        f"BAJA: "
        f"{float(mejor_grueso['multiplicador_baja']):.2f}"
    )
    print(
        f"SUBE: "
        f"{float(mejor_grueso['multiplicador_sube']):.2f}"
    )
    print(
        f"F1-score macro: "
        f"{float(mejor_grueso['f1_macro']):.4f}"
    )

    print("\nBÚSQUEDA FINA")
    print("=" * 70)

    centro_baja = float(
        mejor_grueso["multiplicador_baja"]
    )

    centro_sube = float(
        mejor_grueso["multiplicador_sube"]
    )

    amplitud = paso_grueso

    minimo_baja = max(
        0.05,
        centro_baja - amplitud,
    )

    maximo_baja = (
        centro_baja + amplitud
    )

    minimo_sube = max(
        0.05,
        centro_sube - amplitud,
    )

    maximo_sube = (
        centro_sube + amplitud
    )

    valores_finos_baja = crear_valores_rango(
        minimo=minimo_baja,
        maximo=maximo_baja,
        paso=paso_fino,
    )

    valores_finos_sube = crear_valores_rango(
        minimo=minimo_sube,
        maximo=maximo_sube,
        paso=paso_fino,
    )

    resultados.extend(
        evaluar_combinaciones(
            reales=reales,
            probabilidades=probabilidades,
            valores_baja=valores_finos_baja,
            valores_sube=valores_finos_sube,
            combinaciones_evaluadas=combinaciones_evaluadas,
        )
    )

    resultados_totales = pd.DataFrame(
        resultados
    )

    mejor = seleccionar_mejor(
        resultados=resultados_totales,
        accuracy_minima=accuracy_minima,
    )

    guardar_resultados(
        simbolo=simbolo,
        reales=reales,
        probabilidades=probabilidades,
        resultados=resultados_totales,
        resultado_base=resultado_base,
        mejor=mejor,
        perdida_maxima_accuracy=perdida_maxima_accuracy,
    )


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene los argumentos del comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Ajusta la decisión entre BAJA, NEUTRAL "
            "y SUBE utilizando validación de 2025."
        )
    )

    parser.add_argument(
        "--simbolo",
        required=True,
        choices=SIMBOLOS,
    )

    parser.add_argument(
        "--tamano-lote",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--multiplicador-minimo",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--multiplicador-maximo",
        type=float,
        default=3.00,
    )

    parser.add_argument(
        "--paso-grueso",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--paso-fino",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--max-perdida-accuracy",
        type=float,
        default=0.05,
        help=(
            "Pérdida máxima de Accuracy permitida "
            "frente a la decisión original."
        ),
    )

    argumentos = parser.parse_args()

    if argumentos.tamano_lote <= 0:
        parser.error(
            "--tamano-lote debe ser mayor que cero."
        )

    if argumentos.multiplicador_minimo <= 0:
        parser.error(
            "--multiplicador-minimo debe ser mayor que cero."
        )

    if (
        argumentos.multiplicador_maximo
        < argumentos.multiplicador_minimo
    ):
        parser.error(
            "--multiplicador-maximo debe ser mayor "
            "o igual que --multiplicador-minimo."
        )

    if argumentos.paso_grueso <= 0:
        parser.error(
            "--paso-grueso debe ser mayor que cero."
        )

    if argumentos.paso_fino <= 0:
        parser.error(
            "--paso-fino debe ser mayor que cero."
        )

    if not (
        0
        <= argumentos.max_perdida_accuracy
        < 1
    ):
        parser.error(
            "--max-perdida-accuracy debe estar "
            "entre cero y uno."
        )

    return argumentos


def main() -> None:
    """Punto de entrada."""

    argumentos = obtener_argumentos()

    try:
        ajustar_decision(
            simbolo=argumentos.simbolo,
            tamano_lote=argumentos.tamano_lote,
            multiplicador_minimo=(
                argumentos.multiplicador_minimo
            ),
            multiplicador_maximo=(
                argumentos.multiplicador_maximo
            ),
            paso_grueso=argumentos.paso_grueso,
            paso_fino=argumentos.paso_fino,
            perdida_maxima_accuracy=(
                argumentos.max_perdida_accuracy
            ),
        )

    except (
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(
            "\nNo se pudo ajustar la regla de decisión."
        )
        print(
            f"Detalle: {error}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()