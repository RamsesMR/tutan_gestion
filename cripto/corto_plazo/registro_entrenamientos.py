from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from cripto.corto_plazo.configuracion import RUTA_PROYECTO


RUTA_MODELOS = (
    RUTA_PROYECTO
    / "modelos_entrenados"
    / "cripto"
    / "corto_plazo"
)

RUTA_HISTORIAL = (
    RUTA_MODELOS
    / "historial_entrenamientos.csv"
)

RUTA_RESUMEN = (
    RUTA_MODELOS
    / "resumen_evolucion_entrenamientos.md"
)

RUTA_DETALLES = (
    RUTA_MODELOS
    / "historial"
)


def normalizar_nombre(texto: str) -> str:
    """Convierte una etiqueta en una parte segura para una ruta."""

    texto = texto.strip().lower()

    texto = re.sub(
        r"[^a-z0-9áéíóúñü_-]+",
        "_",
        texto,
    )

    texto = re.sub(
        r"_+",
        "_",
        texto,
    )

    return texto.strip("_") or "entrenamiento"


def convertir_valor_json(valor: Any) -> Any:
    """Convierte tipos de pandas y NumPy en valores serializables."""

    if pd.isna(valor):
        return None

    if hasattr(valor, "item"):
        try:
            return valor.item()
        except (ValueError, AttributeError):
            pass

    return valor


def calcular_huella(
    rutas: list[Path],
) -> str:
    """Calcula una huella de los archivos que forman el resultado."""

    calculo = hashlib.sha256()

    for ruta in rutas:
        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe el archivo necesario: {ruta}"
            )

        calculo.update(
            ruta.name.encode("utf-8")
        )

        with ruta.open("rb") as archivo:
            while bloque := archivo.read(1024 * 1024):
                calculo.update(bloque)

    return calculo.hexdigest()


def cargar_metricas(
    ruta: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Carga y normaliza las métricas del modelo y del DummyClassifier."""

    metricas = pd.read_csv(ruta)

    metricas = metricas.rename(
        columns={
            "exactitud": "accuracy",
            "exactitud_balanceada": "balanced_accuracy",
        }
    )

    columnas_requeridas = {
        "modelo",
        "muestras",
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
    }

    columnas_faltantes = columnas_requeridas.difference(
        metricas.columns
    )

    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas en el archivo de métricas: "
            + ", ".join(sorted(columnas_faltantes))
        )

    nombres_modelo = (
        metricas["modelo"]
        .astype(str)
        .str.lower()
    )

    es_dummy = nombres_modelo.str.contains(
        "dummy|mayoritaria",
        regex=True,
    )

    filas_modelo = metricas.loc[~es_dummy]
    filas_dummy = metricas.loc[es_dummy]

    if filas_modelo.empty or filas_dummy.empty:
        if len(metricas) != 2:
            raise ValueError(
                "No se pudo identificar el modelo y la referencia."
            )

        filas_modelo = metricas.iloc[[0]]
        filas_dummy = metricas.iloc[[1]]

    modelo = {
        clave: convertir_valor_json(valor)
        for clave, valor in filas_modelo.iloc[0].to_dict().items()
    }

    dummy = {
        clave: convertir_valor_json(valor)
        for clave, valor in filas_dummy.iloc[0].to_dict().items()
    }

    modelo["modelo"] = "SGDClassifier"
    dummy["modelo"] = "DummyClassifier (most_frequent)"

    return modelo, dummy


def cargar_informe(
    ruta: Path,
) -> dict[str, Any]:
    """Carga el classification report del modelo."""

    with ruta.open(
        "r",
        encoding="utf-8",
    ) as archivo:
        return json.load(archivo)


def cargar_matriz(
    ruta: Path,
) -> pd.DataFrame:
    """Carga y valida una matriz de confusión."""

    matriz = pd.read_csv(
        ruta,
        index_col=0,
    )

    filas = [
        "real_BAJA",
        "real_NEUTRAL",
        "real_SUBE",
    ]

    columnas = [
        "predicha_BAJA",
        "predicha_NEUTRAL",
        "predicha_SUBE",
    ]

    faltan_filas = set(filas).difference(
        matriz.index
    )

    faltan_columnas = set(columnas).difference(
        matriz.columns
    )

    if faltan_filas or faltan_columnas:
        raise ValueError(
            f"La matriz de confusión no tiene el formato esperado: {ruta}"
        )

    return matriz.loc[
        filas,
        columnas,
    ]


def cargar_paquete_modelo(
    ruta: Path,
) -> dict[str, Any]:
    """Carga los metadatos guardados junto al modelo."""

    paquete = joblib.load(ruta)

    claves_requeridas = {
        "simbolo",
        "columnas_modelo",
        "clases",
        "umbral_clase",
        "conteos_entrenamiento",
        "pesos_clases",
    }

    claves_faltantes = claves_requeridas.difference(
        paquete
    )

    if claves_faltantes:
        raise ValueError(
            "Faltan datos en el paquete del modelo: "
            + ", ".join(sorted(claves_faltantes))
        )

    return paquete


def obtener_metrica_clase(
    informe: dict[str, Any],
    clase: str,
    metrica: str,
) -> float:
    """Obtiene una métrica concreta del classification report."""

    if clase not in informe:
        raise ValueError(
            f"El informe no contiene la clase {clase}."
        )

    if metrica not in informe[clase]:
        raise ValueError(
            f"El informe no contiene {metrica} para {clase}."
        )

    return float(
        informe[clase][metrica]
    )


def construir_registro(
    simbolo: str,
    etiqueta: str,
    epocas: int,
    tamano_lote: int,
    notas: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construye el registro completo de un entrenamiento."""

    ruta_metricas = (
        RUTA_MODELOS
        / f"metricas_base_{simbolo}_4h.csv"
    )

    ruta_informe = (
        RUTA_MODELOS
        / f"informe_clasificacion_base_{simbolo}_4h.json"
    )

    ruta_matriz = (
        RUTA_MODELOS
        / f"matriz_confusion_base_{simbolo}_4h.csv"
    )

    ruta_matriz_dummy = (
        RUTA_MODELOS
        / f"matriz_confusion_referencia_{simbolo}_4h.csv"
    )

    ruta_modelo = (
        RUTA_MODELOS
        / f"modelo_base_{simbolo}_4h.joblib"
    )

    rutas_resultado = [
        ruta_metricas,
        ruta_informe,
        ruta_matriz,
        ruta_matriz_dummy,
        ruta_modelo,
    ]

    huella = calcular_huella(
        rutas=rutas_resultado
    )

    metricas_modelo, metricas_dummy = cargar_metricas(
        ruta=ruta_metricas
    )

    informe = cargar_informe(
        ruta=ruta_informe
    )

    matriz = cargar_matriz(
        ruta=ruta_matriz
    )

    matriz_dummy = cargar_matriz(
        ruta=ruta_matriz_dummy
    )

    paquete = cargar_paquete_modelo(
        ruta=ruta_modelo
    )

    fecha_utc = datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    )

    etiqueta_normalizada = normalizar_nombre(
        etiqueta
    )

    id_entrenamiento = (
        f"{fecha_utc:%Y%m%dT%H%M%SZ}_"
        f"{simbolo}_{etiqueta_normalizada}"
    )

    conteos = {
        str(clase): int(cantidad)
        for clase, cantidad
        in paquete["conteos_entrenamiento"].items()
    }

    pesos = {
        str(clase): float(peso)
        for clase, peso
        in paquete["pesos_clases"].items()
    }

    registro = {
        "id_entrenamiento": id_entrenamiento,
        "fecha_utc": fecha_utc.isoformat(),
        "simbolo": simbolo,
        "etiqueta": etiqueta,
        "modelo": "SGDClassifier",
        "horizonte": "4h",
        "umbral_clase": float(
            paquete["umbral_clase"]
        ),
        "umbral_porcentaje": float(
            paquete["umbral_clase"]
        ) * 100,
        "variables": len(
            paquete["columnas_modelo"]
        ),
        "epocas": epocas,
        "tamano_lote": tamano_lote,
        "muestras_entrenamiento": sum(
            conteos.values()
        ),
        "entrenamiento_baja": conteos.get(
            "BAJA",
            0,
        ),
        "entrenamiento_neutral": conteos.get(
            "NEUTRAL",
            0,
        ),
        "entrenamiento_sube": conteos.get(
            "SUBE",
            0,
        ),
        "peso_baja": pesos.get(
            "BAJA",
            0.0,
        ),
        "peso_neutral": pesos.get(
            "NEUTRAL",
            0.0,
        ),
        "peso_sube": pesos.get(
            "SUBE",
            0.0,
        ),
        "muestras_validacion": int(
            metricas_modelo["muestras"]
        ),
        "accuracy": float(
            metricas_modelo["accuracy"]
        ),
        "balanced_accuracy": float(
            metricas_modelo["balanced_accuracy"]
        ),
        "f1_macro": float(
            metricas_modelo["f1_macro"]
        ),
        "dummy_accuracy": float(
            metricas_dummy["accuracy"]
        ),
        "dummy_balanced_accuracy": float(
            metricas_dummy["balanced_accuracy"]
        ),
        "dummy_f1_macro": float(
            metricas_dummy["f1_macro"]
        ),
        "mejora_accuracy": (
            float(metricas_modelo["accuracy"])
            - float(metricas_dummy["accuracy"])
        ),
        "mejora_balanced_accuracy": (
            float(metricas_modelo["balanced_accuracy"])
            - float(metricas_dummy["balanced_accuracy"])
        ),
        "mejora_f1_macro": (
            float(metricas_modelo["f1_macro"])
            - float(metricas_dummy["f1_macro"])
        ),
        "precision_baja": obtener_metrica_clase(
            informe,
            "BAJA",
            "precision",
        ),
        "recall_baja": obtener_metrica_clase(
            informe,
            "BAJA",
            "recall",
        ),
        "f1_baja": obtener_metrica_clase(
            informe,
            "BAJA",
            "f1-score",
        ),
        "support_baja": int(
            obtener_metrica_clase(
                informe,
                "BAJA",
                "support",
            )
        ),
        "precision_neutral": obtener_metrica_clase(
            informe,
            "NEUTRAL",
            "precision",
        ),
        "recall_neutral": obtener_metrica_clase(
            informe,
            "NEUTRAL",
            "recall",
        ),
        "f1_neutral": obtener_metrica_clase(
            informe,
            "NEUTRAL",
            "f1-score",
        ),
        "support_neutral": int(
            obtener_metrica_clase(
                informe,
                "NEUTRAL",
                "support",
            )
        ),
        "precision_sube": obtener_metrica_clase(
            informe,
            "SUBE",
            "precision",
        ),
        "recall_sube": obtener_metrica_clase(
            informe,
            "SUBE",
            "recall",
        ),
        "f1_sube": obtener_metrica_clase(
            informe,
            "SUBE",
            "f1-score",
        ),
        "support_sube": int(
            obtener_metrica_clase(
                informe,
                "SUBE",
                "support",
            )
        ),
        "cm_baja_baja": int(
            matriz.loc[
                "real_BAJA",
                "predicha_BAJA",
            ]
        ),
        "cm_baja_neutral": int(
            matriz.loc[
                "real_BAJA",
                "predicha_NEUTRAL",
            ]
        ),
        "cm_baja_sube": int(
            matriz.loc[
                "real_BAJA",
                "predicha_SUBE",
            ]
        ),
        "cm_neutral_baja": int(
            matriz.loc[
                "real_NEUTRAL",
                "predicha_BAJA",
            ]
        ),
        "cm_neutral_neutral": int(
            matriz.loc[
                "real_NEUTRAL",
                "predicha_NEUTRAL",
            ]
        ),
        "cm_neutral_sube": int(
            matriz.loc[
                "real_NEUTRAL",
                "predicha_SUBE",
            ]
        ),
        "cm_sube_baja": int(
            matriz.loc[
                "real_SUBE",
                "predicha_BAJA",
            ]
        ),
        "cm_sube_neutral": int(
            matriz.loc[
                "real_SUBE",
                "predicha_NEUTRAL",
            ]
        ),
        "cm_sube_sube": int(
            matriz.loc[
                "real_SUBE",
                "predicha_SUBE",
            ]
        ),
        "notas": notas,
        "huella_resultados": huella,
    }

    detalle = {
        "registro": registro,
        "columnas_modelo": list(
            paquete["columnas_modelo"]
        ),
        "clases": list(
            paquete["clases"]
        ),
        "conteos_entrenamiento": conteos,
        "pesos_clases": pesos,
        "metricas_modelo": metricas_modelo,
        "metricas_dummy": metricas_dummy,
        "informe_clasificacion": informe,
        "matriz_confusion": matriz.to_dict(),
        "matriz_confusion_dummy": matriz_dummy.to_dict(),
        "archivos_origen": [
            str(ruta)
            for ruta in rutas_resultado
        ],
    }

    return registro, detalle


def guardar_historial(
    registro: dict[str, Any],
    detalle: dict[str, Any],
) -> bool:
    """
    Añade el entrenamiento al historial.

    Devuelve False cuando los mismos resultados ya estaban registrados.
    """

    RUTA_MODELOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUTA_DETALLES.mkdir(
        parents=True,
        exist_ok=True,
    )

    if RUTA_HISTORIAL.exists():
        historial = pd.read_csv(
            RUTA_HISTORIAL
        )

        misma_huella = (
            historial[
                "huella_resultados"
            ].astype(str)
            == str(
                registro[
                    "huella_resultados"
                ]
            )
        )

        if misma_huella.any():
            print(
                "\nEstos resultados ya estaban registrados."
            )

            generar_resumen_markdown(
                historial=historial
            )

            return False

        historial = pd.concat(
            [
                historial,
                pd.DataFrame([registro]),
            ],
            ignore_index=True,
        )

    else:
        historial = pd.DataFrame(
            [registro]
        )

    historial = historial.sort_values(
        [
            "fecha_utc",
            "simbolo",
        ]
    ).reset_index(drop=True)

    historial.to_csv(
        RUTA_HISTORIAL,
        index=False,
        encoding="utf-8-sig",
    )

    ruta_detalle = (
        RUTA_DETALLES
        / f"{registro['id_entrenamiento']}.json"
    )

    with ruta_detalle.open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            detalle,
            archivo,
            ensure_ascii=False,
            indent=4,
        )

    generar_resumen_markdown(
        historial=historial
    )

    print(
        "\nEntrenamiento registrado correctamente."
    )

    print(
        f"Historial: {RUTA_HISTORIAL}"
    )

    print(
        f"Resumen: {RUTA_RESUMEN}"
    )

    print(
        f"Detalle: {ruta_detalle}"
    )

    return True


def generar_resumen_markdown(
    historial: pd.DataFrame,
) -> None:
    """Genera un resumen legible de la evolución de los modelos."""

    historial = historial.copy()

    historial = historial.sort_values(
        [
            "fecha_utc",
            "simbolo",
        ]
    ).reset_index(drop=True)

    lineas = [
        "# Evolución de entrenamientos",
        "",
        "Este archivo resume los entrenamientos registrados "
        "para los modelos cripto de corto plazo.",
        "",
        "## Resumen general",
        "",
        "| Fecha UTC | Símbolo | Etiqueta | Accuracy | Balanced Accuracy | F1-score macro | Δ Balanced Accuracy | Δ F1 macro |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]

    for _, fila in historial.iterrows():
        lineas.append(
            "| "
            f"{fila['fecha_utc']} | "
            f"{fila['simbolo']} | "
            f"{fila['etiqueta']} | "
            f"{float(fila['accuracy']):.4f} | "
            f"{float(fila['balanced_accuracy']):.4f} | "
            f"{float(fila['f1_macro']):.4f} | "
            f"{float(fila['mejora_balanced_accuracy']):+.4f} | "
            f"{float(fila['mejora_f1_macro']):+.4f} |"
        )

    for _, fila in historial.iterrows():
        lineas.extend(
            [
                "",
                "---",
                "",
                f"## {fila['simbolo']} — {fila['etiqueta']}",
                "",
                f"- **ID:** `{fila['id_entrenamiento']}`",
                f"- **Fecha UTC:** {fila['fecha_utc']}",
                f"- **Modelo:** {fila['modelo']}",
                f"- **Horizonte:** {fila['horizonte']}",
                f"- **Umbral:** ±{float(fila['umbral_porcentaje']):.2f} %",
                f"- **Variables:** {int(fila['variables'])}",
                f"- **Épocas:** {int(fila['epocas'])}",
                f"- **Tamaño de lote:** {int(fila['tamano_lote']):,}".replace(",", "."),
                f"- **Muestras de entrenamiento:** {int(fila['muestras_entrenamiento']):,}".replace(",", "."),
                f"- **Muestras de validación:** {int(fila['muestras_validacion']):,}".replace(",", "."),
                "",
                "### Métricas principales",
                "",
                "| Modelo | Accuracy | Balanced Accuracy | F1-score macro |",
                "|---|---:|---:|---:|",
                "| SGDClassifier | "
                f"{float(fila['accuracy']):.4f} | "
                f"{float(fila['balanced_accuracy']):.4f} | "
                f"{float(fila['f1_macro']):.4f} |",
                "| DummyClassifier (most_frequent) | "
                f"{float(fila['dummy_accuracy']):.4f} | "
                f"{float(fila['dummy_balanced_accuracy']):.4f} | "
                f"{float(fila['dummy_f1_macro']):.4f} |",
                "",
                "### Métricas por clase",
                "",
                "| Clase | Precision | Recall | F1-score | Support |",
                "|---|---:|---:|---:|---:|",
                "| BAJA | "
                f"{float(fila['precision_baja']):.4f} | "
                f"{float(fila['recall_baja']):.4f} | "
                f"{float(fila['f1_baja']):.4f} | "
                f"{int(fila['support_baja']):,} |".replace(",", "."),
                "| NEUTRAL | "
                f"{float(fila['precision_neutral']):.4f} | "
                f"{float(fila['recall_neutral']):.4f} | "
                f"{float(fila['f1_neutral']):.4f} | "
                f"{int(fila['support_neutral']):,} |".replace(",", "."),
                "| SUBE | "
                f"{float(fila['precision_sube']):.4f} | "
                f"{float(fila['recall_sube']):.4f} | "
                f"{float(fila['f1_sube']):.4f} | "
                f"{int(fila['support_sube']):,} |".replace(",", "."),
                "",
                "### Matriz de confusión",
                "",
                "| Real \\ Predicha | BAJA | NEUTRAL | SUBE |",
                "|---|---:|---:|---:|",
                "| BAJA | "
                f"{int(fila['cm_baja_baja']):,} | "
                f"{int(fila['cm_baja_neutral']):,} | "
                f"{int(fila['cm_baja_sube']):,} |".replace(",", "."),
                "| NEUTRAL | "
                f"{int(fila['cm_neutral_baja']):,} | "
                f"{int(fila['cm_neutral_neutral']):,} | "
                f"{int(fila['cm_neutral_sube']):,} |".replace(",", "."),
                "| SUBE | "
                f"{int(fila['cm_sube_baja']):,} | "
                f"{int(fila['cm_sube_neutral']):,} | "
                f"{int(fila['cm_sube_sube']):,} |".replace(",", "."),
            ]
        )

        notas = str(
            fila.get(
                "notas",
                "",
            )
        ).strip()

        if notas and notas.lower() != "nan":
            lineas.extend(
                [
                    "",
                    "### Notas",
                    "",
                    notas,
                ]
            )

    RUTA_RESUMEN.write_text(
        "\n".join(lineas) + "\n",
        encoding="utf-8",
    )


def registrar_entrenamiento(
    simbolo: str,
    etiqueta: str,
    epocas: int,
    tamano_lote: int,
    notas: str = "",
) -> bool:
    """Registra el último resultado generado para un símbolo."""

    simbolo = simbolo.upper().strip()

    if simbolo not in {
        "BTCUSDT",
        "ETHUSDT",
    }:
        raise ValueError(
            f"Símbolo no permitido: {simbolo}"
        )

    if epocas <= 0:
        raise ValueError(
            "Las épocas deben ser mayores que cero."
        )

    if tamano_lote <= 0:
        raise ValueError(
            "El tamaño de lote debe ser mayor que cero."
        )

    registro, detalle = construir_registro(
        simbolo=simbolo,
        etiqueta=etiqueta,
        epocas=epocas,
        tamano_lote=tamano_lote,
        notas=notas,
    )

    return guardar_historial(
        registro=registro,
        detalle=detalle,
    )


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene los argumentos del comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Registra las métricas y el detalle de "
            "un entrenamiento de corto plazo."
        )
    )

    parser.add_argument(
        "--simbolo",
        required=True,
        choices=(
            "BTCUSDT",
            "ETHUSDT",
        ),
    )

    parser.add_argument(
        "--etiqueta",
        required=True,
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
        "--notas",
        default="",
    )

    return parser.parse_args()


def main() -> None:
    """Punto de entrada."""

    argumentos = obtener_argumentos()

    try:
        registrar_entrenamiento(
            simbolo=argumentos.simbolo,
            etiqueta=argumentos.etiqueta,
            epocas=argumentos.epocas,
            tamano_lote=argumentos.tamano_lote,
            notas=argumentos.notas,
        )

    except (
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(
            "\nNo se pudo registrar el entrenamiento."
        )

        print(
            f"Detalle: {error}"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
