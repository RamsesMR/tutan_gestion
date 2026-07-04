from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from cripto.corto_plazo.gestionar_para_analisis import (
    actualizar_carpeta_para_analisis,
)
from cripto.corto_plazo.validar_estabilidad_mensual import (
    CLASES,
    CODIGOS_CLASES,
    aplicar_decision_ajustada,
    aplicar_decision_original,
    calcular_metricas_periodo,
    clasificar_objetivo,
    crear_comparacion,
    recorrer_lotes,
)


SIMBOLOS = (
    "BTCUSDT",
    "ETHUSDT",
)

RUTA_MODELOS = (
    RUTA_PROYECTO
    / "modelos_entrenados"
    / "cripto"
    / "corto_plazo"
)

RUTA_MANIFIESTO = (
    RUTA_DATOS_PREPARADOS
    / "manifiesto_divisiones_4h.csv"
)

RUTA_CONFIGURACION_CONGELADA = (
    RUTA_MODELOS
    / "configuracion_final_congelada_2026.json"
)

RUTA_RESULTADO_FINAL = (
    RUTA_MODELOS
    / "evaluacion_final_2026.json"
)

RUTA_METRICAS_FINAL = (
    RUTA_MODELOS
    / "metricas_evaluacion_final_2026.csv"
)

RUTA_COMPARACION_MENSUAL = (
    RUTA_MODELOS
    / "comparacion_mensual_evaluacion_final_2026.csv"
)

RUTA_RESUMEN_FINAL = (
    RUTA_MODELOS
    / "resumen_evaluacion_final_2026.md"
)


def calcular_sha256(
    ruta: Path,
    tamano_bloque: int = 1024 * 1024,
) -> str:
    """Calcula la huella SHA-256 de un archivo."""

    calculador = hashlib.sha256()

    with ruta.open("rb") as archivo:
        while True:
            bloque = archivo.read(
                tamano_bloque
            )

            if not bloque:
                break

            calculador.update(
                bloque
            )

    return calculador.hexdigest()


def convertir_serializable(
    valor: Any,
) -> Any:
    """Convierte tipos de NumPy y pandas a tipos compatibles con JSON."""

    if isinstance(
        valor,
        dict,
    ):
        return {
            str(clave): convertir_serializable(
                contenido
            )
            for clave, contenido in valor.items()
        }

    if isinstance(
        valor,
        (list, tuple),
    ):
        return [
            convertir_serializable(
                contenido
            )
            for contenido in valor
        ]

    if isinstance(
        valor,
        np.ndarray,
    ):
        return convertir_serializable(
            valor.tolist()
        )

    if isinstance(
        valor,
        (np.integer,),
    ):
        return int(
            valor
        )

    if isinstance(
        valor,
        (np.floating,),
    ):
        return float(
            valor
        )

    if isinstance(
        valor,
        (pd.Timestamp, datetime),
    ):
        return valor.isoformat()

    if pd.isna(
        valor
    ):
        return None

    return valor


def cargar_json(
    ruta: Path,
) -> dict[str, Any]:
    """Carga un objeto JSON."""

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {ruta}"
        )

    with ruta.open(
        "r",
        encoding="utf-8",
    ) as archivo:
        datos = json.load(
            archivo
        )

    if not isinstance(
        datos,
        dict,
    ):
        raise ValueError(
            f"El archivo no contiene un objeto JSON válido: {ruta}"
        )

    return datos


def cargar_configuracion_congelada() -> dict[str, Any]:
    """Carga la configuración congelada y comprueba su estado."""

    configuracion = cargar_json(
        RUTA_CONFIGURACION_CONGELADA
    )

    if configuracion.get(
        "estado"
    ) != "configuracion_congelada_antes_prueba":
        raise ValueError(
            "La configuración no tiene el estado de congelación esperado."
        )

    if configuracion.get(
        "division_prueba_utilizada"
    ) is not False:
        raise ValueError(
            "La configuración congelada no confirma que la prueba "
            "permanecía sin utilizar."
        )

    simbolos = configuracion.get(
        "simbolos"
    )

    if not isinstance(
        simbolos,
        list,
    ):
        raise ValueError(
            "La configuración congelada no contiene la lista de símbolos."
        )

    simbolos_encontrados = {
        str(
            datos.get(
                "simbolo",
                "",
            )
        )
        for datos in simbolos
        if isinstance(
            datos,
            dict,
        )
    }

    if simbolos_encontrados != set(
        SIMBOLOS
    ):
        raise ValueError(
            "La configuración congelada debe contener exactamente "
            "BTCUSDT y ETHUSDT."
        )

    return configuracion


def obtener_configuracion_simbolo(
    configuracion: dict[str, Any],
    simbolo: str,
) -> dict[str, Any]:
    """Obtiene la configuración congelada correspondiente a un símbolo."""

    for datos in configuracion[
        "simbolos"
    ]:
        if datos.get(
            "simbolo"
        ) == simbolo:
            return datos

    raise ValueError(
        f"No existe configuración congelada para {simbolo}."
    )


def verificar_integridad(
    datos_congelados: dict[str, Any],
) -> tuple[Path, Path]:
    """Verifica que el modelo y el ajuste no cambiaron después de congelarse."""

    simbolo = str(
        datos_congelados[
            "simbolo"
        ]
    )

    ruta_modelo = Path(
        datos_congelados[
            "modelo"
        ][
            "archivo"
        ]
    )

    ruta_ajuste = Path(
        datos_congelados[
            "ajuste_decision"
        ][
            "archivo"
        ]
    )

    for ruta in (
        ruta_modelo,
        ruta_ajuste,
    ):
        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe el archivo congelado de {simbolo}: {ruta}"
            )

    huella_modelo_actual = calcular_sha256(
        ruta_modelo
    )

    huella_ajuste_actual = calcular_sha256(
        ruta_ajuste
    )

    if huella_modelo_actual != datos_congelados[
        "modelo"
    ][
        "sha256"
    ]:
        raise ValueError(
            f"El modelo de {simbolo} cambió después de congelarse."
        )

    if huella_ajuste_actual != datos_congelados[
        "ajuste_decision"
    ][
        "sha256"
    ]:
        raise ValueError(
            f"El ajuste de {simbolo} cambió después de congelarse."
        )

    return (
        ruta_modelo,
        ruta_ajuste,
    )


def obtener_rutas_prueba(
    simbolo: str,
    datos_congelados: dict[str, Any],
) -> list[Path]:
    """Obtiene únicamente los archivos declarados como prueba."""

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

    faltantes = columnas_requeridas.difference(
        manifiesto.columns
    )

    if faltantes:
        raise ValueError(
            "Faltan columnas en el manifiesto: "
            + ", ".join(
                sorted(faltantes)
            )
        )

    registros = (
        manifiesto
        .loc[
            (manifiesto["simbolo"] == simbolo)
            & (manifiesto["division"] == "prueba")
        ]
        .sort_values(
            "desde_archivo"
        )
    )

    if registros.empty:
        raise ValueError(
            f"No existen archivos de prueba para {simbolo}."
        )

    rutas: list[
        Path
    ] = []

    for archivo in registros[
        "archivo"
    ]:
        nombre_archivo = (
            str(
                archivo
            )
            .replace(
                "\\",
                "/",
            )
            .rsplit(
                "/",
                1,
            )[-1]
        )

        ruta = (
            RUTA_DATOS_PREPARADOS
            / nombre_archivo
        )

        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe el archivo de prueba: {ruta}"
            )

        rutas.append(
            ruta
        )

    archivo_declarado = (
        str(
            datos_congelados[
                "periodo_prueba_reservado"
            ][
                "archivo_declarado"
            ]
        )
        .replace(
            "\\",
            "/",
        )
        .rsplit(
            "/",
            1,
        )[-1]
    )

    nombres_encontrados = {
        ruta.name
        for ruta in rutas
    }

    if archivo_declarado not in nombres_encontrados:
        raise ValueError(
            f"El archivo de prueba congelado de {simbolo} "
            "no coincide con el manifiesto actual."
        )

    return rutas


def cargar_prueba(
    rutas: list[Path],
    paquete: dict[str, Any],
    datos_congelados: dict[str, Any],
    tamano_lote: int,
) -> tuple[
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    """Abre la prueba una sola vez y genera probabilidades sin ajustar nada."""

    modelo = paquete[
        "modelo"
    ]

    escalador = paquete[
        "escalador"
    ]

    columnas_modelo = [
        str(
            columna
        )
        for columna in paquete[
            "columnas_modelo"
        ]
    ]

    columnas_congeladas = [
        str(
            columna
        )
        for columna in datos_congelados[
            "modelo"
        ][
            "columnas_modelo"
        ]
    ]

    if columnas_modelo != columnas_congeladas:
        raise ValueError(
            f"Las columnas actuales de {datos_congelados['simbolo']} "
            "no coinciden con la configuración congelada."
        )

    umbral = float(
        paquete[
            "umbral_clase"
        ]
    )

    clases_modelo = [
        str(
            clase
        )
        for clase in modelo.classes_
    ]

    if set(
        clases_modelo
    ) != set(
        CLASES
    ):
        raise ValueError(
            "Las clases del modelo no coinciden con BAJA, NEUTRAL y SUBE."
        )

    orden_columnas = [
        clases_modelo.index(
            str(
                clase
            )
        )
        for clase in CLASES
    ]

    fechas_totales: list[
        np.ndarray
    ] = []

    reales_totales: list[
        np.ndarray
    ] = []

    probabilidades_totales: list[
        np.ndarray
    ] = []

    archivos_utilizados: list[
        dict[str, Any]
    ] = []

    print(
        f"\nABRIENDO PRUEBA FINAL DE {datos_congelados['simbolo']}"
    )
    print("=" * 70)

    for ruta in rutas:
        print(
            f"Procesando: {ruta.name}"
        )

        columnas = [
            "fecha_apertura",
            *columnas_modelo,
            "rendimiento_objetivo",
        ]

        datos = pd.read_parquet(
            ruta,
            columns=columnas,
        )

        fechas = pd.to_datetime(
            datos[
                "fecha_apertura"
            ],
            utc=True,
            errors="coerce",
        )

        if fechas.isna().any():
            raise ValueError(
                f"Existen fechas inválidas en: {ruta}"
            )

        variables = (
            datos[
                columnas_modelo
            ]
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
            umbral=umbral,
        )

        fechas_totales.append(
            fechas.to_numpy()
        )

        reales_totales.append(
            reales
        )

        for inicio, final in recorrer_lotes(
            total=len(
                variables
            ),
            tamano_lote=tamano_lote,
        ):
            variables_lote = escalador.transform(
                variables[
                    inicio:final
                ]
            )

            probabilidades_lote = modelo.predict_proba(
                variables_lote
            )[
                :,
                orden_columnas,
            ]

            probabilidades_totales.append(
                probabilidades_lote.astype(
                    "float32"
                )
            )

        archivos_utilizados.append(
            {
                "archivo": str(
                    ruta
                ),
                "filas": len(
                    datos
                ),
                "bytes": ruta.stat().st_size,
                "sha256": calcular_sha256(
                    ruta
                ),
            }
        )

        del datos
        del variables
        del reales
        gc.collect()

    fechas = pd.to_datetime(
        np.concatenate(
            fechas_totales
        ),
        utc=True,
    )

    reales = np.concatenate(
        reales_totales
    )

    probabilidades = np.concatenate(
        probabilidades_totales
    )

    orden = np.argsort(
        fechas.asi8
    )

    fechas = fechas[
        orden
    ]

    reales = reales[
        orden
    ]

    probabilidades = probabilidades[
        orden
    ]

    if fechas.duplicated().any():
        raise ValueError(
            f"La prueba de {datos_congelados['simbolo']} "
            "contiene fechas duplicadas."
        )

    periodo = datos_congelados[
        "periodo_prueba_reservado"
    ]

    desde = pd.to_datetime(
        periodo[
            "desde"
        ],
        utc=True,
    )

    hasta = pd.to_datetime(
        periodo[
            "hasta_exclusivo"
        ],
        utc=True,
    )

    if fechas.min() < desde:
        raise ValueError(
            f"La prueba de {datos_congelados['simbolo']} "
            "empieza antes del periodo congelado."
        )

    if fechas.max() >= hasta:
        raise ValueError(
            f"La prueba de {datos_congelados['simbolo']} "
            "termina fuera del periodo congelado."
        )

    filas_esperadas = int(
        periodo[
            "filas_utilizables"
        ]
    )

    if len(
        fechas
    ) != filas_esperadas:
        raise ValueError(
            f"La prueba de {datos_congelados['simbolo']} contiene "
            f"{len(fechas)} filas, pero se congelaron {filas_esperadas}."
        )

    return (
        fechas,
        reales,
        probabilidades,
        archivos_utilizados,
    )


def calcular_metricas_globales(
    reales: np.ndarray,
    predicciones: np.ndarray,
) -> dict[str, Any]:
    """Calcula métricas globales, por clase y matriz de confusión."""

    informe = classification_report(
        reales,
        predicciones,
        labels=CODIGOS_CLASES,
        target_names=[
            str(
                clase
            )
            for clase in CLASES
        ],
        output_dict=True,
        zero_division=0,
    )

    matriz = confusion_matrix(
        reales,
        predicciones,
        labels=CODIGOS_CLASES,
    )

    muestras = len(
        reales
    )

    reales_por_clase = {
        str(
            clase
        ): int(
            (
                reales
                == codigo
            ).sum()
        )
        for clase, codigo in zip(
            CLASES,
            CODIGOS_CLASES,
        )
    }

    predicciones_por_clase = {
        str(
            clase
        ): int(
            (
                predicciones
                == codigo
            ).sum()
        )
        for clase, codigo in zip(
            CLASES,
            CODIGOS_CLASES,
        )
    }

    return {
        "muestras": muestras,
        "accuracy": float(
            accuracy_score(
                reales,
                predicciones,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                reales,
                predicciones,
            )
        ),
        "f1_macro": float(
            f1_score(
                reales,
                predicciones,
                labels=CODIGOS_CLASES,
                average="macro",
                zero_division=0,
            )
        ),
        "metricas_por_clase": {
            str(
                clase
            ): {
                "precision": float(
                    informe[
                        str(
                            clase
                        )
                    ][
                        "precision"
                    ]
                ),
                "recall": float(
                    informe[
                        str(
                            clase
                        )
                    ][
                        "recall"
                    ]
                ),
                "f1_score": float(
                    informe[
                        str(
                            clase
                        )
                    ][
                        "f1-score"
                    ]
                ),
                "support": int(
                    informe[
                        str(
                            clase
                        )
                    ][
                        "support"
                    ]
                ),
            }
            for clase in CLASES
        },
        "distribucion_real": {
            clase: {
                "cantidad": cantidad,
                "porcentaje": (
                    cantidad
                    / muestras
                    * 100
                ),
            }
            for clase, cantidad in reales_por_clase.items()
        },
        "distribucion_predicha": {
            clase: {
                "cantidad": cantidad,
                "porcentaje": (
                    cantidad
                    / muestras
                    * 100
                ),
            }
            for clase, cantidad in predicciones_por_clase.items()
        },
        "matriz_confusion": {
            "filas_reales": [
                str(
                    clase
                )
                for clase in CLASES
            ],
            "columnas_predichas": [
                str(
                    clase
                )
                for clase in CLASES
            ],
            "valores": matriz.tolist(),
        },
    }


def evaluar_por_mes(
    fechas: pd.DatetimeIndex,
    reales: np.ndarray,
    predicciones_originales: np.ndarray,
    predicciones_ajustadas: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula las métricas original y ajustada para cada mes de prueba."""

    periodos = (
        fechas
        .tz_convert(
            None
        )
        .to_period(
            "M"
        )
        .astype(
            str
        )
    )

    resultados: list[
        dict[str, Any]
    ] = []

    for mes in sorted(
        np.unique(
            periodos
        )
    ):
        mascara = (
            periodos
            == mes
        )

        reales_mes = reales[
            mascara
        ]

        resultados.append(
            calcular_metricas_periodo(
                mes=mes,
                tipo_decision="original",
                reales=reales_mes,
                predicciones=predicciones_originales[
                    mascara
                ],
            )
        )

        resultados.append(
            calcular_metricas_periodo(
                mes=mes,
                tipo_decision="ajustada_congelada",
                reales=reales_mes,
                predicciones=predicciones_ajustadas[
                    mascara
                ],
            )
        )

    detallados = pd.DataFrame(
        resultados
    )

    comparacion = crear_comparacion(
        detallados.replace(
            {
                "ajustada_congelada": "ajustada",
            }
        )
    )

    return (
        detallados,
        comparacion,
    )


def crear_fila_metricas(
    simbolo: str,
    tipo_decision: str,
    metricas: dict[str, Any],
) -> dict[str, Any]:
    """Convierte las métricas globales en una fila tabular."""

    return {
        "simbolo": simbolo,
        "tipo_decision": tipo_decision,
        "muestras": metricas[
            "muestras"
        ],
        "accuracy": metricas[
            "accuracy"
        ],
        "balanced_accuracy": metricas[
            "balanced_accuracy"
        ],
        "f1_macro": metricas[
            "f1_macro"
        ],
        "precision_baja": metricas[
            "metricas_por_clase"
        ][
            "BAJA"
        ][
            "precision"
        ],
        "recall_baja": metricas[
            "metricas_por_clase"
        ][
            "BAJA"
        ][
            "recall"
        ],
        "f1_baja": metricas[
            "metricas_por_clase"
        ][
            "BAJA"
        ][
            "f1_score"
        ],
        "precision_neutral": metricas[
            "metricas_por_clase"
        ][
            "NEUTRAL"
        ][
            "precision"
        ],
        "recall_neutral": metricas[
            "metricas_por_clase"
        ][
            "NEUTRAL"
        ][
            "recall"
        ],
        "f1_neutral": metricas[
            "metricas_por_clase"
        ][
            "NEUTRAL"
        ][
            "f1_score"
        ],
        "precision_sube": metricas[
            "metricas_por_clase"
        ][
            "SUBE"
        ][
            "precision"
        ],
        "recall_sube": metricas[
            "metricas_por_clase"
        ][
            "SUBE"
        ][
            "recall"
        ],
        "f1_sube": metricas[
            "metricas_por_clase"
        ][
            "SUBE"
        ][
            "f1_score"
        ],
    }


def evaluar_simbolo(
    simbolo: str,
    datos_congelados: dict[str, Any],
    tamano_lote: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    pd.DataFrame,
]:
    """Evalúa un símbolo sin modificar el modelo ni los multiplicadores."""

    ruta_modelo, _ = verificar_integridad(
        datos_congelados=datos_congelados
    )

    paquete = joblib.load(
        ruta_modelo
    )

    rutas_prueba = obtener_rutas_prueba(
        simbolo=simbolo,
        datos_congelados=datos_congelados,
    )

    (
        fechas,
        reales,
        probabilidades,
        archivos_utilizados,
    ) = cargar_prueba(
        rutas=rutas_prueba,
        paquete=paquete,
        datos_congelados=datos_congelados,
        tamano_lote=tamano_lote,
    )

    multiplicadores = {
        clase: float(
            datos_congelados[
                "ajuste_decision"
            ][
                "multiplicadores"
            ][
                clase
            ]
        )
        for clase in (
            "BAJA",
            "NEUTRAL",
            "SUBE",
        )
    }

    predicciones_originales = aplicar_decision_original(
        probabilidades=probabilidades
    )

    predicciones_ajustadas = aplicar_decision_ajustada(
        probabilidades=probabilidades,
        multiplicadores=multiplicadores,
    )

    metricas_originales = calcular_metricas_globales(
        reales=reales,
        predicciones=predicciones_originales,
    )

    metricas_ajustadas = calcular_metricas_globales(
        reales=reales,
        predicciones=predicciones_ajustadas,
    )

    (
        resultados_mensuales,
        comparacion_mensual,
    ) = evaluar_por_mes(
        fechas=fechas,
        reales=reales,
        predicciones_originales=predicciones_originales,
        predicciones_ajustadas=predicciones_ajustadas,
    )

    comparacion_mensual.insert(
        0,
        "simbolo",
        simbolo,
    )

    resultado = {
        "simbolo": simbolo,
        "horizonte": "4h",
        "periodo_evaluado": {
            "desde": fechas.min().isoformat(),
            "hasta": fechas.max().isoformat(),
            "muestras": len(
                fechas
            ),
        },
        "multiplicadores_congelados": multiplicadores,
        "archivos_prueba": archivos_utilizados,
        "decision_original": metricas_originales,
        "decision_ajustada_congelada": metricas_ajustadas,
        "cambios_ajustada_menos_original": {
            "accuracy": (
                metricas_ajustadas[
                    "accuracy"
                ]
                - metricas_originales[
                    "accuracy"
                ]
            ),
            "balanced_accuracy": (
                metricas_ajustadas[
                    "balanced_accuracy"
                ]
                - metricas_originales[
                    "balanced_accuracy"
                ]
            ),
            "f1_macro": (
                metricas_ajustadas[
                    "f1_macro"
                ]
                - metricas_originales[
                    "f1_macro"
                ]
            ),
        },
        "resultados_mensuales": resultados_mensuales.to_dict(
            orient="records"
        ),
        "comparacion_mensual": comparacion_mensual.to_dict(
            orient="records"
        ),
    }

    filas_metricas = [
        crear_fila_metricas(
            simbolo=simbolo,
            tipo_decision="original",
            metricas=metricas_originales,
        ),
        crear_fila_metricas(
            simbolo=simbolo,
            tipo_decision="ajustada_congelada",
            metricas=metricas_ajustadas,
        ),
    ]

    return (
        resultado,
        filas_metricas,
        comparacion_mensual,
    )


def nombre_mes(
    periodo: str,
) -> str:
    """Convierte YYYY-MM en un nombre legible."""

    fecha = pd.Period(
        periodo,
        freq="M",
    )

    meses = {
        1: "enero",
        2: "febrero",
        3: "marzo",
        4: "abril",
        5: "mayo",
        6: "junio",
        7: "julio",
        8: "agosto",
        9: "septiembre",
        10: "octubre",
        11: "noviembre",
        12: "diciembre",
    }

    return (
        f"{meses[fecha.month]} "
        f"{fecha.year}"
    )


def generar_resumen(
    resultado_final: dict[str, Any],
    metricas: pd.DataFrame,
    comparacion_mensual: pd.DataFrame,
    ruta: Path,
) -> None:
    """Genera el resumen legible de la evaluación final."""

    lineas = [
        "# Evaluación final única — prueba 2026",
        "",
        f"- **Fecha UTC:** {resultado_final['fecha_utc']}",
        "- **Estado:** evaluación final completada.",
        "- **Configuración:** congelada antes de abrir la prueba.",
        "- **Ajustes posteriores sobre 2026:** ninguno.",
        "",
        "## Resultado global",
        "",
        "| Símbolo | Decisión | Accuracy | Balanced Accuracy | F1-score macro |",
        "|---|---|---:|---:|---:|",
    ]

    for _, fila in metricas.iterrows():
        nombre_decision = (
            "Original"
            if fila[
                "tipo_decision"
            ] == "original"
            else "Ajustada congelada"
        )

        lineas.append(
            "| "
            f"{fila['simbolo']} | "
            f"{nombre_decision} | "
            f"{float(fila['accuracy']):.4f} | "
            f"{float(fila['balanced_accuracy']):.4f} | "
            f"{float(fila['f1_macro']):.4f} |"
        )

    lineas.extend(
        [
            "",
            "## Cambios de la decisión ajustada",
            "",
            "| Símbolo | Δ Accuracy | Δ Balanced Accuracy | Δ F1-score macro |",
            "|---|---:|---:|---:|",
        ]
    )

    for simbolo in SIMBOLOS:
        datos = next(
            resultado
            for resultado in resultado_final[
                "resultados"
            ]
            if resultado[
                "simbolo"
            ] == simbolo
        )

        cambios = datos[
            "cambios_ajustada_menos_original"
        ]

        lineas.append(
            "| "
            f"{simbolo} | "
            f"{float(cambios['accuracy']):+.4f} | "
            f"{float(cambios['balanced_accuracy']):+.4f} | "
            f"{float(cambios['f1_macro']):+.4f} |"
        )

    for simbolo in SIMBOLOS:
        datos = next(
            resultado
            for resultado in resultado_final[
                "resultados"
            ]
            if resultado[
                "simbolo"
            ] == simbolo
        )

        ajustadas = datos[
            "decision_ajustada_congelada"
        ]

        lineas.extend(
            [
                "",
                f"## {simbolo} — métricas por clase ajustadas",
                "",
                "| Clase | Precision | Recall | F1-score | Support |",
                "|---|---:|---:|---:|---:|",
            ]
        )

        for clase in (
            "BAJA",
            "NEUTRAL",
            "SUBE",
        ):
            clase_datos = ajustadas[
                "metricas_por_clase"
            ][
                clase
            ]

            lineas.append(
                "| "
                f"{clase} | "
                f"{float(clase_datos['precision']):.4f} | "
                f"{float(clase_datos['recall']):.4f} | "
                f"{float(clase_datos['f1_score']):.4f} | "
                f"{int(clase_datos['support']):,}".replace(
                    ",",
                    ".",
                )
                + " |"
            )

        lineas.extend(
            [
                "",
                f"## {simbolo} — comparación mensual",
                "",
                "| Mes | Accuracy original | Accuracy ajustada | Δ Accuracy | Balanced original | Balanced ajustada | Δ Balanced | F1 original | F1 ajustada | Δ F1 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )

        filas_simbolo = comparacion_mensual.loc[
            comparacion_mensual[
                "simbolo"
            ]
            == simbolo
        ]

        for _, fila in filas_simbolo.iterrows():
            lineas.append(
                "| "
                f"{nombre_mes(str(fila['mes']))} | "
                f"{float(fila['accuracy_original']):.4f} | "
                f"{float(fila['accuracy_ajustada']):.4f} | "
                f"{float(fila['delta_accuracy']):+.4f} | "
                f"{float(fila['balanced_accuracy_original']):.4f} | "
                f"{float(fila['balanced_accuracy_ajustada']):.4f} | "
                f"{float(fila['delta_balanced_accuracy']):+.4f} | "
                f"{float(fila['f1_macro_original']):.4f} | "
                f"{float(fila['f1_macro_ajustada']):.4f} | "
                f"{float(fila['delta_f1_macro']):+.4f} |"
            )

    lineas.extend(
        [
            "",
            "## Regla posterior",
            "",
            "Los resultados de 2026 no deben utilizarse para modificar esta "
            "misma configuración y volver a medirla sobre el mismo periodo.",
            "",
            "Cualquier modelo nuevo deberá evaluarse con un periodo temporal "
            "posterior que permanezca sin utilizar durante su desarrollo.",
            "",
        ]
    )

    ruta.write_text(
        "\n".join(
            lineas
        ),
        encoding="utf-8",
    )


def escribir_archivos_atomicos(
    resultado_final: dict[str, Any],
    metricas: pd.DataFrame,
    comparacion_mensual: pd.DataFrame,
) -> None:
    """Escribe todos los resultados y deja el JSON principal para el final."""

    RUTA_MODELOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    rutas_temporales = {
        "metricas": RUTA_METRICAS_FINAL.with_suffix(
            ".csv.tmp"
        ),
        "comparacion": RUTA_COMPARACION_MENSUAL.with_suffix(
            ".csv.tmp"
        ),
        "resumen": RUTA_RESUMEN_FINAL.with_suffix(
            ".md.tmp"
        ),
        "resultado": RUTA_RESULTADO_FINAL.with_suffix(
            ".json.tmp"
        ),
    }

    try:
        metricas.to_csv(
            rutas_temporales[
                "metricas"
            ],
            index=False,
            encoding="utf-8-sig",
        )

        comparacion_mensual.to_csv(
            rutas_temporales[
                "comparacion"
            ],
            index=False,
            encoding="utf-8-sig",
        )

        generar_resumen(
            resultado_final=resultado_final,
            metricas=metricas,
            comparacion_mensual=comparacion_mensual,
            ruta=rutas_temporales[
                "resumen"
            ],
        )

        with rutas_temporales[
            "resultado"
        ].open(
            "w",
            encoding="utf-8",
        ) as archivo:
            json.dump(
                convertir_serializable(
                    resultado_final
                ),
                archivo,
                ensure_ascii=False,
                indent=4,
            )

        os.replace(
            rutas_temporales[
                "metricas"
            ],
            RUTA_METRICAS_FINAL,
        )

        os.replace(
            rutas_temporales[
                "comparacion"
            ],
            RUTA_COMPARACION_MENSUAL,
        )

        os.replace(
            rutas_temporales[
                "resumen"
            ],
            RUTA_RESUMEN_FINAL,
        )

        os.replace(
            rutas_temporales[
                "resultado"
            ],
            RUTA_RESULTADO_FINAL,
        )

    finally:
        for ruta in rutas_temporales.values():
            ruta.unlink(
                missing_ok=True
            )


def ejecutar_evaluacion_final(
    tamano_lote: int,
) -> None:
    """Evalúa ambos símbolos una sola vez con la configuración congelada."""

    if RUTA_RESULTADO_FINAL.exists():
        raise FileExistsError(
            "La evaluación final ya fue completada. "
            f"Archivo existente: {RUTA_RESULTADO_FINAL}"
        )

    configuracion = cargar_configuracion_congelada()

    print(
        "\nEVALUACIÓN FINAL ÚNICA — PRUEBA 2026"
    )
    print("=" * 70)
    print(
        "Se verificarán las huellas congeladas antes de abrir la prueba."
    )
    print(
        "No se buscarán multiplicadores ni se modificará el modelo."
    )

    resultados: list[
        dict[str, Any]
    ] = []

    filas_metricas: list[
        dict[str, Any]
    ] = []

    comparaciones: list[
        pd.DataFrame
    ] = []

    for simbolo in SIMBOLOS:
        datos_congelados = obtener_configuracion_simbolo(
            configuracion=configuracion,
            simbolo=simbolo,
        )

        (
            resultado,
            filas,
            comparacion,
        ) = evaluar_simbolo(
            simbolo=simbolo,
            datos_congelados=datos_congelados,
            tamano_lote=tamano_lote,
        )

        resultados.append(
            resultado
        )

        filas_metricas.extend(
            filas
        )

        comparaciones.append(
            comparacion
        )

    metricas = pd.DataFrame(
        filas_metricas
    )

    comparacion_mensual = pd.concat(
        comparaciones,
        ignore_index=True,
    )

    resultado_final = {
        "fecha_utc": datetime.now(
            timezone.utc
        ).isoformat(
            timespec="seconds"
        ),
        "estado": "evaluacion_final_2026_completada",
        "configuracion_congelada": {
            "archivo": str(
                RUTA_CONFIGURACION_CONGELADA
            ),
            "sha256": calcular_sha256(
                RUTA_CONFIGURACION_CONGELADA
            ),
            "fecha_congelacion_utc": configuracion[
                "fecha_utc"
            ],
        },
        "ajustes_realizados_con_prueba": False,
        "entrenamiento_realizado_con_prueba": False,
        "resultados": resultados,
    }

    escribir_archivos_atomicos(
        resultado_final=resultado_final,
        metricas=metricas,
        comparacion_mensual=comparacion_mensual,
    )

    archivos = actualizar_carpeta_para_analisis()

    print(
        "\nEVALUACIÓN FINAL COMPLETADA"
    )
    print("=" * 70)

    for simbolo in SIMBOLOS:
        resultado = next(
            datos
            for datos in resultados
            if datos[
                "simbolo"
            ] == simbolo
        )

        originales = resultado[
            "decision_original"
        ]

        ajustadas = resultado[
            "decision_ajustada_congelada"
        ]

        print(
            f"\n{simbolo}"
        )
        print(
            f"Original  | Accuracy {originales['accuracy']:.4f} | "
            f"Balanced {originales['balanced_accuracy']:.4f} | "
            f"F1 macro {originales['f1_macro']:.4f}"
        )
        print(
            f"Ajustada  | Accuracy {ajustadas['accuracy']:.4f} | "
            f"Balanced {ajustadas['balanced_accuracy']:.4f} | "
            f"F1 macro {ajustadas['f1_macro']:.4f}"
        )

    print(
        "\nArchivos generados:"
    )
    print(
        f"- {RUTA_RESULTADO_FINAL}"
    )
    print(
        f"- {RUTA_METRICAS_FINAL}"
    )
    print(
        f"- {RUTA_COMPARACION_MENSUAL}"
    )
    print(
        f"- {RUTA_RESUMEN_FINAL}"
    )

    print(
        "\nCarpeta para_analisis actualizada:"
    )
    print(
        f"- {len(archivos)} archivos"
    )

    print(
        "\nLa prueba de 2026 ya fue utilizada y no debe reutilizarse "
        "para ajustar esta configuración."
    )


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene y valida los argumentos del comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Evalúa una sola vez los modelos congelados "
            "sobre la prueba reservada de 2026."
        )
    )

    parser.add_argument(
        "--confirmar-evaluacion-final",
        action="store_true",
        help=(
            "Confirma que se abrirá la prueba final reservada de 2026."
        ),
    )

    parser.add_argument(
        "--tamano-lote",
        type=int,
        default=100000,
    )

    argumentos = parser.parse_args()

    if not argumentos.confirmar_evaluacion_final:
        parser.error(
            "Debes incluir --confirmar-evaluacion-final."
        )

    if argumentos.tamano_lote <= 0:
        parser.error(
            "--tamano-lote debe ser mayor que cero."
        )

    return argumentos


def main() -> None:
    """Punto de entrada."""

    argumentos = obtener_argumentos()

    try:
        ejecutar_evaluacion_final(
            tamano_lote=argumentos.tamano_lote
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(
            "\nNo se pudo completar la evaluación final."
        )
        print(
            f"Detalle: {error}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
