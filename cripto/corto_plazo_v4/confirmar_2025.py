from __future__ import annotations

import argparse
import gc
import json
import uuid
from datetime import datetime, timezone

import joblib
import pandas as pd

from cripto.corto_plazo_v4.configuracion import (
    DETECTORES,
    PERIODO_CONFIRMACION_2025,
    PERFILES_UMBRALES,
    RUTA_CONFIRMACION_2025,
    RUTA_HISTORIAL_CONFIRMACION_2025,
    RUTA_SELECCION_UMBRALES,
    SIMBOLOS,
    VARIANTES,
    VERSION_MODELO,
)
from cripto.corto_plazo_v4.utilidades import (
    ajustar_escalador_y_contar,
    calcular_metricas_generales,
    calcular_pesos_binarios,
    construir_tabla_coeficientes,
    convertir_fechas_ns,
    entrenar_modelo_binario,
    evaluar_umbral,
    predecir_periodo,
)


def resolver_valores(
    valor: str,
    disponibles: tuple[str, ...],
    todos: str,
) -> list[str]:
    """Convierte TODO/AMBOS en una lista."""

    if valor == todos:
        return list(
            disponibles
        )

    return [
        valor
    ]


def ejecutar(
    perfil: str,
    simbolo_solicitado: str,
    detector_solicitado: str,
    epocas: int,
    tamano_lote: int,
) -> None:
    """Entrena 2021-2024 y confirma en 2025 sin retocar umbrales."""

    if not RUTA_SELECCION_UMBRALES.exists():
        raise FileNotFoundError(
            f"No existe: {RUTA_SELECCION_UMBRALES}"
        )

    seleccion = pd.read_csv(
        RUTA_SELECCION_UMBRALES
    )

    simbolos = resolver_valores(
        valor=simbolo_solicitado,
        disponibles=SIMBOLOS,
        todos="TODOS",
    )

    detectores = resolver_valores(
        valor=detector_solicitado,
        disponibles=DETECTORES,
        todos="AMBOS",
    )

    seleccion = seleccion.loc[
        (seleccion["perfil"] == perfil)
        & (
            seleccion["simbolo"].isin(
                simbolos
            )
        )
        & (
            seleccion["detector"].isin(
                detectores
            )
        )
    ].copy()

    if seleccion.empty:
        raise ValueError(
            "No existen configuraciones seleccionadas para esos filtros."
        )

    identificador = (
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        + "_"
        + uuid.uuid4().hex[
            :8
        ]
    )

    filas_historial = []

    print(
        "\nCONFIRMACIÓN SECUNDARIA V4 EN 2025"
    )
    print("=" * 72)
    print(
        f"Perfil: {perfil}"
    )
    print(
        "Los umbrales vienen congelados desde 2023/2024."
    )
    print(
        "2025 ya fue observado en versiones anteriores y no es "
        "una prueba totalmente intacta."
    )
    print(
        "2026 no será utilizado."
    )

    for _, fila_seleccion in seleccion.iterrows():
        simbolo = str(
            fila_seleccion[
                "simbolo"
            ]
        )

        detector = str(
            fila_seleccion[
                "detector"
            ]
        )

        nombre_variante = str(
            fila_seleccion[
                "variante"
            ]
        )

        umbral = float(
            fila_seleccion[
                "umbral"
            ]
        )

        configuracion_variante = VARIANTES[
            nombre_variante
        ]

        columnas_modelo = tuple(
            configuracion_variante[
                "columnas"
            ]
        )

        print(
            "\n"
            + "=" * 72
        )
        print(
            f"{simbolo} | {detector} | {nombre_variante}"
        )
        print(
            f"Umbral congelado: {umbral:.2f}"
        )
        print(
            "=" * 72
        )

        escalador, conteos = ajustar_escalador_y_contar(
            simbolo=simbolo,
            detector=detector,
            configuracion_periodo=PERIODO_CONFIRMACION_2025,
            columnas_modelo=columnas_modelo,
            tamano_lote=tamano_lote,
        )

        pesos = calcular_pesos_binarios(
            conteos=conteos,
            potencia_peso_positivo=float(
                configuracion_variante[
                    "potencia_peso_positivo"
                ]
            ),
        )

        modelo = entrenar_modelo_binario(
            simbolo=simbolo,
            detector=detector,
            configuracion_periodo=PERIODO_CONFIRMACION_2025,
            configuracion_variante=configuracion_variante,
            escalador=escalador,
            pesos=pesos,
            epocas=epocas,
            tamano_lote=tamano_lote,
        )

        (
            objetivo,
            probabilidades,
            fechas_ns,
            rendimientos,
        ) = predecir_periodo(
            simbolo=simbolo,
            detector=detector,
            configuracion_periodo=PERIODO_CONFIRMACION_2025,
            columnas_modelo=columnas_modelo,
            modelo=modelo,
            escalador=escalador,
            tamano_lote=tamano_lote,
        )

        metricas_generales = calcular_metricas_generales(
            objetivo=objetivo,
            probabilidades=probabilidades,
        )

        coste_operacion = float(
            fila_seleccion.get(
                "coste_operacion_validacion_2023",
                fila_seleccion.get(
                    "coste_operacion_validacion_2024",
                    0.0,
                ),
            )
        )

        metricas_umbral = evaluar_umbral(
            objetivo=objetivo,
            probabilidades=probabilidades,
            fechas_ns=fechas_ns,
            rendimientos=rendimientos,
            detector=detector,
            umbral=umbral,
            coste_operacion=coste_operacion,
        )

        ruta = (
            RUTA_CONFIRMACION_2025
            / perfil
            / simbolo
            / detector
            / identificador
        )

        ruta.mkdir(
            parents=True,
            exist_ok=False,
        )

        predicciones = pd.DataFrame(
            {
                "fecha_apertura": convertir_fechas_ns(
                    fechas_ns
                ),
                "objetivo_positivo": objetivo,
                "probabilidad": probabilidades,
                "rendimiento_objetivo": rendimientos,
            }
        )

        ruta_predicciones = (
            ruta
            / "predicciones_2025.parquet"
        )

        predicciones.to_parquet(
            ruta_predicciones,
            index=False,
        )

        ruta_coeficientes = (
            ruta
            / "coeficientes.csv"
        )

        construir_tabla_coeficientes(
            modelo=modelo,
            columnas_modelo=columnas_modelo,
        ).to_csv(
            ruta_coeficientes,
            index=False,
            encoding="utf-8-sig",
        )

        detalle = {
            "fecha_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "identificador": identificador,
            "version_modelo": VERSION_MODELO,
            "perfil": perfil,
            "simbolo": simbolo,
            "detector": detector,
            "variante": nombre_variante,
            "variables": len(
                columnas_modelo
            ),
            "umbral_congelado": umbral,
            "cumplia_filtros_desarrollo": bool(
                fila_seleccion[
                    "cumple_filtros"
                ]
            ),
            "epocas": epocas,
            "tamano_lote": tamano_lote,
            "pesos": pesos,
            "metricas_generales_2025": metricas_generales,
            "metricas_umbral_2025": metricas_umbral,
            "ruta_predicciones": str(
                ruta_predicciones
            ),
            "ruta_coeficientes": str(
                ruta_coeficientes
            ),
            "confirmacion_secundaria": True,
            "prueba_totalmente_intacta": False,
            "division_2026_utilizada": False,
        }

        ruta_detalle = (
            ruta
            / "detalle.json"
        )

        ruta_detalle.write_text(
            json.dumps(
                detalle,
                ensure_ascii=False,
                indent=4,
            ),
            encoding="utf-8",
        )

        ruta_modelo = (
            ruta
            / "modelo.joblib"
        )

        joblib.dump(
            {
                "detalle": detalle,
                "modelo": modelo,
                "escalador": escalador,
            },
            ruta_modelo,
        )

        fila_historial = {
            "fecha_utc": detalle[
                "fecha_utc"
            ],
            "identificador": identificador,
            "perfil": perfil,
            "simbolo": simbolo,
            "detector": detector,
            "variante": nombre_variante,
            "variables": len(
                columnas_modelo
            ),
            "umbral": umbral,
            **{
                f"general_{clave}": valor
                for clave, valor in metricas_generales.items()
            },
            **{
                f"umbral_{clave}": valor
                for clave, valor in metricas_umbral.items()
            },
            "ruta": str(
                ruta
            ),
            "ruta_modelo": str(
                ruta_modelo
            ),
            "ruta_predicciones": str(
                ruta_predicciones
            ),
            "confirmacion_secundaria": True,
            "prueba_totalmente_intacta": False,
            "division_2026_utilizada": False,
        }

        filas_historial.append(
            fila_historial
        )

        print(
            "\nResultado 2025:"
        )
        print(
            f"  Prevalencia: {metricas_generales['prevalencia']:.4f}"
        )
        print(
            f"  PR-AUC: {metricas_generales['pr_auc']:.4f}"
        )
        print(
            f"  PR-AUC lift: {metricas_generales['pr_auc_lift']:.4f}"
        )
        print(
            f"  Precision al umbral: {metricas_umbral['precision']:.4f}"
        )
        print(
            f"  Recall al umbral: {metricas_umbral['recall']:.4f}"
        )
        print(
            f"  F1 al umbral: {metricas_umbral['f1']:.4f}"
        )
        print(
            "  Operaciones no solapadas: "
            f"{metricas_umbral['operaciones_no_solapadas']}"
        )
        print(
            f"  Ruta: {ruta}"
        )

        del modelo
        del escalador
        del objetivo
        del probabilidades
        del fechas_ns
        del rendimientos
        del predicciones
        gc.collect()

    RUTA_CONFIRMACION_2025.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        filas_historial
    ).to_csv(
        RUTA_HISTORIAL_CONFIRMACION_2025,
        mode="a",
        header=not RUTA_HISTORIAL_CONFIRMACION_2025.exists(),
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nCONFIRMACIÓN 2025 COMPLETADA"
    )
    print(
        f"Historial: {RUTA_HISTORIAL_CONFIRMACION_2025}"
    )


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene argumentos."""

    parser = argparse.ArgumentParser(
        description=(
            "Confirma en 2025 configuraciones seleccionadas con 2023/2024."
        )
    )

    parser.add_argument(
        "--perfil",
        choices=PERFILES_UMBRALES,
        default="equilibrado",
    )

    parser.add_argument(
        "--simbolo",
        choices=(
            *SIMBOLOS,
            "TODOS",
        ),
        default="TODOS",
    )

    parser.add_argument(
        "--detector",
        choices=(
            *DETECTORES,
            "AMBOS",
        ),
        default="AMBOS",
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

    return parser.parse_args()


def main() -> None:
    """Punto de entrada."""

    argumentos = obtener_argumentos()

    try:
        ejecutar(
            perfil=argumentos.perfil,
            simbolo_solicitado=argumentos.simbolo,
            detector_solicitado=argumentos.detector,
            epocas=argumentos.epocas,
            tamano_lote=argumentos.tamano_lote,
        )

    except (
        FileNotFoundError,
        KeyError,
        OSError,
        ValueError,
    ) as error:
        print(
            "\nNo se pudo completar la confirmación V4 en 2025."
        )
        print(
            f"Detalle: {error}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
