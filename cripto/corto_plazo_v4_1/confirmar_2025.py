from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4.configuracion import (
    PERIODO_CONFIRMACION_2025,
    VARIANTES,
)
from cripto.corto_plazo_v4.utilidades import (
    ajustar_escalador_y_contar,
    calcular_pesos_binarios,
    construir_ruta_datos,
    entrenar_modelo_binario,
    predecir_periodo,
)
from cripto.corto_plazo_v4_1.configuracion import (
    RUTA_CONFIRMACION_2025,
    RUTA_SELECCION,
    SIMBOLO,
    UMBRAL_CLASE_SUBE,
    VARIANTE_BAJA,
    VARIANTE_SUBE,
)
from cripto.corto_plazo_v4_1.utilidades import (
    calcular_salida_barrera,
    calcular_salida_fija,
    construir_mascara_entrada,
    convertir_fechas_ns,
    evaluar_estrategia,
)


def entrenar_y_predecir(
    detector: str,
    variante: str,
    epocas: int,
    tamano_lote: int,
) -> np.ndarray:
    """Entrena 2021-2024 y predice 2025."""

    configuracion_variante = VARIANTES[
        variante
    ]

    columnas = tuple(
        configuracion_variante[
            "columnas"
        ]
    )

    escalador, conteos = ajustar_escalador_y_contar(
        simbolo=SIMBOLO,
        detector=detector,
        configuracion_periodo=PERIODO_CONFIRMACION_2025,
        columnas_modelo=columnas,
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
        simbolo=SIMBOLO,
        detector=detector,
        configuracion_periodo=PERIODO_CONFIRMACION_2025,
        configuracion_variante=configuracion_variante,
        escalador=escalador,
        pesos=pesos,
        epocas=epocas,
        tamano_lote=tamano_lote,
    )

    (
        _,
        probabilidades,
        _,
        _,
    ) = predecir_periodo(
        simbolo=SIMBOLO,
        detector=detector,
        configuracion_periodo=PERIODO_CONFIRMACION_2025,
        columnas_modelo=columnas,
        modelo=modelo,
        escalador=escalador,
        tamano_lote=tamano_lote,
    )

    return probabilidades


def cargar_2025() -> pd.DataFrame:
    """Carga precios de 2025 con la misma purga V4."""

    desde_archivo, hasta_archivo = PERIODO_CONFIRMACION_2025[
        "archivo_validacion"
    ]

    ruta = construir_ruta_datos(
        simbolo=SIMBOLO,
        desde=desde_archivo,
        hasta=hasta_archivo,
    )

    datos = pd.read_parquet(
        ruta,
        columns=[
            "fecha_apertura",
            "fecha_objetivo",
            "precio_apertura",
            "precio_maximo",
            "precio_minimo",
            "precio_cierre",
            "rendimiento_objetivo",
        ],
    )

    datos[
        "fecha_apertura"
    ] = pd.to_datetime(
        datos[
            "fecha_apertura"
        ],
        utc=True,
        errors="raise",
    )

    datos[
        "fecha_objetivo"
    ] = pd.to_datetime(
        datos[
            "fecha_objetivo"
        ],
        utc=True,
        errors="raise",
    )

    desde = pd.Timestamp(
        PERIODO_CONFIRMACION_2025[
            "validacion_desde"
        ],
        tz="UTC",
    )

    hasta = pd.Timestamp(
        PERIODO_CONFIRMACION_2025[
            "validacion_hasta"
        ],
        tz="UTC",
    )

    mascara = (
        (
            datos[
                "fecha_apertura"
            ]
            >= desde
        )
        & (
            datos[
                "fecha_apertura"
            ]
            < hasta
        )
        & (
            datos[
                "fecha_objetivo"
            ]
            > datos[
                "fecha_apertura"
            ]
        )
        & (
            datos[
                "fecha_objetivo"
            ]
            < hasta
        )
    )

    return datos.loc[
        mascara
    ].reset_index(
        drop=True
    )


def main() -> None:
    """Confirma una estrategia congelada en 2025."""

    parser = argparse.ArgumentParser()

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

    argumentos = parser.parse_args()

    if not RUTA_SELECCION.exists():
        raise FileNotFoundError(
            f"No existe: {RUTA_SELECCION}"
        )

    seleccion = pd.read_csv(
        RUTA_SELECCION
    ).iloc[
        0
    ]

    if not bool(
        seleccion[
            "cumple_filtros"
        ]
    ):
        raise RuntimeError(
            "La estrategia seleccionada no cumple los filtros de desarrollo. "
            "No debe abrirse 2025."
        )

    print(
        "\nCONFIRMACIÓN V4.1 EN 2025"
    )
    print("=" * 72)
    print(
        "La estrategia está congelada desde 2023/2024."
    )

    probabilidades_sube = entrenar_y_predecir(
        detector="SUBE",
        variante=VARIANTE_SUBE,
        epocas=argumentos.epocas,
        tamano_lote=argumentos.tamano_lote,
    )

    probabilidades_baja = entrenar_y_predecir(
        detector="BAJA",
        variante=VARIANTE_BAJA,
        epocas=argumentos.epocas,
        tamano_lote=argumentos.tamano_lote,
    )

    datos = cargar_2025()

    if not (
        len(
            datos
        )
        == len(
            probabilidades_sube
        )
        == len(
            probabilidades_baja
        )
    ):
        raise RuntimeError(
            "Longitudes incompatibles."
        )

    mascara = construir_mascara_entrada(
        probabilidades_sube=probabilidades_sube,
        probabilidades_baja=probabilidades_baja,
        umbral_sube=float(
            seleccion[
                "umbral_sube"
            ]
        ),
        umbral_veto_baja=(
            float(
                seleccion[
                    "umbral_veto_baja"
                ]
            )
            if bool(
                seleccion[
                    "usa_veto_baja"
                ]
            )
            else None
        ),
        modo=str(
            seleccion[
                "modo_entrada"
            ]
        ),
    )

    cierres = datos[
        "precio_cierre"
    ].to_numpy(
        dtype="float64"
    )

    if str(
        seleccion[
            "tipo_salida"
        ]
    ) == "fija":
        retornos, duraciones = calcular_salida_fija(
            cierres=cierres,
            minutos=int(
                seleccion[
                    "horizonte_salida"
                ]
            ),
        )

    else:
        nombre = str(
            seleccion[
                "nombre_salida"
            ]
        )

        partes = nombre.replace(
            "tp",
            "",
        ).replace(
            "sl",
            "",
        ).replace(
            "m",
            "",
        ).split(
            "_"
        )

        take_profit = int(
            partes[
                0
            ]
        ) / 10000

        stop_loss = int(
            partes[
                1
            ]
        ) / 10000

        horizonte = int(
            partes[
                2
            ]
        )

        retornos, duraciones = calcular_salida_barrera(
            cierres=cierres,
            maximos=datos[
                "precio_maximo"
            ].to_numpy(
                dtype="float64"
            ),
            minimos=datos[
                "precio_minimo"
            ].to_numpy(
                dtype="float64"
            ),
            take_profit=take_profit,
            stop_loss=stop_loss,
            horizonte=horizonte,
        )

    objetivo = (
        datos[
            "rendimiento_objetivo"
        ].to_numpy(
            dtype="float64"
        )
        >= UMBRAL_CLASE_SUBE
    ).astype(
        "int8"
    )

    metricas = evaluar_estrategia(
        fechas_ns=convertir_fechas_ns(
            datos[
                "fecha_apertura"
            ]
        ),
        mascara_entrada=mascara,
        retornos_brutos=retornos,
        minutos_salida=duraciones,
        objetivo_sube_4h=objetivo,
        coste=float(
            seleccion[
                "coste"
            ]
        ),
    )

    fecha = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    ruta = (
        RUTA_CONFIRMACION_2025
        / fecha
    )

    ruta.mkdir(
        parents=True,
        exist_ok=False,
    )

    detalle = {
        "fecha_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "estrategia": seleccion.to_dict(),
        "metricas_2025": metricas,
        "confirmacion_secundaria": True,
        "uso_2026": False,
    }

    (
        ruta
        / "detalle.json"
    ).write_text(
        json.dumps(
            detalle,
            ensure_ascii=False,
            indent=4,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"Precisión: {metricas['precision_clasificacion']:.4f}"
    )
    print(
        "Porcentaje de acierto: "
        f"{metricas['porcentaje_acierto_clasificacion']:.2f} %"
    )
    print(
        "Operaciones positivas: "
        f"{metricas['porcentaje_operaciones_positivas']:.2f} %"
    )
    print(
        f"Operaciones: {metricas['operaciones']}"
    )
    print(
        f"Retorno neto medio: {metricas['retorno_neto_medio']:.4%}"
    )
    print(
        f"Retorno neto mediano: {metricas['retorno_neto_mediano']:.4%}"
    )
    print(
        f"Factor beneficio: {metricas['factor_beneficio']:.3f}"
    )
    print(
        f"Drawdown: {metricas['maximo_drawdown']:.2%}"
    )
    print(
        f"- {ruta}"
    )


if __name__ == "__main__":
    main()
