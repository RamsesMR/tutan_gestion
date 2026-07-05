from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4.configuracion import VARIANTES
from cripto.corto_plazo_v4.utilidades import (
    ajustar_escalador_y_contar,
    calcular_pesos_binarios,
    construir_ruta_datos,
    entrenar_modelo_binario,
    predecir_periodo,
)
from cripto.corto_plazo_v4_2.configuracion import (
    COSTE_TOTAL,
    RUTA_CONFIRMACIONES,
    RUTA_POLITICA,
    SIMBOLO,
    UMBRAL_SUBE_BASE,
)
from cripto.corto_plazo_v4_2.utilidades import (
    aplicar_rearme,
    calcular_retorno_fijo,
    convertir_fechas_ns,
    cruces_desde_abajo,
    evaluar_indices,
    seleccionar_no_solapadas,
)


PERIODO_CONFIRMACION_2025 = {
    "entrenamiento_desde": "2021-01-01",
    "entrenamiento_hasta": "2025-01-01",
    "archivos_entrenamiento": (
        ("2021-01-01", "2022-01-01"),
        ("2022-01-01", "2023-01-01"),
        ("2023-01-01", "2024-01-01"),
        ("2024-01-01", "2025-01-01"),
    ),
    "validacion_desde": "2025-01-01",
    "validacion_hasta": "2026-01-01",
    "archivo_validacion": (
        "2025-01-01",
        "2026-01-01",
    ),
}

POLITICA_CONGELADA = {
    "horizonte": 720,
    "umbral_rearme": 0.40,
    "enfriamiento": 60,
    "distancia_minima": 0.01,
    "diferencia_minima": None,
    "exige_pendiente_positiva_5m": False,
}

VARIANTE_SUBE = "v4_63_sin_pesos"


def validar_politica() -> None:
    """Comprueba que la política guardada coincide con la congelada."""

    if not RUTA_POLITICA.exists():
        raise FileNotFoundError(
            f"No existe: {RUTA_POLITICA}"
        )

    politica = json.loads(
        RUTA_POLITICA.read_text(
            encoding="utf-8"
        )
    )

    for clave, valor_esperado in POLITICA_CONGELADA.items():
        valor_real = politica.get(clave)

        if isinstance(valor_esperado, float):
            if not np.isclose(
                float(valor_real),
                valor_esperado,
            ):
                raise RuntimeError(
                    f"La política cambió en {clave}: "
                    f"{valor_real} != {valor_esperado}"
                )
        else:
            if valor_real != valor_esperado:
                raise RuntimeError(
                    f"La política cambió en {clave}: "
                    f"{valor_real} != {valor_esperado}"
                )


def entrenar_y_predecir_sube(
    epocas: int,
    tamano_lote: int,
) -> np.ndarray:
    """Entrena BTC SUBE con 2021-2024 y predice 2025."""

    configuracion_variante = VARIANTES[
        VARIANTE_SUBE
    ]

    columnas = tuple(
        configuracion_variante["columnas"]
    )

    escalador, conteos = ajustar_escalador_y_contar(
        simbolo=SIMBOLO,
        detector="SUBE",
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
        detector="SUBE",
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
        detector="SUBE",
        configuracion_periodo=PERIODO_CONFIRMACION_2025,
        columnas_modelo=columnas,
        modelo=modelo,
        escalador=escalador,
        tamano_lote=tamano_lote,
    )

    return probabilidades


def cargar_2025() -> pd.DataFrame:
    """Carga 2025 con la misma purga temporal usada por V4."""

    desde, hasta = PERIODO_CONFIRMACION_2025[
        "archivo_validacion"
    ]

    ruta = construir_ruta_datos(
        simbolo=SIMBOLO,
        desde=desde,
        hasta=hasta,
    )

    datos = pd.read_parquet(
        ruta,
        columns=[
            "fecha_apertura",
            "fecha_objetivo",
            "precio_cierre",
            "rendimiento_objetivo",
        ],
    )

    datos["fecha_apertura"] = pd.to_datetime(
        datos["fecha_apertura"],
        utc=True,
        errors="raise",
    )

    datos["fecha_objetivo"] = pd.to_datetime(
        datos["fecha_objetivo"],
        utc=True,
        errors="raise",
    )

    desde_ts = pd.Timestamp(
        PERIODO_CONFIRMACION_2025[
            "validacion_desde"
        ],
        tz="UTC",
    )

    hasta_ts = pd.Timestamp(
        PERIODO_CONFIRMACION_2025[
            "validacion_hasta"
        ],
        tz="UTC",
    )

    mascara = (
        (datos["fecha_apertura"] >= desde_ts)
        & (datos["fecha_apertura"] < hasta_ts)
        & (
            datos["fecha_objetivo"]
            > datos["fecha_apertura"]
        )
        & (datos["fecha_objetivo"] < hasta_ts)
    )

    return datos.loc[
        mascara
    ].reset_index(
        drop=True
    )


def construir_mascara_politica(
    probabilidades: np.ndarray,
) -> np.ndarray:
    """Aplica la política V4.2 sin meta-modelo."""

    cruces = cruces_desde_abajo(
        probabilidades,
        UMBRAL_SUBE_BASE,
    )

    cruces = aplicar_rearme(
        cruces=cruces,
        probabilidades=probabilidades,
        umbral_rearme=POLITICA_CONGELADA[
            "umbral_rearme"
        ],
    )

    distancia_umbral = (
        probabilidades
        - UMBRAL_SUBE_BASE
    )

    mascara = (
        cruces
        & (
            distancia_umbral
            >= POLITICA_CONGELADA[
                "distancia_minima"
            ]
        )
    )

    return mascara


def main() -> None:
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

    validar_politica()

    print(
        "\nCONFIRMACIÓN V4.2 SIN META-MODELO EN 2025"
    )
    print("=" * 72)
    print(
        "Detector SUBE: v4_63_sin_pesos"
    )
    print(
        "Horizonte: 720 minutos"
    )
    print(
        "Rearme: 0,40"
    )
    print(
        "Entrada efectiva mínima: 0,45"
    )
    print(
        "Enfriamiento: 60 minutos"
    )
    print(
        "Coste total: 0,10 %"
    )
    print(
        "Meta-modelo: desactivado"
    )

    probabilidades = entrenar_y_predecir_sube(
        epocas=argumentos.epocas,
        tamano_lote=argumentos.tamano_lote,
    )

    datos = cargar_2025()

    if len(datos) != len(probabilidades):
        raise RuntimeError(
            "Las probabilidades y los precios "
            "no tienen la misma longitud."
        )

    mascara = construir_mascara_politica(
        probabilidades
    )

    cierres = datos[
        "precio_cierre"
    ].to_numpy(
        dtype="float64"
    )

    retornos = calcular_retorno_fijo(
        cierres=cierres,
        horizonte=POLITICA_CONGELADA[
            "horizonte"
        ],
    )

    mascara &= np.isfinite(
        retornos
    )

    fechas_ns = convertir_fechas_ns(
        datos[
            "fecha_apertura"
        ]
    )

    indices = seleccionar_no_solapadas(
        fechas_ns=fechas_ns,
        mascara=mascara,
        horizonte=POLITICA_CONGELADA[
            "horizonte"
        ],
        enfriamiento=POLITICA_CONGELADA[
            "enfriamiento"
        ],
    )

    objetivo_sube = (
        datos[
            "rendimiento_objetivo"
        ].to_numpy(
            dtype="float64"
        )
        >= 0.005
    ).astype(
        "int8"
    )

    metricas = evaluar_indices(
        indices=indices,
        retorno_bruto=retornos,
        objetivo_sube=objetivo_sube,
        coste=COSTE_TOTAL,
    )

    fecha = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    ruta_salida = (
        RUTA_CONFIRMACIONES
        / "confirmacion_2025_sin_meta"
        / fecha
    )

    ruta_salida.mkdir(
        parents=True,
        exist_ok=False,
    )

    detalle = {
        "fecha_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "periodo_entrenamiento_detector": {
            "desde": "2021-01-01",
            "hasta": "2025-01-01",
        },
        "periodo_confirmacion": {
            "desde": "2025-01-01",
            "hasta": "2026-01-01",
        },
        "politica_congelada": POLITICA_CONGELADA,
        "variante_sube": VARIANTE_SUBE,
        "meta_modelo": False,
        "filas_evaluadas": len(datos),
        "senales_politica": int(
            mascara.sum()
        ),
        "metricas_2025": metricas,
    }

    (
        ruta_salida
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
        "\nRESULTADOS V4.2 SIN META EN 2025"
    )
    print("=" * 72)
    print(
        f"Filas evaluadas: {len(datos):,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"Señales de política: {int(mascara.sum())}"
    )
    print(
        f"Operaciones no solapadas: {metricas['operaciones']}"
    )
    print(
        f"Precisión clasificatoria: {metricas['precision']:.4f}"
    )
    print(
        "Porcentaje de acierto clasificatorio: "
        f"{metricas['porcentaje_acierto']:.2f} %"
    )
    print(
        "Operaciones positivas netas: "
        f"{metricas['porcentaje_positivas']:.2f} %"
    )
    print(
        "Retorno neto medio: "
        f"{metricas['retorno_neto_medio']:.4%}"
    )
    print(
        "Retorno neto mediano: "
        f"{metricas['retorno_neto_mediano']:.4%}"
    )
    print(
        "Factor beneficio: "
        f"{metricas['factor_beneficio']:.3f}"
    )
    print(
        f"Drawdown: {metricas['maximo_drawdown']:.2%}"
    )
    print(
        f"\n- {ruta_salida}"
    )


if __name__ == "__main__":
    main()
