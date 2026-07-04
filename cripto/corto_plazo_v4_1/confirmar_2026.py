from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4.configuracion import (
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
    RUTA_MODELOS_V4_1,
    RUTA_SELECCION,
    SIMBOLO,
    UMBRAL_CLASE_SUBE,
    VARIANTE_SUBE,
)
from cripto.corto_plazo_v4_1.utilidades import (
    calcular_salida_fija,
    construir_mascara_entrada,
    convertir_fechas_ns,
    evaluar_estrategia,
)


PERIODO_CONFIRMACION_2026 = {
    "entrenamiento_desde": "2021-01-01",
    "entrenamiento_hasta": "2026-01-01",
    "archivos_entrenamiento": (
        ("2021-01-01", "2022-01-01"),
        ("2022-01-01", "2023-01-01"),
        ("2023-01-01", "2024-01-01"),
        ("2024-01-01", "2025-01-01"),
        ("2025-01-01", "2026-01-01"),
    ),
    "validacion_desde": "2026-01-01",
    "validacion_hasta": "2026-06-01",
    "archivo_validacion": (
        "2026-01-01",
        "2026-06-01",
    ),
}

ESTRATEGIA_CONGELADA = {
    "umbral_sube": 0.44,
    "modo_entrada": "cruce",
    "usa_veto_baja": False,
    "tipo_salida": "fija",
    "nombre_salida": "fija_480m",
    "horizonte_salida": 480,
    "coste": 0.001,
}


def validar_estrategia_congelada(
    seleccion: pd.Series,
) -> None:
    """Impide evaluar 2026 con parámetros distintos a los aprobados."""

    comprobaciones = {
        "umbral_sube": np.isclose(
            float(seleccion["umbral_sube"]),
            ESTRATEGIA_CONGELADA["umbral_sube"],
        ),
        "modo_entrada": (
            str(seleccion["modo_entrada"])
            == ESTRATEGIA_CONGELADA["modo_entrada"]
        ),
        "usa_veto_baja": (
            bool(seleccion["usa_veto_baja"])
            == ESTRATEGIA_CONGELADA["usa_veto_baja"]
        ),
        "tipo_salida": (
            str(seleccion["tipo_salida"])
            == ESTRATEGIA_CONGELADA["tipo_salida"]
        ),
        "nombre_salida": (
            str(seleccion["nombre_salida"])
            == ESTRATEGIA_CONGELADA["nombre_salida"]
        ),
        "horizonte_salida": (
            int(seleccion["horizonte_salida"])
            == ESTRATEGIA_CONGELADA["horizonte_salida"]
        ),
        "coste": np.isclose(
            float(seleccion["coste"]),
            ESTRATEGIA_CONGELADA["coste"],
        ),
    }

    incorrectas = [
        nombre
        for nombre, correcta in comprobaciones.items()
        if not correcta
    ]

    if incorrectas:
        raise RuntimeError(
            "La estrategia guardada ya no coincide con la estrategia "
            "congelada. Parámetros distintos: "
            + ", ".join(incorrectas)
        )

    if not bool(seleccion["cumple_filtros"]):
        raise RuntimeError(
            "La estrategia seleccionada no cumple los filtros de desarrollo."
        )


def entrenar_y_predecir_sube(
    epocas: int,
    tamano_lote: int,
) -> np.ndarray:
    """Entrena BTC SUBE con 2021-2025 y predice enero-mayo de 2026."""

    configuracion_variante = VARIANTES[
        VARIANTE_SUBE
    ]

    columnas = tuple(
        configuracion_variante["columnas"]
    )

    escalador, conteos = ajustar_escalador_y_contar(
        simbolo=SIMBOLO,
        detector="SUBE",
        configuracion_periodo=PERIODO_CONFIRMACION_2026,
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
        configuracion_periodo=PERIODO_CONFIRMACION_2026,
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
        configuracion_periodo=PERIODO_CONFIRMACION_2026,
        columnas_modelo=columnas,
        modelo=modelo,
        escalador=escalador,
        tamano_lote=tamano_lote,
    )

    return probabilidades


def cargar_2026() -> pd.DataFrame:
    """Carga el periodo intacto de enero-mayo de 2026."""

    desde_archivo, hasta_archivo = (
        PERIODO_CONFIRMACION_2026[
            "archivo_validacion"
        ]
    )

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

    desde = pd.Timestamp(
        PERIODO_CONFIRMACION_2026[
            "validacion_desde"
        ],
        tz="UTC",
    )

    hasta = pd.Timestamp(
        PERIODO_CONFIRMACION_2026[
            "validacion_hasta"
        ],
        tz="UTC",
    )

    mascara = (
        (datos["fecha_apertura"] >= desde)
        & (datos["fecha_apertura"] < hasta)
        & (
            datos["fecha_objetivo"]
            > datos["fecha_apertura"]
        )
        & (datos["fecha_objetivo"] < hasta)
    )

    return datos.loc[
        mascara
    ].reset_index(
        drop=True
    )


def main() -> None:
    """Ejecuta una única prueba final con la estrategia congelada."""

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
    ).iloc[0]

    validar_estrategia_congelada(
        seleccion
    )

    print(
        "\nCONFIRMACIÓN FINAL V4.1 EN 2026"
    )
    print("=" * 72)
    print(
        "Periodo intacto: 2026-01-01 a 2026-06-01"
    )
    print(
        "Entrenamiento: 2021-01-01 a 2026-01-01"
    )
    print(
        "Detector: BTCUSDT SUBE — v4_63_sin_pesos"
    )
    print(
        "Entrada congelada: cruce desde abajo de 0,44"
    )
    print(
        "Salida congelada: cierre a 480 minutos"
    )
    print(
        "Coste total congelado: 0,10 %"
    )
    print(
        "Sin veto BAJA"
    )

    probabilidades_sube = entrenar_y_predecir_sube(
        epocas=argumentos.epocas,
        tamano_lote=argumentos.tamano_lote,
    )

    datos = cargar_2026()

    if len(datos) != len(probabilidades_sube):
        raise RuntimeError(
            "Las predicciones y los precios de 2026 "
            "no tienen la misma longitud."
        )

    probabilidades_baja_neutras = np.zeros(
        len(probabilidades_sube),
        dtype="float64",
    )

    mascara = construir_mascara_entrada(
        probabilidades_sube=probabilidades_sube,
        probabilidades_baja=probabilidades_baja_neutras,
        umbral_sube=ESTRATEGIA_CONGELADA[
            "umbral_sube"
        ],
        umbral_veto_baja=None,
        modo=ESTRATEGIA_CONGELADA[
            "modo_entrada"
        ],
    )

    cierres = datos[
        "precio_cierre"
    ].to_numpy(
        dtype="float64"
    )

    retornos, duraciones = calcular_salida_fija(
        cierres=cierres,
        minutos=ESTRATEGIA_CONGELADA[
            "horizonte_salida"
        ],
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
            datos["fecha_apertura"]
        ),
        mascara_entrada=mascara,
        retornos_brutos=retornos,
        minutos_salida=duraciones,
        objetivo_sube_4h=objetivo,
        coste=ESTRATEGIA_CONGELADA[
            "coste"
        ],
    )

    fecha = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    ruta_salida = (
        RUTA_MODELOS_V4_1
        / "confirmacion_2026"
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
        "periodo_entrenamiento": {
            "desde": "2021-01-01",
            "hasta": "2026-01-01",
        },
        "periodo_prueba_final": {
            "desde": "2026-01-01",
            "hasta": "2026-06-01",
        },
        "estrategia_congelada": ESTRATEGIA_CONGELADA,
        "variante_sube": VARIANTE_SUBE,
        "epocas": argumentos.epocas,
        "tamano_lote": argumentos.tamano_lote,
        "filas_evaluadas": len(datos),
        "metricas_2026": metricas,
        "prueba_final_intacta": True,
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
        "\nRESULTADOS 2026"
    )
    print("=" * 72)
    print(
        f"Filas evaluadas: {len(datos):,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"Precisión: {metricas['precision_clasificacion']:.4f}"
    )
    print(
        "Porcentaje de acierto clasificatorio: "
        f"{metricas['porcentaje_acierto_clasificacion']:.2f} %"
    )
    print(
        "Operaciones positivas netas: "
        f"{metricas['porcentaje_operaciones_positivas']:.2f} %"
    )
    print(
        f"Operaciones: {metricas['operaciones']}"
    )
    print(
        "Retorno bruto medio: "
        f"{metricas['retorno_bruto_medio']:.4%}"
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
