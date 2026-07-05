from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cripto.corto_plazo_v4_3.configuracion import (
    ESTRATEGIA_CONGELADA,
    SIMBOLO,
    VARIANTE_SUBE,
    VERSION_MODELO,
)
from cripto.corto_plazo_v4_3.entrenamiento import (
    cargar_validacion_completa,
    entrenar_y_predecir_sube,
)
from cripto.corto_plazo_v4_3.evaluacion import (
    evaluar_datos,
)


def evaluar_periodo_conocido(
    nombre_periodo: str,
    configuracion_periodo: dict[str, Any],
    ruta_salida: Path,
    epocas: int,
    tamano_lote: int,
) -> dict[str, Any]:
    """Evalúa un periodo ya observado sin presentarlo como prueba virgen."""

    print(
        f"\nEVALUACIÓN CONOCIDA {nombre_periodo} - V4.3"
    )
    print("=" * 72)
    print(
        "Este periodo ya fue observado. Se usa solo para comparar con V4.1."
    )
    print(
        "No se modifican la estrategia, el umbral ni la salida."
    )

    probabilidades, detalle_entrenamiento = entrenar_y_predecir_sube(
        configuracion_periodo=configuracion_periodo,
        epocas=epocas,
        tamano_lote=tamano_lote,
    )

    datos = cargar_validacion_completa(
        configuracion_periodo=configuracion_periodo
    )

    if len(datos) != len(probabilidades):
        raise RuntimeError(
            "Las predicciones y los precios no tienen la misma longitud."
        )

    datos[
        "probabilidad_sube"
    ] = probabilidades.astype(
        "float32"
    )

    metricas = evaluar_datos(
        datos=datos
    )

    ruta_salida.mkdir(
        parents=True,
        exist_ok=True,
    )

    detalle = {
        "fecha_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "version": VERSION_MODELO,
        "periodo": nombre_periodo,
        "periodo_ya_observado": True,
        "no_es_prueba_virgen": True,
        "simbolo": SIMBOLO,
        "variante_sube": VARIANTE_SUBE,
        "estrategia_congelada": ESTRATEGIA_CONGELADA,
        "configuracion_periodo": configuracion_periodo,
        "epocas": epocas,
        "tamano_lote": tamano_lote,
        "filas_evaluadas": len(datos),
        "metricas": metricas,
        "entrenamiento": detalle_entrenamiento,
    }

    ruta_json = (
        ruta_salida
        / "detalle.json"
    )

    ruta_json.write_text(
        json.dumps(
            detalle,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        f"Filas evaluadas: {len(datos):,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"Operaciones: {metricas['operaciones']}"
    )
    print(
        "Precisión: "
        f"{metricas['precision_clasificacion']:.4f}"
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
        "Retorno neto medio: "
        f"{metricas['retorno_neto_medio']:.4%}"
    )
    print(
        "Factor de beneficio: "
        f"{metricas['factor_beneficio']:.4f}"
    )
    print(
        "Drawdown máximo: "
        f"{metricas['maximo_drawdown']:.2%}"
    )
    print(f"\n- {ruta_json}")

    return detalle
