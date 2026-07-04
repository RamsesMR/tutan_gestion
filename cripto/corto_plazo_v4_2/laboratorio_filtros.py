from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from cripto.corto_plazo_v4_2.configuracion import (
    COSTE_TOTAL,
    DISTANCIAS_CRUCE,
    ENFRIAMIENTOS_MINUTOS,
    HORIZONTES_MINUTOS,
    RUTA_BASES,
    RUTA_RESULTADOS,
    RUTA_RESULTADOS_FILTROS,
    UMBRAL_REARME,
    UMBRAL_SUBE_BASE,
    UMBRALES_DIFERENCIA_SUBE_BAJA,
)
from cripto.corto_plazo_v4_2.utilidades import (
    aplicar_rearme,
    calcular_retorno_fijo,
    convertir_fechas_ns,
    cruces_desde_abajo,
    evaluar_indices,
    seleccionar_no_solapadas,
)


def main() -> None:
    RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    registros = []

    print("\nLABORATORIO DE FILTROS V4.2")
    print("=" * 72)

    for nombre in ("validacion_2023", "validacion_2024"):
        datos = pd.read_parquet(RUTA_BASES / f"{nombre}.parquet")

        fechas_ns = convertir_fechas_ns(datos["fecha_apertura"])
        probabilidades = datos["probabilidad_sube"].to_numpy(float)
        diferencia = datos["diferencia_sube_baja"].to_numpy(float)
        distancia = datos["distancia_umbral"].to_numpy(float)
        pendiente_5m = datos["pendiente_sube_5m"].to_numpy(float)
        cierres = datos["precio_cierre"].to_numpy(float)
        objetivo = datos["objetivo_sube_4h"].to_numpy("int8")

        cruces_base = cruces_desde_abajo(
            probabilidades,
            UMBRAL_SUBE_BASE,
        )

        for (
            horizonte,
            rearme,
            enfriamiento,
            distancia_minima,
            diferencia_minima,
            exige_pendiente,
        ) in itertools.product(
            HORIZONTES_MINUTOS,
            UMBRAL_REARME,
            ENFRIAMIENTOS_MINUTOS,
            DISTANCIAS_CRUCE,
            UMBRALES_DIFERENCIA_SUBE_BAJA,
            (False, True),
        ):
            cruces = aplicar_rearme(
                cruces_base,
                probabilidades,
                rearme,
            )

            mascara = cruces & (distancia >= distancia_minima)

            if diferencia_minima is not None:
                mascara &= diferencia >= diferencia_minima

            if exige_pendiente:
                mascara &= pendiente_5m > 0

            retornos = calcular_retorno_fijo(
                cierres,
                horizonte,
            )

            mascara &= np.isfinite(retornos)

            indices = seleccionar_no_solapadas(
                fechas_ns,
                mascara,
                horizonte,
                enfriamiento,
            )

            metricas = evaluar_indices(
                indices,
                retornos,
                objetivo,
                COSTE_TOTAL,
            )

            registros.append(
                {
                    "periodo": nombre,
                    "horizonte": horizonte,
                    "umbral_rearme": rearme,
                    "enfriamiento": enfriamiento,
                    "distancia_minima": distancia_minima,
                    "diferencia_minima": diferencia_minima,
                    "exige_pendiente_positiva_5m": exige_pendiente,
                    **metricas,
                }
            )

    resultados = pd.DataFrame(registros)

    resultados.to_csv(
        RUTA_RESULTADOS_FILTROS,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Combinaciones guardadas: {len(resultados):,}".replace(",", ".")
    )
    print(f"- {RUTA_RESULTADOS_FILTROS}")


if __name__ == "__main__":
    main()
