from __future__ import annotations

import pandas as pd

from cripto.corto_plazo_v4_2.configuracion import (
    COSTE_TOTAL,
    HORIZONTES_MINUTOS,
    RUTA_BASES,
    RUTA_RESULTADOS,
    RUTA_RESULTADOS_SALIDAS,
    UMBRAL_SUBE_BASE,
)
from cripto.corto_plazo_v4_2.utilidades import (
    calcular_retorno_fijo,
    convertir_fechas_ns,
    cruces_desde_abajo,
    evaluar_indices,
    seleccionar_no_solapadas,
)


def main() -> None:
    RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    registros = []

    print("\nLABORATORIO DE SALIDAS V4.2")
    print("=" * 72)

    for nombre in ("validacion_2023", "validacion_2024"):
        datos = pd.read_parquet(RUTA_BASES / f"{nombre}.parquet")

        fechas_ns = convertir_fechas_ns(datos["fecha_apertura"])
        probabilidades = datos["probabilidad_sube"].to_numpy(float)
        cierres = datos["precio_cierre"].to_numpy(float)
        objetivo = datos["objetivo_sube_4h"].to_numpy("int8")

        cruces = cruces_desde_abajo(
            probabilidades,
            UMBRAL_SUBE_BASE,
        )

        for horizonte in HORIZONTES_MINUTOS:
            retornos = calcular_retorno_fijo(cierres, horizonte)
            mascara = cruces & pd.notna(retornos)

            indices = seleccionar_no_solapadas(
                fechas_ns,
                mascara,
                horizonte,
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
                    **metricas,
                }
            )

            print(
                f"{nombre} | {horizonte}m | "
                f"ops={metricas['operaciones']} | "
                f"acierto={metricas['porcentaje_acierto']:.2f}% | "
                f"neto={metricas['retorno_neto_medio']:.4%} | "
                f"PF={metricas['factor_beneficio']:.3f}"
            )

    pd.DataFrame(registros).to_csv(
        RUTA_RESULTADOS_SALIDAS,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n- {RUTA_RESULTADOS_SALIDAS}")


if __name__ == "__main__":
    main()
