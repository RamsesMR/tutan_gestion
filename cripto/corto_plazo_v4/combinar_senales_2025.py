from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cripto.corto_plazo_v4.configuracion import (
    PERFILES_UMBRALES,
    RUTA_COMBINACION_2025,
    RUTA_HISTORIAL_CONFIRMACION_2025,
    SIMBOLOS,
)
from cripto.corto_plazo_v4.utilidades import (
    calcular_factor_beneficio,
    calcular_maximo_drawdown,
)


def obtener_ultimo(
    historial: pd.DataFrame,
    simbolo: str,
    detector: str,
    perfil: str,
) -> pd.Series:
    """Obtiene la confirmación más reciente."""

    subconjunto = historial.loc[
        (historial["simbolo"] == simbolo)
        & (historial["detector"] == detector)
        & (historial["perfil"] == perfil)
    ].copy()

    if subconjunto.empty:
        raise ValueError(
            f"No existe confirmación para {simbolo} {detector} {perfil}."
        )

    subconjunto["fecha_utc"] = pd.to_datetime(
        subconjunto["fecha_utc"],
        utc=True,
        errors="coerce",
    )

    return subconjunto.sort_values(
        "fecha_utc"
    ).iloc[
        -1
    ]


def aplicar_cooldown(
    datos: pd.DataFrame,
    minutos: int,
) -> pd.DataFrame:
    """Evita operaciones solapadas."""

    datos = datos.sort_values(
        "fecha_apertura"
    ).reset_index(
        drop=True
    )

    seleccionados = []
    siguiente_fecha = None

    for indice, fila in datos.iterrows():
        fecha = fila[
            "fecha_apertura"
        ]

        if siguiente_fecha is not None and fecha < siguiente_fecha:
            continue

        seleccionados.append(
            indice
        )

        siguiente_fecha = (
            fecha
            + pd.Timedelta(
                minutes=minutos
            )
        )

    return datos.loc[
        seleccionados
    ].reset_index(
        drop=True
    )


def ejecutar(
    simbolo: str,
    perfil: str,
    conflicto: str,
    coste_operacion: float,
) -> None:
    """Combina detectores de BAJA y SUBE confirmados en 2025."""

    if not RUTA_HISTORIAL_CONFIRMACION_2025.exists():
        raise FileNotFoundError(
            f"No existe: {RUTA_HISTORIAL_CONFIRMACION_2025}"
        )

    historial = pd.read_csv(
        RUTA_HISTORIAL_CONFIRMACION_2025
    )

    fila_baja = obtener_ultimo(
        historial=historial,
        simbolo=simbolo,
        detector="BAJA",
        perfil=perfil,
    )

    fila_sube = obtener_ultimo(
        historial=historial,
        simbolo=simbolo,
        detector="SUBE",
        perfil=perfil,
    )

    baja = pd.read_parquet(
        fila_baja[
            "ruta_predicciones"
        ]
    ).rename(
        columns={
            "objetivo_positivo": "objetivo_baja",
            "probabilidad": "probabilidad_baja",
        }
    )

    sube = pd.read_parquet(
        fila_sube[
            "ruta_predicciones"
        ]
    ).rename(
        columns={
            "objetivo_positivo": "objetivo_sube",
            "probabilidad": "probabilidad_sube",
        }
    )

    sube = sube.drop(
        columns=[
            "rendimiento_objetivo",
        ]
    )

    datos = baja.merge(
        sube,
        on="fecha_apertura",
        how="inner",
        validate="one_to_one",
    )

    umbral_baja = float(
        fila_baja[
            "umbral"
        ]
    )

    umbral_sube = float(
        fila_sube[
            "umbral"
        ]
    )

    activa_baja = (
        datos[
            "probabilidad_baja"
        ]
        >= umbral_baja
    )

    activa_sube = (
        datos[
            "probabilidad_sube"
        ]
        >= umbral_sube
    )

    senal = np.full(
        len(
            datos
        ),
        "NO_OPERAR",
        dtype=object,
    )

    senal[
        activa_baja
        & ~activa_sube
    ] = "CORTO"

    senal[
        activa_sube
        & ~activa_baja
    ] = "LARGO"

    conflictos = (
        activa_baja
        & activa_sube
    )

    if conflicto == "mayor_exceso":
        exceso_baja = (
            datos[
                "probabilidad_baja"
            ]
            / umbral_baja
        )

        exceso_sube = (
            datos[
                "probabilidad_sube"
            ]
            / umbral_sube
        )

        senal[
            conflictos
            & (
                exceso_baja
                > exceso_sube
            )
        ] = "CORTO"

        senal[
            conflictos
            & (
                exceso_sube
                > exceso_baja
            )
        ] = "LARGO"

    datos[
        "senal"
    ] = senal

    operaciones = datos.loc[
        datos["senal"] != "NO_OPERAR"
    ].copy()

    operaciones = aplicar_cooldown(
        datos=operaciones,
        minutos=240,
    )

    operaciones[
        "retorno_bruto_direccion"
    ] = np.where(
        operaciones[
            "senal"
        ] == "LARGO",
        operaciones[
            "rendimiento_objetivo"
        ],
        -operaciones[
            "rendimiento_objetivo"
        ],
    )

    operaciones[
        "retorno_neto"
    ] = (
        operaciones[
            "retorno_bruto_direccion"
        ]
        - coste_operacion
    )

    cantidad_largos = int(
        np.sum(
            operaciones[
                "senal"
            ]
            == "LARGO"
        )
    )

    cantidad_cortos = int(
        np.sum(
            operaciones[
                "senal"
            ]
            == "CORTO"
        )
    )

    retornos = operaciones[
        "retorno_neto"
    ].to_numpy(
        dtype="float64"
    )

    metricas = {
        "fecha_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "simbolo": simbolo,
        "perfil": perfil,
        "conflicto": conflicto,
        "umbral_baja": umbral_baja,
        "umbral_sube": umbral_sube,
        "coste_operacion": coste_operacion,
        "filas_comunes": len(
            datos
        ),
        "conflictos_brutos": int(
            conflictos.sum()
        ),
        "operaciones_no_solapadas": len(
            operaciones
        ),
        "largos": cantidad_largos,
        "cortos": cantidad_cortos,
        "tasa_retorno_positivo": (
            float(
                np.mean(
                    retornos > 0
                )
            )
            if len(
                retornos
            )
            else 0.0
        ),
        "retorno_neto_medio": (
            float(
                retornos.mean()
            )
            if len(
                retornos
            )
            else 0.0
        ),
        "retorno_neto_mediano": (
            float(
                np.median(
                    retornos
                )
            )
            if len(
                retornos
            )
            else 0.0
        ),
        "factor_beneficio": float(
            calcular_factor_beneficio(
                retornos
            )
        ),
        "maximo_drawdown": calcular_maximo_drawdown(
            retornos
        ),
        "confirmacion_secundaria": True,
        "prueba_totalmente_intacta": False,
    }

    ruta = (
        RUTA_COMBINACION_2025
        / perfil
        / simbolo
    )

    ruta.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_senales = (
        ruta
        / "operaciones_no_solapadas.parquet"
    )

    operaciones.to_parquet(
        ruta_senales,
        index=False,
    )

    ruta_metricas = (
        ruta
        / "metricas_combinadas.json"
    )

    ruta_metricas.write_text(
        json.dumps(
            metricas,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\nCOMBINACIÓN DE SEÑALES COMPLETADA"
    )
    print("=" * 72)
    print(
        f"Símbolo: {simbolo}"
    )
    print(
        f"Perfil: {perfil}"
    )
    print(
        f"Operaciones no solapadas: {len(operaciones)}"
    )
    print(
        f"Largos: {cantidad_largos}"
    )
    print(
        f"Cortos: {cantidad_cortos}"
    )
    print(
        f"Tasa retorno positivo: "
        f"{metricas['tasa_retorno_positivo']:.4f}"
    )
    print(
        f"Retorno neto medio: "
        f"{metricas['retorno_neto_medio']:.4%}"
    )
    print(
        f"- {ruta_senales}"
    )
    print(
        f"- {ruta_metricas}"
    )


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene argumentos."""

    parser = argparse.ArgumentParser(
        description=(
            "Combina detectores de BAJA y SUBE sobre la confirmación 2025."
        )
    )

    parser.add_argument(
        "--simbolo",
        required=True,
        choices=SIMBOLOS,
    )

    parser.add_argument(
        "--perfil",
        choices=PERFILES_UMBRALES,
        default="equilibrado",
    )

    parser.add_argument(
        "--conflicto",
        choices=(
            "no_operar",
            "mayor_exceso",
        ),
        default="no_operar",
    )

    parser.add_argument(
        "--coste-operacion",
        type=float,
        default=0.0,
    )

    return parser.parse_args()


def main() -> None:
    """Punto de entrada."""

    argumentos = obtener_argumentos()

    try:
        ejecutar(
            simbolo=argumentos.simbolo,
            perfil=argumentos.perfil,
            conflicto=argumentos.conflicto,
            coste_operacion=argumentos.coste_operacion,
        )

    except (
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(
            "\nNo se pudieron combinar las señales V4."
        )
        print(
            f"Detalle: {error}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
