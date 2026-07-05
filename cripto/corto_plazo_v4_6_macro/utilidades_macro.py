from __future__ import annotations

from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


ZONA_NUEVA_YORK = ZoneInfo("America/New_York")

DIA_HABIL_FEDERAL = CustomBusinessDay(
    calendar=USFederalHolidayCalendar()
)


def detectar_columnas_fred(
    datos: pd.DataFrame,
    serie: str,
) -> tuple[str, str]:
    candidatas_fecha = (
        "observation_date",
        "DATE",
        "date",
    )

    columna_fecha = next(
        (
            columna
            for columna in candidatas_fecha
            if columna in datos.columns
        ),
        None,
    )

    if columna_fecha is None:
        raise ValueError(
            "No se encontró la columna de fecha en el CSV de FRED."
        )

    if serie in datos.columns:
        columna_valor = serie
    else:
        columnas_valor = [
            columna
            for columna in datos.columns
            if columna != columna_fecha
        ]

        if len(columnas_valor) != 1:
            raise ValueError(
                f"No se pudo identificar la columna de valores de {serie}."
            )

        columna_valor = columnas_valor[0]

    return columna_fecha, columna_valor


def leer_serie_fred(
    ruta: Path,
    serie: str,
) -> pd.DataFrame:
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe la serie descargada: {ruta}"
        )

    datos = pd.read_csv(ruta)

    columna_fecha, columna_valor = detectar_columnas_fred(
        datos,
        serie,
    )

    datos = datos[
        [
            columna_fecha,
            columna_valor,
        ]
    ].rename(
        columns={
            columna_fecha: "fecha_observacion",
            columna_valor: "valor",
        }
    )

    datos["fecha_observacion"] = pd.to_datetime(
        datos["fecha_observacion"],
        errors="raise",
    ).dt.normalize()

    datos["valor"] = pd.to_numeric(
        datos["valor"],
        errors="coerce",
    )

    datos = (
        datos.dropna(
            subset=[
                "fecha_observacion",
                "valor",
            ]
        )
        .drop_duplicates(
            subset=["fecha_observacion"],
            keep="last",
        )
        .sort_values("fecha_observacion")
        .reset_index(drop=True)
    )

    if datos.empty:
        raise ValueError(
            f"La serie {serie} no contiene observaciones válidas."
        )

    if not np.isfinite(
        datos["valor"].to_numpy(dtype="float64")
    ).all():
        raise ValueError(
            f"La serie {serie} contiene valores no finitos."
        )

    return datos


def siguiente_dia_habil(
    fecha: pd.Timestamp,
) -> pd.Timestamp:
    fecha = pd.Timestamp(fecha).normalize()
    return fecha + DIA_HABIL_FEDERAL


def ajustar_a_dia_habil(
    fecha: pd.Timestamp,
) -> pd.Timestamp:
    fecha = pd.Timestamp(fecha).normalize()

    if DIA_HABIL_FEDERAL.is_on_offset(fecha):
        return fecha

    return DIA_HABIL_FEDERAL.rollforward(fecha)


def siguiente_lunes_publicable(
    fecha: pd.Timestamp,
) -> pd.Timestamp:
    fecha = pd.Timestamp(fecha).normalize()

    dias = 7 - int(fecha.weekday())
    candidato = fecha + pd.Timedelta(days=dias)

    return ajustar_a_dia_habil(candidato)


def siguiente_miercoles_publicable(
    fecha: pd.Timestamp,
) -> pd.Timestamp:
    fecha = pd.Timestamp(fecha).normalize()

    dias = (2 - int(fecha.weekday())) % 7

    if dias == 0:
        dias = 7

    candidato = fecha + pd.Timedelta(days=dias)

    return ajustar_a_dia_habil(candidato)


def localizar_nueva_york(
    fecha: pd.Timestamp,
    hora: int,
    minuto: int,
) -> pd.Timestamp:
    naive = pd.Timestamp.combine(
        pd.Timestamp(fecha).date(),
        time(
            hour=hora,
            minute=minuto,
        ),
    )

    return naive.tz_localize(
        ZONA_NUEVA_YORK,
        ambiguous="raise",
        nonexistent="shift_forward",
    ).tz_convert("UTC")


def calcular_fecha_disponibilidad(
    fechas: pd.Series,
    regla: str,
) -> pd.Series:
    resultado: list[pd.Timestamp] = []

    for fecha in pd.to_datetime(
        fechas,
        errors="raise",
    ):
        fecha = pd.Timestamp(fecha).normalize()

        if regla == "cierre_mercado":
            disponible = localizar_nueva_york(
                fecha,
                16,
                5,
            )
        elif regla == "h15_siguiente_habil":
            disponible = localizar_nueva_york(
                siguiente_dia_habil(fecha),
                16,
                16,
            )
        elif regla == "siguiente_habil_conservador":
            disponible = localizar_nueva_york(
                siguiente_dia_habil(fecha),
                18,
                0,
            )
        elif regla == "h10_semanal":
            disponible = localizar_nueva_york(
                siguiente_lunes_publicable(fecha),
                16,
                16,
            )
        elif regla == "h41_jueves":
            disponible = localizar_nueva_york(
                ajustar_a_dia_habil(
                    fecha + pd.Timedelta(days=1)
                ),
                16,
                31,
            )
        elif regla == "nfci_miercoles":
            disponible = localizar_nueva_york(
                siguiente_miercoles_publicable(fecha),
                8,
                31,
            )
        else:
            raise ValueError(
                f"Regla de disponibilidad desconocida: {regla}"
            )

        resultado.append(disponible)

    return pd.Series(
        pd.DatetimeIndex(resultado).as_unit("ns"),
        index=fechas.index,
        dtype="datetime64[ns, UTC]",
    )


def construir_variable(
    datos: pd.DataFrame,
    columna_salida: str,
    transformacion: str,
    periodos: int,
    regla_disponibilidad: str,
) -> pd.DataFrame:
    resultado = datos.copy()

    valores = resultado["valor"].astype("float64")

    if transformacion == "log_ratio":
        if (valores <= 0).any():
            raise ValueError(
                f"{columna_salida} requiere valores estrictamente positivos."
            )

        resultado[columna_salida] = np.log(
            valores
            / valores.shift(periodos)
        )
    elif transformacion == "diferencia":
        resultado[columna_salida] = (
            valores
            - valores.shift(periodos)
        )
    else:
        raise ValueError(
            f"Transformación desconocida: {transformacion}"
        )

    resultado["fecha_disponibilidad"] = (
        calcular_fecha_disponibilidad(
            resultado["fecha_observacion"],
            regla_disponibilidad,
        )
    )

    resultado = (
        resultado[
            [
                "fecha_observacion",
                "fecha_disponibilidad",
                "valor",
                columna_salida,
            ]
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna(
            subset=[
                "fecha_disponibilidad",
                columna_salida,
            ]
        )
        .sort_values("fecha_disponibilidad")
        .drop_duplicates(
            subset=["fecha_disponibilidad"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if resultado.empty:
        raise ValueError(
            f"No se pudo construir {columna_salida}."
        )

    if not np.isfinite(
        resultado[columna_salida].to_numpy(
            dtype="float64"
        )
    ).all():
        raise ValueError(
            f"{columna_salida} contiene valores no finitos."
        )

    return resultado
