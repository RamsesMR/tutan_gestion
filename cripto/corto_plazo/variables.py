from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VENTANAS_MINUTOS = (
    5,
    15,
    30,
    60,
    240,
)

COLUMNAS_REQUERIDAS = {
    "simbolo",
    "fecha_apertura",
    "precio_apertura",
    "precio_maximo",
    "precio_minimo",
    "precio_cierre",
    "volumen",
    "volumen_activo_cotizacion",
    "numero_operaciones",
    "volumen_comprador_base",
    "volumen_comprador_cotizacion",
    "fecha_objetivo",
    "precio_cierre_futuro",
    "rendimiento_objetivo",
    "es_muestra_periodo",
}


def validar_columnas(datos: pd.DataFrame) -> None:
    """Comprueba que el conjunto tenga todas las columnas necesarias."""

    columnas_faltantes = COLUMNAS_REQUERIDAS.difference(
        datos.columns
    )

    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas obligatorias: "
            + ", ".join(sorted(columnas_faltantes))
        )


def division_segura(
    numerador: pd.Series,
    denominador: pd.Series,
    valor_si_cero: float = 0.0,
) -> np.ndarray:
    """Divide evitando infinitos cuando el denominador sea cero."""

    valores_numerador = numerador.to_numpy(
        dtype="float64"
    )

    valores_denominador = denominador.to_numpy(
        dtype="float64"
    )

    resultado = np.full(
        shape=len(numerador),
        fill_value=valor_si_cero,
        dtype="float64",
    )

    np.divide(
        valores_numerador,
        valores_denominador,
        out=resultado,
        where=valores_denominador != 0,
    )

    return resultado


def crear_grupos_continuos(
    datos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Separa las velas en grupos continuos.

    Si falta un minuto, las variables posteriores no utilizarán
    información situada al otro lado del hueco.
    """

    datos = datos.copy()

    diferencias = (
        datos
        .groupby(
            "simbolo",
            sort=False,
        )["fecha_apertura"]
        .diff()
    )

    inicio_nuevo_grupo = diferencias.ne(
        pd.Timedelta(minutes=1)
    )

    datos["grupo_continuo"] = (
        inicio_nuevo_grupo
        .groupby(datos["simbolo"])
        .cumsum()
        .astype("int64")
    )

    return datos


def crear_variables(
    datos: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], int]:
    """
    Crea variables utilizando únicamente la vela actual
    y las velas anteriores.
    """

    validar_columnas(datos)

    datos = datos.copy()

    datos["fecha_apertura"] = pd.to_datetime(
        datos["fecha_apertura"],
        utc=True,
    )

    datos["fecha_objetivo"] = pd.to_datetime(
        datos["fecha_objetivo"],
        utc=True,
    )

    datos["es_muestra_periodo"] = datos[
        "es_muestra_periodo"
    ].astype(bool)

    datos = (
        datos
        .sort_values(
            [
                "simbolo",
                "fecha_apertura",
            ]
        )
        .reset_index(drop=True)
    )

    datos = crear_grupos_continuos(datos)

    claves_grupo = [
        "simbolo",
        "grupo_continuo",
    ]

    columnas_variables: list[str] = []

    # ---------------------------------------------------------
    # Forma de la vela
    # ---------------------------------------------------------

    datos["rango_relativo"] = (
        datos["precio_maximo"]
        - datos["precio_minimo"]
    ) / datos["precio_apertura"]

    datos["cuerpo_relativo"] = (
        datos["precio_cierre"]
        - datos["precio_apertura"]
    ) / datos["precio_apertura"]

    maximo_cuerpo = datos[
        [
            "precio_apertura",
            "precio_cierre",
        ]
    ].max(axis=1)

    minimo_cuerpo = datos[
        [
            "precio_apertura",
            "precio_cierre",
        ]
    ].min(axis=1)

    datos["mecha_superior_relativa"] = (
        datos["precio_maximo"]
        - maximo_cuerpo
    ) / datos["precio_apertura"]

    datos["mecha_inferior_relativa"] = (
        minimo_cuerpo
        - datos["precio_minimo"]
    ) / datos["precio_apertura"]

    columnas_variables.extend(
        [
            "rango_relativo",
            "cuerpo_relativo",
            "mecha_superior_relativa",
            "mecha_inferior_relativa",
        ]
    )

    # ---------------------------------------------------------
    # Presión compradora
    # ---------------------------------------------------------

    datos["proporcion_compradora_base"] = division_segura(
        datos["volumen_comprador_base"],
        datos["volumen"],
    )

    datos["proporcion_compradora_cotizacion"] = division_segura(
        datos["volumen_comprador_cotizacion"],
        datos["volumen_activo_cotizacion"],
    )

    columnas_variables.extend(
        [
            "proporcion_compradora_base",
            "proporcion_compradora_cotizacion",
        ]
    )

    # ---------------------------------------------------------
    # Rendimiento de un minuto
    # ---------------------------------------------------------

    datos["rendimiento_1m"] = (
        datos
        .groupby(
            claves_grupo,
            sort=False,
        )["precio_cierre"]
        .transform(
            lambda serie: (
                serie / serie.shift(1) - 1
            )
        )
    )

    columnas_variables.append("rendimiento_1m")

    # ---------------------------------------------------------
    # Variables por ventanas temporales
    # ---------------------------------------------------------

    for ventana in VENTANAS_MINUTOS:
        columna_rendimiento = (
            f"rendimiento_{ventana}m"
        )

        columna_volatilidad = (
            f"volatilidad_{ventana}m"
        )

        columna_desviacion_media = (
            f"desviacion_media_cierre_{ventana}m"
        )

        columna_volumen_relativo = (
            f"volumen_relativo_{ventana}m"
        )

        columna_operaciones_relativas = (
            f"operaciones_relativas_{ventana}m"
        )

        datos[columna_rendimiento] = (
            datos
            .groupby(
                claves_grupo,
                sort=False,
            )["precio_cierre"]
            .transform(
                lambda serie, periodo=ventana: (
                    serie / serie.shift(periodo) - 1
                )
            )
        )

        datos[columna_volatilidad] = (
            datos
            .groupby(
                claves_grupo,
                sort=False,
            )["rendimiento_1m"]
            .transform(
                lambda serie, periodo=ventana: (
                    serie
                    .rolling(
                        window=periodo,
                        min_periods=periodo,
                    )
                    .std()
                )
            )
        )

        media_cierre = (
            datos
            .groupby(
                claves_grupo,
                sort=False,
            )["precio_cierre"]
            .transform(
                lambda serie, periodo=ventana: (
                    serie
                    .rolling(
                        window=periodo,
                        min_periods=periodo,
                    )
                    .mean()
                )
            )
        )

        datos[columna_desviacion_media] = (
            datos["precio_cierre"]
            / media_cierre
            - 1
        )

        media_volumen = (
            datos
            .groupby(
                claves_grupo,
                sort=False,
            )["volumen"]
            .transform(
                lambda serie, periodo=ventana: (
                    serie
                    .rolling(
                        window=periodo,
                        min_periods=periodo,
                    )
                    .mean()
                )
            )
        )

        proporcion_volumen = division_segura(
            datos["volumen"],
            media_volumen,
        )

        datos[columna_volumen_relativo] = np.where(
            media_volumen.to_numpy() != 0,
            proporcion_volumen - 1,
            0.0,
        )

        media_operaciones = (
            datos
            .groupby(
                claves_grupo,
                sort=False,
            )["numero_operaciones"]
            .transform(
                lambda serie, periodo=ventana: (
                    serie
                    .rolling(
                        window=periodo,
                        min_periods=periodo,
                    )
                    .mean()
                )
            )
        )

        proporcion_operaciones = division_segura(
            datos["numero_operaciones"],
            media_operaciones,
        )

        datos[columna_operaciones_relativas] = np.where(
            media_operaciones.to_numpy() != 0,
            proporcion_operaciones - 1,
            0.0,
        )

        columnas_variables.extend(
            [
                columna_rendimiento,
                columna_volatilidad,
                columna_desviacion_media,
                columna_volumen_relativo,
                columna_operaciones_relativas,
            ]
        )

    # ---------------------------------------------------------
    # Variables temporales cíclicas
    # ---------------------------------------------------------

    minuto_del_dia = (
        datos["fecha_apertura"].dt.hour * 60
        + datos["fecha_apertura"].dt.minute
    )

    dia_semana = (
        datos["fecha_apertura"].dt.dayofweek
    )

    datos["minuto_dia_seno"] = np.sin(
        2 * np.pi * minuto_del_dia / 1440
    )

    datos["minuto_dia_coseno"] = np.cos(
        2 * np.pi * minuto_del_dia / 1440
    )

    datos["dia_semana_seno"] = np.sin(
        2 * np.pi * dia_semana / 7
    )

    datos["dia_semana_coseno"] = np.cos(
        2 * np.pi * dia_semana / 7
    )

    columnas_variables.extend(
        [
            "minuto_dia_seno",
            "minuto_dia_coseno",
            "dia_semana_seno",
            "dia_semana_coseno",
        ]
    )

    # ---------------------------------------------------------
    # Eliminar filas sin historial y conservar solo el periodo
    # ---------------------------------------------------------

    total_periodo = int(
        datos["es_muestra_periodo"].sum()
    )

    datos = datos.dropna(
        subset=columnas_variables,
    )

    datos = (
        datos
        .loc[datos["es_muestra_periodo"]]
        .reset_index(drop=True)
    )

    muestras_descartadas = (
        total_periodo - len(datos)
    )

    datos = datos.drop(
        columns=[
            "grupo_continuo",
            "es_muestra_periodo",
        ]
    )

    return (
        datos,
        columnas_variables,
        muestras_descartadas,
    )


def construir_ruta_salida(
    ruta_entrada: Path,
) -> Path:
    """Crea el nombre del archivo que contiene las variables."""

    return ruta_entrada.with_name(
        f"{ruta_entrada.stem}_variables.parquet"
    )


def procesar_archivo(
    ruta_entrada: Path,
) -> Path:
    """Lee un Parquet, crea variables y guarda el resultado."""

    if not ruta_entrada.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {ruta_entrada}"
        )

    print("\nCREACIÓN DE VARIABLES DE CORTO PLAZO")
    print("=" * 60)
    print(f"Archivo de entrada: {ruta_entrada}")

    datos = pd.read_parquet(ruta_entrada)

    print(f"Muestras recibidas: {len(datos)}")

    (
        datos_variables,
        columnas_variables,
        muestras_descartadas,
    ) = crear_variables(datos)

    ruta_salida = construir_ruta_salida(
        ruta_entrada
    )

    datos_variables.to_parquet(
        ruta_salida,
        index=False,
        compression="snappy",
    )

    print("\nVariables creadas correctamente.")
    print(
        "Variables generadas: "
        f"{len(columnas_variables)}"
    )
    print(
        "Muestras válidas: "
        f"{len(datos_variables)}"
    )
    print(
        "Muestras descartadas: "
        f"{muestras_descartadas}"
    )
    print(f"Archivo: {ruta_salida}")
    print(
        "Tamaño: "
        f"{ruta_salida.stat().st_size / (1024 * 1024):.2f} MB"
    )

    return ruta_salida


def obtener_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crea variables para el modelo cripto "
            "de corto plazo."
        )
    )

    parser.add_argument(
        "--archivo",
        required=True,
        type=Path,
        help="Ruta del archivo Parquet preparado.",
    )

    return parser.parse_args()


def main() -> None:
    argumentos = obtener_argumentos()

    try:
        procesar_archivo(
            ruta_entrada=argumentos.archivo,
        )

    except (
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print("\nNo se pudieron crear las variables.")
        print(f"Detalle: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()