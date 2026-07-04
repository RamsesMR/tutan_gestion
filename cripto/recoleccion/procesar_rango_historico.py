from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from typing import Iterator

from cripto.recoleccion.procesar_historico import procesar_historico


INTERVALOS_PERMITIDOS = {
    "1m",
    "1h",
    "1d",
}


def convertir_periodo(valor: str) -> tuple[int, int]:
    """
    Convierte un periodo con formato YYYY-MM en una tupla:

    (año, mes)
    """

    if not re.fullmatch(r"\d{4}-\d{2}", valor):
        raise argparse.ArgumentTypeError(
            "El periodo debe tener el formato YYYY-MM."
        )

    anio, mes = map(int, valor.split("-"))

    if anio < 2017:
        raise argparse.ArgumentTypeError(
            "El año no puede ser anterior a 2017."
        )

    if mes < 1 or mes > 12:
        raise argparse.ArgumentTypeError(
            "El mes debe estar comprendido entre 01 y 12."
        )

    return anio, mes


def periodo_a_numero(periodo: tuple[int, int]) -> int:
    """Convierte año y mes en un número comparable."""

    anio, mes = periodo

    return anio * 12 + mes


def obtener_mes_siguiente(
    anio: int,
    mes: int,
) -> tuple[int, int]:
    """Devuelve el año y mes siguientes."""

    if mes == 12:
        return anio + 1, 1

    return anio, mes + 1


def recorrer_meses(
    desde: tuple[int, int],
    hasta: tuple[int, int],
) -> Iterator[tuple[int, int]]:
    """Genera todos los meses comprendidos en el rango."""

    anio, mes = desde
    anio_final, mes_final = hasta

    while (anio, mes) <= (anio_final, mes_final):
        yield anio, mes
        anio, mes = obtener_mes_siguiente(anio, mes)


def validar_rango(
    desde: tuple[int, int],
    hasta: tuple[int, int],
) -> None:
    """Comprueba que el rango sea válido y esté cerrado."""

    if periodo_a_numero(desde) > periodo_a_numero(hasta):
        raise ValueError(
            "El periodo inicial no puede ser posterior al final."
        )

    ahora = datetime.now(timezone.utc)
    mes_actual = (ahora.year, ahora.month)

    if periodo_a_numero(hasta) >= periodo_a_numero(mes_actual):
        raise ValueError(
            "El periodo final debe ser anterior al mes actual. "
            "No se procesarán meses todavía incompletos."
        )


def procesar_rango(
    simbolo: str,
    intervalo: str,
    desde: tuple[int, int],
    hasta: tuple[int, int],
) -> None:
    """Procesa todos los meses del rango indicado."""

    simbolo = simbolo.upper().strip()
    intervalo = intervalo.strip()

    validar_rango(
        desde=desde,
        hasta=hasta,
    )

    meses = list(
        recorrer_meses(
            desde=desde,
            hasta=hasta,
        )
    )

    print("\n" + "=" * 60)
    print("PROCESAMIENTO DE RANGO HISTÓRICO")
    print("=" * 60)
    print(f"Símbolo: {simbolo}")
    print(f"Intervalo: {intervalo}")
    print(
        f"Desde: {desde[0]}-{desde[1]:02d}"
    )
    print(
        f"Hasta: {hasta[0]}-{hasta[1]:02d}"
    )
    print(f"Meses que se procesarán: {len(meses)}")

    for posicion, (anio, mes) in enumerate(
        meses,
        start=1,
    ):
        print("\n" + "#" * 60)
        print(
            f"MES {posicion} DE {len(meses)}: "
            f"{anio}-{mes:02d}"
        )
        print("#" * 60)

        procesar_historico(
            simbolo=simbolo,
            intervalo=intervalo,
            anio=anio,
            mes=mes,
        )

    print("\n" + "=" * 60)
    print("RANGO COMPLETADO CORRECTAMENTE")
    print("=" * 60)
    print(f"Meses procesados: {len(meses)}")


def obtener_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Descarga, importa y valida un rango de meses "
            "históricos de Binance Spot."
        )
    )

    parser.add_argument(
        "--simbolo",
        default="BTCUSDT",
        help="Símbolo del mercado. Ejemplo: BTCUSDT.",
    )

    parser.add_argument(
        "--intervalo",
        default="1m",
        choices=sorted(INTERVALOS_PERMITIDOS),
        help="Intervalo de las velas.",
    )

    parser.add_argument(
        "--desde",
        required=True,
        type=convertir_periodo,
        help="Primer mes del rango. Formato: YYYY-MM.",
    )

    parser.add_argument(
        "--hasta",
        required=True,
        type=convertir_periodo,
        help="Último mes del rango. Formato: YYYY-MM.",
    )

    return parser.parse_args()


def main() -> None:
    argumentos = obtener_argumentos()

    try:
        procesar_rango(
            simbolo=argumentos.simbolo,
            intervalo=argumentos.intervalo,
            desde=argumentos.desde,
            hasta=argumentos.hasta,
        )

    except (OSError, ValueError) as error:
        print("\n" + "=" * 60)
        print("EL RANGO NO PUDO COMPLETARSE")
        print("=" * 60)
        print(f"Detalle: {error}")

        raise SystemExit(1)


if __name__ == "__main__":
    main()