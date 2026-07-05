from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import pandas as pd
import requests

from cripto.corto_plazo_v4_5.configuracion import (
    INTERVALO,
    RUTA_KLINES_FUTUROS,
    SIMBOLOS,
    URL_BASE_ARCHIVOS_BINANCE,
)


def descargar_texto(
    sesion: requests.Session,
    url: str,
) -> str:
    respuesta = sesion.get(
        url,
        timeout=60,
    )
    respuesta.raise_for_status()
    return respuesta.text.strip()


def sha256_archivo(
    ruta: Path,
) -> str:
    hash_archivo = hashlib.sha256()

    with ruta.open("rb") as archivo:
        while bloque := archivo.read(1024 * 1024):
            hash_archivo.update(bloque)

    return hash_archivo.hexdigest()


def descargar_archivo(
    sesion: requests.Session,
    url: str,
    ruta: Path,
) -> None:
    ruta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporal = ruta.with_suffix(
        ruta.suffix + ".part"
    )

    with sesion.get(
        url,
        stream=True,
        timeout=120,
    ) as respuesta:
        respuesta.raise_for_status()

        with temporal.open("wb") as archivo:
            for bloque in respuesta.iter_content(
                chunk_size=1024 * 1024
            ):
                if bloque:
                    archivo.write(bloque)

    temporal.replace(ruta)


def periodos_mensuales(
    desde: str,
    hasta: str,
) -> list[pd.Period]:
    inicio = pd.Period(desde, freq="M")
    final = pd.Period(hasta, freq="M")

    if inicio > final:
        raise ValueError(
            "El periodo inicial debe ser anterior o igual al final."
        )

    return list(
        pd.period_range(
            inicio,
            final,
            freq="M",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--desde",
        default="2020-12",
        help="Primer mes YYYY-MM. Incluye contexto para 2021.",
    )
    parser.add_argument(
        "--hasta",
        default="2024-12",
        help="Último mes YYYY-MM. No incluye 2025 por defecto.",
    )
    parser.add_argument(
        "--sobrescribir",
        action="store_true",
    )
    argumentos = parser.parse_args()

    periodos = periodos_mensuales(
        argumentos.desde,
        argumentos.hasta,
    )

    sesion = requests.Session()
    sesion.headers.update(
        {
            "User-Agent": "tutanGestion-v4.5",
        }
    )

    print("\nDESCARGA DE KLINES USD-M FUTURES")
    print("=" * 72)
    print(
        f"Periodo: {periodos[0]} a {periodos[-1]}"
    )

    descargados = 0
    omitidos = 0

    for simbolo in SIMBOLOS:
        for periodo in periodos:
            mes = periodo.strftime("%Y-%m")
            nombre = f"{simbolo}-{INTERVALO}-{mes}.zip"
            carpeta = (
                RUTA_KLINES_FUTUROS
                / simbolo
                / INTERVALO
            )
            ruta = carpeta / nombre

            url = (
                f"{URL_BASE_ARCHIVOS_BINANCE}"
                f"/futures/um/monthly/klines/"
                f"{simbolo}/{INTERVALO}/{nombre}"
            )
            url_checksum = url + ".CHECKSUM"

            checksum_texto = descargar_texto(
                sesion,
                url_checksum,
            )
            checksum_esperado = checksum_texto.split()[0].lower()

            if (
                ruta.exists()
                and not argumentos.sobrescribir
                and sha256_archivo(ruta) == checksum_esperado
            ):
                omitidos += 1
                print(f"Correcto, se conserva: {nombre}")
                continue

            print(f"Descargando: {nombre}")
            descargar_archivo(
                sesion,
                url,
                ruta,
            )

            checksum_obtenido = sha256_archivo(ruta)

            if checksum_obtenido != checksum_esperado:
                ruta.unlink(missing_ok=True)
                raise ValueError(
                    f"Checksum incorrecto para {nombre}."
                )

            descargados += 1
            time.sleep(0.05)

    print("\nDESCARGA DE FUTUROS COMPLETADA")
    print(f"Descargados: {descargados}")
    print(f"Conservados: {omitidos}")
    print(f"- {RUTA_KLINES_FUTUROS}")


if __name__ == "__main__":
    main()
