from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests


URL_BASE = "https://data.binance.vision/data"

INTERVALOS_PERMITIDOS = {
    "1m",
    "1h",
    "1d",
}

RUTA_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_DATOS_CRIPTO = RUTA_PROYECTO / "datos" / "cripto" / "bruto"


def calcular_sha256(ruta_archivo: Path) -> str:
    """Calcula el hash SHA-256 de un archivo."""

    sha256 = hashlib.sha256()

    with ruta_archivo.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            sha256.update(bloque)

    return sha256.hexdigest()


def leer_checksum(ruta_checksum: Path) -> str:
    """Obtiene el hash esperado desde el archivo .CHECKSUM."""

    contenido = ruta_checksum.read_text(encoding="utf-8").strip()

    if not contenido:
        raise ValueError("El archivo CHECKSUM está vacío.")

    hash_esperado = contenido.split()[0].lower()

    if len(hash_esperado) != 64:
        raise ValueError("El CHECKSUM descargado no tiene un SHA-256 válido.")

    return hash_esperado


def verificar_integridad(
    ruta_zip: Path,
    ruta_checksum: Path,
) -> bool:
    """Compara el SHA-256 del ZIP con el publicado por Binance."""

    hash_esperado = leer_checksum(ruta_checksum)
    hash_calculado = calcular_sha256(ruta_zip)

    return hash_calculado == hash_esperado


def descargar_archivo(url: str, ruta_destino: Path) -> None:
    """Descarga un archivo utilizando una ruta temporal."""

    ruta_temporal = ruta_destino.with_name(
        f"{ruta_destino.name}.part"
    )

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(10, 120),
            headers={
                "User-Agent": "tutan-gestion/0.1",
            },
        ) as respuesta:

            if respuesta.status_code == 404:
                raise FileNotFoundError(
                    f"Binance no encontró el archivo solicitado: {url}"
                )

            respuesta.raise_for_status()

            with ruta_temporal.open("wb") as archivo:
                for bloque in respuesta.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if bloque:
                        archivo.write(bloque)

        ruta_temporal.replace(ruta_destino)

    except Exception:
        ruta_temporal.unlink(missing_ok=True)
        raise


def construir_url_mensual(
    simbolo: str,
    intervalo: str,
    anio: int,
    mes: int,
) -> tuple[str, str, str]:
    """Construye el nombre y las URLs oficiales del archivo mensual."""

    nombre_archivo = (
        f"{simbolo}-{intervalo}-{anio}-{mes:02d}.zip"
    )

    ruta_remota = (
        f"{URL_BASE}/spot/monthly/klines/"
        f"{simbolo}/{intervalo}/{nombre_archivo}"
    )

    url_zip = ruta_remota
    url_checksum = f"{ruta_remota}.CHECKSUM"

    return nombre_archivo, url_zip, url_checksum


def descargar_mes(
    simbolo: str,
    intervalo: str,
    anio: int,
    mes: int,
) -> Path:
    """Descarga y verifica un mes de velas Spot de Binance."""

    simbolo = simbolo.upper().strip()
    intervalo = intervalo.strip()

    if intervalo not in INTERVALOS_PERMITIDOS:
        raise ValueError(
            f"Intervalo no permitido: {intervalo}. "
            f"Permitidos: {', '.join(sorted(INTERVALOS_PERMITIDOS))}"
        )

    if mes < 1 or mes > 12:
        raise ValueError("El mes debe estar comprendido entre 1 y 12.")

    if anio < 2017:
        raise ValueError("El año indicado no es válido para Binance.")

    nombre_archivo, url_zip, url_checksum = construir_url_mensual(
        simbolo=simbolo,
        intervalo=intervalo,
        anio=anio,
        mes=mes,
    )

    ruta_destino = (
        RUTA_DATOS_CRIPTO
        / "binance"
        / "spot"
        / "klines"
        / simbolo
        / intervalo
    )

    ruta_destino.mkdir(parents=True, exist_ok=True)

    ruta_zip = ruta_destino / nombre_archivo
    ruta_checksum = ruta_destino / f"{nombre_archivo}.CHECKSUM"

    print(f"Símbolo: {simbolo}")
    print(f"Intervalo: {intervalo}")
    print(f"Periodo: {anio}-{mes:02d}")
    print(f"Destino: {ruta_destino}")

    print("\nDescargando CHECKSUM...")
    descargar_archivo(url_checksum, ruta_checksum)

    if ruta_zip.exists():
        print("El archivo ZIP ya existe. Verificando integridad...")

        if verificar_integridad(ruta_zip, ruta_checksum):
            print("El archivo existente es correcto.")
            return ruta_zip

        print("El archivo existente no es válido. Se descargará nuevamente.")
        ruta_zip.unlink()

    print("Descargando archivo histórico...")
    descargar_archivo(url_zip, ruta_zip)

    print("Verificando integridad...")

    if not verificar_integridad(ruta_zip, ruta_checksum):
        ruta_zip.unlink(missing_ok=True)

        raise ValueError(
            "El SHA-256 del archivo descargado no coincide "
            "con el publicado por Binance."
        )

    print("\nDescarga completada correctamente.")
    print(f"Archivo: {ruta_zip}")
    print(f"Tamaño: {ruta_zip.stat().st_size / (1024 * 1024):.2f} MB")

    return ruta_zip


def obtener_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Descarga velas históricas mensuales "
            "de Binance Spot."
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
        "--anio",
        type=int,
        required=True,
        help="Año que se desea descargar.",
    )

    parser.add_argument(
        "--mes",
        type=int,
        required=True,
        help="Mes que se desea descargar.",
    )

    return parser.parse_args()


def main() -> None:
    argumentos = obtener_argumentos()

    try:
        descargar_mes(
            simbolo=argumentos.simbolo,
            intervalo=argumentos.intervalo,
            anio=argumentos.anio,
            mes=argumentos.mes,
        )

    except (
        requests.RequestException,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print("\nNo se pudo completar la descarga.")
        print(f"Detalle: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()