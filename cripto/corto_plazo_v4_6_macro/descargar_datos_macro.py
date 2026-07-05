from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from cripto.corto_plazo_v4_6_macro.configuracion import (
    DESCARGA_DESDE,
    DESCARGA_HASTA,
    RUTA_MACRO_BRUTO,
    SERIES_MACRO,
    URL_FRED_CSV,
)
from cripto.corto_plazo_v4_6_macro.utilidades_macro import (
    detectar_columnas_fred,
)


def crear_sesion() -> requests.Session:
    reintentos = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(
            429,
            500,
            502,
            503,
            504,
        ),
        allowed_methods=("GET",),
    )

    adaptador = HTTPAdapter(
        max_retries=reintentos
    )

    sesion = requests.Session()
    sesion.mount(
        "https://",
        adaptador,
    )
    sesion.headers.update(
        {
            "User-Agent": (
                "tutanGestion-v4.6-macro/1.0"
            )
        }
    )

    return sesion


def descargar_serie(
    sesion: requests.Session,
    serie: str,
    desde: str,
    hasta: str,
    sobrescribir: bool,
) -> dict[str, object]:
    ruta = RUTA_MACRO_BRUTO / f"{serie}.csv"

    if ruta.exists() and not sobrescribir:
        estado = "conservado"
    else:
        respuesta = sesion.get(
            URL_FRED_CSV,
            params={
                "id": serie,
                "cosd": desde,
                "coed": hasta,
            },
            timeout=60,
        )
        respuesta.raise_for_status()

        ruta.write_bytes(
            respuesta.content
        )
        estado = "descargado"

    datos = pd.read_csv(ruta)

    columna_fecha, columna_valor = detectar_columnas_fred(
        datos,
        serie,
    )

    fechas = pd.to_datetime(
        datos[columna_fecha],
        errors="coerce",
    )

    valores = pd.to_numeric(
        datos[columna_valor],
        errors="coerce",
    )

    validos = fechas.notna() & valores.notna()

    if not validos.any():
        raise ValueError(
            f"La descarga de {serie} no contiene datos válidos."
        )

    return {
        "serie": serie,
        "estado": estado,
        "filas_validas": int(validos.sum()),
        "primera_fecha": str(
            fechas.loc[validos].min().date()
        ),
        "ultima_fecha": str(
            fechas.loc[validos].max().date()
        ),
        "archivo": str(ruta),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--desde",
        default=DESCARGA_DESDE,
    )
    parser.add_argument(
        "--hasta",
        default=DESCARGA_HASTA,
    )
    parser.add_argument(
        "--sobrescribir",
        action="store_true",
    )

    argumentos = parser.parse_args()

    RUTA_MACRO_BRUTO.mkdir(
        parents=True,
        exist_ok=True,
    )

    sesion = crear_sesion()
    resultados = []

    print(
        "\nDESCARGA DE DATOS MACRO V4.6"
    )
    print("=" * 72)
    print(
        f"Periodo solicitado: {argumentos.desde} a {argumentos.hasta}"
    )

    for detalle in SERIES_MACRO.values():
        serie = str(
            detalle["serie"]
        )

        resultado = descargar_serie(
            sesion=sesion,
            serie=serie,
            desde=argumentos.desde,
            hasta=argumentos.hasta,
            sobrescribir=argumentos.sobrescribir,
        )

        resultados.append(resultado)

        print(
            f"{serie}: "
            f"{resultado['filas_validas']:,} registros, "
            f"{resultado['primera_fecha']} a "
            f"{resultado['ultima_fecha']} "
            f"({resultado['estado']})"
        )

    manifiesto = {
        "fecha_descarga_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "desde": argumentos.desde,
        "hasta": argumentos.hasta,
        "fuente": URL_FRED_CSV,
        "series": resultados,
    }

    ruta_manifiesto = (
        RUTA_MACRO_BRUTO
        / "detalle_descarga.json"
    )

    ruta_manifiesto.write_text(
        json.dumps(
            manifiesto,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\nDATOS MACRO DESCARGADOS CORRECTAMENTE"
    )
    print(f"- {ruta_manifiesto}")


if __name__ == "__main__":
    main()
