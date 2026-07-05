from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import requests

from cripto.corto_plazo_v4_5.configuracion import (
    RUTA_FUNDING,
    SIMBOLOS,
    URL_FUNDING_BINANCE,
)
from cripto.corto_plazo_v4_5.utilidades_datos import (
    ruta_funding,
)


def timestamp_ms(
    valor: str,
) -> int:
    return int(
        pd.Timestamp(
            valor,
            tz="UTC",
        ).timestamp()
        * 1000
    )


def descargar_simbolo(
    sesion: requests.Session,
    simbolo: str,
    desde: str,
    hasta: str,
) -> pd.DataFrame:
    inicio = timestamp_ms(desde)
    final = timestamp_ms(hasta) - 1
    registros: list[dict] = []

    while inicio <= final:
        respuesta = sesion.get(
            URL_FUNDING_BINANCE,
            params={
                "symbol": simbolo,
                "startTime": inicio,
                "endTime": final,
                "limit": 1000,
            },
            timeout=60,
        )
        respuesta.raise_for_status()
        lote = respuesta.json()

        if not lote:
            break

        registros.extend(lote)

        ultimo = int(
            lote[-1]["fundingTime"]
        )

        if ultimo < inicio:
            raise RuntimeError(
                "Binance devolvió funding fuera de orden."
            )

        inicio = ultimo + 1
        time.sleep(0.1)

        if len(lote) < 1000:
            break

    if not registros:
        raise ValueError(
            f"No se obtuvo funding para {simbolo}."
        )

    datos = pd.DataFrame(registros)

    datos["fecha_funding"] = pd.to_datetime(
        pd.to_numeric(
            datos["fundingTime"],
            errors="raise",
        ).astype("int64"),
        unit="ms",
        utc=True,
    )

    datos["funding_rate"] = pd.to_numeric(
        datos["fundingRate"],
        errors="raise",
    )

    if "markPrice" in datos.columns:
        datos["mark_price"] = pd.to_numeric(
            datos["markPrice"],
            errors="coerce",
        )
    else:
        datos["mark_price"] = pd.NA

    return (
        datos[
            [
                "fecha_funding",
                "funding_rate",
                "mark_price",
            ]
        ]
        .sort_values("fecha_funding")
        .drop_duplicates(
            subset=["fecha_funding"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--desde",
        default="2020-12-01",
    )
    parser.add_argument(
        "--hasta",
        default="2025-01-01",
    )
    parser.add_argument(
        "--sobrescribir",
        action="store_true",
    )
    argumentos = parser.parse_args()

    RUTA_FUNDING.mkdir(
        parents=True,
        exist_ok=True,
    )

    sesion = requests.Session()
    sesion.headers.update(
        {
            "User-Agent": "tutanGestion-v4.5",
        }
    )

    manifiesto = {
        "desde": argumentos.desde,
        "hasta": argumentos.hasta,
        "simbolos": {},
    }

    print("\nDESCARGA DE FUNDING USD-M FUTURES")
    print("=" * 72)

    for simbolo in SIMBOLOS:
        ruta = ruta_funding(simbolo)

        if ruta.exists() and not argumentos.sobrescribir:
            raise FileExistsError(
                f"Ya existe: {ruta}. Usa --sobrescribir."
            )

        datos = descargar_simbolo(
            sesion,
            simbolo,
            argumentos.desde,
            argumentos.hasta,
        )

        temporal = ruta.with_suffix(".parquet.tmp")
        datos.to_parquet(
            temporal,
            index=False,
        )
        temporal.replace(ruta)

        manifiesto["simbolos"][simbolo] = {
            "filas": len(datos),
            "primera_fecha": datos[
                "fecha_funding"
            ].min().isoformat(),
            "ultima_fecha": datos[
                "fecha_funding"
            ].max().isoformat(),
            "archivo": str(ruta),
        }

        print(
            f"{simbolo}: {len(datos):,} registros".replace(",", ".")
        )
        print(f"- {ruta}")

    (
        RUTA_FUNDING / "manifiesto_funding.json"
    ).write_text(
        json.dumps(
            manifiesto,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print("\nFUNDING DESCARGADO CORRECTAMENTE")


if __name__ == "__main__":
    main()
