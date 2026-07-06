from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .configuracion import ARCHIVO_EVENTOS_NORMALIZADOS
from .proveedores.whale_alert import convertir_alerta
from .utilidades import guardar_tabla, leer_json


def _payloads(ruta: Path) -> list[dict]:
    if ruta.suffix.lower() in {".jsonl", ".ndjson"}:
        salida = []
        with ruta.open("r", encoding="utf-8") as archivo:
            for linea in archivo:
                linea = linea.strip()
                if linea:
                    salida.append(json.loads(linea))
        return salida
    contenido = leer_json(ruta)
    if isinstance(contenido, list):
        return [x for x in contenido if isinstance(x, dict)]
    if isinstance(contenido, dict):
        for clave in ("alerts", "data", "events", "transactions"):
            if isinstance(contenido.get(clave), list):
                return [x for x in contenido[clave] if isinstance(x, dict)]
        return [contenido]
    return []


def _disponibilidad(payload: dict, latencia_minutos: float) -> datetime:
    for clave in ("published_at", "alert_timestamp", "received_at", "created_at"):
        valor = payload.get(clave)
        if valor is None:
            continue
        try:
            if isinstance(valor, (int, float)):
                return datetime.fromtimestamp(float(valor), tz=timezone.utc)
            return datetime.fromisoformat(str(valor).replace("Z", "+00:00")).astimezone(timezone.utc)
        except (TypeError, ValueError, OSError):
            pass
    marca = payload.get("timestamp")
    try:
        blockchain = datetime.fromtimestamp(float(marca), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        blockchain = datetime.now(timezone.utc)
    return blockchain + timedelta(minutes=float(latencia_minutos))


def importar(ruta: Path, latencia_minutos: float) -> pd.DataFrame:
    eventos = []
    for payload in _payloads(ruta):
        recibido = _disponibilidad(payload, latencia_minutos)
        eventos.extend(evento.a_dict() for evento in convertir_alerta(payload, recibido))
    return pd.DataFrame(eventos)


def main() -> None:
    parser = argparse.ArgumentParser(description="Importa el archivo histórico oficial de Whale Alert.")
    parser.add_argument("--entrada", required=True, type=Path)
    parser.add_argument("--salida", type=Path, default=ARCHIVO_EVENTOS_NORMALIZADOS)
    parser.add_argument(
        "--latencia-minutos",
        type=float,
        default=5.0,
        help="Latencia conservadora si el archivo no incluye fecha de publicación.",
    )
    args = parser.parse_args()
    df = importar(args.entrada, args.latencia_minutos)
    guardar_tabla(df, args.salida)
    print(f"Eventos importados: {len(df):,}")
    print(f"Salida: {args.salida}")


if __name__ == "__main__":
    main()
