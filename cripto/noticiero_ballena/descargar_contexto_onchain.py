from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .configuracion import RUTA_CRUDOS
from .proveedores.cryptoquant import ClienteCryptoQuant
from .proveedores.glassnode import ClienteGlassnode
from .utilidades import guardar_tabla


def _fecha(valor: str) -> datetime:
    return datetime.fromisoformat(valor.replace("Z", "+00:00")).astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga contexto on-chain opcional.")
    parser.add_argument("--proveedor", choices=("cryptoquant", "glassnode"), required=True)
    parser.add_argument("--desde", required=True)
    parser.add_argument("--hasta", required=True)
    parser.add_argument("--salida", type=Path)
    parser.add_argument("--ventana", default="hour")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--endpoint")
    args = parser.parse_args()

    desde, hasta = _fecha(args.desde), _fecha(args.hasta)
    if args.proveedor == "cryptoquant":
        cliente = ClienteCryptoQuant()
        endpoint = args.endpoint or "btc/flow-indicator/exchange-whale-ratio"
        df = cliente.obtener_metrica(
            endpoint,
            desde,
            hasta,
            ventana=args.ventana,
            exchange=args.exchange,
        )
        salida = args.salida or RUTA_CRUDOS / "cryptoquant_contexto.parquet"
    else:
        if not args.endpoint:
            raise SystemExit("Glassnode requiere --endpoint con una métrica autorizada por tu plan.")
        cliente = ClienteGlassnode()
        df = cliente.obtener(args.endpoint, desde, hasta, intervalo=args.ventana)
        salida = args.salida or RUTA_CRUDOS / "glassnode_contexto.parquet"
    guardar_tabla(df, salida)
    print(f"Filas descargadas: {len(df):,}")
    print(f"Salida: {salida}")


if __name__ == "__main__":
    main()
