from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .configuracion import RUTA_CRUDOS, UMBRAL_BALLENA_USD
from .proveedores.whale_alert import escuchar_alertas


def main() -> None:
    parser = argparse.ArgumentParser(description="Recolecta alertas BTC en tiempo real.")
    parser.add_argument("--salida", type=Path, default=RUTA_CRUDOS / "whale_alert_tiempo_real.jsonl")
    parser.add_argument("--min-usd", type=float, default=UMBRAL_BALLENA_USD)
    args = parser.parse_args()
    asyncio.run(escuchar_alertas(args.salida, args.min_usd))


if __name__ == "__main__":
    main()
