from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .configuracion import RUTA_CRUDOS, UMBRAL_BALLENA_BTC, UMBRAL_BALLENA_USD


def main() -> None:
    parser = argparse.ArgumentParser(description="Recolecta movimientos BTC grandes en tiempo real.")
    parser.add_argument(
        "--proveedor",
        choices=("blockchain_com", "whale_alert"),
        default="blockchain_com",
        help="blockchain_com es gratuito y no requiere token; whale_alert es opcional y de pago.",
    )
    parser.add_argument("--salida", type=Path)
    parser.add_argument("--min-usd", type=float, default=UMBRAL_BALLENA_USD)
    parser.add_argument("--min-btc", type=float, default=UMBRAL_BALLENA_BTC)
    args = parser.parse_args()

    if args.proveedor == "blockchain_com":
        from .proveedores.blockchain_com import escuchar_transacciones

        salida = args.salida or (RUTA_CRUDOS / "blockchain_com_tiempo_real.jsonl")
        asyncio.run(escuchar_transacciones(
            salida,
            min_value_usd=args.min_usd,
            min_btc_respaldo=args.min_btc,
        ))
        return

    from .proveedores.whale_alert import escuchar_alertas

    salida = args.salida or (RUTA_CRUDOS / "whale_alert_tiempo_real.jsonl")
    asyncio.run(escuchar_alertas(salida, args.min_usd))


if __name__ == "__main__":
    main()
