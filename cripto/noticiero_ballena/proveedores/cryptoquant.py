from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd

from .http import ClienteHTTP


class ClienteCryptoQuant:
    BASE_URL = "https://api.cryptoquant.com/v1"

    def __init__(self, cliente: ClienteHTTP | None = None):
        self.token = os.getenv("CRYPTOQUANT_API_KEY")
        if not self.token:
            raise RuntimeError("Falta CRYPTOQUANT_API_KEY")
        self.cliente = cliente or ClienteHTTP()

    def obtener_metrica(
        self,
        endpoint: str,
        desde: datetime,
        hasta: datetime,
        *,
        ventana: str,
        exchange: str | None = None,
        limite: int = 100000,
    ) -> pd.DataFrame:
        parametros: dict[str, Any] = {
            "from": desde.strftime("%Y%m%dT%H%M%S"),
            "to": hasta.strftime("%Y%m%dT%H%M%S"),
            "window": ventana,
            "limit": int(limite),
            "format": "json",
        }
        if exchange:
            parametros["exchange"] = exchange
        datos = self.cliente.get_json(
            f"{self.BASE_URL}/{endpoint.lstrip('/')}",
            headers={"Authorization": f"Bearer {self.token}"},
            params=parametros,
        )
        filas: Any = datos.get("result", {}).get("data") if isinstance(datos, dict) else datos
        if filas is None and isinstance(datos, dict):
            filas = datos.get("data", [])
        df = pd.DataFrame(filas or [])
        if "date" in df:
            df["fecha_disponible"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        return df

    def exchange_whale_ratio(self, desde: datetime, hasta: datetime, *, ventana: str, exchange: str) -> pd.DataFrame:
        return self.obtener_metrica(
            "btc/flow-indicator/exchange-whale-ratio",
            desde,
            hasta,
            ventana=ventana,
            exchange=exchange,
        )
