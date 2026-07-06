from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from .http import ClienteHTTP


class ClienteGlassnode:
    BASE_URL = "https://api.glassnode.com/v1/metrics"

    def __init__(self, cliente: ClienteHTTP | None = None):
        self.api_key = os.getenv("GLASSNODE_API_KEY")
        if not self.api_key:
            raise RuntimeError("Falta GLASSNODE_API_KEY")
        self.cliente = cliente or ClienteHTTP()

    def obtener(
        self,
        ruta_metrica: str,
        desde: datetime,
        hasta: datetime,
        *,
        intervalo: str = "10m",
        activo: str = "BTC",
    ) -> pd.DataFrame:
        datos = self.cliente.get_json(
            f"{self.BASE_URL}/{ruta_metrica.lstrip('/')}",
            params={
                "api_key": self.api_key,
                "a": activo,
                "s": int(desde.timestamp()),
                "u": int(hasta.timestamp()),
                "i": intervalo,
            },
        )
        df = pd.DataFrame(datos or [])
        if "t" in df:
            df["fecha_disponible"] = pd.to_datetime(df["t"], unit="s", utc=True, errors="coerce")
        return df
