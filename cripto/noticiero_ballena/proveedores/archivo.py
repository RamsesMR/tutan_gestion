from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utilidades import leer_tabla


class ProveedorArchivo:
    """Entrada universal para históricos CSV, JSON, JSONL o Parquet."""

    nombre = "archivo"

    def __init__(self, ruta: Path):
        self.ruta = ruta

    def obtener_dataframe(self) -> pd.DataFrame:
        return leer_tabla(self.ruta)
