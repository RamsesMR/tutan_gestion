from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable

from ..esquemas import EventoBallena


class ProveedorEventosBallena(ABC):
    nombre: str

    @abstractmethod
    def obtener_eventos(self, desde: datetime, hasta: datetime) -> Iterable[EventoBallena]:
        raise NotImplementedError
