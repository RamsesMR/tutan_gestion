from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class EventoBallena:
    evento_id: str
    tx_hash: str
    fecha_disponible: str
    activo: str
    cantidad_activo: float
    valor_usd: float
    tipo_origen: str
    tipo_destino: str
    direccion_flujo: str
    fuente: str
    fecha_blockchain: str | None = None
    fecha_detectada: str | None = None
    cadena: str = "bitcoin"
    direccion_origen: str | None = None
    direccion_destino: str | None = None
    entidad_origen: str | None = None
    entidad_destino: str | None = None
    exchange_origen: str | None = None
    exchange_destino: str | None = None
    identificador_entidad_origen: str | None = None
    identificador_entidad_destino: str | None = None
    confianza_etiquetado: float = 0.0
    identidad_verificada: bool = False
    es_movimiento_interno_probable: bool = False
    fuente_id: str | None = None
    metadata_json: str | None = None

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)
