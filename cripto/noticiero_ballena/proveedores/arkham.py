from __future__ import annotations

import os
from typing import Any

from .http import ClienteHTTP


class ClienteArkham:
    """Enriquecimiento opcional de direcciones con Arkham.

    El endpoint concreto se configura porque el acceso y la referencia pueden
    variar por cuenta. El módulo nunca inventa atribuciones.
    """

    def __init__(self, cliente: ClienteHTTP | None = None):
        self.api_key = os.getenv("ARKHAM_API_KEY")
        self.url_template = os.getenv(
            "ARKHAM_ADDRESS_URL_TEMPLATE",
            "https://api.arkm.com/intelligence/address/{address}/all",
        )
        if not self.api_key:
            raise RuntimeError("Falta ARKHAM_API_KEY")
        self.cliente = cliente or ClienteHTTP()

    def resolver_direccion(self, address: str, chain: str = "bitcoin") -> dict[str, Any]:
        url = self.url_template.format(chain=chain, address=address)
        respuesta = self.cliente.get_json(
            url,
            headers={"API-Key": self.api_key},
            params={"chain": chain} if "/all" not in url else None,
        )
        return respuesta if isinstance(respuesta, dict) else {"raw": respuesta}
