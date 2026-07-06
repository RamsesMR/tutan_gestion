from __future__ import annotations

import time
from typing import Any

import requests


class ClienteHTTP:
    def __init__(self, timeout: int = 30, reintentos: int = 4, espera_base: float = 1.0):
        self.timeout = timeout
        self.reintentos = reintentos
        self.espera_base = espera_base
        self.sesion = requests.Session()
        self.sesion.headers.update({"User-Agent": "tutanGestion-noticiero-ballena/1.0"})

    def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        ultimo_error: Exception | None = None
        for intento in range(self.reintentos):
            try:
                respuesta = self.sesion.get(url, headers=headers, params=params, timeout=self.timeout)
                if respuesta.status_code == 429:
                    espera = float(respuesta.headers.get("Retry-After", self.espera_base * (2**intento)))
                    time.sleep(espera)
                    continue
                respuesta.raise_for_status()
                return respuesta.json()
            except (requests.RequestException, ValueError) as exc:
                ultimo_error = exc
                if intento + 1 < self.reintentos:
                    time.sleep(self.espera_base * (2**intento))
        raise RuntimeError(f"No fue posible consultar {url}: {ultimo_error}")
