from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

TABLA_TRANSACCIONES = "bigquery-public-data.crypto_bitcoin.transactions"
SATOSHIS_POR_BTC = 100_000_000


# IMPORTANTE:
# La tabla transactions está particionada por block_timestamp_month.
# Consultar la vista outputs directamente obliga a escanear prácticamente todo
# el histórico. Esta consulta entra por la tabla particionada y desanida outputs.
CONSULTA_OUTPUTS_GRANDES = f"""
SELECT
  t.hash AS tx_hash,
  t.block_hash,
  t.block_number,
  t.block_timestamp,
  o.`index` AS output_index,
  ARRAY_TO_STRING(o.addresses, '|') AS direcciones_destino,
  o.type AS tipo_salida,
  CAST(o.value AS BIGNUMERIC) AS valor_satoshis
FROM `{TABLA_TRANSACCIONES}` AS t
CROSS JOIN UNNEST(t.outputs) AS o
WHERE t.block_timestamp_month = @mes_particion
  AND t.block_timestamp >= @inicio
  AND t.block_timestamp < @fin
  AND o.value >= @min_satoshis
ORDER BY t.block_timestamp, t.hash, o.`index`
""".strip()


@dataclass(frozen=True, slots=True)
class EstimacionConsulta:
    inicio: datetime
    fin: datetime
    min_satoshis: int
    bytes_procesados: int

    @property
    def gib_procesados(self) -> float:
        return self.bytes_procesados / (1024 ** 3)


class ClienteBigQueryBitcoin:
    """Acceso controlado al histórico público de Bitcoin en BigQuery.

    No almacena credenciales. Usa Application Default Credentials (ADC),
    poda la consulta por partición mensual y aplica un límite máximo de bytes
    facturables a cada consulta real.
    """

    def __init__(self, proyecto: str | None = None, *, ubicacion: str = "US") -> None:
        try:
            from google.cloud import bigquery
        except ImportError as exc:  # pragma: no cover - mensaje operativo
            raise RuntimeError(
                "Falta google-cloud-bigquery. Instala requirements_noticiero_ballena.txt"
            ) from exc

        self._bigquery = bigquery
        try:
            self.cliente = bigquery.Client(project=proyecto, location=ubicacion)
        except Exception as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError(
                "No se pudieron cargar credenciales de Google Cloud. "
                "Ejecuta: gcloud auth application-default login"
            ) from exc
        self.ubicacion = ubicacion

    def _configuracion(
        self,
        inicio: datetime,
        fin: datetime,
        min_satoshis: int,
        *,
        dry_run: bool,
        max_bytes: int | None = None,
    ) -> Any:
        bigquery = self._bigquery
        mes_particion = inicio.date().replace(day=1)
        config = bigquery.QueryJobConfig(
            dry_run=dry_run,
            use_query_cache=False,
            query_parameters=[
                bigquery.ScalarQueryParameter("mes_particion", "DATE", mes_particion),
                bigquery.ScalarQueryParameter("inicio", "TIMESTAMP", inicio),
                bigquery.ScalarQueryParameter("fin", "TIMESTAMP", fin),
                bigquery.ScalarQueryParameter("min_satoshis", "BIGNUMERIC", min_satoshis),
            ],
        )
        if max_bytes is not None:
            config.maximum_bytes_billed = int(max_bytes)
        return config

    def estimar(
        self,
        inicio: datetime,
        fin: datetime,
        min_satoshis: int,
    ) -> EstimacionConsulta:
        trabajo = self.cliente.query(
            CONSULTA_OUTPUTS_GRANDES,
            job_config=self._configuracion(
                inicio,
                fin,
                min_satoshis,
                dry_run=True,
            ),
            location=self.ubicacion,
        )
        return EstimacionConsulta(
            inicio=inicio,
            fin=fin,
            min_satoshis=int(min_satoshis),
            bytes_procesados=int(trabajo.total_bytes_processed or 0),
        )

    def descargar(
        self,
        inicio: datetime,
        fin: datetime,
        min_satoshis: int,
        *,
        max_bytes: int,
    ) -> pd.DataFrame:
        trabajo = self.cliente.query(
            CONSULTA_OUTPUTS_GRANDES,
            job_config=self._configuracion(
                inicio,
                fin,
                min_satoshis,
                dry_run=False,
                max_bytes=max_bytes,
            ),
            location=self.ubicacion,
        )
        filas = [dict(fila.items()) for fila in trabajo.result(page_size=20_000)]
        return pd.DataFrame(filas)
