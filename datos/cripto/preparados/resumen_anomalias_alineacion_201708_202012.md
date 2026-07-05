# Auditoría de anomalías tras normalizar al minuto

- **Desde:** 2017-08-01T00:00:00+00:00
- **Hasta exclusivo:** 2021-01-01T00:00:00+00:00
- **Intervalo:** 1m
- **Datos modificados:** no.

## Minutos problemáticos

| Minuto | BTC | ETH | Tipo |
|---|---:|---:|---|
| 2017-12-04 06:00:00+00:00 | 2 | 2 | COLISION_AMBOS |
| 2020-12-21 14:08:00+00:00 | 1 | 0 | FALTA_ETH |

## Resultado

- **Minutos problemáticos:** 2
- **Velas originales implicadas:** 5

Esta auditoría no corrige ni elimina registros. Su objetivo es definir después una regla determinista para preparar los datos sin alterar PostgreSQL ni los archivos originales.
