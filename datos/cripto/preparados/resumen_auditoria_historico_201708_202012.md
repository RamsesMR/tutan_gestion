# Auditoría histórica BTCUSDT y ETHUSDT

- **Desde:** 2017-08-01T00:00:00+00:00
- **Hasta exclusivo:** 2021-01-01T00:00:00+00:00
- **Intervalo:** 1m
- **Fuente:** Binance Spot almacenado en PostgreSQL.
- **Relleno de huecos:** no se inventaron velas.

## Resumen por símbolo

| Símbolo | Primera vela | Última vela | Encontradas | Faltantes internas | Duplicados | Incoherentes | Huecos |
|---|---|---|---:|---:|---:|---:|---:|
| BTCUSDT | 2017-08-17 04:00:00+00:00 | 2020-12-31 23:59:00+00:00 | 1767791 | 7489 | 0 | 0 | 27 |
| ETHUSDT | 2017-08-17 04:00:00+00:00 | 2020-12-31 23:59:00+00:00 | 1767790 | 7490 | 0 | 0 | 27 |

## Solapamiento BTC/ETH

- **Primera vela común:** 2017-08-17 04:00:00+00:00
- **Última vela común:** 2020-12-31 23:59:00+00:00
- **Velas presentes en ambos:** 1746188
- **Velas presentes solo en BTC:** 21603
- **Velas presentes solo en ETH:** 21602

## Criterio previo al entrenamiento

Esta auditoría no aprueba automáticamente los datos para entrenar. Primero deben revisarse los huecos, el solapamiento entre símbolos y la cantidad de filas que sobrevivirá a las ventanas de variables.
