# Comparación directa V4.1 original frente a V4.5

## Configuración comparada

- **V4.1 original:** detector `v4_63_sin_pesos`, 63 variables y umbral `0,44`.
- **V4.5 candidata:** variante `flujo_spot`, 90 variables y umbral `0.42`.
- **Entrada:** cruce desde abajo.
- **Salida:** fija a 480 minutos.
- **Coste:** 0,10 %.
- **Selección:** 2023 y 2024.
- **2025 utilizado para selección:** no.
- **2026 utilizado para selección:** no.

## Resultados

| Pliegue | Modelo | Operaciones | Precisión | Positivas | Retorno neto medio | Factor beneficio | Drawdown |
|---|---|---:|---:|---:|---:|---:|---:|
| validacion_2023 | V4.1 original | 70 | 50.00 % | 61.43 % | 0.5191 % | 1.8054 | -8.8724 % |
| validacion_2023 | V4.5 candidata | 79 | 49.37 % | 65.82 % | 0.5339 % | 1.9582 | -8.7463 % |
| validacion_2024 | V4.1 original | 127 | 48.03 % | 61.42 % | 0.3497 % | 1.4338 | -12.5804 % |
| validacion_2024 | V4.5 candidata | 130 | 46.15 % | 59.23 % | 0.4128 % | 1.5702 | -11.8612 % |

## Diferencias de V4.5 respecto a V4.1

| Pliegue | Mejora relativa retorno | Mejora relativa factor | Mejora drawdown |
|---|---:|---:|---:|
| validacion_2023 | 2.8573 % | 8.4631 % | 0.1261 % |
| validacion_2024 | 18.0391 % | 9.5076 % | 0.7192 % |

## Decisión

**V4.5 queda seleccionada como candidata de desarrollo, pero todavía no sustituye a V4.1 hasta completar las confirmaciones posteriores con la configuración congelada.**
