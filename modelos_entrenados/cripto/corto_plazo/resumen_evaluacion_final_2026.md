# Evaluación final única — prueba 2026

- **Fecha UTC:** 2026-07-04T17:33:43+00:00
- **Estado:** evaluación final completada.
- **Configuración:** congelada antes de abrir la prueba.
- **Ajustes posteriores sobre 2026:** ninguno.

## Resultado global

| Símbolo | Decisión | Accuracy | Balanced Accuracy | F1-score macro |
|---|---|---:|---:|---:|
| BTCUSDT | Original | 0.5580 | 0.3981 | 0.3747 |
| BTCUSDT | Ajustada congelada | 0.5281 | 0.4124 | 0.4118 |
| ETHUSDT | Original | 0.4873 | 0.3970 | 0.3608 |
| ETHUSDT | Ajustada congelada | 0.4685 | 0.4136 | 0.4119 |

## Cambios de la decisión ajustada

| Símbolo | Δ Accuracy | Δ Balanced Accuracy | Δ F1-score macro |
|---|---:|---:|---:|
| BTCUSDT | -0.0299 | +0.0142 | +0.0371 |
| ETHUSDT | -0.0187 | +0.0166 | +0.0512 |

## BTCUSDT — métricas por clase ajustadas

| Clase | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| BAJA | 0.2880 | 0.2044 | 0.2391 | 48.113 |
| NEUTRAL | 0.6399 | 0.7490 | 0.6901 | 122.297 |
| SUBE | 0.3327 | 0.2837 | 0.3062 | 46.790 |

## BTCUSDT — comparación mensual

| Mes | Accuracy original | Accuracy ajustada | Δ Accuracy | Balanced original | Balanced ajustada | Δ Balanced | F1 original | F1 ajustada | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| enero 2026 | 0.6347 | 0.6014 | -0.0334 | 0.3985 | 0.4085 | +0.0101 | 0.3832 | 0.4115 | +0.0283 |
| febrero 2026 | 0.4137 | 0.4170 | +0.0033 | 0.3918 | 0.4084 | +0.0166 | 0.3503 | 0.4022 | +0.0519 |
| marzo 2026 | 0.4648 | 0.4347 | -0.0301 | 0.3885 | 0.3911 | +0.0026 | 0.3552 | 0.3854 | +0.0303 |
| abril 2026 | 0.5941 | 0.5369 | -0.0572 | 0.3746 | 0.3793 | +0.0047 | 0.3567 | 0.3723 | +0.0156 |
| mayo 2026 | 0.6706 | 0.6408 | -0.0298 | 0.3518 | 0.3734 | +0.0215 | 0.3243 | 0.3719 | +0.0476 |

## ETHUSDT — métricas por clase ajustadas

| Clase | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| BAJA | 0.3220 | 0.2585 | 0.2867 | 57.847 |
| NEUTRAL | 0.5856 | 0.6557 | 0.6187 | 105.669 |
| SUBE | 0.3343 | 0.3267 | 0.3304 | 53.684 |

## ETHUSDT — comparación mensual

| Mes | Accuracy original | Accuracy ajustada | Δ Accuracy | Balanced original | Balanced ajustada | Δ Balanced | F1 original | F1 ajustada | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| enero 2026 | 0.5349 | 0.5256 | -0.0093 | 0.3925 | 0.4251 | +0.0326 | 0.3566 | 0.4241 | +0.0675 |
| febrero 2026 | 0.3709 | 0.3941 | +0.0232 | 0.3781 | 0.3965 | +0.0184 | 0.3285 | 0.3904 | +0.0619 |
| marzo 2026 | 0.4230 | 0.3933 | -0.0298 | 0.3825 | 0.3797 | -0.0028 | 0.3479 | 0.3782 | +0.0302 |
| abril 2026 | 0.5051 | 0.4572 | -0.0479 | 0.3738 | 0.3789 | +0.0051 | 0.3403 | 0.3749 | +0.0347 |
| mayo 2026 | 0.5924 | 0.5656 | -0.0268 | 0.3643 | 0.3875 | +0.0232 | 0.3316 | 0.3843 | +0.0527 |

## Regla posterior

Los resultados de 2026 no deben utilizarse para modificar esta misma configuración y volver a medirla sobre el mismo periodo.

Cualquier modelo nuevo deberá evaluarse con un periodo temporal posterior que permanezca sin utilizar durante su desarrollo.
