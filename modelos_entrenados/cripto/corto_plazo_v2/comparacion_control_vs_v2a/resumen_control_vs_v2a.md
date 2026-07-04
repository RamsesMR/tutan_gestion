# Comparación control V1 vs V2A

El control y la V2A utilizan exactamente las mismas filas, periodos, clases, épocas, lotes y algoritmo.

## BTCUSDT

| Métrica | Control 36 | V2A 63 | Cambio |
|---|---:|---:|---:|
| Accuracy | 0.5857 | 0.5734 | -0.0124 |
| Balanced Accuracy | 0.3927 | 0.3965 | +0.0038 |
| F1 macro | 0.3668 | 0.3846 | +0.0179 |

**Veredicto automático:** `VARIABLES_CRUZADAS_APORTAN`

| Clase | Métrica | Control | V2A | Cambio |
|---|---|---:|---:|---:|
| BAJA | precision | 0.3030 | 0.2553 | -0.0477 |
| BAJA | recall | 0.0391 | 0.0913 | +0.0522 |
| BAJA | f1_score | 0.0693 | 0.1345 | +0.0652 |
| NEUTRAL | precision | 0.6359 | 0.6463 | +0.0104 |
| NEUTRAL | recall | 0.8975 | 0.8598 | -0.0377 |
| NEUTRAL | f1_score | 0.7444 | 0.7379 | -0.0065 |
| SUBE | precision | 0.3528 | 0.3436 | -0.0092 |
| SUBE | recall | 0.2415 | 0.2384 | -0.0030 |
| SUBE | f1_score | 0.2867 | 0.2815 | -0.0052 |

## ETHUSDT

| Métrica | Control 36 | V2A 63 | Cambio |
|---|---:|---:|---:|
| Accuracy | 0.4335 | 0.4335 | -0.0000 |
| Balanced Accuracy | 0.3998 | 0.4059 | +0.0061 |
| F1 macro | 0.3619 | 0.3873 | +0.0254 |

**Veredicto automático:** `VARIABLES_CRUZADAS_APORTAN`

| Clase | Métrica | Control | V2A | Cambio |
|---|---|---:|---:|---:|
| BAJA | precision | 0.3554 | 0.3550 | -0.0003 |
| BAJA | recall | 0.0846 | 0.1671 | +0.0825 |
| BAJA | f1_score | 0.1367 | 0.2273 | +0.0905 |
| NEUTRAL | precision | 0.4609 | 0.4728 | +0.0119 |
| NEUTRAL | recall | 0.7507 | 0.6987 | -0.0521 |
| NEUTRAL | f1_score | 0.5711 | 0.5640 | -0.0072 |
| SUBE | precision | 0.3926 | 0.3915 | -0.0011 |
| SUBE | recall | 0.3641 | 0.3519 | -0.0122 |
| SUBE | f1_score | 0.3778 | 0.3707 | -0.0072 |
