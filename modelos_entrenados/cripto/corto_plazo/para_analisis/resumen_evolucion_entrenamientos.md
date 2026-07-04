# Evolución de entrenamientos

Este archivo resume los entrenamientos registrados para los modelos cripto de corto plazo.

## Resumen general

| Fecha UTC | Símbolo | Etiqueta | Accuracy | Balanced Accuracy | F1-score macro | Δ Balanced Accuracy | Δ F1 macro |
|---|---|---|---:|---:|---:|---:|---:|
| 2026-07-04T16:24:25+00:00 | BTCUSDT | sgd_base_v2 | 0.5863 | 0.3907 | 0.3685 | +0.0574 | +0.1220 |
| 2026-07-04T16:24:43+00:00 | ETHUSDT | sgd_base_v1 | 0.4348 | 0.4013 | 0.3592 | +0.0679 | +0.1715 |

---

## BTCUSDT — sgd_base_v2

- **ID:** `20260704T162425Z_BTCUSDT_sgd_base_v2`
- **Fecha UTC:** 2026-07-04T16:24:25+00:00
- **Modelo:** SGDClassifier
- **Horizonte:** 4h
- **Umbral:** ±0.50 %
- **Variables:** 36
- **Épocas:** 3
- **Tamaño de lote:** 100.000
- **Muestras de entrenamiento:** 2.099.167
- **Muestras de validación:** 525.600

### Métricas principales

| Modelo | Accuracy | Balanced Accuracy | F1-score macro |
|---|---:|---:|---:|
| SGDClassifier | 0.5863 | 0.3907 | 0.3685 |
| DummyClassifier (most_frequent) | 0.5864 | 0.3333 | 0.2464 |

### Métricas por clase

| Clase | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| BAJA | 0.2982 | 0.0542 | 0.0917 | 104.660 |
| NEUTRAL | 0.6341 | 0.9027 | 0.7449 | 308.235 |
| SUBE | 0.3579 | 0.2153 | 0.2688 | 112.705 |

### Matriz de confusión

| Real \ Predicha | BAJA | NEUTRAL | SUBE |
|---|---:|---:|---:|
| BAJA | 5.668 | 77.829 | 21.163 |
| NEUTRAL | 7.620 | 278.250 | 22.365 |
| SUBE | 5.718 | 82.726 | 24.261 |

### Notas

Segundo entrenamiento del modelo base

---

## ETHUSDT — sgd_base_v1

- **ID:** `20260704T162443Z_ETHUSDT_sgd_base_v1`
- **Fecha UTC:** 2026-07-04T16:24:43+00:00
- **Modelo:** SGDClassifier
- **Horizonte:** 4h
- **Umbral:** ±0.50 %
- **Variables:** 36
- **Épocas:** 3
- **Tamaño de lote:** 100.000
- **Muestras de entrenamiento:** 2.099.167
- **Muestras de validación:** 525.600

### Métricas principales

| Modelo | Accuracy | Balanced Accuracy | F1-score macro |
|---|---:|---:|---:|
| SGDClassifier | 0.4348 | 0.4013 | 0.3592 |
| DummyClassifier (most_frequent) | 0.3920 | 0.3333 | 0.1877 |

### Métricas por clase

| Clase | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| BAJA | 0.3630 | 0.0688 | 0.1156 | 154.166 |
| NEUTRAL | 0.4633 | 0.7430 | 0.5707 | 206.023 |
| SUBE | 0.3907 | 0.3921 | 0.3914 | 165.411 |

### Matriz de confusión

| Real \ Predicha | BAJA | NEUTRAL | SUBE |
|---|---:|---:|---:|
| BAJA | 10.599 | 87.477 | 56.090 |
| NEUTRAL | 7.913 | 153.069 | 45.041 |
| SUBE | 10.687 | 89.863 | 64.861 |

### Notas

Primer entrenamiento base registrado de ETHUSDT
