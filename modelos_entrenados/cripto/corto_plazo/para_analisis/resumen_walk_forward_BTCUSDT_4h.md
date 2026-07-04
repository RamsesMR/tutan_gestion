# Validación walk-forward 2025 — BTCUSDT

Cada mes fue evaluado con multiplicadores elegidos exclusivamente mediante los meses anteriores.

Antes de ajustar cada mes se excluyeron las últimas cuatro horas del periodo anterior para impedir que el objetivo cruce la frontera temporal.

La división de prueba de 2026 no fue utilizada.

## Configuración

- **Primer mes evaluado:** abril 2025
- **Criterio:** mayor F1-score macro; después Balanced Accuracy y Accuracy.
- **Pérdida máxima de Accuracy durante el ajuste:** 0.0300

## Resumen

- F1-score macro mejoró en **9 de 9 meses**.
- Balanced Accuracy mejoró en **9 de 9 meses**.
- Accuracy mejoró en **0 de 9 meses**.
- Cambio medio de Accuracy: **-0.0387**.
- Cambio medio de Balanced Accuracy: **+0.0211**.
- Cambio medio de F1-score macro: **+0.0433**.

## Comparación mensual fuera de muestra

| Mes evaluado | Multiplicador BAJA | Multiplicador SUBE | Accuracy original | Accuracy ajustada | Δ Accuracy | Balanced original | Balanced ajustada | Δ Balanced | F1 original | F1 ajustada | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| abril 2025 | 1.55 | 1.35 | 0.5406 | 0.4998 | -0.0408 | 0.4081 | 0.4337 | +0.0255 | 0.3892 | 0.4249 | +0.0356 |
| mayo 2025 | 1.55 | 1.40 | 0.5953 | 0.5385 | -0.0569 | 0.3663 | 0.3983 | +0.0320 | 0.3340 | 0.3880 | +0.0541 |
| junio 2025 | 1.50 | 1.40 | 0.6471 | 0.5935 | -0.0536 | 0.3675 | 0.3883 | +0.0208 | 0.3397 | 0.3878 | +0.0481 |
| julio 2025 | 1.45 | 1.35 | 0.6676 | 0.6146 | -0.0530 | 0.3455 | 0.3746 | +0.0292 | 0.3134 | 0.3706 | +0.0572 |
| agosto 2025 | 1.40 | 1.30 | 0.6482 | 0.6174 | -0.0308 | 0.3493 | 0.3588 | +0.0095 | 0.3197 | 0.3511 | +0.0313 |
| septiembre 2025 | 1.40 | 1.30 | 0.7219 | 0.6945 | -0.0274 | 0.3466 | 0.3703 | +0.0237 | 0.3182 | 0.3623 | +0.0441 |
| octubre 2025 | 1.40 | 1.30 | 0.5439 | 0.5175 | -0.0263 | 0.3754 | 0.3971 | +0.0217 | 0.3478 | 0.3938 | +0.0461 |
| noviembre 2025 | 1.40 | 1.30 | 0.5033 | 0.4903 | -0.0130 | 0.4036 | 0.4253 | +0.0216 | 0.3622 | 0.4177 | +0.0555 |
| diciembre 2025 | 1.40 | 1.30 | 0.6039 | 0.5570 | -0.0469 | 0.4012 | 0.4067 | +0.0054 | 0.3898 | 0.4073 | +0.0176 |

## Recall por clase

| Mes | BAJA original | BAJA ajustada | NEUTRAL original | NEUTRAL ajustada | SUBE original | SUBE ajustada |
|---|---:|---:|---:|---:|---:|---:|
| abril 2025 | 0.0869 | 0.3949 | 0.8604 | 0.6899 | 0.2771 | 0.2162 |
| mayo 2025 | 0.0489 | 0.3092 | 0.9402 | 0.7595 | 0.1098 | 0.1263 |
| junio 2025 | 0.0377 | 0.1799 | 0.9631 | 0.8285 | 0.1017 | 0.1565 |
| julio 2025 | 0.0250 | 0.1810 | 0.9551 | 0.8322 | 0.0563 | 0.1107 |
| agosto 2025 | 0.0152 | 0.1186 | 0.9371 | 0.8644 | 0.0955 | 0.0934 |
| septiembre 2025 | 0.0295 | 0.1385 | 0.9755 | 0.9158 | 0.0348 | 0.0565 |
| octubre 2025 | 0.0693 | 0.2304 | 0.8880 | 0.7695 | 0.1688 | 0.1914 |
| noviembre 2025 | 0.0403 | 0.2091 | 0.8363 | 0.7040 | 0.3343 | 0.3628 |
| diciembre 2025 | 0.0880 | 0.2072 | 0.8841 | 0.7648 | 0.2317 | 0.2481 |

## Distribución de predicciones

| Mes | BAJA original | BAJA ajustada | NEUTRAL original | NEUTRAL ajustada | SUBE original | SUBE ajustada |
|---|---:|---:|---:|---:|---:|---:|
| abril 2025 | 5.70 % | 29.22 % | 76.65 % | 57.40 % | 17.66 % | 13.38 % |
| mayo 2025 | 2.78 % | 22.95 % | 91.23 % | 68.89 % | 5.99 % | 8.17 % |
| junio 2025 | 1.84 % | 13.22 % | 93.83 % | 78.50 % | 4.34 % | 8.28 % |
| julio 2025 | 1.89 % | 13.91 % | 94.55 % | 79.35 % | 3.56 % | 6.74 % |
| agosto 2025 | 2.71 % | 10.52 % | 91.93 % | 82.27 % | 5.37 % | 7.20 % |
| septiembre 2025 | 1.59 % | 7.76 % | 96.63 % | 89.43 % | 1.78 % | 2.81 % |
| octubre 2025 | 3.32 % | 14.59 % | 82.97 % | 68.94 % | 13.71 % | 16.47 % |
| noviembre 2025 | 4.52 % | 16.16 % | 74.10 % | 57.09 % | 21.38 % | 26.75 % |
| diciembre 2025 | 4.40 % | 15.50 % | 81.19 % | 67.47 % | 14.41 % | 17.03 % |
