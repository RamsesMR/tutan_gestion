# Validación walk-forward 2025 — ETHUSDT

Cada mes fue evaluado con multiplicadores elegidos exclusivamente mediante los meses anteriores.

Antes de ajustar cada mes se excluyeron las últimas cuatro horas del periodo anterior para impedir que el objetivo cruce la frontera temporal.

La división de prueba de 2026 no fue utilizada.

## Configuración

- **Primer mes evaluado:** abril 2025
- **Criterio:** mayor F1-score macro; después Balanced Accuracy y Accuracy.
- **Pérdida máxima de Accuracy durante el ajuste:** 0.0500

## Resumen

- F1-score macro mejoró en **9 de 9 meses**.
- Balanced Accuracy mejoró en **8 de 9 meses**.
- Accuracy mejoró en **1 de 9 meses**.
- Cambio medio de Accuracy: **-0.0166**.
- Cambio medio de Balanced Accuracy: **+0.0150**.
- Cambio medio de F1-score macro: **+0.0502**.

## Comparación mensual fuera de muestra

| Mes evaluado | Multiplicador BAJA | Multiplicador SUBE | Accuracy original | Accuracy ajustada | Δ Accuracy | Balanced original | Balanced ajustada | Δ Balanced | F1 original | F1 ajustada | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| abril 2025 | 1.50 | 1.35 | 0.4113 | 0.4080 | -0.0033 | 0.4033 | 0.4080 | +0.0047 | 0.3616 | 0.4085 | +0.0468 |
| mayo 2025 | 1.50 | 1.35 | 0.4130 | 0.4086 | -0.0045 | 0.3999 | 0.4117 | +0.0118 | 0.3587 | 0.4104 | +0.0517 |
| junio 2025 | 1.40 | 1.25 | 0.4343 | 0.4322 | -0.0021 | 0.3901 | 0.4165 | +0.0264 | 0.3502 | 0.4102 | +0.0600 |
| julio 2025 | 1.45 | 1.30 | 0.4131 | 0.3828 | -0.0304 | 0.3696 | 0.3855 | +0.0160 | 0.3367 | 0.3750 | +0.0383 |
| agosto 2025 | 1.40 | 1.25 | 0.3945 | 0.3643 | -0.0302 | 0.3711 | 0.3615 | -0.0096 | 0.3307 | 0.3589 | +0.0282 |
| septiembre 2025 | 1.40 | 1.25 | 0.5321 | 0.5079 | -0.0242 | 0.3721 | 0.4166 | +0.0445 | 0.3350 | 0.4123 | +0.0773 |
| octubre 2025 | 1.40 | 1.30 | 0.4262 | 0.4199 | -0.0063 | 0.3871 | 0.4079 | +0.0208 | 0.3420 | 0.4074 | +0.0653 |
| noviembre 2025 | 1.40 | 1.30 | 0.4266 | 0.4357 | +0.0092 | 0.4315 | 0.4383 | +0.0068 | 0.3807 | 0.4324 | +0.0517 |
| diciembre 2025 | 1.40 | 1.30 | 0.5434 | 0.4855 | -0.0579 | 0.4196 | 0.4332 | +0.0136 | 0.3997 | 0.4318 | +0.0321 |

## Recall por clase

| Mes | BAJA original | BAJA ajustada | NEUTRAL original | NEUTRAL ajustada | SUBE original | SUBE ajustada |
|---|---:|---:|---:|---:|---:|---:|
| abril 2025 | 0.0837 | 0.4030 | 0.6448 | 0.4052 | 0.4814 | 0.4157 |
| mayo 2025 | 0.0722 | 0.4452 | 0.6529 | 0.4275 | 0.4747 | 0.3624 |
| junio 2025 | 0.0760 | 0.4241 | 0.7729 | 0.5752 | 0.3215 | 0.2501 |
| julio 2025 | 0.0752 | 0.4510 | 0.7200 | 0.4513 | 0.3134 | 0.2542 |
| agosto 2025 | 0.0657 | 0.3411 | 0.6933 | 0.4625 | 0.3542 | 0.2809 |
| septiembre 2025 | 0.0426 | 0.3546 | 0.8908 | 0.7130 | 0.1830 | 0.1823 |
| octubre 2025 | 0.0541 | 0.3118 | 0.7546 | 0.5217 | 0.3525 | 0.3902 |
| noviembre 2025 | 0.0941 | 0.3111 | 0.7051 | 0.4967 | 0.4951 | 0.5071 |
| diciembre 2025 | 0.0808 | 0.3313 | 0.8203 | 0.6081 | 0.3578 | 0.3603 |

## Distribución de predicciones

| Mes | BAJA original | BAJA ajustada | NEUTRAL original | NEUTRAL ajustada | SUBE original | SUBE ajustada |
|---|---:|---:|---:|---:|---:|---:|
| abril 2025 | 6.50 % | 36.45 % | 53.91 % | 30.59 % | 39.59 % | 32.96 % |
| mayo 2025 | 7.49 % | 39.40 % | 51.99 % | 28.52 % | 40.51 % | 32.08 % |
| junio 2025 | 4.50 % | 34.03 % | 70.62 % | 47.26 % | 24.88 % | 18.71 % |
| julio 2025 | 6.99 % | 39.31 % | 64.56 % | 37.35 % | 28.45 % | 23.34 % |
| agosto 2025 | 6.60 % | 34.58 % | 61.51 % | 38.11 % | 31.89 % | 27.31 % |
| septiembre 2025 | 2.63 % | 26.04 % | 84.38 % | 62.69 % | 13.00 % | 11.27 % |
| octubre 2025 | 5.19 % | 26.72 % | 65.60 % | 42.35 % | 29.21 % | 30.93 % |
| noviembre 2025 | 7.10 % | 24.34 % | 54.13 % | 34.56 % | 38.77 % | 41.09 % |
| diciembre 2025 | 4.14 % | 24.89 % | 71.34 % | 49.43 % | 24.53 % | 25.68 % |
