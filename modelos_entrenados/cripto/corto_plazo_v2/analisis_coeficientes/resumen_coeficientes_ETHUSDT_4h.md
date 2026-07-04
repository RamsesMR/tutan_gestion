# Coeficientes V2A — ETHUSDT

Los coeficientes corresponden a variables previamente estandarizadas. Un valor alto indica mayor influencia lineal, pero no demuestra causalidad.

## Variables con mayor importancia media absoluta

| Posición | Variable | Grupo | Cruzada | Importancia |
|---:|---|---|---|---:|
| 1 | volatilidad_240m | volatilidad_propia | no | 0.309427 |
| 2 | beta_propio_otro_240m | beta | sí | 0.213267 |
| 3 | ratio_volatilidad_240m | volatilidad_relativa | sí | 0.137508 |
| 4 | otro_volatilidad_60m | volatilidad_otro | sí | 0.135902 |
| 5 | rango_relativo | forma_vela | no | 0.112489 |
| 6 | dia_semana_seno | tiempo | no | 0.100287 |
| 7 | minuto_dia_coseno | tiempo | no | 0.094381 |
| 8 | volatilidad_30m | volatilidad_propia | no | 0.090839 |
| 9 | correlacion_btc_eth_240m | correlacion | sí | 0.086212 |
| 10 | beta_propio_otro_1440m | beta | sí | 0.074088 |
| 11 | cuerpo_relativo | forma_vela | no | 0.067718 |
| 12 | otro_volatilidad_240m | volatilidad_otro | sí | 0.062549 |
| 13 | operaciones_relativas_240m | operaciones_relativas | no | 0.061928 |
| 14 | volatilidad_60m | volatilidad_propia | no | 0.061656 |
| 15 | volatilidad_15m | volatilidad_propia | no | 0.061184 |
| 16 | beta_propio_otro_60m | beta | sí | 0.053435 |
| 17 | residual_propio_otro_240m | residual | sí | 0.050557 |
| 18 | volumen_relativo_240m | volumen_relativo | no | 0.046913 |
| 19 | residual_propio_otro_60m | residual | sí | 0.046432 |
| 20 | residual_propio_otro_1440m | residual | sí | 0.046308 |

## Importancia por grupo

| Grupo | Variables | Media | Total | Máxima |
|---|---:|---:|---:|---:|
| volatilidad_propia | 5 | 0.109593 | 0.547965 | 0.536310 |
| beta | 3 | 0.113597 | 0.340790 | 0.333318 |
| tiempo | 4 | 0.061494 | 0.245976 | 0.134653 |
| forma_vela | 4 | 0.057478 | 0.229914 | 0.215920 |
| volatilidad_otro | 2 | 0.099225 | 0.198451 | 0.201936 |
| volatilidad_relativa | 2 | 0.088274 | 0.176548 | 0.215999 |
| operaciones_relativas | 5 | 0.031246 | 0.156231 | 0.079080 |
| residual | 3 | 0.047766 | 0.143297 | 0.073771 |
| correlacion | 3 | 0.046825 | 0.140474 | 0.144083 |
| divergencia_rendimiento | 6 | 0.018266 | 0.109598 | 0.058968 |
| volumen_relativo | 5 | 0.020863 | 0.104314 | 0.102207 |
| desviacion_media | 5 | 0.019844 | 0.099219 | 0.045815 |
| rendimiento_propio | 6 | 0.014362 | 0.086171 | 0.052142 |
| rendimiento_otro | 6 | 0.011021 | 0.066127 | 0.061422 |
| zscore_divergencia | 2 | 0.015776 | 0.031552 | 0.022969 |
| presion_compradora | 2 | 0.004890 | 0.009780 | 0.008902 |

## Coeficientes más positivos para BAJA

| Variable | Grupo | Coeficiente |
|---|---|---:|
| volatilidad_240m | volatilidad_propia | 0.272848 |
| ratio_volatilidad_240m | volatilidad_relativa | 0.215999 |
| correlacion_btc_eth_240m | correlacion | 0.144083 |
| dia_semana_seno | tiempo | 0.091776 |
| beta_propio_otro_1440m | beta | 0.089305 |
| residual_propio_otro_1440m | residual | 0.073771 |
| correlacion_btc_eth_60m | correlacion | 0.071352 |
| operaciones_relativas_240m | operaciones_relativas | 0.067608 |
| rendimiento_240m | rendimiento_propio | 0.052142 |
| volatilidad_60m | volatilidad_propia | 0.046569 |

## Coeficientes más negativos para BAJA

| Variable | Grupo | Coeficiente |
|---|---|---:|
| beta_propio_otro_240m | beta | -0.333318 |
| otro_volatilidad_60m | volatilidad_otro | -0.142617 |
| minuto_dia_coseno | tiempo | -0.083013 |
| residual_propio_otro_240m | residual | -0.062442 |
| otro_volatilidad_240m | volatilidad_otro | -0.060082 |
| operaciones_relativas_60m | operaciones_relativas | -0.053322 |
| operaciones_relativas_30m | operaciones_relativas | -0.040166 |
| minuto_dia_seno | tiempo | -0.034962 |
| volumen_relativo_60m | volumen_relativo | -0.034959 |
| ratio_volatilidad_60m | volatilidad_relativa | -0.024904 |

## Coeficientes más positivos para NEUTRAL

| Variable | Grupo | Coeficiente |
|---|---|---:|
| beta_propio_otro_240m | beta | 0.220848 |
| minuto_dia_coseno | tiempo | 0.134653 |
| otro_volatilidad_240m | volatilidad_otro | 0.089891 |
| cuerpo_relativo | forma_vela | 0.050304 |
| operaciones_relativas_5m | operaciones_relativas | 0.048628 |
| divergencia_rendimiento_60m | divergencia_rendimiento | 0.044822 |
| ratio_volatilidad_60m | volatilidad_relativa | 0.039878 |
| desviacion_media_cierre_240m | desviacion_media | 0.035436 |
| minuto_dia_seno | tiempo | 0.034118 |
| divergencia_rendimiento_240m | divergencia_rendimiento | 0.027688 |

## Coeficientes más negativos para NEUTRAL

| Variable | Grupo | Coeficiente |
|---|---|---:|
| volatilidad_240m | volatilidad_propia | -0.536310 |
| rango_relativo | forma_vela | -0.215920 |
| volatilidad_30m | volatilidad_propia | -0.184794 |
| ratio_volatilidad_240m | volatilidad_relativa | -0.144295 |
| volatilidad_15m | volatilidad_propia | -0.125758 |
| dia_semana_seno | tiempo | -0.119119 |
| beta_propio_otro_1440m | beta | -0.092731 |
| correlacion_btc_eth_240m | correlacion | -0.092457 |
| operaciones_relativas_240m | operaciones_relativas | -0.079080 |
| otro_volatilidad_60m | volatilidad_otro | -0.063151 |

## Coeficientes más positivos para SUBE

| Variable | Grupo | Coeficiente |
|---|---|---:|
| otro_volatilidad_60m | volatilidad_otro | 0.201936 |
| volatilidad_240m | volatilidad_propia | 0.119124 |
| rango_relativo | forma_vela | 0.112660 |
| dia_semana_seno | tiempo | 0.089967 |
| beta_propio_otro_240m | beta | 0.085636 |
| volatilidad_30m | volatilidad_propia | 0.072767 |
| beta_propio_otro_60m | beta | 0.072524 |
| residual_propio_otro_240m | residual | 0.068158 |
| otro_rendimiento_1m | rendimiento_otro | 0.061422 |
| residual_propio_otro_60m | residual | 0.057310 |

## Coeficientes más negativos para SUBE

| Variable | Grupo | Coeficiente |
|---|---|---:|
| cuerpo_relativo | forma_vela | -0.139284 |
| volumen_relativo_240m | volumen_relativo | -0.102207 |
| volatilidad_60m | volatilidad_propia | -0.085597 |
| minuto_dia_coseno | tiempo | -0.065476 |
| divergencia_rendimiento_240m | divergencia_rendimiento | -0.058968 |
| residual_propio_otro_1440m | residual | -0.057561 |
| ratio_volatilidad_60m | volatilidad_relativa | -0.052339 |
| ratio_volatilidad_240m | volatilidad_relativa | -0.052230 |
| desviacion_media_cierre_240m | desviacion_media | -0.045815 |
| divergencia_rendimiento_60m | divergencia_rendimiento | -0.040441 |