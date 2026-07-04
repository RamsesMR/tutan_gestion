# Coeficientes V2A — BTCUSDT

Los coeficientes corresponden a variables previamente estandarizadas. Un valor alto indica mayor influencia lineal, pero no demuestra causalidad.

## Variables con mayor importancia media absoluta

| Posición | Variable | Grupo | Cruzada | Importancia |
|---:|---|---|---|---:|
| 1 | otro_volatilidad_240m | volatilidad_otro | sí | 0.285407 |
| 2 | otro_volatilidad_60m | volatilidad_otro | sí | 0.211986 |
| 3 | volatilidad_60m | volatilidad_propia | no | 0.193997 |
| 4 | ratio_volatilidad_240m | volatilidad_relativa | sí | 0.170119 |
| 5 | beta_propio_otro_240m | beta | sí | 0.160081 |
| 6 | ratio_volatilidad_60m | volatilidad_relativa | sí | 0.155118 |
| 7 | minuto_dia_coseno | tiempo | no | 0.141293 |
| 8 | beta_propio_otro_60m | beta | sí | 0.140244 |
| 9 | dia_semana_seno | tiempo | no | 0.137548 |
| 10 | volatilidad_240m | volatilidad_propia | no | 0.119607 |
| 11 | correlacion_btc_eth_60m | correlacion | sí | 0.110999 |
| 12 | rango_relativo | forma_vela | no | 0.091010 |
| 13 | volatilidad_30m | volatilidad_propia | no | 0.077923 |
| 14 | cuerpo_relativo | forma_vela | no | 0.069636 |
| 15 | volatilidad_15m | volatilidad_propia | no | 0.063347 |
| 16 | correlacion_btc_eth_1440m | correlacion | sí | 0.057926 |
| 17 | residual_propio_otro_240m | residual | sí | 0.057624 |
| 18 | operaciones_relativas_60m | operaciones_relativas | no | 0.054848 |
| 19 | correlacion_btc_eth_240m | correlacion | sí | 0.053740 |
| 20 | volumen_relativo_240m | volumen_relativo | no | 0.053188 |

## Importancia por grupo

| Grupo | Variables | Media | Total | Máxima |
|---|---:|---:|---:|---:|
| volatilidad_otro | 2 | 0.248697 | 0.497394 | 0.417920 |
| volatilidad_propia | 5 | 0.096045 | 0.480223 | 0.320634 |
| tiempo | 4 | 0.084468 | 0.337871 | 0.175780 |
| beta | 3 | 0.109133 | 0.327399 | 0.207703 |
| volatilidad_relativa | 2 | 0.162618 | 0.325237 | 0.272969 |
| correlacion | 3 | 0.074222 | 0.222665 | 0.132478 |
| forma_vela | 4 | 0.052818 | 0.211271 | 0.176449 |
| operaciones_relativas | 5 | 0.031073 | 0.155365 | 0.074295 |
| divergencia_rendimiento | 6 | 0.021891 | 0.131344 | 0.094019 |
| desviacion_media | 5 | 0.024743 | 0.123716 | 0.069854 |
| volumen_relativo | 5 | 0.023881 | 0.119405 | 0.076469 |
| residual | 3 | 0.037230 | 0.111690 | 0.094830 |
| rendimiento_propio | 6 | 0.018392 | 0.110349 | 0.070743 |
| rendimiento_otro | 6 | 0.012208 | 0.073250 | 0.039460 |
| zscore_divergencia | 2 | 0.033541 | 0.067082 | 0.072412 |
| presion_compradora | 2 | 0.002448 | 0.004896 | 0.007028 |

## Coeficientes más positivos para BAJA

| Variable | Grupo | Coeficiente |
|---|---|---:|
| otro_volatilidad_60m | volatilidad_otro | 0.277246 |
| ratio_volatilidad_60m | volatilidad_relativa | 0.241127 |
| beta_propio_otro_240m | beta | 0.207703 |
| volatilidad_240m | volatilidad_propia | 0.148335 |
| dia_semana_seno | tiempo | 0.135012 |
| otro_volatilidad_240m | volatilidad_otro | 0.128693 |
| correlacion_btc_eth_60m | correlacion | 0.123010 |
| cuerpo_relativo | forma_vela | 0.095235 |
| residual_propio_otro_240m | residual | 0.094830 |
| desviacion_media_cierre_240m | desviacion_media | 0.069854 |

## Coeficientes más negativos para BAJA

| Variable | Grupo | Coeficiente |
|---|---|---:|
| volatilidad_60m | volatilidad_propia | -0.320634 |
| ratio_volatilidad_240m | volatilidad_relativa | -0.218214 |
| beta_propio_otro_60m | beta | -0.186900 |
| minuto_dia_coseno | tiempo | -0.130399 |
| divergencia_rendimiento_240m | divergencia_rendimiento | -0.094019 |
| rendimiento_1m | rendimiento_propio | -0.070743 |
| operaciones_relativas_60m | operaciones_relativas | -0.061665 |
| correlacion_btc_eth_1440m | correlacion | -0.045703 |
| volumen_relativo_60m | volumen_relativo | -0.042724 |
| minuto_dia_seno | tiempo | -0.042360 |

## Coeficientes más positivos para NEUTRAL

| Variable | Grupo | Coeficiente |
|---|---|---:|
| minuto_dia_coseno | tiempo | 0.175780 |
| volatilidad_60m | volatilidad_propia | 0.132018 |
| beta_propio_otro_60m | beta | 0.111812 |
| rendimiento_1m | rendimiento_propio | 0.054404 |
| divergencia_rendimiento_240m | divergencia_rendimiento | 0.049374 |
| operaciones_relativas_5m | operaciones_relativas | 0.039595 |
| minuto_dia_seno | tiempo | 0.039133 |
| otro_rendimiento_1m | rendimiento_otro | 0.030210 |
| divergencia_rendimiento_1m | divergencia_rendimiento | 0.024954 |
| divergencia_rendimiento_60m | divergencia_rendimiento | 0.023602 |

## Coeficientes más negativos para NEUTRAL

| Variable | Grupo | Coeficiente |
|---|---|---:|
| otro_volatilidad_240m | volatilidad_otro | -0.417920 |
| otro_volatilidad_60m | volatilidad_otro | -0.332643 |
| rango_relativo | forma_vela | -0.176449 |
| dia_semana_seno | tiempo | -0.157769 |
| ratio_volatilidad_60m | volatilidad_relativa | -0.141836 |
| beta_propio_otro_240m | beta | -0.127902 |
| volatilidad_30m | volatilidad_propia | -0.118387 |
| volatilidad_15m | volatilidad_propia | -0.110906 |
| correlacion_btc_eth_60m | correlacion | -0.077507 |
| volumen_relativo_240m | volumen_relativo | -0.076469 |

## Coeficientes más positivos para SUBE

| Variable | Grupo | Coeficiente |
|---|---|---:|
| otro_volatilidad_240m | volatilidad_otro | 0.309609 |
| ratio_volatilidad_240m | volatilidad_relativa | 0.272969 |
| correlacion_btc_eth_60m | correlacion | 0.132478 |
| volatilidad_60m | volatilidad_propia | 0.129338 |
| correlacion_btc_eth_240m | correlacion | 0.126529 |
| dia_semana_seno | tiempo | 0.119864 |
| correlacion_btc_eth_1440m | correlacion | 0.101174 |
| rango_relativo | forma_vela | 0.093518 |
| ratio_volatilidad_60m | volatilidad_relativa | 0.082390 |
| operaciones_relativas_60m | operaciones_relativas | 0.074295 |

## Coeficientes más negativos para SUBE

| Variable | Grupo | Coeficiente |
|---|---|---:|
| volatilidad_240m | volatilidad_propia | -0.160942 |
| beta_propio_otro_240m | beta | -0.144639 |
| beta_propio_otro_60m | beta | -0.122020 |
| minuto_dia_coseno | tiempo | -0.117701 |
| zscore_divergencia_1440m | zscore_divergencia | -0.072412 |
| desviacion_media_cierre_30m | desviacion_media | -0.054497 |
| cuerpo_relativo | forma_vela | -0.048230 |
| operaciones_relativas_5m | operaciones_relativas | -0.046358 |
| beta_propio_otro_1440m | beta | -0.043659 |
| volumen_relativo_240m | volumen_relativo | -0.040682 |