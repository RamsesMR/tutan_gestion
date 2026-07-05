# Detector BAJA de corto plazo — V2 por eventos, EV y macro

Laboratorio independiente para decidir si una operación short sobre BTCUSDT
tiene valor esperado neto positivo. No predice un precio futuro exacto.

## Protección

No modifica:

- `cripto/corto_plazo_v4_1`
- `cripto/corto_plazo_v4_5`
- `cripto/corto_plazo_v4_6_macro`
- `cripto/corto_plazo_baja_v1`
- sus datos, modelos o resultados.

## Fuente

Reutiliza los Parquet generados por BAJA V1:

```text
datos/cripto/preparados/corto_plazo_baja_v1
```

BAJA V1 ya contiene las 63 variables V2A/V4, OHLC y el objetivo short
`corto_tp100_sl050_480m`.

## Cambio arquitectónico

BAJA V1 etiquetaba cada minuto. BAJA V2 compara rejillas de eventos:

- 5 minutos;
- 10 minutos;
- 15 minutos;
- 30 minutos.

Cada evento conserva:

- `event_id`;
- fecha de inicio;
- fecha final del horizonte;
- `overlap_group`;
- clase `SL`, `TP` o `TIMEOUT`;
- retorno bruto y neto;
- duración.

## Objetivo económico

- TP: +1,00 %.
- SL: -0,50 %.
- Horizonte: 480 minutos.
- Coste oficial: 0,10 %.

El modelo estima:

```text
EV = P(TP) × retorno_neto_TP
   + P(SL) × retorno_neto_SL
   + P(TIMEOUT) × retorno_neto_timeout_estimado
```

Se abre una operación solo cuando EV supera un margen:

- 0;
- 0,025 %;
- 0,05 %;
- 0,10 %.

## Modelos

- HistGradientBoosting multiclase para `SL/TP/TIMEOUT`.
- HistGradientBoostingRegressor para retorno del TIMEOUT.
- Control: 63 variables.
- Macro: seis variables de V4.6, primero individualmente y luego juntas.

## Variables macro V4.6

- `impulso_dolar_amplio_20s`
- `impulso_rendimiento_real_10y_20s`
- `impulso_spread_baa_20s`
- `impulso_vix_20s`
- `retorno_nasdaq100_20s`
- `impulso_reservas_fed_4s`

El generador busca Parquet macro ya integrados en:

```text
datos/cripto/preparados/corto_plazo_v4_6_macro
```

No interpola ni rellena hacia atrás. Si no encuentra macro, genera el control
pero deja las columnas macro sin cobertura; la auditoría lo registra.

## Desarrollo temporal

```text
2021–2022 -> validación 2023
2021–2023 -> validación 2024
```

Se purgan eventos cuyo horizonte cruza el límite temporal.

2025 y 2026 no participan en selección.

## Resultados

```text
modelos_entrenados/cripto/corto_plazo_baja_v2/desarrollo
```

Archivos principales:

```text
resultados/historial_desarrollo.csv
seleccion/seleccion_baja_v2.csv
seleccion/decision_baja_v2.json
auditorias/auditoria_eventos.json
```

## Orden obligatorio

Lee y ejecuta `COMANDOS_BAJA_V2.txt`.

Primero:

```powershell
python -m cripto.corto_plazo_baja_v2.generar_eventos
python -m cripto.corto_plazo_baja_v2.auditar_eventos
python -m cripto.corto_plazo_baja_v2.entrenar_modelos --rejillas 5,10,15,30 --variantes control_63_eventos --semillas 42
python -m cripto.corto_plazo_baja_v2.seleccionar_modelo
```

Detenerse después del control y revisar resultados antes de entrenar macro.

## Criterios provisionales

Por cada pliegue:

- mínimo 50 operaciones;
- retorno neto medio positivo;
- mediana no negativa;
- PF > 1,10;
- drawdown >= -20 %;
- concentración mensual <= 35 %.

La promoción automática permanece en `false`.
