# V5 — Oportunidades financieras BTC SUBE

La V5 cambia el objetivo. Ya no intenta adivinar únicamente el precio al finalizar cuatro horas. Evalúa si una operación larga alcanza un take profit antes que un stop loss.

## Qué incluye

- 63 variables de la V2A.
- 38 variables técnicas nuevas.
- Tres objetivos de triple barrera.
- Control lineal con SGD.
- Modelo no lineal HistGradientBoosting.
- Costes de 0,05 %, 0,10 % y 0,20 %.
- Validaciones walk-forward en 2023 y 2024.
- Confirmación secundaria en 2025.
- Recolector de futuros de Binance para la siguiente ampliación.

## Importante

La primera versión no usa probabilidades de V4 como variable porque hacerlo correctamente exige probabilidades fuera de muestra. Meter la predicción del modelo entrenado con los mismos datos produciría filtración. Esa capa se añadirá después mediante stacking temporal.

## Instalación

Copiar la carpeta completa en:

```text
cripto/corto_plazo_v5/
```

Comprobar imports:

```powershell
python -c "from cripto.corto_plazo_v5.configuracion import COLUMNAS_MODELO_V5; print(len(COLUMNAS_MODELO_V5))"
```

## Paso 1 — Inspeccionar la fuente

```powershell
python -m cripto.corto_plazo_v5.inspeccionar_fuentes `
    --simbolo BTCUSDT
```

Si detecta `apertura`, `maximo`, `minimo` y `cierre`, continúa directamente.

Si faltan, vuelve a ejecutar el generador indicando la carpeta donde guardas los Parquet originales de velas de un minuto.

## Paso 2 — Generar datos V5

Cuando OHLC ya esté dentro de los Parquet V2A:

```powershell
python -m cripto.corto_plazo_v5.generar_datos_v5 `
    --simbolo BTCUSDT
```

Cuando OHLC esté en otra carpeta:

```powershell
python -m cripto.corto_plazo_v5.generar_datos_v5 `
    --simbolo BTCUSDT `
    --ruta-precios "C:\RUTA\A\TUS\PARQUET\DE\PRECIOS"
```

Para regenerar:

```powershell
python -m cripto.corto_plazo_v5.generar_datos_v5 `
    --simbolo BTCUSDT `
    --sobrescribir
```

## Paso 3 — Primer experimento

Empieza únicamente con el objetivo base:

```powershell
python -m cripto.corto_plazo_v5.entrenar_modelos `
    --simbolo BTCUSDT `
    --objetivo largo_tp070_sl035_240m `
    --variante TODAS `
    --epocas 3 `
    --tamano-lote 100000
```

Esto compara:

- `sgd_sin_pesos`
- `sgd_moderado`
- `histgb_moderado`

en validación 2023 y 2024.

## Paso 4 — Seleccionar modelo y umbral

```powershell
python -m cripto.corto_plazo_v5.seleccionar_modelo
```

Genera:

```text
modelos_entrenados/cripto/corto_plazo_v5/desarrollo/resumen/
├── comparacion_completa_v5.csv
├── ranking_v5.csv
└── seleccion_v5.csv
```

## Paso 5 — Probar las otras barreras

Solo después de analizar el objetivo base:

```powershell
python -m cripto.corto_plazo_v5.entrenar_modelos `
    --simbolo BTCUSDT `
    --objetivo TODOS `
    --variante TODAS `
    --epocas 3 `
    --tamano-lote 100000
```

Y volver a ejecutar:

```powershell
python -m cripto.corto_plazo_v5.seleccionar_modelo
```

## Paso 6 — Confirmación secundaria en 2025

No ejecutar hasta congelar objetivo, modelo y umbral con 2023/2024.

```powershell
python -m cripto.corto_plazo_v5.confirmar_2025 `
    --simbolo BTCUSDT `
    --epocas 3 `
    --tamano-lote 100000
```

## Recolector de futuros

Instalar dependencias:

```powershell
pip install -r .\cripto\corto_plazo_v5\requirements_v5.txt
```

Descarga el histórico disponible y actualiza archivos:

```powershell
python -m cripto.corto_plazo_v5.recolector_futuros_binance `
    --simbolos BTCUSDT ETHUSDT `
    --periodo 5m `
    --dias 30 `
    --dias-funding 3650
```

Los endpoints de interés abierto agregado, taker, ratios y basis solo ofrecen aproximadamente los últimos 30 días. Conviene ejecutar el recolector diariamente.

## Criterio de éxito

La V5 no pasa por tener mejor accuracy. Debe mostrar, en 2023 y 2024:

- PR-AUC superior a la prevalencia.
- Retorno neto medio y mediano positivos con coste de 0,10 %.
- Factor de beneficio superior a 1.
- Al menos 50 operaciones no solapadas por pliegue.
- Resultado razonable alrededor del umbral elegido.
- Drawdown aceptable.
- Confirmación secundaria positiva en 2025 sin reajustar.
