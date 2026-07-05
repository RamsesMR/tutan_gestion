# Confirmación de V4.5-Spot en 2025 y 2026

Esta fase compara:

```text
V4.1:
63 variables
umbral 0,44

V4.5-Spot:
90 variables
umbral 0,42
```

Ambas configuraciones utilizan:

```text
BTCUSDT
entrada por cruce desde abajo
sin veto BAJA
salida fija de 480 minutos
coste 0,001
SGD logístico sin pesos
3 épocas
lote 100000
semilla 42
```

## Interpretación temporal

### Confirmación 2025

```text
Entrenamiento:
2021-01-01 a 2025-01-01

Evaluación:
2025-01-01 a 2026-01-01
```

### Confirmación enero-mayo de 2026

```text
Entrenamiento:
2021-01-01 a 2026-01-01

Evaluación:
2026-01-01 a 2026-06-01
```

2025 y 2026 ya fueron observados dentro del proyecto con versiones
anteriores, por lo que no son periodos vírgenes para el proyecto. Sin
embargo, no se utilizaron para seleccionar las variables, el umbral ni
la política de V4.5-Spot.

## Archivos

Añadir dentro de:

```text
cripto/corto_plazo_v4_5
```

los siguientes archivos:

```text
configuracion_confirmacion.py
preparar_datos_confirmacion.py
validar_datos_confirmacion.py
confirmar_2025_2026.py
```

No se reemplaza ningún archivo existente.

## Ejecución

### 1. Preparar las 27 variables Spot faltantes

```powershell
python -m cripto.corto_plazo_v4_5.preparar_datos_confirmacion
```

Los archivos ya existentes se conservan. Para regenerarlos:

```powershell
python -m cripto.corto_plazo_v4_5.preparar_datos_confirmacion --sobrescribir
```

### 2. Validar todos los años necesarios

```powershell
python -m cripto.corto_plazo_v4_5.validar_datos_confirmacion
```

El control de 63 variables se leerá desde los mismos Parquet Spot que
la candidata de 90 variables. Así se garantiza una comparación con
exactamente las mismas velas.

### 3. Confirmar 2025 y 2026

```powershell
python -m cripto.corto_plazo_v4_5.confirmar_2025_2026 --epocas 3 --tamano-lote 100000
```

El proceso conserva ejecuciones completas ya existentes. Para
regenerarlo:

```powershell
python -m cripto.corto_plazo_v4_5.confirmar_2025_2026 --epocas 3 --tamano-lote 100000 --sobrescribir
```

## Resultados

Se guardan en:

```text
modelos_entrenados/cripto/corto_plazo_v4_5/
confirmacion_2025_2026
```

Archivos principales:

```text
resultados/metricas_confirmacion.csv
resultados/comparacion_v45_vs_v41.csv
resultados/decision_confirmacion.json
```

La decisión automática distingue entre:

```text
V4.5 supera V4.1 en ambos periodos
V4.5 confirma rentabilidad pero no supera de forma concluyente
V4.5 es rentable pero no justifica sustituir V4.1
V4.5 no confirma fuera del desarrollo
```

No se promociona automáticamente ningún modelo. La decisión definitiva
se toma después de revisar los resultados completos.
