# V4.2 — Mejora de ejecución de la V4.1

Esta versión mantiene congelado el detector base:

```text
BTCUSDT SUBE
v4_63_sin_pesos
umbral base 0,44
coste total 0,10 %
```

La V4.2 no sustituye el detector. Mejora la política de entrada y salida.

## Qué prueba

### Salidas

- 300 minutos
- 360 minutos
- 420 minutos
- 480 minutos
- 540 minutos
- 600 minutos
- 720 minutos

### Calidad del cruce

- distancia por encima de 0,44;
- pendiente positiva en cinco minutos;
- rearme después de volver a 0,42, 0,40 o 0,38;
- enfriamiento de 0, 30, 60 o 120 minutos.

### Relación SUBE–BAJA

- diferencia mínima entre probabilidades:
  - 0,00;
  - 0,05;
  - 0,10;
  - 0,15.

### Meta-modelo

Un clasificador secundario aprende a decidir si una señal debe convertirse en operación.

La etiqueta del meta-modelo es:

```text
retorno neto al horizonte seleccionado > 0
```

## Orden de ejecución

### 1. Construir las bases de 2023 y 2024

```powershell
python -m cripto.corto_plazo_v4_2.construir_bases
```

### 2. Probar duraciones

```powershell
python -m cripto.corto_plazo_v4_2.laboratorio_salidas
```

### 3. Probar filtros y calidad del cruce

```powershell
python -m cripto.corto_plazo_v4_2.laboratorio_filtros
```

### 4. Seleccionar política

```powershell
python -m cripto.corto_plazo_v4_2.seleccionar_politica
```

### 5. Entrenar meta-modelo

```powershell
python -m cripto.corto_plazo_v4_2.entrenar_meta_modelo
```

## Regla metodológica

- 2023 y 2024 se usan para desarrollo;
- 2025 se usa para confirmación;
- 2026 se usa como referencia posterior;
- no deben cambiarse parámetros después de ver confirmación.

## Dependencia adicional

```powershell
python -m pip install joblib scikit-learn
```
