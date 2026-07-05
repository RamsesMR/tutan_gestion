# V4.5 — Flujo Spot, futuros, basis y funding

## Objetivo

V4.5 estudia si nuevas variables de microestructura histórica mejoran la estrategia campeona V4.1.

Se mantienen:

- BTCUSDT como símbolo operativo.
- Horizonte objetivo de cuatro horas.
- Detector logístico incremental.
- Tres épocas por defecto.
- Sin ponderación de clases para SUBE.
- Entrada por cruce.
- Umbral congelado principal `0.44`.
- Salida fija de `480` minutos.
- Coste de `0.001`.
- Desarrollo únicamente con validaciones 2023 y 2024.

No se modifica V4.1.

## Variantes

### `control_63`

Reproduce las 63 variables usadas por V4.1. Sirve como control de integridad.

### `flujo_spot`

Añade 27 variables creadas con:

- volumen comprador taker;
- volumen vendedor estimado;
- desequilibrio comprador/vendedor;
- ratio compra/venta;
- aceleración del flujo;
- confirmación entre precio y flujo;
- tamaño relativo de las operaciones.

### `flujo_spot_futuros`

Añade además:

- rendimiento del perpetuo USD-M;
- flujo comprador/vendedor de futuros;
- divergencia Spot-futuros;
- basis perpetuo-Spot;
- cambios del basis;
- funding realizado;
- cambio del funding;
- media de funding de 24 horas.

## Variables excluidas por ahora

No se incluyen libro de órdenes ni open interest porque no existe en el proyecto un histórico completo y homogéneo para 2021-2024. Incorporarlos sin ese histórico impediría una comparación limpia.

## Orden de ejecución

### 1. Auditoría

```powershell
python -m cripto.corto_plazo_v4_5.auditar_fuentes
```

### 2. Variables Spot

```powershell
python -m cripto.corto_plazo_v4_5.generar_variables_flujo_spot
```

Para repetir:

```powershell
python -m cripto.corto_plazo_v4_5.generar_variables_flujo_spot --sobrescribir
```

### 3. Primera comparación: control y flujo Spot

```powershell
python -m cripto.corto_plazo_v4_5.generar_predicciones_desarrollo --variantes control_63,flujo_spot --epocas 3 --tamano-lote 100000
```

```powershell
python -m cripto.corto_plazo_v4_5.evaluar_desarrollo
```

Esta primera fase permite saber si el flujo Spot aporta valor antes de descargar futuros.

### 4. Descargar futuros USD-M

Incluye diciembre de 2020 como contexto para las primeras variables de 2021.

```powershell
python -m cripto.corto_plazo_v4_5.descargar_futuros_binance --desde 2020-12 --hasta 2024-12
```

### 5. Descargar funding

```powershell
python -m cripto.corto_plazo_v4_5.descargar_funding_binance --desde 2020-12-01 --hasta 2025-01-01
```

### 6. Integrar futuros

```powershell
python -m cripto.corto_plazo_v4_5.integrar_variables_futuros
```

### 7. Validar

```powershell
python -m cripto.corto_plazo_v4_5.validar_integracion
```

### 8. Entrenar variante completa

```powershell
python -m cripto.corto_plazo_v4_5.generar_predicciones_desarrollo --variantes flujo_spot_futuros --epocas 3 --tamano-lote 100000
```

### 9. Evaluar todo

```powershell
python -m cripto.corto_plazo_v4_5.evaluar_desarrollo
```

### 10. Seleccionar

```powershell
python -m cripto.corto_plazo_v4_5.seleccionar_candidata
```

### 11. Comparar

```powershell
python -m cripto.corto_plazo_v4_5.comparar_v4_1
```

## Directorios nuevos

Estos directorios aún no existían antes de V4.5 y serán creados por los scripts:

```text
datos/cripto/preparados/corto_plazo_v4_5
datos/cripto/bruto/binance/futuros_um
modelos_entrenados/cripto/corto_plazo_v4_5
```

## Regla de trabajo

Primero se ejecuta la fase Spot. Solo después de revisar su resultado se ejecuta la fase de futuros. Esto permite atribuir cualquier mejora a la familia de variables correcta.
