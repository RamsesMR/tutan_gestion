# V4.1 — Laboratorio de ejecución

La V4.1 no reemplaza el detector V4. Parte del mejor detector actual:

```text
BTCUSDT SUBE
v4_63_sin_pesos
```

y usa el detector:

```text
BTCUSDT BAJA
v4_63_moderado
```

como posible veto.

## Qué investiga

Entradas:

- nivel sobre umbral;
- cruce desde abajo;
- confirmación 3, 5 y 10 minutos;
- probabilidad creciente 3, 5 y 10 minutos;
- umbrales SUBE 0,44–0,60;
- veto BAJA opcional.

Salidas:

- cierre fijo a 60, 120, 240 y 480 minutos;
- seis combinaciones de take profit y stop loss.

Costes:

- 0,05 %;
- 0,10 %;
- 0,20 %.

Métricas:

- precisión;
- porcentaje de acierto;
- porcentaje de operaciones positivas;
- operaciones no solapadas;
- retorno bruto y neto;
- retorno mediano;
- factor de beneficio;
- drawdown.

## Instalación

Copiar la carpeta como:

```text
cripto/corto_plazo_v4_1/
```

## Paso 1 — Generar probabilidades fuera de muestra

```powershell
python -m cripto.corto_plazo_v4_1.generar_predicciones_desarrollo `
    --epocas 3 `
    --tamano-lote 100000
```

Genera probabilidades de 2023 y 2024 sin utilizar 2025 ni 2026.

## Paso 2 — Ejecutar laboratorio

Primero se recomienda una prueba rápida:

```powershell
python -m cripto.corto_plazo_v4_1.laboratorio_ejecucion `
    --modo-rapido
```

Después, si termina correctamente, ejecutar la rejilla completa:

```powershell
python -m cripto.corto_plazo_v4_1.laboratorio_ejecucion
```

## Paso 3 — Seleccionar estrategia

```powershell
python -m cripto.corto_plazo_v4_1.seleccionar_estrategia
```

El selector exige con coste total de 0,10 %:

- al menos 50 operaciones en cada año;
- retorno neto medio positivo en ambos años;
- retorno mediano no negativo en ambos años;
- factor de beneficio superior a 1 en ambos años.

La consola muestra precisión y porcentaje de acierto.

## Paso 4 — Confirmación secundaria en 2025

No ejecutar si la estrategia no cumple filtros.

```powershell
python -m cripto.corto_plazo_v4_1.confirmar_2025 `
    --epocas 3 `
    --tamano-lote 100000
```

El script se bloquea automáticamente si la estrategia seleccionada no cumple los filtros.
