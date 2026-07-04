# V4 — Detectores binarios especializados

La V4 separa el problema en cuatro detectores:

- BTCUSDT BAJA
- BTCUSDT SUBE
- ETHUSDT BAJA
- ETHUSDT SUBE

Cada detector responde una sola pregunta:

- BAJA: ¿el rendimiento de las próximas cuatro horas será menor o igual que -0,50 %?
- SUBE: ¿el rendimiento de las próximas cuatro horas será mayor o igual que +0,50 %?

## Estructura

Copiar todos los archivos dentro de:

```text
cripto/corto_plazo_v4/
```

## Fase 1 — Variantes base

BTC:

```powershell
python -m cripto.corto_plazo_v4.entrenar_detectores `
    --simbolo BTCUSDT `
    --detector AMBOS `
    --grupo base `
    --epocas 3 `
    --tamano-lote 100000
```

ETH:

```powershell
python -m cripto.corto_plazo_v4.entrenar_detectores `
    --simbolo ETHUSDT `
    --detector AMBOS `
    --grupo base `
    --epocas 3 `
    --tamano-lote 100000
```

El grupo base compara:

- `v4_63_balanceado`
- `v4_52_balanceado`

Cada variante se valida en 2023 y 2024. No usa 2025 ni 2026.

## Fase 2 — Resumen y umbrales

```powershell
python -m cripto.corto_plazo_v4.resumir_detectores
```

```powershell
python -m cripto.corto_plazo_v4.seleccionar_umbrales `
    --operaciones-minimas 50 `
    --recall-minimo-precision 0.03
```

Genera tres perfiles por detector:

- equilibrado
- precision
- operativo

## Fase 3 — Ponderaciones alternativas

Solo después de analizar la fase base:

```powershell
python -m cripto.corto_plazo_v4.entrenar_detectores `
    --simbolo BTCUSDT `
    --detector AMBOS `
    --grupo pesos `
    --epocas 3 `
    --tamano-lote 100000
```

```powershell
python -m cripto.corto_plazo_v4.entrenar_detectores `
    --simbolo ETHUSDT `
    --detector AMBOS `
    --grupo pesos `
    --epocas 3 `
    --tamano-lote 100000
```

Después se vuelven a ejecutar el resumen y la selección de umbrales.

## Fase 4 — Confirmación secundaria en 2025

No ejecutar hasta haber revisado la selección de 2023/2024.

```powershell
python -m cripto.corto_plazo_v4.confirmar_2025 `
    --perfil equilibrado `
    --simbolo TODOS `
    --detector AMBOS `
    --epocas 3 `
    --tamano-lote 100000
```

El umbral queda congelado. No se recalibra con 2025.

## Fase 5 — Combinar largos y cortos

BTC:

```powershell
python -m cripto.corto_plazo_v4.combinar_senales_2025 `
    --simbolo BTCUSDT `
    --perfil equilibrado `
    --conflicto no_operar `
    --coste-operacion 0
```

ETH:

```powershell
python -m cripto.corto_plazo_v4.combinar_senales_2025 `
    --simbolo ETHUSDT `
    --perfil equilibrado `
    --conflicto no_operar `
    --coste-operacion 0
```

`coste-operacion` representa el coste total estimado de entrada, salida y deslizamiento. Debe configurarse con el coste real antes de interpretar resultados operativos.

## Advertencias

- Las operaciones se filtran con un cooldown de 240 minutos para evitar solapamiento.
- La evaluación de retorno es una simulación simple, no un backtest completo de ejecución.
- 2025 ya fue observado durante versiones anteriores, por lo que solo sirve como confirmación secundaria.
- Los datos posteriores a mayo de 2026 deben reservarse para una prueba realmente no observada.
