# V4.3 — V4.1 con histórico ampliado

## Objetivo

V4.3 conserva exactamente la estrategia final de V4.1:

- BTCUSDT;
- detector SUBE `v4_63_sin_pesos`;
- 63 variables;
- umbral `0.44`;
- entrada por cruce desde abajo;
- sin veto BAJA;
- salida fija a `480` minutos;
- coste total `0.001` (0,10 %);
- 3 épocas y lotes de 100.000 por defecto.

La única diferencia es el histórico de entrenamiento, que comienza el 17 de agosto de 2017.

V4.1 no se modifica y continúa siendo la campeona hasta revisar explícitamente la comparación.

## Instalación

Crear la carpeta:

```text
cripto/corto_plazo_v4_3
```

Extraer dentro de ella todos los archivos de este ZIP.

## Paso 1 — Integrar el histórico 2017-2020

```powershell
python -m cripto.corto_plazo_v4_3.integrar_historico --fase todo
```

Este paso:

- normaliza al minuto solo en los archivos derivados;
- exige una vela de BTC y una de ETH por minuto;
- elimina los dos minutos problemáticos ya auditados;
- no rellena huecos;
- no modifica PostgreSQL ni los ZIP originales;
- no modifica V4.1;
- genera los archivos históricos con el mismo esquema que consume V4.

Para repetir únicamente después de revisar un error:

```powershell
python -m cripto.corto_plazo_v4_3.integrar_historico --fase todo --sobrescribir
```

## Paso 2 — Generar predicciones de desarrollo

```powershell
python -m cripto.corto_plazo_v4_3.generar_predicciones_desarrollo --epocas 3 --tamano-lote 100000
```

Entrena con el histórico ampliado y genera predicciones fuera de muestra para 2023 y 2024.

## Paso 3 — Evaluar la estrategia congelada

```powershell
python -m cripto.corto_plazo_v4_3.evaluar_desarrollo
```

No ejecuta una nueva selección. Aplica directamente la estrategia de V4.1.

## Paso 4 — Evaluar 2025 conocido

```powershell
python -m cripto.corto_plazo_v4_3.evaluar_2025 --epocas 3 --tamano-lote 100000
```

2025 ya fue observado. Este resultado es una comparación retrospectiva, no una confirmación virgen.

## Paso 5 — Evaluar enero-mayo de 2026 conocido

```powershell
python -m cripto.corto_plazo_v4_3.evaluar_2026 --epocas 3 --tamano-lote 100000
```

Enero-mayo de 2026 también fue observado anteriormente. No se presenta como prueba virgen.

## Paso 6 — Comparar con V4.1

```powershell
python -m cripto.corto_plazo_v4_3.comparar_v4_1
```

La comparación se guarda en:

```text
modelos_entrenados/cripto/corto_plazo_v4_3/comparacion_v4_1
```

## Orden obligatorio

No ejecutes los pasos 2 a 6 hasta que el paso 1 termine con:

```text
INTEGRACIÓN HISTÓRICA APROBADA
```

No cambies umbral, variables, estrategia, coste, salida, épocas o lote durante esta primera comparación. El propósito es medir únicamente el efecto del histórico adicional.
