# Robustez por semillas — V4.6-Macro

Esta fase comprueba si las candidatas macro dependen de la semilla
aleatoria del SGD.

## Archivos

Reemplazar:

- `utilidades_modelo.py`

Añadir:

- `configuracion_robustez.py`
- `generar_robustez_semillas.py`
- `evaluar_robustez_semillas.py`
- `seleccionar_robustez_semillas.py`

Todos van dentro de:

```text
cripto/corto_plazo_v4_6_macro
```

## Semillas

```text
11, 23, 42, 77, 101
```

## Comparaciones pareadas

```text
v41_control_macro 0,44
vs.
v41_reservas_fed 0,44

v41_control_macro 0,42
vs.
v41_vix 0,42

v45_control_macro 0,44
vs.
v45_reservas_fed 0,44
```

Los controles y las candidatas usan la misma semilla y el mismo orden
de lotes en cada comparación.

El escalador es determinista y no depende de la semilla. Se genera una
sola vez por variante y pliegue y se reutiliza en las cinco semillas.

## Comandos

```powershell
Remove-Item -Recurse -Force .\cripto\corto_plazo_v4_6_macro\__pycache__ -ErrorAction SilentlyContinue

python -m cripto.corto_plazo_v4_6_macro.generar_robustez_semillas --semillas todas --epocas 3 --tamano-lote 100000

python -m cripto.corto_plazo_v4_6_macro.evaluar_robustez_semillas

python -m cripto.corto_plazo_v4_6_macro.seleccionar_robustez_semillas
```

La generación es reanudable. Si todos los artefactos de una ejecución
ya existen, se conservan. Si existe una ejecución parcial, el script
solicita `--sobrescribir`.

## Criterios

Una candidata robusta debe:

- aprobar al menos cuatro de las cinco semillas;
- conservar al menos 50 operaciones en ambos años;
- mantener retorno positivo y factor mayor que uno;
- conservar al menos 95 % del retorno del control en el peor año;
- conservar al menos 98 % del factor del control en el peor año;
- mejorar en mediana retorno y factor;
- no empeorar el drawdown mediano más de un punto;
- mantener el signo del coeficiente entre 2023 y 2024;
- no concentrar más del 35 % del resultado absoluto en un solo mes.

No se utilizan 2025 ni 2026. Ningún modelo se promueve
automáticamente: la variable ganadora deberá confirmarse con datos
vintage de ALFRED.
