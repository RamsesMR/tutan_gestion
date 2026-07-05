# V4.4 — Gestión dinámica del riesgo con detector BAJA

## Objetivo

V4.4 conserva completamente la entrada de la estrategia campeona V4.1:

- BTCUSDT.
- Detector SUBE `v4_63_sin_pesos`.
- Entrada por cruce de `0.44`.
- Sin veto BAJA antes de entrar.
- Coste `0.001`.
- Salida máxima a `480` minutos.

La única modificación estudiada es una posible salida anticipada cuando el detector BAJA cruza un umbral después de abrir la posición.

V4.4 no reentrena ningún modelo. Reutiliza las predicciones fuera de muestra de V4.1 para 2023 y 2024.

## Principios del experimento

- No usa 2025 ni 2026 para seleccionar.
- No modifica V4.1.
- No cambia la entrada.
- No cambia las variables.
- No cambia el algoritmo.
- Evalúa primero una cohorte con las mismas entradas de V4.1.
- Evalúa después el comportamiento operativo permitiendo nuevas entradas tras una salida anticipada.
- No promueve V4.4 si reduce demasiado la rentabilidad o el factor de beneficio.

## Archivos

- `configuracion.py`: parámetros congelados y rutas.
- `verificar_entorno.py`: comprueba los artefactos de V4.1.
- `utilidades.py`: simulación causal de salidas.
- `diagnosticar_baja.py`: estudia si BAJA anticipa operaciones perdedoras.
- `laboratorio_salidas_baja.py`: prueba un laboratorio pequeño y predefinido.
- `seleccionar_estrategia.py`: aplica filtros duros de promoción.
- `comparar_v4_1.py`: genera el informe final.
- `__init__.py`: paquete Python.

## Orden de ejecución

Desde la raíz de `tutan_gestion`:

```powershell
python -m cripto.corto_plazo_v4_4.verificar_entorno
```

```powershell
python -m cripto.corto_plazo_v4_4.diagnosticar_baja
```

```powershell
python -m cripto.corto_plazo_v4_4.laboratorio_salidas_baja
```

```powershell
python -m cripto.corto_plazo_v4_4.seleccionar_estrategia
```

```powershell
python -m cripto.corto_plazo_v4_4.comparar_v4_1
```

## Criterio de promoción

Una candidata debe, en ambos pliegues de desarrollo:

- conservar al menos el 90 % del retorno neto medio de V4.1;
- conservar al menos el 95 % del factor de beneficio;
- mantener factor de beneficio superior a 1;
- mantener retorno neto mediano no negativo;
- no empeorar el drawdown;
- mejorar el drawdown al menos un punto porcentual en algún pliegue;
- usar exactamente las mismas operaciones de entrada en la evaluación primaria.

Si ninguna candidata cumple, V4.1 permanece como campeona.
