# Noticiero Ballena V1

Modelo auxiliar independiente y reutilizable del proyecto `tutan_gestion`.

## Objetivo

`noticiero_ballena` detecta movimientos grandes de BTC, construye variables
causales y estima su impacto a 15, 60 y 240 minutos.

No abre operaciones. BAJA V3 será quien combine estas señales con BTC, ETH,
Nasdaq, VIX, Spot, derivados y otras variables para decidir `SHORT` o
`NO OPERAR`.

## Fuente gratuita predeterminada

La V1 arranca con el WebSocket público de Blockchain.com:

```text
wss://ws.blockchain.info/inv
```

No requiere:

- cuenta;
- tarjeta bancaria;
- API key;
- archivo `.env`.

El recolector se suscribe a transacciones Bitcoin sin confirmar y conserva las
salidas cuyo valor supera el umbral configurado.

Para calcular su valor aproximado en dólares usa el precio Spot público de
Coinbase. Si temporalmente no puede obtener el precio, aplica el umbral de BTC
configurado como respaldo.

## Limitación importante de la fuente gratuita

La blockchain muestra direcciones y cantidades, pero no revela automáticamente
quién controla cada dirección ni cuál era la intención económica.

Por eso una salida grande puede ser:

- pago real;
- dirección de cambio;
- batching;
- reorganización de custodia;
- transferencia interna;
- movimiento hacia o desde un exchange.

La V1 no inventa identidades. Las direcciones de personas, fondos, ETF,
gobiernos, empresas o exchanges solo se etiquetan cuando se incorporan a un
registro verificable.

El archivo para ese registro es:

```text
cripto/noticiero_ballena/config/entidades_verificadas.example.json
```

## Proveedores opcionales

Se mantienen conectores opcionales para Whale Alert, Arkham, CryptoQuant y
Glassnode, pero el funcionamiento básico de V1 no depende de ellos. No deben
activarse sin revisar antes condiciones y posibles costos.

## Principios temporales

1. Usar `fecha_disponible`, no solo la hora de blockchain.
2. No usar información futura para construir variables.
3. No interpolar ni rellenar hacia atrás.
4. Publicar predicciones históricas fuera de muestra (`OOF`).
5. Mantener variables crudas junto con las salidas del modelo.
6. Bloquear 2025 y 2026 para desarrollo hasta autorización expresa.

## Estructura

```text
cripto/noticiero_ballena/
├── config/
│   ├── entidades_verificadas.example.json
│   └── proveedores.example.json
├── proveedores/
│   ├── blockchain_com.py
│   ├── archivo.py
│   ├── arkham.py
│   ├── cryptoquant.py
│   ├── glassnode.py
│   ├── http.py
│   └── whale_alert.py
├── recolectar_tiempo_real.py
├── normalizar_eventos.py
├── auditar_eventos.py
├── generar_variables.py
├── etiquetar_impacto.py
├── entrenar_modelo.py
├── inferir.py
├── exportar_para_modelos.py
├── auditar_exportacion.py
├── configuracion.py
├── esquemas.py
├── registro_entidades.py
├── utilidades.py
└── tests/
```

La única carpeta oficial es:

```text
cripto/noticiero_ballena/
```

## Primer arranque

Desde la raíz del proyecto:

```powershell
python -m pip install -r cripto/noticiero_ballena/requirements_noticiero_ballena.txt
python -m pytest cripto/noticiero_ballena/tests -q
```

Después:

```powershell
python -m cripto.noticiero_ballena.recolectar_tiempo_real `
  --proveedor blockchain_com `
  --min-usd 5000000 `
  --min-btc 50
```

El proceso queda escuchando. Se detiene con `Ctrl + C`.

Los eventos se guardan en:

```text
datos/cripto/crudos/noticiero_ballena/blockchain_com_tiempo_real.jsonl
```

## Normalización

Después de recopilar eventos:

```powershell
python -m cripto.noticiero_ballena.normalizar_eventos `
  --entrada datos/cripto/crudos/noticiero_ballena/blockchain_com_tiempo_real.jsonl `
  --fuente blockchain_com_public `
  --registro-entidades cripto/noticiero_ballena/config/entidades_verificadas.example.json
```

Luego:

```powershell
python -m cripto.noticiero_ballena.auditar_eventos
```

## Variables generadas

Ventanas: 5, 15, 60 y 240 minutos.

Familias principales:

```text
nb_raw_inflow_btc_*
nb_raw_outflow_btc_*
nb_raw_netflow_btc_*
nb_raw_eventos_*
nb_raw_institucional_btc_*
nb_raw_gobierno_btc_*
nb_raw_persona_publica_btc_*
nb_raw_minero_btc_*
nb_raw_ratio_interno_*
nb_raw_ratio_verificado_*
nb_raw_concentracion_top1_*
nb_raw_inflow_zscore_robusto_30d
nb_raw_calidad_datos
nb_raw_minutos_desde_evento
```

Cuando las entidades sean desconocidas, las variables de dirección de flujo no
se fuerzan artificialmente. El modelo conserva la incertidumbre.

## Salidas del especialista

Para 15, 60 y 240 minutos:

```text
nb_p_bajista_*
nb_p_alcista_*
nb_p_sin_impacto_*
nb_retorno_estimado_*
nb_confianza_*
```

Además publica variables crudas para poder comparar:

```text
V3 sin ballenas
V3 con variables crudas
V3 con salidas del modelo
V3 con variables crudas + salidas del modelo
```

## Entrenamiento

La recolección gratuita comienza desde el momento en que se inicia el programa.
No incluye automáticamente años anteriores.

Por tanto, no debe ejecutarse `entrenar_modelo` hasta disponer de suficiente
histórico real y una auditoría aprobada. Para construir varios años gratis, la
ruta robusta será un nodo Bitcoin Core propio y un indexador histórico; eso no
requiere suscripción, pero sí almacenamiento, ancho de banda y tiempo de
sincronización.

## Contrato con BAJA V3

Salida histórica:

```text
datos/cripto/preparados/noticiero_ballena/exportaciones/
noticiero_ballena_oof_1m.parquet
```

Salida operativa:

```text
datos/cripto/preparados/noticiero_ballena/exportaciones/
noticiero_ballena_tiempo_real_1m.parquet
```

BAJA V3 hará un `merge_asof` hacia atrás usando la hora real de disponibilidad.
`noticiero_ballena` informa; BAJA V3 decide.

## Comandos

El orden completo está en:

```text
cripto/noticiero_ballena/COMANDOS_NOTICIERO_BALLENA.txt
```
