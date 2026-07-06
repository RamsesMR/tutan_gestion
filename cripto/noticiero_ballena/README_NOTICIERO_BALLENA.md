# Noticiero Ballena V1

Modelo auxiliar independiente y reutilizable del proyecto `tutan_gestion`.

## Separación de responsabilidades

### Noticiero Ballena

Procesa movimientos grandes de BTC, identifica su contexto y estima su impacto:

- probabilidad de impacto bajista;
- probabilidad de impacto alcista;
- probabilidad de ausencia de impacto;
- retorno estimado a 15, 60 y 240 minutos;
- confianza de la señal;
- variables crudas de inflow, outflow, netflow, identidad y concentración.

### BAJA V3

BAJA V3 será el modelo decisor. Combinará las salidas del Noticiero Ballena con
BTC, ETH, Nasdaq, VIX, flujo Spot, derivados y otras variables. Solo BAJA V3
decidirá `SHORT` o `NO OPERAR`.

## Principios de V1

1. No inventar identidades ni wallets.
2. Registrar `fecha_disponible`, no solo la fecha de blockchain.
3. No interpolar hacia el futuro ni rellenar hacia atrás.
4. Las predicciones históricas entregadas a otros modelos deben ser fuera de
   muestra (`OOF`).
5. Mantener también las variables crudas para comprobar si el modelo auxiliar
   aporta más que las reglas deterministas.
6. Ser reutilizable por futuros modelos de ALZA, BAJA, volatilidad o régimen.

## Fuentes previstas

- **Whale Alert:** descubrimiento de movimientos grandes e información en tiempo
  real. Es la fuente operativa inicial recomendada.
- **Arkham:** enriquecimiento opcional de direcciones, entidades y etiquetas.
- **CryptoQuant:** métricas agregadas como Exchange Whale Ratio.
- **Glassnode:** contexto histórico y on-chain agregado cuando el plan contratado
  permita acceso API.
- **Archivo:** CSV, JSON, JSONL o Parquet para históricos y pruebas.

El núcleo no depende de una sola API. Las claves se leen desde variables de
entorno y nunca se guardan en el repositorio.

## Estructura

```text
cripto/noticiero_ballena/
├── config/
│   ├── entidades_verificadas.example.json
│   └── proveedores.example.json
├── proveedores/
│   ├── archivo.py
│   ├── arkham.py
│   ├── cryptoquant.py
│   ├── glassnode.py
│   ├── http.py
│   └── whale_alert.py
├── normalizar_eventos.py
├── importar_whale_alert.py
├── enriquecer_arkham.py
├── auditar_eventos.py
├── generar_variables.py
├── descargar_contexto_onchain.py
├── integrar_contexto.py
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

## Rutas generadas

```text
datos/cripto/crudos/noticiero_ballena/
datos/cripto/preparados/noticiero_ballena/
modelos_entrenados/cripto/noticiero_ballena/
```

No crea carpetas `noticiero_ballena_v1` ni `noticiero_ballena_v1_1` dentro de
`cripto`. La única carpeta de código es:

```text
cripto/noticiero_ballena/
```

## Esquema de evento normalizado

Campos principales:

```text
evento_id
tx_hash
fecha_blockchain
fecha_detectada
fecha_disponible
activo
cantidad_activo
valor_usd
direccion_origen
direccion_destino
entidad_origen
entidad_destino
tipo_origen
tipo_destino
direccion_flujo
confianza_etiquetado
identidad_verificada
es_movimiento_interno_probable
fuente
metadata_json
```

`fecha_disponible` representa el momento real en el que el sistema pudo conocer
el dato. Es la única clave temporal válida para construir variables.

## Identidades verificadas

El archivo `config/entidades_verificadas.example.json` está vacío a propósito.
Para una figura pública, fondo, ETF, gobierno o empresa solo debe añadirse una
dirección cuando exista una fuente pública y verificable.

Una atribución debe conservar:

```text
dirección
entidad
tipo de entidad
fuente de verificación
confianza
```

Una etiqueta del proveedor puede cambiar; por eso la confianza y la fuente
forman parte del dato.

## Variables generadas

Ventanas: 5, 15, 60 y 240 minutos.

Familias:

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

Los movimientos entre exchanges y los movimientos internos probables se
mantienen separados para que el modelo pueda aprender que no equivalen a venta.

La tabla de variables es temporalmente dispersa: solo conserva los minutos que
pertenecen a una ventana activa de 240 minutos y añade un minuto de reinicio al
terminar cada ventana. Así se evita generar millones de filas vacías y también
se evita que un `merge_asof` mantenga una señal antigua indefinidamente.

## Etiquetas del especialista

Para cada horizonte de 15, 60 y 240 minutos:

```text
BAJISTA
ALCISTA
SIN_IMPACTO
AMBIGUO
```

`AMBIGUO` se conserva para auditoría, pero se excluye del entrenamiento. Las
barreras se calculan desde el primer minuto negociable después de que la
información estaba disponible.

## Entrenamiento temporal

- Desarrollo: 2021–2024.
- 2025 y 2026: bloqueados.
- Predicciones históricas: pliegues expansivos de 90 días después de un mínimo
  inicial de 180 días.
- Modelo final: se entrena únicamente después de generar las predicciones OOF.

El archivo histórico para BAJA V3 será:

```text
datos/cripto/preparados/noticiero_ballena/exportaciones/
noticiero_ballena_oof_1m.parquet
```

BAJA V3 nunca debe entrenarse con predicciones in-sample del modelo final.

La salida operativa será:

```text
datos/cripto/preparados/noticiero_ballena/exportaciones/
noticiero_ballena_tiempo_real_1m.parquet
```

## Contrato para modelos consumidores

Salidas principales:

```text
fecha_apertura
nb_p_bajista_15m
nb_p_alcista_15m
nb_p_sin_impacto_15m
nb_retorno_estimado_15m
nb_confianza_15m
```

También se publican equivalentes a 60 y 240 minutos y una selección de variables
crudas. Los consumidores deben hacer un `merge_asof` hacia atrás y respetar
`nb_raw_minutos_desde_evento` para no mantener señales antiguas indefinidamente.

## Proceso completo

Consulta `COMANDOS_NOTICIERO_BALLENA.txt`. El orden obligatorio es:

1. instalar requisitos;
2. incorporar histórico real;
3. normalizar o importar;
4. enriquecer opcionalmente con Arkham;
5. auditar eventos;
6. generar variables;
7. integrar contexto opcional con su retraso real;
8. etiquetar usando BTC 1m;
9. entrenar y producir OOF;
10. exportar y auditar;
11. revisar resultados antes de conectar BAJA V3.

## Tiempo real

Whale Alert se recolecta en JSONL. Después se ejecuta el mismo normalizador,
generador de variables e inferencia. No existe un pipeline alternativo oculto:
histórico y tiempo real comparten el mismo contrato.

## Límites conocidos

- Una transferencia grande no implica venta.
- Las etiquetas de exchanges y entidades pueden revisarse.
- Los históricos de distintos proveedores no deben mezclarse sin una auditoría
  de metodología y cobertura.
- Las métricas agregadas deben alinearse por su hora real de disponibilidad,
  incluyendo el retraso del proveedor.
- El modelo puede no aportar más que las variables crudas; por eso ambas salidas
  se conservan para ablación.
