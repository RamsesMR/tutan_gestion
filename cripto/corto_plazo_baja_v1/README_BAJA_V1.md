# Detector BAJA de corto plazo — V1

Nueva línea independiente para desarrollar operaciones cortas sobre
BTCUSDT. No modifica V4.1, que permanece congelada como campeona de
ALZA.

## Carpeta nueva

Esta carpeta todavía no existe en el repositorio:

```text
cripto/corto_plazo_baja_v1
```

## Fuente de datos

El laboratorio reutiliza los Parquet preparados por V5 porque ya
contienen:

- las 63 variables V2A/V4;
- 38 variables de estructura de velas;
- OHLC de un minuto;
- tratamiento causal de huecos.

No utiliza los objetivos largos de V5. Genera objetivos y retornos
short propios.

## Objetivos

### Control histórico

```text
baja_cierre_4h
```

Clase positiva cuando el cierre futuro a cuatro horas cae al menos
0,5 %. La estrategia de control usa salida fija a 480 minutos, igual
que la línea operativa V4.1.

### Triple barrera short

```text
corto_tp050_sl030_120m
corto_tp070_sl035_240m
corto_tp100_sl050_480m
```

Para una posición corta:

- el take profit se toca cuando el mínimo cae;
- el stop se toca cuando el máximo sube;
- si ambas barreras se tocan en la misma vela, gana el stop;
- si no se toca ninguna, se cierra al terminar el horizonte;
- el PnL lineal es `(entrada - salida) / entrada`.

## Variantes

```text
control_v4_63_moderado
sgd_63_sin_pesos
histgb_63_moderado
sgd_101_moderado
histgb_101_moderado
```

Esto separa:

- efecto de pesos;
- efecto del algoritmo;
- efecto de las 38 variables de estructura de velas.

## Desarrollo temporal

```text
Entrenamiento 2021-2022 → validación 2023
Entrenamiento 2021-2023 → validación 2024
```

2025 se utiliza únicamente como contexto futuro para calcular las
últimas salidas de 2024. No se usa para seleccionar modelos. 2026 no
se utiliza.

## Instalación

Copia todos los archivos del ZIP dentro de:

```text
cripto/corto_plazo_baja_v1
```

## Ejecución recomendada

### 1. Generar objetivos short

```powershell
python -m cripto.corto_plazo_baja_v1.generar_datos_baja
```

La fuente V5 debe existir previamente en:

```text
datos/cripto/preparados/corto_plazo_v5
```

### 2. Auditar datos

```powershell
python -m cripto.corto_plazo_baja_v1.auditar_datos_baja
```

Debe terminar con:

```text
AUDITORÍA BAJA V1 APROBADA
```

### 3. Primera ejecución base

Empieza por el control y el objetivo central:

```powershell
python -m cripto.corto_plazo_baja_v1.entrenar_modelos --objetivos baja_cierre_4h,corto_tp070_sl035_240m --variantes control_v4_63_moderado,sgd_63_sin_pesos,histgb_63_moderado --epocas 3 --tamano-lote 100000
```

### 4. Familia técnica

Después:

```powershell
python -m cripto.corto_plazo_baja_v1.entrenar_modelos --objetivos baja_cierre_4h,corto_tp070_sl035_240m --variantes sgd_101_moderado,histgb_101_moderado --epocas 3 --tamano-lote 100000
```

### 5. Objetivos alternativos

Solo después de revisar el objetivo central:

```powershell
python -m cripto.corto_plazo_baja_v1.entrenar_modelos --objetivos corto_tp050_sl030_120m,corto_tp100_sl050_480m --variantes control_v4_63_moderado,sgd_63_sin_pesos,histgb_63_moderado,sgd_101_moderado,histgb_101_moderado --epocas 3 --tamano-lote 100000
```

También es posible ejecutar todo:

```powershell
python -m cripto.corto_plazo_baja_v1.entrenar_modelos --objetivos todos --variantes todos --epocas 3 --tamano-lote 100000
```

### 6. Seleccionar

```powershell
python -m cripto.corto_plazo_baja_v1.seleccionar_modelo
```

## Resultados

```text
modelos_entrenados/cripto/corto_plazo_baja_v1/desarrollo
```

Archivos principales:

```text
resultados/historial_desarrollo.csv
seleccion/seleccion_baja_v1.csv
seleccion/decision_baja_v1.json
seleccion/informe_seleccion_baja_v1.md
```

## Métricas

Clasificación:

- prevalencia;
- PR-AUC;
- lift de PR-AUC;
- ROC-AUC;
- precision;
- porcentaje de acierto;
- recall;
- F1;
- matriz de confusión.

Estrategia short:

- operaciones no solapadas;
- precisión entre operaciones;
- porcentaje de operaciones positivas;
- retorno neto medio y mediano;
- factor de beneficio;
- drawdown máximo;
- retorno compuesto;
- duración media;
- concentración mensual.

## Criterios iniciales de aprobación

- al menos 50 operaciones en cada pliegue;
- retorno neto medio y mediano no negativos;
- factor de beneficio mínimo 1,10;
- drawdown no inferior a -20 %;
- al menos 50 % de operaciones netas positivas;
- concentración mensual máxima de 35 %.

Ningún modelo se promociona automáticamente. Primero se revisan los
resultados y después se realiza robustez por semillas. Solo una
configuración congelada puede pasar a 2025 y 2026.
