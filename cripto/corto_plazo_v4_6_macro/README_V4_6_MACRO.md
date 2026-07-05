# V4.6-Macro

Laboratorio independiente para comprobar si variables macroeconómicas
mejoran V4.1 y V4.5-Spot.

## Modelos que permanecen intactos

- V4.1: campeona oficial, 63 variables y umbral 0,44.
- V4.5-Spot: candidata congelada, 90 variables y umbral 0,42.

V4.6 no reemplaza ni modifica sus datos, modelos o selecciones.

## Variables estudiadas

1. `impulso_dolar_amplio_20s`
2. `impulso_rendimiento_real_10y_20s`
3. `impulso_spread_baa_20s`
4. `impulso_vix_20s`
5. `retorno_nasdaq100_20s`
6. `impulso_reservas_fed_4s`

El spread high-yield inicialmente considerado no se utiliza porque,
desde abril de 2026, su histórico público en FRED está limitado a tres
años. Se sustituye por `BAA10Y`, que conserva cobertura reproducible.

## Alineación temporal

Las transformaciones se calculan en la frecuencia original de cada
serie. Después se asigna una fecha de disponibilidad.

- Nasdaq 100 y VIX: después del cierre de las 16:00 ET.
- DFII10: siguiente día hábil, después de H.15.
- DTWEXBGS: siguiente publicación semanal H.10.
- BAA10Y: siguiente día hábil a las 18:00 ET, de forma conservadora.
- WRESBAL: jueves a las 16:31 ET, después de H.4.1.

Cada vela recibe mediante `merge_asof` el último dato que ya había sido
publicado. No se interpola y no se rellena hacia el pasado.

## Fase 1

Se crean dos controles y doce pruebas individuales:

- `v41_control_macro`
- `v45_control_macro`
- seis variables individuales sobre V4.1
- seis variables individuales sobre V4.5-Spot

La primera selección utiliza los umbrales congelados:

- V4.1: 0,44
- V4.5-Spot: 0,42

## Comandos

```powershell
python -m cripto.corto_plazo_v4_6_macro.descargar_datos_macro
python -m cripto.corto_plazo_v4_6_macro.construir_variables_macro
python -m cripto.corto_plazo_v4_6_macro.auditar_datos_macro
python -m cripto.corto_plazo_v4_6_macro.integrar_variables_macro
python -m cripto.corto_plazo_v4_6_macro.validar_integracion
python -m cripto.corto_plazo_v4_6_macro.generar_predicciones_desarrollo --variantes todas --epocas 3 --tamano-lote 100000
python -m cripto.corto_plazo_v4_6_macro.evaluar_desarrollo
python -m cripto.corto_plazo_v4_6_macro.analizar_estabilidad
python -m cripto.corto_plazo_v4_6_macro.seleccionar_variables_macro
```

No se deben construir combinaciones hasta revisar la selección
individual. El selector solamente marca candidatas preliminares.

## Directorios nuevos

Estos directorios no existían antes de V4.6:

```text
cripto/corto_plazo_v4_6_macro
datos/cripto/bruto/macro/fred
datos/cripto/preparados/corto_plazo_v4_6_macro
modelos_entrenados/cripto/corto_plazo_v4_6_macro
```


## Limitación de datos revisados

La descarga pública de FRED entrega el histórico vigente. El laboratorio
evita fuga por fecha y hora de publicación, pero algunas series pueden
haber recibido revisiones posteriores. Por eso, cualquier variable que
supere la fase individual deberá confirmarse con datos vintage de ALFRED
antes de entrar en una combinación final o sustituir modelos existentes.
