from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

from compartido.base_datos.conexion import obtener_conexion


SIMBOLOS = (
    "BTCUSDT",
    "ETHUSDT",
)

INTERVALO = "1m"

RUTA_PROYECTO = Path(__file__).resolve().parents[2]

RUTA_SALIDA = (
    RUTA_PROYECTO
    / "datos"
    / "cripto"
    / "preparados"
)


def convertir_mes(
    valor: str,
) -> datetime:
    """Convierte AAAA-MM en el primer instante UTC del mes."""

    try:
        fecha = datetime.strptime(
            valor,
            "%Y-%m",
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "El mes debe tener el formato AAAA-MM."
        ) from error

    return fecha.replace(
        tzinfo=timezone.utc
    )


def mes_siguiente(
    fecha: datetime,
) -> datetime:
    """Devuelve el primer instante del mes siguiente."""

    if fecha.month == 12:
        return fecha.replace(
            year=fecha.year + 1,
            month=1,
        )

    return fecha.replace(
        month=fecha.month + 1,
    )


def obtener_mercado_id(
    cursor: psycopg.Cursor,
    simbolo: str,
) -> int:
    """Obtiene el identificador de Binance Spot."""

    cursor.execute(
        """
        SELECT mercados.id
        FROM mercados
        INNER JOIN fuentes_datos
            ON fuentes_datos.id = mercados.fuente_datos_id
        WHERE UPPER(mercados.simbolo_proveedor) = UPPER(%s)
          AND LOWER(mercados.tipo_mercado) = 'spot'
          AND LOWER(fuentes_datos.nombre) = 'binance'
          AND mercados.activo = TRUE
          AND fuentes_datos.activo = TRUE;
        """,
        (
            simbolo,
        ),
    )

    resultado = cursor.fetchone()

    if resultado is None:
        raise ValueError(
            f"No existe el mercado {simbolo} de Binance Spot."
        )

    return int(
        resultado[0]
    )


def auditar_simbolo(
    cursor: psycopg.Cursor,
    mercado_id: int,
    inicio: datetime,
    fin: datetime,
) -> dict[str, Any]:
    """Audita la alineación de las aperturas al minuto."""

    cursor.execute(
        """
        WITH base AS (
            SELECT
                fecha_apertura,
                DATE_TRUNC(
                    'minute',
                    fecha_apertura
                ) AS minuto
            FROM velas
            WHERE mercado_id = %s
              AND intervalo = %s
              AND fecha_apertura >= %s
              AND fecha_apertura < %s
        )
        SELECT
            COUNT(*) AS registros,
            COUNT(*) FILTER (
                WHERE fecha_apertura = minuto
            ) AS alineadas,
            COUNT(*) FILTER (
                WHERE fecha_apertura <> minuto
            ) AS no_alineadas,
            COUNT(
                DISTINCT minuto
            ) AS minutos_distintos,
            COUNT(*)
                - COUNT(
                    DISTINCT minuto
                ) AS colisiones_al_normalizar,
            MIN(
                EXTRACT(
                    EPOCH FROM (
                        fecha_apertura
                        - minuto
                    )
                )
            ) FILTER (
                WHERE fecha_apertura <> minuto
            ) AS desplazamiento_minimo_segundos,
            MAX(
                EXTRACT(
                    EPOCH FROM (
                        fecha_apertura
                        - minuto
                    )
                )
            ) FILTER (
                WHERE fecha_apertura <> minuto
            ) AS desplazamiento_maximo_segundos
        FROM base;
        """,
        (
            mercado_id,
            INTERVALO,
            inicio,
            fin,
        ),
    )

    fila = cursor.fetchone()

    return {
        "registros": int(
            fila[0]
        ),
        "alineadas": int(
            fila[1]
        ),
        "no_alineadas": int(
            fila[2]
        ),
        "minutos_distintos": int(
            fila[3]
        ),
        "colisiones_al_normalizar": int(
            fila[4]
        ),
        "desplazamiento_minimo_segundos": (
            float(fila[5])
            if fila[5] is not None
            else None
        ),
        "desplazamiento_maximo_segundos": (
            float(fila[6])
            if fila[6] is not None
            else None
        ),
    }


def obtener_muestras_no_alineadas(
    cursor: psycopg.Cursor,
    mercado_id: int,
    simbolo: str,
    inicio: datetime,
    fin: datetime,
    limite: int = 50,
) -> list[dict[str, Any]]:
    """Obtiene ejemplos de aperturas que no comienzan en segundo cero."""

    cursor.execute(
        """
        SELECT
            fecha_apertura,
            DATE_TRUNC(
                'minute',
                fecha_apertura
            ) AS minuto_normalizado,
            EXTRACT(
                EPOCH FROM (
                    fecha_apertura
                    - DATE_TRUNC(
                        'minute',
                        fecha_apertura
                    )
                )
            ) AS desplazamiento_segundos
        FROM velas
        WHERE mercado_id = %s
          AND intervalo = %s
          AND fecha_apertura >= %s
          AND fecha_apertura < %s
          AND fecha_apertura <> DATE_TRUNC(
                'minute',
                fecha_apertura
          )
        ORDER BY fecha_apertura
        LIMIT %s;
        """,
        (
            mercado_id,
            INTERVALO,
            inicio,
            fin,
            limite,
        ),
    )

    return [
        {
            "simbolo": simbolo,
            "fecha_apertura_original": fila[0],
            "minuto_normalizado": fila[1],
            "desplazamiento_segundos": float(
                fila[2]
            ),
        }
        for fila in cursor.fetchall()
    ]


def obtener_solapamiento_exactamente(
    cursor: psycopg.Cursor,
    mercado_btc_id: int,
    mercado_eth_id: int,
    inicio: datetime,
    fin: datetime,
) -> dict[str, int]:
    """Compara las marcas temporales originales sin normalizarlas."""

    cursor.execute(
        """
        WITH btc AS (
            SELECT DISTINCT fecha_apertura
            FROM velas
            WHERE mercado_id = %s
              AND intervalo = %s
              AND fecha_apertura >= %s
              AND fecha_apertura < %s
        ),
        eth AS (
            SELECT DISTINCT fecha_apertura
            FROM velas
            WHERE mercado_id = %s
              AND intervalo = %s
              AND fecha_apertura >= %s
              AND fecha_apertura < %s
        ),
        comparacion AS (
            SELECT
                COALESCE(
                    btc.fecha_apertura,
                    eth.fecha_apertura
                ) AS fecha,
                btc.fecha_apertura IS NOT NULL AS existe_btc,
                eth.fecha_apertura IS NOT NULL AS existe_eth
            FROM btc
            FULL OUTER JOIN eth
                ON eth.fecha_apertura = btc.fecha_apertura
        )
        SELECT
            COUNT(*) FILTER (
                WHERE existe_btc
                  AND existe_eth
            ) AS comunes,
            COUNT(*) FILTER (
                WHERE existe_btc
                  AND NOT existe_eth
            ) AS solo_btc,
            COUNT(*) FILTER (
                WHERE existe_eth
                  AND NOT existe_btc
            ) AS solo_eth
        FROM comparacion;
        """,
        (
            mercado_btc_id,
            INTERVALO,
            inicio,
            fin,
            mercado_eth_id,
            INTERVALO,
            inicio,
            fin,
        ),
    )

    fila = cursor.fetchone()

    return {
        "comunes": int(
            fila[0]
        ),
        "solo_btc": int(
            fila[1]
        ),
        "solo_eth": int(
            fila[2]
        ),
    }


def obtener_solapamiento_normalizado(
    cursor: psycopg.Cursor,
    mercado_btc_id: int,
    mercado_eth_id: int,
    inicio: datetime,
    fin: datetime,
) -> dict[str, Any]:
    """Compara ambos símbolos después de agrupar por minuto natural."""

    cursor.execute(
        """
        WITH btc AS (
            SELECT DISTINCT
                DATE_TRUNC(
                    'minute',
                    fecha_apertura
                ) AS minuto
            FROM velas
            WHERE mercado_id = %s
              AND intervalo = %s
              AND fecha_apertura >= %s
              AND fecha_apertura < %s
        ),
        eth AS (
            SELECT DISTINCT
                DATE_TRUNC(
                    'minute',
                    fecha_apertura
                ) AS minuto
            FROM velas
            WHERE mercado_id = %s
              AND intervalo = %s
              AND fecha_apertura >= %s
              AND fecha_apertura < %s
        ),
        comparacion AS (
            SELECT
                COALESCE(
                    btc.minuto,
                    eth.minuto
                ) AS minuto,
                btc.minuto IS NOT NULL AS existe_btc,
                eth.minuto IS NOT NULL AS existe_eth
            FROM btc
            FULL OUTER JOIN eth
                ON eth.minuto = btc.minuto
        )
        SELECT
            MIN(minuto) FILTER (
                WHERE existe_btc
                  AND existe_eth
            ) AS primera_comun,
            MAX(minuto) FILTER (
                WHERE existe_btc
                  AND existe_eth
            ) AS ultima_comun,
            COUNT(*) FILTER (
                WHERE existe_btc
                  AND existe_eth
            ) AS comunes,
            COUNT(*) FILTER (
                WHERE existe_btc
                  AND NOT existe_eth
            ) AS solo_btc,
            COUNT(*) FILTER (
                WHERE existe_eth
                  AND NOT existe_btc
            ) AS solo_eth
        FROM comparacion;
        """,
        (
            mercado_btc_id,
            INTERVALO,
            inicio,
            fin,
            mercado_eth_id,
            INTERVALO,
            inicio,
            fin,
        ),
    )

    fila = cursor.fetchone()

    return {
        "primera_comun": fila[0],
        "ultima_comun": fila[1],
        "comunes": int(
            fila[2]
        ),
        "solo_btc": int(
            fila[3]
        ),
        "solo_eth": int(
            fila[4]
        ),
    }


def obtener_diferencias_dentro_del_mismo_minuto(
    cursor: psycopg.Cursor,
    mercado_btc_id: int,
    mercado_eth_id: int,
    inicio: datetime,
    fin: datetime,
) -> dict[str, Any]:
    """Cuenta minutos compartidos cuyos timestamps originales son distintos."""

    cursor.execute(
        """
        WITH btc AS (
            SELECT
                DATE_TRUNC(
                    'minute',
                    fecha_apertura
                ) AS minuto,
                MIN(
                    fecha_apertura
                ) AS fecha_btc,
                COUNT(*) AS registros_btc
            FROM velas
            WHERE mercado_id = %s
              AND intervalo = %s
              AND fecha_apertura >= %s
              AND fecha_apertura < %s
            GROUP BY 1
        ),
        eth AS (
            SELECT
                DATE_TRUNC(
                    'minute',
                    fecha_apertura
                ) AS minuto,
                MIN(
                    fecha_apertura
                ) AS fecha_eth,
                COUNT(*) AS registros_eth
            FROM velas
            WHERE mercado_id = %s
              AND intervalo = %s
              AND fecha_apertura >= %s
              AND fecha_apertura < %s
            GROUP BY 1
        )
        SELECT
            COUNT(*) FILTER (
                WHERE btc.fecha_btc <> eth.fecha_eth
            ) AS timestamps_distintos,
            MAX(
                ABS(
                    EXTRACT(
                        EPOCH FROM (
                            btc.fecha_btc
                            - eth.fecha_eth
                        )
                    )
                )
            ) FILTER (
                WHERE btc.fecha_btc <> eth.fecha_eth
            ) AS diferencia_maxima_segundos,
            COUNT(*) FILTER (
                WHERE btc.registros_btc > 1
                   OR eth.registros_eth > 1
            ) AS minutos_con_colision
        FROM btc
        INNER JOIN eth
            ON eth.minuto = btc.minuto;
        """,
        (
            mercado_btc_id,
            INTERVALO,
            inicio,
            fin,
            mercado_eth_id,
            INTERVALO,
            inicio,
            fin,
        ),
    )

    fila = cursor.fetchone()

    return {
        "timestamps_distintos_mismo_minuto": int(
            fila[0]
        ),
        "diferencia_maxima_segundos": (
            float(fila[1])
            if fila[1] is not None
            else None
        ),
        "minutos_con_colision": int(
            fila[2]
        ),
    }


def obtener_solapamiento_mensual(
    cursor: psycopg.Cursor,
    mercado_btc_id: int,
    mercado_eth_id: int,
    inicio: datetime,
    fin: datetime,
) -> list[dict[str, Any]]:
    """Resume el solapamiento exacto y normalizado por mes."""

    cursor.execute(
        """
        WITH btc_original AS (
            SELECT DISTINCT fecha_apertura
            FROM velas
            WHERE mercado_id = %s
              AND intervalo = %s
              AND fecha_apertura >= %s
              AND fecha_apertura < %s
        ),
        eth_original AS (
            SELECT DISTINCT fecha_apertura
            FROM velas
            WHERE mercado_id = %s
              AND intervalo = %s
              AND fecha_apertura >= %s
              AND fecha_apertura < %s
        ),
        exacto AS (
            SELECT
                DATE_TRUNC(
                    'month',
                    COALESCE(
                        btc_original.fecha_apertura,
                        eth_original.fecha_apertura
                    )
                ) AS mes,
                COUNT(*) FILTER (
                    WHERE btc_original.fecha_apertura IS NOT NULL
                      AND eth_original.fecha_apertura IS NOT NULL
                ) AS comunes_exactos,
                COUNT(*) FILTER (
                    WHERE btc_original.fecha_apertura IS NOT NULL
                      AND eth_original.fecha_apertura IS NULL
                ) AS solo_btc_exacto,
                COUNT(*) FILTER (
                    WHERE eth_original.fecha_apertura IS NOT NULL
                      AND btc_original.fecha_apertura IS NULL
                ) AS solo_eth_exacto
            FROM btc_original
            FULL OUTER JOIN eth_original
                ON eth_original.fecha_apertura
                    = btc_original.fecha_apertura
            GROUP BY 1
        ),
        btc_minuto AS (
            SELECT DISTINCT
                DATE_TRUNC(
                    'minute',
                    fecha_apertura
                ) AS minuto
            FROM velas
            WHERE mercado_id = %s
              AND intervalo = %s
              AND fecha_apertura >= %s
              AND fecha_apertura < %s
        ),
        eth_minuto AS (
            SELECT DISTINCT
                DATE_TRUNC(
                    'minute',
                    fecha_apertura
                ) AS minuto
            FROM velas
            WHERE mercado_id = %s
              AND intervalo = %s
              AND fecha_apertura >= %s
              AND fecha_apertura < %s
        ),
        normalizado AS (
            SELECT
                DATE_TRUNC(
                    'month',
                    COALESCE(
                        btc_minuto.minuto,
                        eth_minuto.minuto
                    )
                ) AS mes,
                COUNT(*) FILTER (
                    WHERE btc_minuto.minuto IS NOT NULL
                      AND eth_minuto.minuto IS NOT NULL
                ) AS comunes_normalizados,
                COUNT(*) FILTER (
                    WHERE btc_minuto.minuto IS NOT NULL
                      AND eth_minuto.minuto IS NULL
                ) AS solo_btc_normalizado,
                COUNT(*) FILTER (
                    WHERE eth_minuto.minuto IS NOT NULL
                      AND btc_minuto.minuto IS NULL
                ) AS solo_eth_normalizado
            FROM btc_minuto
            FULL OUTER JOIN eth_minuto
                ON eth_minuto.minuto
                    = btc_minuto.minuto
            GROUP BY 1
        )
        SELECT
            COALESCE(
                exacto.mes,
                normalizado.mes
            ) AS mes,
            COALESCE(
                exacto.comunes_exactos,
                0
            ) AS comunes_exactos,
            COALESCE(
                exacto.solo_btc_exacto,
                0
            ) AS solo_btc_exacto,
            COALESCE(
                exacto.solo_eth_exacto,
                0
            ) AS solo_eth_exacto,
            COALESCE(
                normalizado.comunes_normalizados,
                0
            ) AS comunes_normalizados,
            COALESCE(
                normalizado.solo_btc_normalizado,
                0
            ) AS solo_btc_normalizado,
            COALESCE(
                normalizado.solo_eth_normalizado,
                0
            ) AS solo_eth_normalizado
        FROM exacto
        FULL OUTER JOIN normalizado
            ON normalizado.mes = exacto.mes
        ORDER BY mes;
        """,
        (
            mercado_btc_id,
            INTERVALO,
            inicio,
            fin,
            mercado_eth_id,
            INTERVALO,
            inicio,
            fin,
            mercado_btc_id,
            INTERVALO,
            inicio,
            fin,
            mercado_eth_id,
            INTERVALO,
            inicio,
            fin,
        ),
    )

    return [
        {
            "periodo": fila[0].strftime(
                "%Y-%m"
            ),
            "comunes_exactos": int(
                fila[1]
            ),
            "solo_btc_exacto": int(
                fila[2]
            ),
            "solo_eth_exacto": int(
                fila[3]
            ),
            "comunes_normalizados": int(
                fila[4]
            ),
            "solo_btc_normalizado": int(
                fila[5]
            ),
            "solo_eth_normalizado": int(
                fila[6]
            ),
        }
        for fila in cursor.fetchall()
    ]


def escribir_csv(
    ruta: Path,
    filas: list[dict[str, Any]],
) -> None:
    """Escribe una lista de diccionarios en CSV."""

    if not filas:
        return

    ruta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ruta.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=list(
                filas[0].keys()
            ),
        )

        escritor.writeheader()
        escritor.writerows(
            filas
        )


def generar_resumen(
    ruta: Path,
    inicio: datetime,
    fin: datetime,
    resumenes: dict[str, dict[str, Any]],
    exacto: dict[str, int],
    normalizado: dict[str, Any],
    diferencias: dict[str, Any],
) -> None:
    """Genera un resumen en Markdown."""

    lineas = [
        "# Auditoría de alineación temporal histórica",
        "",
        f"- **Desde:** {inicio.isoformat()}",
        f"- **Hasta exclusivo:** {fin.isoformat()}",
        f"- **Intervalo:** {INTERVALO}",
        "- **Datos modificados:** no.",
        "",
        "## Alineación por símbolo",
        "",
        "| Símbolo | Registros | Alineadas al minuto | No alineadas | Minutos distintos | Colisiones al normalizar |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for simbolo in SIMBOLOS:
        datos = resumenes[
            simbolo
        ]

        lineas.append(
            "| "
            f"{simbolo} | "
            f"{datos['registros']} | "
            f"{datos['alineadas']} | "
            f"{datos['no_alineadas']} | "
            f"{datos['minutos_distintos']} | "
            f"{datos['colisiones_al_normalizar']} |"
        )

    lineas.extend(
        [
            "",
            "## Solapamiento original",
            "",
            f"- **Comunes exactos:** {exacto['comunes']}",
            f"- **Solo BTC:** {exacto['solo_btc']}",
            f"- **Solo ETH:** {exacto['solo_eth']}",
            "",
            "## Solapamiento por minuto natural",
            "",
            f"- **Primera vela común:** {normalizado['primera_comun']}",
            f"- **Última vela común:** {normalizado['ultima_comun']}",
            f"- **Comunes normalizados:** {normalizado['comunes']}",
            f"- **Solo BTC normalizado:** {normalizado['solo_btc']}",
            f"- **Solo ETH normalizado:** {normalizado['solo_eth']}",
            "",
            "## Diferencias dentro del mismo minuto",
            "",
            f"- **Timestamps originales distintos:** {diferencias['timestamps_distintos_mismo_minuto']}",
            f"- **Diferencia máxima en segundos:** {diferencias['diferencia_maxima_segundos']}",
            f"- **Minutos con colisión:** {diferencias['minutos_con_colision']}",
            "",
            "La normalización solo se está auditando. Este proceso no altera "
            "las velas originales ni autoriza todavía su uso en entrenamiento.",
            "",
        ]
    )

    ruta.write_text(
        "\n".join(
            lineas
        ),
        encoding="utf-8",
    )


def auditar(
    desde: str,
    hasta: str,
) -> None:
    """Ejecuta la auditoría de alineación temporal."""

    inicio = convertir_mes(
        desde
    )

    fin = mes_siguiente(
        convertir_mes(
            hasta
        )
    )

    if inicio >= fin:
        raise ValueError(
            "El periodo indicado no es válido."
        )

    with obtener_conexion() as conexion:
        with conexion.cursor() as cursor:
            mercados = {
                simbolo: obtener_mercado_id(
                    cursor=cursor,
                    simbolo=simbolo,
                )
                for simbolo in SIMBOLOS
            }

            resumenes = {
                simbolo: auditar_simbolo(
                    cursor=cursor,
                    mercado_id=mercados[
                        simbolo
                    ],
                    inicio=inicio,
                    fin=fin,
                )
                for simbolo in SIMBOLOS
            }

            muestras: list[
                dict[str, Any]
            ] = []

            for simbolo in SIMBOLOS:
                muestras.extend(
                    obtener_muestras_no_alineadas(
                        cursor=cursor,
                        mercado_id=mercados[
                            simbolo
                        ],
                        simbolo=simbolo,
                        inicio=inicio,
                        fin=fin,
                    )
                )

            exacto = obtener_solapamiento_exactamente(
                cursor=cursor,
                mercado_btc_id=mercados[
                    "BTCUSDT"
                ],
                mercado_eth_id=mercados[
                    "ETHUSDT"
                ],
                inicio=inicio,
                fin=fin,
            )

            normalizado = obtener_solapamiento_normalizado(
                cursor=cursor,
                mercado_btc_id=mercados[
                    "BTCUSDT"
                ],
                mercado_eth_id=mercados[
                    "ETHUSDT"
                ],
                inicio=inicio,
                fin=fin,
            )

            diferencias = obtener_diferencias_dentro_del_mismo_minuto(
                cursor=cursor,
                mercado_btc_id=mercados[
                    "BTCUSDT"
                ],
                mercado_eth_id=mercados[
                    "ETHUSDT"
                ],
                inicio=inicio,
                fin=fin,
            )

            mensual = obtener_solapamiento_mensual(
                cursor=cursor,
                mercado_btc_id=mercados[
                    "BTCUSDT"
                ],
                mercado_eth_id=mercados[
                    "ETHUSDT"
                ],
                inicio=inicio,
                fin=fin,
            )

    RUTA_SALIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    sufijo = (
        f"{desde.replace('-', '')}_"
        f"{hasta.replace('-', '')}"
    )

    ruta_mensual = (
        RUTA_SALIDA
        / f"auditoria_alineacion_mensual_{sufijo}.csv"
    )

    ruta_muestras = (
        RUTA_SALIDA
        / f"muestras_timestamps_no_alineados_{sufijo}.csv"
    )

    ruta_resumen = (
        RUTA_SALIDA
        / f"resumen_auditoria_alineacion_{sufijo}.md"
    )

    escribir_csv(
        ruta=ruta_mensual,
        filas=mensual,
    )

    escribir_csv(
        ruta=ruta_muestras,
        filas=muestras,
    )

    generar_resumen(
        ruta=ruta_resumen,
        inicio=inicio,
        fin=fin,
        resumenes=resumenes,
        exacto=exacto,
        normalizado=normalizado,
        diferencias=diferencias,
    )

    print("\nAUDITORÍA DE ALINEACIÓN COMPLETADA")
    print("=" * 70)

    for simbolo in SIMBOLOS:
        datos = resumenes[
            simbolo
        ]

        print(f"\n{simbolo}")
        print(
            f"Registros: "
            f"{datos['registros']:,}"
            .replace(
                ",",
                ".",
            )
        )
        print(
            f"Alineados al minuto: "
            f"{datos['alineadas']:,}"
            .replace(
                ",",
                ".",
            )
        )
        print(
            f"No alineados: "
            f"{datos['no_alineadas']:,}"
            .replace(
                ",",
                ".",
            )
        )
        print(
            f"Colisiones al normalizar: "
            f"{datos['colisiones_al_normalizar']:,}"
            .replace(
                ",",
                ".",
            )
        )

    print("\nSOLAPAMIENTO ORIGINAL")
    print("-" * 70)
    print(
        f"Comunes exactos: "
        f"{exacto['comunes']:,}"
        .replace(
            ",",
            ".",
        )
    )
    print(
        f"Solo BTC: "
        f"{exacto['solo_btc']:,}"
        .replace(
            ",",
            ".",
        )
    )
    print(
        f"Solo ETH: "
        f"{exacto['solo_eth']:,}"
        .replace(
            ",",
            ".",
        )
    )

    print("\nSOLAPAMIENTO NORMALIZADO AL MINUTO")
    print("-" * 70)
    print(
        f"Comunes: "
        f"{normalizado['comunes']:,}"
        .replace(
            ",",
            ".",
        )
    )
    print(
        f"Solo BTC: "
        f"{normalizado['solo_btc']:,}"
        .replace(
            ",",
            ".",
        )
    )
    print(
        f"Solo ETH: "
        f"{normalizado['solo_eth']:,}"
        .replace(
            ",",
            ".",
        )
    )
    print(
        f"Timestamps distintos dentro del mismo minuto: "
        f"{diferencias['timestamps_distintos_mismo_minuto']:,}"
        .replace(
            ",",
            ".",
        )
    )
    print(
        f"Diferencia máxima en segundos: "
        f"{diferencias['diferencia_maxima_segundos']}"
    )
    print(
        f"Minutos con colisión: "
        f"{diferencias['minutos_con_colision']:,}"
        .replace(
            ",",
            ".",
        )
    )

    print("\nArchivos generados:")
    print(f"- {ruta_mensual}")
    print(f"- {ruta_muestras}")
    print(f"- {ruta_resumen}")
    print(
        "\nNo se modificaron datos ni se entrenó ningún modelo."
    )


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene los argumentos del comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Audita la alineación temporal de BTCUSDT y ETHUSDT "
            "sin modificar las velas originales."
        )
    )

    parser.add_argument(
        "--desde",
        default="2017-08",
        help="Primer mes incluido, con formato AAAA-MM.",
    )

    parser.add_argument(
        "--hasta",
        default="2020-12",
        help="Último mes incluido, con formato AAAA-MM.",
    )

    return parser.parse_args()


def main() -> None:
    """Punto de entrada."""

    argumentos = obtener_argumentos()

    try:
        auditar(
            desde=argumentos.desde,
            hasta=argumentos.hasta,
        )
    except (
        argparse.ArgumentTypeError,
        psycopg.Error,
        OSError,
        ValueError,
    ) as error:
        print(
            "\nNo se pudo completar la auditoría de alineación."
        )
        print(
            f"Detalle: {error}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
