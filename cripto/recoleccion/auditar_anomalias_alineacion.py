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


def obtener_minutos_problematicos(
    cursor: psycopg.Cursor,
    mercado_btc_id: int,
    mercado_eth_id: int,
    inicio: datetime,
    fin: datetime,
) -> list[dict[str, Any]]:
    """
    Localiza minutos ausentes o con más de una vela en alguno de los símbolos.
    """

    cursor.execute(
        """
        WITH btc AS (
            SELECT
                DATE_TRUNC(
                    'minute',
                    fecha_apertura
                ) AS minuto,
                COUNT(*) AS conteo_btc
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
                COUNT(*) AS conteo_eth
            FROM velas
            WHERE mercado_id = %s
              AND intervalo = %s
              AND fecha_apertura >= %s
              AND fecha_apertura < %s
            GROUP BY 1
        )
        SELECT
            COALESCE(
                btc.minuto,
                eth.minuto
            ) AS minuto,
            COALESCE(
                btc.conteo_btc,
                0
            ) AS conteo_btc,
            COALESCE(
                eth.conteo_eth,
                0
            ) AS conteo_eth
        FROM btc
        FULL OUTER JOIN eth
            ON eth.minuto = btc.minuto
        WHERE COALESCE(
                btc.conteo_btc,
                0
              ) <> 1
           OR COALESCE(
                eth.conteo_eth,
                0
              ) <> 1
        ORDER BY minuto;
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

    return [
        {
            "minuto": fila[0],
            "conteo_btc": int(
                fila[1]
            ),
            "conteo_eth": int(
                fila[2]
            ),
            "tipo_anomalia": clasificar_anomalia(
                conteo_btc=int(
                    fila[1]
                ),
                conteo_eth=int(
                    fila[2]
                ),
            ),
        }
        for fila in cursor.fetchall()
    ]


def clasificar_anomalia(
    conteo_btc: int,
    conteo_eth: int,
) -> str:
    """Clasifica el tipo de anomalía encontrada."""

    if conteo_btc == 0:
        return "FALTA_BTC"

    if conteo_eth == 0:
        return "FALTA_ETH"

    if conteo_btc > 1 and conteo_eth > 1:
        return "COLISION_AMBOS"

    if conteo_btc > 1:
        return "COLISION_BTC"

    if conteo_eth > 1:
        return "COLISION_ETH"

    return "OTRA"


def obtener_detalle_velas(
    cursor: psycopg.Cursor,
    mercados: dict[str, int],
    minutos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Obtiene las velas originales de cada minuto problemático."""

    detalles: list[
        dict[str, Any]
    ] = []

    for anomalia in minutos:
        minuto = anomalia[
            "minuto"
        ]

        for simbolo in SIMBOLOS:
            cursor.execute(
                """
                SELECT
                    fecha_apertura,
                    fecha_cierre,
                    precio_apertura,
                    precio_maximo,
                    precio_minimo,
                    precio_cierre,
                    volumen,
                    volumen_activo_cotizacion,
                    numero_operaciones,
                    volumen_comprador_base,
                    volumen_comprador_cotizacion
                FROM velas
                WHERE mercado_id = %s
                  AND intervalo = %s
                  AND DATE_TRUNC(
                        'minute',
                        fecha_apertura
                      ) = %s
                ORDER BY fecha_apertura;
                """,
                (
                    mercados[
                        simbolo
                    ],
                    INTERVALO,
                    minuto,
                ),
            )

            filas = cursor.fetchall()

            for posicion, fila in enumerate(
                filas,
                start=1,
            ):
                detalles.append(
                    {
                        "minuto_normalizado": minuto,
                        "tipo_anomalia": anomalia[
                            "tipo_anomalia"
                        ],
                        "simbolo": simbolo,
                        "posicion_en_minuto": posicion,
                        "fecha_apertura": fila[0],
                        "fecha_cierre": fila[1],
                        "desplazamiento_segundos": (
                            fila[0]
                            - minuto
                        ).total_seconds(),
                        "precio_apertura": fila[2],
                        "precio_maximo": fila[3],
                        "precio_minimo": fila[4],
                        "precio_cierre": fila[5],
                        "volumen": fila[6],
                        "volumen_activo_cotizacion": fila[7],
                        "numero_operaciones": fila[8],
                        "volumen_comprador_base": fila[9],
                        "volumen_comprador_cotizacion": fila[10],
                    }
                )

    return detalles


def escribir_csv(
    ruta: Path,
    filas: list[dict[str, Any]],
) -> None:
    """Escribe una lista de diccionarios en CSV."""

    ruta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not filas:
        ruta.write_text(
            "",
            encoding="utf-8-sig",
        )
        return

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
    minutos: list[dict[str, Any]],
    detalles: list[dict[str, Any]],
) -> None:
    """Genera un resumen legible de las anomalías."""

    lineas = [
        "# Auditoría de anomalías tras normalizar al minuto",
        "",
        f"- **Desde:** {inicio.isoformat()}",
        f"- **Hasta exclusivo:** {fin.isoformat()}",
        f"- **Intervalo:** {INTERVALO}",
        "- **Datos modificados:** no.",
        "",
        "## Minutos problemáticos",
        "",
        "| Minuto | BTC | ETH | Tipo |",
        "|---|---:|---:|---|",
    ]

    for anomalia in minutos:
        lineas.append(
            "| "
            f"{anomalia['minuto']} | "
            f"{anomalia['conteo_btc']} | "
            f"{anomalia['conteo_eth']} | "
            f"{anomalia['tipo_anomalia']} |"
        )

    lineas.extend(
        [
            "",
            "## Resultado",
            "",
            f"- **Minutos problemáticos:** {len(minutos)}",
            f"- **Velas originales implicadas:** {len(detalles)}",
            "",
            "Esta auditoría no corrige ni elimina registros. Su objetivo es "
            "definir después una regla determinista para preparar los datos "
            "sin alterar PostgreSQL ni los archivos originales.",
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
    """Ejecuta la auditoría puntual de colisiones y minutos ausentes."""

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

            minutos = obtener_minutos_problematicos(
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

            detalles = obtener_detalle_velas(
                cursor=cursor,
                mercados=mercados,
                minutos=minutos,
            )

    RUTA_SALIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    sufijo = (
        f"{desde.replace('-', '')}_"
        f"{hasta.replace('-', '')}"
    )

    ruta_minutos = (
        RUTA_SALIDA
        / f"auditoria_minutos_problematicos_{sufijo}.csv"
    )

    ruta_detalles = (
        RUTA_SALIDA
        / f"auditoria_detalle_anomalias_{sufijo}.csv"
    )

    ruta_resumen = (
        RUTA_SALIDA
        / f"resumen_anomalias_alineacion_{sufijo}.md"
    )

    escribir_csv(
        ruta=ruta_minutos,
        filas=minutos,
    )

    escribir_csv(
        ruta=ruta_detalles,
        filas=detalles,
    )

    generar_resumen(
        ruta=ruta_resumen,
        inicio=inicio,
        fin=fin,
        minutos=minutos,
        detalles=detalles,
    )

    print("\nAUDITORÍA DE ANOMALÍAS COMPLETADA")
    print("=" * 70)
    print(
        f"Minutos problemáticos: "
        f"{len(minutos)}"
    )

    for anomalia in minutos:
        print(
            "\n"
            f"{anomalia['minuto']} | "
            f"BTC: {anomalia['conteo_btc']} | "
            f"ETH: {anomalia['conteo_eth']} | "
            f"{anomalia['tipo_anomalia']}"
        )

    print("\nArchivos generados:")
    print(f"- {ruta_minutos}")
    print(f"- {ruta_detalles}")
    print(f"- {ruta_resumen}")
    print(
        "\nNo se modificaron datos ni se entrenó ningún modelo."
    )


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene los argumentos del comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Localiza colisiones y minutos ausentes después de "
            "normalizar las velas históricas al minuto."
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
            "\nNo se pudo completar la auditoría de anomalías."
        )
        print(
            f"Detalle: {error}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
