from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg

from compartido.base_datos.conexion import obtener_conexion


SIMBOLOS = (
    "BTCUSDT",
    "ETHUSDT",
)

INTERVALO = "1m"
DURACION_VELA = timedelta(minutes=1)

RUTA_PROYECTO = Path(__file__).resolve().parents[2]

RUTA_SALIDA = (
    RUTA_PROYECTO
    / "datos"
    / "cripto"
    / "preparados"
)


@dataclass(frozen=True)
class Periodo:
    inicio: datetime
    fin: datetime


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


def construir_periodo(
    desde: str,
    hasta: str,
) -> Periodo:
    """Construye un periodo mensual inclusivo."""

    inicio = convertir_mes(
        desde
    )

    ultimo_mes = convertir_mes(
        hasta
    )

    fin = mes_siguiente(
        ultimo_mes
    )

    if inicio >= fin:
        raise ValueError(
            "El mes inicial debe ser anterior o igual al mes final."
        )

    return Periodo(
        inicio=inicio,
        fin=fin,
    )


def recorrer_meses(
    periodo: Periodo,
):
    """Recorre todos los meses del periodo."""

    actual = periodo.inicio

    while actual < periodo.fin:
        siguiente = mes_siguiente(
            actual
        )

        yield actual, siguiente

        actual = siguiente


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


def obtener_resumen_global(
    cursor: psycopg.Cursor,
    mercado_id: int,
    periodo: Periodo,
) -> dict[str, Any]:
    """Obtiene el resumen global de un símbolo."""

    cursor.execute(
        """
        SELECT
            COUNT(*) AS total,
            COUNT(DISTINCT fecha_apertura) AS fechas_distintas,
            MIN(fecha_apertura) AS primera_vela,
            MAX(fecha_apertura) AS ultima_vela,
            COUNT(*) FILTER (
                WHERE fecha_cierre <= fecha_apertura
                   OR precio_apertura <= 0
                   OR precio_maximo <= 0
                   OR precio_minimo <= 0
                   OR precio_cierre <= 0
                   OR precio_maximo < GREATEST(
                        precio_apertura,
                        precio_cierre,
                        precio_minimo
                   )
                   OR precio_minimo > LEAST(
                        precio_apertura,
                        precio_cierre,
                        precio_maximo
                   )
                   OR volumen < 0
                   OR volumen_activo_cotizacion < 0
                   OR numero_operaciones < 0
                   OR volumen_comprador_base < 0
                   OR volumen_comprador_cotizacion < 0
            ) AS incoherentes
        FROM velas
        WHERE mercado_id = %s
          AND intervalo = %s
          AND fecha_apertura >= %s
          AND fecha_apertura < %s;
        """,
        (
            mercado_id,
            INTERVALO,
            periodo.inicio,
            periodo.fin,
        ),
    )

    fila = cursor.fetchone()

    total = int(
        fila[0]
    )

    fechas_distintas = int(
        fila[1]
    )

    primera_vela = fila[2]
    ultima_vela = fila[3]
    incoherentes = int(
        fila[4]
    )

    if primera_vela is None or ultima_vela is None:
        raise ValueError(
            "No existen velas dentro del periodo solicitado."
        )

    esperadas_desde_inicio_real = int(
        (
            ultima_vela
            - primera_vela
        )
        / DURACION_VELA
    ) + 1

    return {
        "total": total,
        "fechas_distintas": fechas_distintas,
        "duplicados": (
            total
            - fechas_distintas
        ),
        "primera_vela": primera_vela,
        "ultima_vela": ultima_vela,
        "esperadas_desde_inicio_real": (
            esperadas_desde_inicio_real
        ),
        "faltantes_desde_inicio_real": max(
            esperadas_desde_inicio_real
            - fechas_distintas,
            0,
        ),
        "incoherentes": incoherentes,
    }


def obtener_conteos_mensuales(
    cursor: psycopg.Cursor,
    mercado_id: int,
    periodo: Periodo,
) -> dict[datetime, int]:
    """Obtiene las fechas distintas encontradas por mes."""

    cursor.execute(
        """
        SELECT
            DATE_TRUNC(
                'month',
                fecha_apertura
            ) AS mes,
            COUNT(
                DISTINCT fecha_apertura
            ) AS total
        FROM velas
        WHERE mercado_id = %s
          AND intervalo = %s
          AND fecha_apertura >= %s
          AND fecha_apertura < %s
        GROUP BY 1
        ORDER BY 1;
        """,
        (
            mercado_id,
            INTERVALO,
            periodo.inicio,
            periodo.fin,
        ),
    )

    return {
        fila[0]: int(
            fila[1]
        )
        for fila in cursor.fetchall()
    }


def obtener_huecos(
    cursor: psycopg.Cursor,
    mercado_id: int,
    periodo: Periodo,
) -> list[dict[str, Any]]:
    """Obtiene todos los huecos entre velas consecutivas."""

    cursor.execute(
        """
        WITH ordenadas AS (
            SELECT
                fecha_apertura,
                LAG(
                    fecha_apertura
                ) OVER (
                    ORDER BY fecha_apertura
                ) AS fecha_anterior
            FROM velas
            WHERE mercado_id = %s
              AND intervalo = %s
              AND fecha_apertura >= %s
              AND fecha_apertura < %s
        )
        SELECT
            fecha_anterior,
            fecha_apertura,
            (
                EXTRACT(
                    EPOCH FROM (
                        fecha_apertura
                        - fecha_anterior
                    )
                )::BIGINT / 60
            ) - 1 AS velas_faltantes
        FROM ordenadas
        WHERE fecha_anterior IS NOT NULL
          AND fecha_apertura
              - fecha_anterior
              > INTERVAL '1 minute'
        ORDER BY fecha_anterior;
        """,
        (
            mercado_id,
            INTERVALO,
            periodo.inicio,
            periodo.fin,
        ),
    )

    return [
        {
            "fecha_anterior": fila[0],
            "fecha_siguiente": fila[1],
            "velas_faltantes": int(
                fila[2]
            ),
        }
        for fila in cursor.fetchall()
    ]


def construir_auditoria_mensual(
    simbolo: str,
    periodo: Periodo,
    resumen: dict[str, Any],
    conteos: dict[datetime, int],
) -> list[dict[str, Any]]:
    """Construye el resumen mensual sin contar el tiempo previo al listado."""

    resultados: list[
        dict[str, Any]
    ] = []

    primera_vela = resumen[
        "primera_vela"
    ]

    ultima_vela = resumen[
        "ultima_vela"
    ]

    for inicio_mes, fin_mes in recorrer_meses(
        periodo
    ):
        inicio_efectivo = max(
            inicio_mes,
            primera_vela,
        )

        fin_efectivo = min(
            fin_mes,
            ultima_vela
            + DURACION_VELA,
        )

        encontradas = conteos.get(
            inicio_mes,
            0,
        )

        if inicio_efectivo >= fin_efectivo:
            esperadas = 0
        else:
            esperadas = int(
                (
                    fin_efectivo
                    - inicio_efectivo
                )
                / DURACION_VELA
            )

        faltantes = max(
            esperadas
            - encontradas,
            0,
        )

        resultados.append(
            {
                "simbolo": simbolo,
                "periodo": inicio_mes.strftime(
                    "%Y-%m"
                ),
                "velas_esperadas_desde_inicio_real": esperadas,
                "velas_encontradas": encontradas,
                "velas_faltantes": faltantes,
                "completo": (
                    faltantes == 0
                    and esperadas > 0
                ),
            }
        )

    return resultados


def obtener_solapamiento(
    cursor: psycopg.Cursor,
    mercado_btc_id: int,
    mercado_eth_id: int,
    periodo: Periodo,
) -> dict[str, Any]:
    """Compara las marcas temporales disponibles en BTC y ETH."""

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
            MIN(fecha) FILTER (
                WHERE existe_btc
                  AND existe_eth
            ) AS primera_comun,
            MAX(fecha) FILTER (
                WHERE existe_btc
                  AND existe_eth
            ) AS ultima_comun,
            COUNT(*) FILTER (
                WHERE existe_btc
                  AND existe_eth
            ) AS velas_comunes,
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
            periodo.inicio,
            periodo.fin,
            mercado_eth_id,
            INTERVALO,
            periodo.inicio,
            periodo.fin,
        ),
    )

    fila = cursor.fetchone()

    return {
        "primera_vela_comun": fila[0],
        "ultima_vela_comun": fila[1],
        "velas_comunes": int(
            fila[2]
        ),
        "velas_solo_btc": int(
            fila[3]
        ),
        "velas_solo_eth": int(
            fila[4]
        ),
    }


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


def generar_resumen_markdown(
    ruta: Path,
    periodo: Periodo,
    resumenes: dict[str, dict[str, Any]],
    huecos: dict[str, list[dict[str, Any]]],
    solapamiento: dict[str, Any],
) -> None:
    """Genera el resumen legible de la auditoría."""

    lineas = [
        "# Auditoría histórica BTCUSDT y ETHUSDT",
        "",
        f"- **Desde:** {periodo.inicio.isoformat()}",
        f"- **Hasta exclusivo:** {periodo.fin.isoformat()}",
        f"- **Intervalo:** {INTERVALO}",
        "- **Fuente:** Binance Spot almacenado en PostgreSQL.",
        "- **Relleno de huecos:** no se inventaron velas.",
        "",
        "## Resumen por símbolo",
        "",
        "| Símbolo | Primera vela | Última vela | Encontradas | Faltantes internas | Duplicados | Incoherentes | Huecos |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]

    for simbolo in SIMBOLOS:
        resumen = resumenes[
            simbolo
        ]

        lineas.append(
            "| "
            f"{simbolo} | "
            f"{resumen['primera_vela']} | "
            f"{resumen['ultima_vela']} | "
            f"{resumen['fechas_distintas']} | "
            f"{resumen['faltantes_desde_inicio_real']} | "
            f"{resumen['duplicados']} | "
            f"{resumen['incoherentes']} | "
            f"{len(huecos[simbolo])} |"
        )

    lineas.extend(
        [
            "",
            "## Solapamiento BTC/ETH",
            "",
            f"- **Primera vela común:** {solapamiento['primera_vela_comun']}",
            f"- **Última vela común:** {solapamiento['ultima_vela_comun']}",
            f"- **Velas presentes en ambos:** {solapamiento['velas_comunes']}",
            f"- **Velas presentes solo en BTC:** {solapamiento['velas_solo_btc']}",
            f"- **Velas presentes solo en ETH:** {solapamiento['velas_solo_eth']}",
            "",
            "## Criterio previo al entrenamiento",
            "",
            "Esta auditoría no aprueba automáticamente los datos para entrenar. "
            "Primero deben revisarse los huecos, el solapamiento entre símbolos "
            "y la cantidad de filas que sobrevivirá a las ventanas de variables.",
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
    """Ejecuta la auditoría histórica completa."""

    periodo = construir_periodo(
        desde=desde,
        hasta=hasta,
    )

    resumenes: dict[
        str,
        dict[str, Any],
    ] = {}

    huecos_por_simbolo: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    filas_mensuales: list[
        dict[str, Any]
    ] = []

    filas_huecos: list[
        dict[str, Any]
    ] = []

    with obtener_conexion() as conexion:
        with conexion.cursor() as cursor:
            mercados = {
                simbolo: obtener_mercado_id(
                    cursor=cursor,
                    simbolo=simbolo,
                )
                for simbolo in SIMBOLOS
            }

            for simbolo in SIMBOLOS:
                mercado_id = mercados[
                    simbolo
                ]

                resumen = obtener_resumen_global(
                    cursor=cursor,
                    mercado_id=mercado_id,
                    periodo=periodo,
                )

                conteos = obtener_conteos_mensuales(
                    cursor=cursor,
                    mercado_id=mercado_id,
                    periodo=periodo,
                )

                huecos = obtener_huecos(
                    cursor=cursor,
                    mercado_id=mercado_id,
                    periodo=periodo,
                )

                resumenes[
                    simbolo
                ] = resumen

                huecos_por_simbolo[
                    simbolo
                ] = huecos

                filas_mensuales.extend(
                    construir_auditoria_mensual(
                        simbolo=simbolo,
                        periodo=periodo,
                        resumen=resumen,
                        conteos=conteos,
                    )
                )

                for hueco in huecos:
                    filas_huecos.append(
                        {
                            "simbolo": simbolo,
                            **hueco,
                        }
                    )

            solapamiento = obtener_solapamiento(
                cursor=cursor,
                mercado_btc_id=mercados[
                    "BTCUSDT"
                ],
                mercado_eth_id=mercados[
                    "ETHUSDT"
                ],
                periodo=periodo,
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
        / f"auditoria_historico_mensual_{sufijo}.csv"
    )

    ruta_huecos = (
        RUTA_SALIDA
        / f"auditoria_huecos_{sufijo}.csv"
    )

    ruta_resumen = (
        RUTA_SALIDA
        / f"resumen_auditoria_historico_{sufijo}.md"
    )

    escribir_csv(
        ruta=ruta_mensual,
        filas=filas_mensuales,
    )

    escribir_csv(
        ruta=ruta_huecos,
        filas=filas_huecos,
    )

    generar_resumen_markdown(
        ruta=ruta_resumen,
        periodo=periodo,
        resumenes=resumenes,
        huecos=huecos_por_simbolo,
        solapamiento=solapamiento,
    )

    print("\nAUDITORÍA HISTÓRICA COMPLETADA")
    print("=" * 70)
    print(
        f"Periodo: {desde} a {hasta}"
    )
    print(
        f"Intervalo: {INTERVALO}"
    )

    for simbolo in SIMBOLOS:
        resumen = resumenes[
            simbolo
        ]

        print(
            f"\n{simbolo}"
        )
        print(
            f"Primera vela: "
            f"{resumen['primera_vela']}"
        )
        print(
            f"Última vela: "
            f"{resumen['ultima_vela']}"
        )
        print(
            f"Velas encontradas: "
            f"{resumen['fechas_distintas']:,}"
            .replace(
                ",",
                ".",
            )
        )
        print(
            f"Velas faltantes internas: "
            f"{resumen['faltantes_desde_inicio_real']:,}"
            .replace(
                ",",
                ".",
            )
        )
        print(
            f"Duplicados: "
            f"{resumen['duplicados']}"
        )
        print(
            f"Incoherentes: "
            f"{resumen['incoherentes']}"
        )
        print(
            f"Huecos temporales: "
            f"{len(huecos_por_simbolo[simbolo])}"
        )

    print("\nSOLAPAMIENTO BTC/ETH")
    print("-" * 70)
    print(
        f"Primera vela común: "
        f"{solapamiento['primera_vela_comun']}"
    )
    print(
        f"Última vela común: "
        f"{solapamiento['ultima_vela_comun']}"
    )
    print(
        f"Velas presentes en ambos: "
        f"{solapamiento['velas_comunes']:,}"
        .replace(
            ",",
            ".",
        )
    )
    print(
        f"Velas solo BTC: "
        f"{solapamiento['velas_solo_btc']:,}"
        .replace(
            ",",
            ".",
        )
    )
    print(
        f"Velas solo ETH: "
        f"{solapamiento['velas_solo_eth']:,}"
        .replace(
            ",",
            ".",
        )
    )

    print("\nArchivos generados:")
    print(f"- {ruta_mensual}")
    print(f"- {ruta_huecos}")
    print(f"- {ruta_resumen}")
    print(
        "\nNo se modificaron datos ni se entrenó ningún modelo."
    )


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene los argumentos del comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Audita la integridad y el solapamiento del histórico "
            "BTCUSDT/ETHUSDT almacenado en PostgreSQL."
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
            "\nNo se pudo completar la auditoría histórica."
        )
        print(
            f"Detalle: {error}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
