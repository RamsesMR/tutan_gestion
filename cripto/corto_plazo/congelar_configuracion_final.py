from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from cripto.corto_plazo.configuracion import (
    RUTA_DATOS_PREPARADOS,
    RUTA_PROYECTO,
)
from cripto.corto_plazo.gestionar_para_analisis import (
    actualizar_carpeta_para_analisis,
)


SIMBOLOS = (
    "BTCUSDT",
    "ETHUSDT",
)

RUTA_MODELOS = (
    RUTA_PROYECTO
    / "modelos_entrenados"
    / "cripto"
    / "corto_plazo"
)

RUTA_MANIFIESTO = (
    RUTA_DATOS_PREPARADOS
    / "manifiesto_divisiones_4h.csv"
)

RUTA_CONFIGURACION_CONGELADA = (
    RUTA_MODELOS
    / "configuracion_final_congelada_2026.json"
)

RUTA_RESUMEN_CONGELADO = (
    RUTA_MODELOS
    / "resumen_configuracion_final_congelada_2026.md"
)


def calcular_sha256(
    ruta: Path,
    tamano_bloque: int = 1024 * 1024,
) -> str:
    """Calcula la huella SHA-256 de un archivo."""

    calculador = hashlib.sha256()

    with ruta.open(
        "rb"
    ) as archivo:
        while True:
            bloque = archivo.read(
                tamano_bloque
            )

            if not bloque:
                break

            calculador.update(
                bloque
            )

    return calculador.hexdigest()


def cargar_json(
    ruta: Path,
) -> dict[str, Any]:
    """Carga un archivo JSON y comprueba que sea un objeto."""

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {ruta}"
        )

    with ruta.open(
        "r",
        encoding="utf-8",
    ) as archivo:
        datos = json.load(
            archivo
        )

    if not isinstance(
        datos,
        dict,
    ):
        raise ValueError(
            f"El archivo no contiene un objeto JSON válido: {ruta}"
        )

    return datos


def cargar_periodo_prueba(
    simbolo: str,
) -> dict[str, Any]:
    """
    Obtiene el periodo reservado de prueba desde el manifiesto.

    No abre ni evalúa los datos de prueba.
    """

    if not RUTA_MANIFIESTO.exists():
        raise FileNotFoundError(
            f"No existe el manifiesto: {RUTA_MANIFIESTO}"
        )

    manifiesto = pd.read_csv(
        RUTA_MANIFIESTO
    )

    columnas_requeridas = {
        "simbolo",
        "division",
        "archivo",
        "desde_division",
        "hasta_division",
        "filas_utilizables",
    }

    faltantes = columnas_requeridas.difference(
        manifiesto.columns
    )

    if faltantes:
        raise ValueError(
            "Faltan columnas en el manifiesto: "
            + ", ".join(
                sorted(faltantes)
            )
        )

    registros = manifiesto.loc[
        (manifiesto["simbolo"] == simbolo)
        & (manifiesto["division"] == "prueba")
    ].copy()

    if registros.empty:
        raise ValueError(
            f"No existe una división de prueba para {simbolo}."
        )

    if len(registros) != 1:
        raise ValueError(
            f"Se esperaba un único archivo de prueba para {simbolo}."
        )

    fila = registros.iloc[0]

    return {
        "desde": str(
            fila["desde_division"]
        ),
        "hasta_exclusivo": str(
            fila["hasta_division"]
        ),
        "filas_utilizables": int(
            fila["filas_utilizables"]
        ),
        "archivo_declarado": str(
            fila["archivo"]
        ),
    }


def validar_resultados_previos(
    simbolo: str,
) -> dict[str, str]:
    """Comprueba que existan las validaciones previas necesarias."""

    nombres = {
        "ajuste": (
            f"ajuste_decision_{simbolo}_4h.json"
        ),
        "validacion_mensual": (
            f"resumen_validacion_mensual_{simbolo}_4h.md"
        ),
        "walk_forward": (
            f"resumen_walk_forward_{simbolo}_4h.md"
        ),
    }

    rutas: dict[
        str,
        str,
    ] = {}

    for clave, nombre in nombres.items():
        ruta = (
            RUTA_MODELOS
            / nombre
        )

        if not ruta.exists():
            raise FileNotFoundError(
                f"Falta el resultado previo requerido: {ruta}"
            )

        rutas[clave] = str(
            ruta
        )

    return rutas


def cargar_configuracion_simbolo(
    simbolo: str,
) -> dict[str, Any]:
    """Carga y registra el modelo y el ajuste que quedarán congelados."""

    ruta_modelo = (
        RUTA_MODELOS
        / f"modelo_base_{simbolo}_4h.joblib"
    )

    ruta_ajuste = (
        RUTA_MODELOS
        / f"ajuste_decision_{simbolo}_4h.json"
    )

    if not ruta_modelo.exists():
        raise FileNotFoundError(
            f"No existe el modelo: {ruta_modelo}"
        )

    ajuste = cargar_json(
        ruta=ruta_ajuste
    )

    if ajuste.get(
        "division_prueba_utilizada"
    ) is not False:
        raise ValueError(
            f"El ajuste de {simbolo} no confirma que la prueba "
            "de 2026 permaneció sin utilizar."
        )

    if str(
        ajuste.get(
            "division_utilizada",
            "",
        )
    ) != "validacion_2025":
        raise ValueError(
            f"El ajuste de {simbolo} no fue generado con validación 2025."
        )

    multiplicadores = ajuste.get(
        "multiplicadores",
        {},
    )

    clases_requeridas = {
        "BAJA",
        "NEUTRAL",
        "SUBE",
    }

    faltantes = clases_requeridas.difference(
        multiplicadores
    )

    if faltantes:
        raise ValueError(
            f"Faltan multiplicadores para {simbolo}: "
            + ", ".join(
                sorted(faltantes)
            )
        )

    paquete = joblib.load(
        ruta_modelo
    )

    claves_modelo = {
        "modelo",
        "escalador",
        "columnas_modelo",
        "clases",
        "umbral_clase",
    }

    claves_faltantes = claves_modelo.difference(
        paquete
    )

    if claves_faltantes:
        raise ValueError(
            f"El paquete del modelo de {simbolo} está incompleto: "
            + ", ".join(
                sorted(claves_faltantes)
            )
        )

    modelo = paquete[
        "modelo"
    ]

    columnas_modelo = [
        str(columna)
        for columna in paquete[
            "columnas_modelo"
        ]
    ]

    clases = [
        str(clase)
        for clase in paquete[
            "clases"
        ]
    ]

    return {
        "simbolo": simbolo,
        "modelo": {
            "archivo": str(
                ruta_modelo
            ),
            "sha256": calcular_sha256(
                ruta_modelo
            ),
            "clase": type(
                modelo
            ).__name__,
            "variables": len(
                columnas_modelo
            ),
            "columnas_modelo": columnas_modelo,
            "clases": clases,
            "umbral_clase": float(
                paquete[
                    "umbral_clase"
                ]
            ),
        },
        "ajuste_decision": {
            "archivo": str(
                ruta_ajuste
            ),
            "sha256": calcular_sha256(
                ruta_ajuste
            ),
            "multiplicadores": {
                clase: float(
                    multiplicadores[
                        clase
                    ]
                )
                for clase in (
                    "BAJA",
                    "NEUTRAL",
                    "SUBE",
                )
            },
            "perdida_maxima_accuracy_permitida": float(
                ajuste.get(
                    "perdida_maxima_accuracy_permitida",
                    0.0,
                )
            ),
        },
        "validaciones_previas": validar_resultados_previos(
            simbolo=simbolo
        ),
        "periodo_prueba_reservado": cargar_periodo_prueba(
            simbolo=simbolo
        ),
    }


def generar_resumen(
    configuracion: dict[str, Any],
) -> None:
    """Genera un resumen legible de la configuración congelada."""

    lineas = [
        "# Configuración final congelada antes de la prueba 2026",
        "",
        f"- **Fecha UTC:** {configuracion['fecha_utc']}",
        "- **Estado:** configuración congelada; prueba final todavía no evaluada.",
        "- **Horizonte:** 4 horas.",
        "- **Clases:** BAJA, NEUTRAL y SUBE.",
        "",
        "## Configuraciones por símbolo",
        "",
    ]

    for datos in configuracion[
        "simbolos"
    ]:
        simbolo = datos[
            "simbolo"
        ]

        modelo = datos[
            "modelo"
        ]

        ajuste = datos[
            "ajuste_decision"
        ]

        periodo = datos[
            "periodo_prueba_reservado"
        ]

        multiplicadores = ajuste[
            "multiplicadores"
        ]

        lineas.extend(
            [
                f"### {simbolo}",
                "",
                f"- **Modelo:** {modelo['clase']}",
                f"- **Variables:** {modelo['variables']}",
                f"- **Umbral de clase:** ±{modelo['umbral_clase'] * 100:.2f} %",
                f"- **Multiplicador BAJA:** {multiplicadores['BAJA']:.2f}",
                f"- **Multiplicador NEUTRAL:** {multiplicadores['NEUTRAL']:.2f}",
                f"- **Multiplicador SUBE:** {multiplicadores['SUBE']:.2f}",
                f"- **Prueba reservada desde:** {periodo['desde']}",
                f"- **Prueba reservada hasta:** {periodo['hasta_exclusivo']} (exclusivo)",
                f"- **Muestras declaradas:** {periodo['filas_utilizables']:,}".replace(
                    ",",
                    ".",
                ),
                f"- **SHA-256 modelo:** `{modelo['sha256']}`",
                f"- **SHA-256 ajuste:** `{ajuste['sha256']}`",
                "",
            ]
        )

    lineas.extend(
        [
            "## Regla de integridad",
            "",
            "La evaluación final deberá verificar estas huellas antes de abrir "
            "la división de prueba. Si el modelo o el ajuste cambian, la "
            "evaluación se detendrá.",
            "",
            "La división de prueba de 2026 no fue utilizada al generar este archivo.",
            "",
        ]
    )

    RUTA_RESUMEN_CONGELADO.write_text(
        "\n".join(
            lineas
        ),
        encoding="utf-8",
    )


def congelar_configuracion() -> None:
    """Congela las configuraciones finales antes de evaluar 2026."""

    if RUTA_CONFIGURACION_CONGELADA.exists():
        raise FileExistsError(
            "La configuración final ya fue congelada. "
            f"Archivo existente: {RUTA_CONFIGURACION_CONGELADA}"
        )

    print(
        "\nCONGELANDO CONFIGURACIÓN FINAL"
    )
    print("=" * 70)
    print(
        "La división de prueba de 2026 no será abierta."
    )

    configuraciones = [
        cargar_configuracion_simbolo(
            simbolo=simbolo
        )
        for simbolo in SIMBOLOS
    ]

    configuracion = {
        "fecha_utc": datetime.now(
            timezone.utc
        ).isoformat(
            timespec="seconds"
        ),
        "estado": (
            "configuracion_congelada_antes_prueba"
        ),
        "horizonte": "4h",
        "division_prueba_utilizada": False,
        "simbolos": configuraciones,
    }

    RUTA_MODELOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RUTA_CONFIGURACION_CONGELADA.open(
        "x",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            configuracion,
            archivo,
            ensure_ascii=False,
            indent=4,
        )

    generar_resumen(
        configuracion=configuracion
    )

    print(
        "\nCONFIGURACIÓN CONGELADA"
    )
    print("=" * 70)

    for datos in configuraciones:
        simbolo = datos[
            "simbolo"
        ]

        multiplicadores = datos[
            "ajuste_decision"
        ][
            "multiplicadores"
        ]

        print(
            f"{simbolo}: "
            f"BAJA {multiplicadores['BAJA']:.2f} | "
            f"NEUTRAL {multiplicadores['NEUTRAL']:.2f} | "
            f"SUBE {multiplicadores['SUBE']:.2f}"
        )

    print(
        f"\nArchivo: {RUTA_CONFIGURACION_CONGELADA}"
    )
    print(
        f"Resumen: {RUTA_RESUMEN_CONGELADO}"
    )

    archivos = actualizar_carpeta_para_analisis()

    print(
        "\nCARPETA PARA ANÁLISIS ACTUALIZADA"
    )
    print("=" * 70)

    for ruta in archivos:
        print(f"- {ruta.name}")

    print(
        "\nLa prueba de 2026 continúa sin utilizar."
    )


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene los argumentos del comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Congela los modelos y ajustes definitivos "
            "antes de evaluar la prueba de 2026."
        )
    )

    parser.add_argument(
        "--confirmar",
        action="store_true",
        help=(
            "Confirma que los ajustes actuales son los definitivos."
        ),
    )

    argumentos = parser.parse_args()

    if not argumentos.confirmar:
        parser.error(
            "Debes incluir --confirmar para congelar "
            "la configuración final."
        )

    return argumentos


def main() -> None:
    """Punto de entrada."""

    obtener_argumentos()

    try:
        congelar_configuracion()
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(
            "\nNo se pudo congelar la configuración final."
        )
        print(
            f"Detalle: {error}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
