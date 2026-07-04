from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from cripto.corto_plazo.configuracion import RUTA_PROYECTO


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

RUTA_DETALLES = (
    RUTA_MODELOS
    / "historial"
)

RUTA_PARA_ANALISIS = (
    RUTA_MODELOS
    / "para_analisis"
)


ARCHIVOS_GENERALES = (
    "historial_entrenamientos.csv",
    "resumen_evolucion_entrenamientos.md",
    "configuracion_final_congelada_2026.json",
    "resumen_configuracion_final_congelada_2026.md",
)

ARCHIVOS_POR_SIMBOLO = (
    "ajuste_decision_{simbolo}_4h.json",
    "informe_ajuste_decision_{simbolo}_4h.json",
    "matriz_confusion_ajuste_decision_{simbolo}_4h.csv",
    "validacion_mensual_{simbolo}_4h.csv",
    "comparacion_validacion_mensual_{simbolo}_4h.csv",
    "resumen_validacion_mensual_{simbolo}_4h.md",
    "validacion_walk_forward_{simbolo}_4h.csv",
    "comparacion_walk_forward_{simbolo}_4h.csv",
    "resumen_walk_forward_{simbolo}_4h.md",
)


def leer_json(ruta: Path) -> dict[str, Any] | None:
    """Lee un JSON y devuelve None si no tiene un formato válido."""

    try:
        with ruta.open(
            "r",
            encoding="utf-8",
        ) as archivo:
            datos = json.load(archivo)
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    if not isinstance(datos, dict):
        return None

    return datos


def obtener_fecha_detalle(
    ruta: Path,
) -> float:
    """Obtiene una marca de tiempo para ordenar los detalles."""

    datos = leer_json(ruta)

    if datos:
        registro = datos.get(
            "registro",
            {},
        )

        fecha_texto = registro.get(
            "fecha_utc"
        )

        if fecha_texto:
            try:
                return datetime.fromisoformat(
                    str(fecha_texto)
                ).timestamp()
            except ValueError:
                pass

    return ruta.stat().st_mtime


def detalle_pertenece_simbolo(
    ruta: Path,
    simbolo: str,
) -> bool:
    """Comprueba si un detalle pertenece al símbolo indicado."""

    datos = leer_json(ruta)

    if datos:
        registro = datos.get(
            "registro",
            {},
        )

        simbolo_registro = str(
            registro.get(
                "simbolo",
                "",
            )
        ).upper()

        if simbolo_registro:
            return simbolo_registro == simbolo

    return f"_{simbolo}_" in ruta.name.upper()


def obtener_detalles_simbolo(
    simbolo: str,
) -> list[Path]:
    """Obtiene los detalles completos guardados para un símbolo."""

    if not RUTA_DETALLES.exists():
        return []

    detalles = [
        ruta
        for ruta in RUTA_DETALLES.glob("*.json")
        if detalle_pertenece_simbolo(
            ruta=ruta,
            simbolo=simbolo,
        )
    ]

    return sorted(
        detalles,
        key=obtener_fecha_detalle,
    )


def obtener_ultimo_detalle(
    simbolo: str,
) -> Path | None:
    """Devuelve el detalle más reciente de un símbolo."""

    detalles = obtener_detalles_simbolo(
        simbolo=simbolo
    )

    if not detalles:
        return None

    return detalles[-1]


def eliminar_detalles_anteriores(
    simbolo: str,
    conservar: Path,
) -> None:
    """
    Elimina detalles antiguos del mismo símbolo.

    La evolución numérica permanece en el CSV y en el resumen Markdown.
    """

    conservar = conservar.resolve()

    for ruta in obtener_detalles_simbolo(
        simbolo=simbolo
    ):
        if ruta.resolve() == conservar:
            continue

        ruta.unlink(
            missing_ok=True
        )


def limpiar_carpeta_para_analisis() -> None:
    """Vacía la carpeta que el usuario enviará para analizar."""

    RUTA_PARA_ANALISIS.mkdir(
        parents=True,
        exist_ok=True,
    )

    for elemento in RUTA_PARA_ANALISIS.iterdir():
        if elemento.is_dir():
            shutil.rmtree(
                elemento
            )
        else:
            elemento.unlink(
                missing_ok=True
            )


def copiar_archivo(
    origen: Path,
    nombre_destino: str | None = None,
) -> bool:
    """Copia un archivo si existe."""

    if not origen.exists():
        return False

    destino = (
        RUTA_PARA_ANALISIS
        / (
            nombre_destino
            or origen.name
        )
    )

    shutil.copy2(
        origen,
        destino,
    )

    return True


def actualizar_carpeta_para_analisis() -> list[Path]:
    """
    Regenera la carpeta para análisis con los archivos actuales.

    La carpeta se limpia por completo antes de copiar, por lo que nunca
    quedan versiones antiguas mezcladas con las nuevas.
    """

    limpiar_carpeta_para_analisis()

    archivos_copiados: list[Path] = []

    for nombre in ARCHIVOS_GENERALES:
        origen = RUTA_MODELOS / nombre

        if copiar_archivo(
            origen=origen
        ):
            archivos_copiados.append(
                RUTA_PARA_ANALISIS
                / nombre
            )

    for simbolo in SIMBOLOS:
        ultimo_detalle = obtener_ultimo_detalle(
            simbolo=simbolo
        )

        if ultimo_detalle is not None:
            nombre_destino = (
                f"ultimo_entrenamiento_{simbolo}.json"
            )

            copiar_archivo(
                origen=ultimo_detalle,
                nombre_destino=nombre_destino,
            )

            archivos_copiados.append(
                RUTA_PARA_ANALISIS
                / nombre_destino
            )

        for plantilla in ARCHIVOS_POR_SIMBOLO:
            nombre = plantilla.format(
                simbolo=simbolo
            )

            origen = RUTA_MODELOS / nombre

            if copiar_archivo(
                origen=origen
            ):
                archivos_copiados.append(
                    RUTA_PARA_ANALISIS
                    / nombre
                )

    return archivos_copiados


def mostrar_resultado(
    archivos: list[Path],
) -> None:
    """Muestra los archivos que quedaron listos para enviar."""

    print(
        "\nCARPETA PARA ANÁLISIS ACTUALIZADA"
    )
    print("=" * 70)
    print(f"Carpeta: {RUTA_PARA_ANALISIS}")

    if not archivos:
        print(
            "No había resultados disponibles para copiar."
        )
        return

    for ruta in archivos:
        print(f"- {ruta.name}")


def obtener_argumentos() -> argparse.Namespace:
    """Obtiene los argumentos del comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Regenera la carpeta con los archivos "
            "necesarios para analizar los modelos."
        )
    )

    return parser.parse_args()


def main() -> None:
    """Punto de entrada."""

    obtener_argumentos()

    try:
        archivos = actualizar_carpeta_para_analisis()
        mostrar_resultado(
            archivos=archivos
        )
    except OSError as error:
        print(
            "\nNo se pudo actualizar la carpeta para análisis."
        )
        print(f"Detalle: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
