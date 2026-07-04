from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cripto.corto_plazo_v2.configuracion import (
    CLASES,
    NOMBRE_HORIZONTE,
    RUTA_MODELOS_V2,
    SIMBOLOS,
)


RUTA_CONTROL = (
    RUTA_MODELOS_V2
    / "control_v1_mismas_filas"
)

RUTA_COMPARACION = (
    RUTA_MODELOS_V2
    / "comparacion_control_vs_v2a"
)


def leer_json(
    ruta: Path,
) -> dict[str, Any]:
    """Lee un archivo JSON."""

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe: {ruta}"
        )

    with ruta.open(
        "r",
        encoding="utf-8",
    ) as archivo:
        return json.load(
            archivo
        )


def cargar_metricas_globales(
    simbolo: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Carga las métricas globales del control y de la V2A."""

    ruta_control = (
        RUTA_CONTROL
        / f"metricas_control_{simbolo}_{NOMBRE_HORIZONTE}.csv"
    )

    ruta_v2a = (
        RUTA_MODELOS_V2
        / f"metricas_base_{simbolo}_{NOMBRE_HORIZONTE}.csv"
    )

    if not ruta_control.exists():
        raise FileNotFoundError(
            f"No existe el control de {simbolo}: {ruta_control}"
        )

    if not ruta_v2a.exists():
        raise FileNotFoundError(
            f"No existe el resultado V2A de {simbolo}: {ruta_v2a}"
        )

    control = pd.read_csv(
        ruta_control
    ).iloc[0].to_dict()

    v2a = pd.read_csv(
        ruta_v2a
    ).iloc[0].to_dict()

    return control, v2a


def cargar_informes(
    simbolo: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Carga los informes por clase."""

    control = leer_json(
        RUTA_CONTROL
        / (
            f"informe_clasificacion_control_"
            f"{simbolo}_{NOMBRE_HORIZONTE}.json"
        )
    )

    v2a = leer_json(
        RUTA_MODELOS_V2
        / (
            f"informe_clasificacion_base_"
            f"{simbolo}_{NOMBRE_HORIZONTE}.json"
        )
    )

    return control, v2a


def crear_veredicto(
    delta_f1: float,
    delta_balanced: float,
    delta_accuracy: float,
) -> str:
    """Crea un veredicto prudente para la comparación."""

    if (
        delta_f1 > 0.01
        and delta_balanced > 0
    ):
        return "VARIABLES_CRUZADAS_APORTAN"

    if (
        delta_f1 > 0
        and delta_balanced >= -0.002
    ):
        return "MEJORA_PEQUENA"

    if (
        delta_f1 <= 0
        and delta_balanced <= 0
    ):
        return "NO_MEJORA"

    return "RESULTADO_MIXTO"


def main() -> None:
    """Compara el control de 36 variables contra la V2A de 63."""

    RUTA_COMPARACION.mkdir(
        parents=True,
        exist_ok=True,
    )

    filas_globales: list[
        dict[str, Any]
    ] = []

    filas_clases: list[
        dict[str, Any]
    ] = []

    lineas_resumen = [
        "# Comparación control V1 vs V2A",
        "",
        (
            "El control y la V2A utilizan exactamente las mismas filas, "
            "periodos, clases, épocas, lotes y algoritmo."
        ),
        "",
    ]

    for simbolo in SIMBOLOS:
        control_global, v2a_global = cargar_metricas_globales(
            simbolo
        )

        control_informe, v2a_informe = cargar_informes(
            simbolo
        )

        delta_accuracy = (
            float(v2a_global["accuracy"])
            - float(control_global["accuracy"])
        )

        delta_balanced = (
            float(v2a_global["balanced_accuracy"])
            - float(control_global["balanced_accuracy"])
        )

        delta_f1 = (
            float(v2a_global["f1_macro"])
            - float(control_global["f1_macro"])
        )

        veredicto = crear_veredicto(
            delta_f1=delta_f1,
            delta_balanced=delta_balanced,
            delta_accuracy=delta_accuracy,
        )

        filas_globales.append(
            {
                "simbolo": simbolo,
                "accuracy_control": float(
                    control_global["accuracy"]
                ),
                "accuracy_v2a": float(
                    v2a_global["accuracy"]
                ),
                "delta_accuracy": delta_accuracy,
                "balanced_accuracy_control": float(
                    control_global["balanced_accuracy"]
                ),
                "balanced_accuracy_v2a": float(
                    v2a_global["balanced_accuracy"]
                ),
                "delta_balanced_accuracy": delta_balanced,
                "f1_macro_control": float(
                    control_global["f1_macro"]
                ),
                "f1_macro_v2a": float(
                    v2a_global["f1_macro"]
                ),
                "delta_f1_macro": delta_f1,
                "veredicto": veredicto,
            }
        )

        lineas_resumen.extend(
            [
                f"## {simbolo}",
                "",
                "| Métrica | Control 36 | V2A 63 | Cambio |",
                "|---|---:|---:|---:|",
                (
                    f"| Accuracy | {float(control_global['accuracy']):.4f} | "
                    f"{float(v2a_global['accuracy']):.4f} | "
                    f"{delta_accuracy:+.4f} |"
                ),
                (
                    "| Balanced Accuracy | "
                    f"{float(control_global['balanced_accuracy']):.4f} | "
                    f"{float(v2a_global['balanced_accuracy']):.4f} | "
                    f"{delta_balanced:+.4f} |"
                ),
                (
                    f"| F1 macro | {float(control_global['f1_macro']):.4f} | "
                    f"{float(v2a_global['f1_macro']):.4f} | "
                    f"{delta_f1:+.4f} |"
                ),
                "",
                f"**Veredicto automático:** `{veredicto}`",
                "",
                "| Clase | Métrica | Control | V2A | Cambio |",
                "|---|---|---:|---:|---:|",
            ]
        )

        for clase in CLASES:
            for clave_json, nombre_metrica in (
                ("precision", "precision"),
                ("recall", "recall"),
                ("f1-score", "f1_score"),
            ):
                valor_control = float(
                    control_informe[
                        clase
                    ][
                        clave_json
                    ]
                )

                valor_v2a = float(
                    v2a_informe[
                        clase
                    ][
                        clave_json
                    ]
                )

                delta = (
                    valor_v2a
                    - valor_control
                )

                filas_clases.append(
                    {
                        "simbolo": simbolo,
                        "clase": clase,
                        "metrica": nombre_metrica,
                        "control": valor_control,
                        "v2a": valor_v2a,
                        "delta": delta,
                    }
                )

                lineas_resumen.append(
                    f"| {clase} | {nombre_metrica} | "
                    f"{valor_control:.4f} | "
                    f"{valor_v2a:.4f} | "
                    f"{delta:+.4f} |"
                )

        lineas_resumen.append(
            ""
        )

    ruta_global = (
        RUTA_COMPARACION
        / "comparacion_global_control_vs_v2a.csv"
    )

    ruta_clases = (
        RUTA_COMPARACION
        / "comparacion_por_clase_control_vs_v2a.csv"
    )

    ruta_resumen = (
        RUTA_COMPARACION
        / "resumen_control_vs_v2a.md"
    )

    pd.DataFrame(
        filas_globales
    ).to_csv(
        ruta_global,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        filas_clases
    ).to_csv(
        ruta_clases,
        index=False,
        encoding="utf-8-sig",
    )

    ruta_resumen.write_text(
        "\n".join(
            lineas_resumen
        ),
        encoding="utf-8",
    )

    print(
        "\nCOMPARACIÓN CONTROL VS V2A COMPLETADA"
    )
    print("=" * 70)

    for fila in filas_globales:
        print(
            f"\n{fila['simbolo']}"
        )
        print(
            f"Delta Accuracy: "
            f"{fila['delta_accuracy']:+.4f}"
        )
        print(
            "Delta Balanced Accuracy: "
            f"{fila['delta_balanced_accuracy']:+.4f}"
        )
        print(
            f"Delta F1 macro: "
            f"{fila['delta_f1_macro']:+.4f}"
        )
        print(
            f"Veredicto: {fila['veredicto']}"
        )

    print(
        "\nArchivos:"
    )
    print(
        f"- {ruta_global}"
    )
    print(
        f"- {ruta_clases}"
    )
    print(
        f"- {ruta_resumen}"
    )


if __name__ == "__main__":
    main()
