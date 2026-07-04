from __future__ import annotations

from cripto.corto_plazo.ajustar_decision import (
    main as ejecutar_ajuste_decision,
)
from cripto.corto_plazo.gestionar_para_analisis import (
    actualizar_carpeta_para_analisis,
    mostrar_resultado,
)


def main() -> None:
    """Ejecuta el ajuste y actualiza la carpeta para análisis."""

    ejecutar_ajuste_decision()

    archivos = actualizar_carpeta_para_analisis()

    mostrar_resultado(
        archivos=archivos
    )


if __name__ == "__main__":
    main()
