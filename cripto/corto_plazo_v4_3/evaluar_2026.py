from __future__ import annotations

import argparse

from cripto.corto_plazo_v4_3.configuracion import (
    PERIODO_2026_CONOCIDO,
    RUTA_RESULTADO_2026,
)
from cripto.corto_plazo_v4_3.periodos_conocidos import (
    evaluar_periodo_conocido,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas", type=int, default=3)
    parser.add_argument("--tamano-lote", type=int, default=100000)
    argumentos = parser.parse_args()

    evaluar_periodo_conocido(
        nombre_periodo="2026-01-01_a_2026-06-01",
        configuracion_periodo=PERIODO_2026_CONOCIDO,
        ruta_salida=RUTA_RESULTADO_2026,
        epocas=argumentos.epocas,
        tamano_lote=argumentos.tamano_lote,
    )


if __name__ == "__main__":
    main()
