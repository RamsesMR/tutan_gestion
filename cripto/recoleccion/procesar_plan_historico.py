from __future__ import annotations

from cripto.recoleccion.procesar_rango_historico import procesar_rango


PLAN_HISTORICO = [
    {
        "simbolo": "BTCUSDT",
        "intervalo": "1m",
        "desde": (2026, 4),
        "hasta": (2026, 5),
    },
    {
        "simbolo": "ETHUSDT",
        "intervalo": "1m",
        "desde": (2026, 4),
        "hasta": (2026, 5),
    },
    {
        "simbolo": "BTCUSDT",
        "intervalo": "1h",
        "desde": (2026, 1),
        "hasta": (2026, 5),
    },
    {
        "simbolo": "ETHUSDT",
        "intervalo": "1h",
        "desde": (2026, 1),
        "hasta": (2026, 5),
    },
    {
        "simbolo": "BTCUSDT",
        "intervalo": "1d",
        "desde": (2025, 1),
        "hasta": (2026, 5),
    },
    {
        "simbolo": "ETHUSDT",
        "intervalo": "1d",
        "desde": (2025, 1),
        "hasta": (2026, 5),
    },
]


def procesar_plan_historico() -> None:
    """
    Ejecuta el plan inicial de descarga, importación y validación
    para BTCUSDT y ETHUSDT.
    """

    total_procesos = len(PLAN_HISTORICO)

    print("\n" + "=" * 60)
    print("PLAN HISTÓRICO DE CRIPTOMONEDAS")
    print("=" * 60)
    print(f"Procesos configurados: {total_procesos}")

    for posicion, configuracion in enumerate(
        PLAN_HISTORICO,
        start=1,
    ):
        simbolo = configuracion["simbolo"]
        intervalo = configuracion["intervalo"]
        desde = configuracion["desde"]
        hasta = configuracion["hasta"]

        print("\n" + "#" * 60)
        print(f"PROCESO {posicion} DE {total_procesos}")
        print("#" * 60)
        print(f"Símbolo: {simbolo}")
        print(f"Intervalo: {intervalo}")
        print(f"Desde: {desde[0]}-{desde[1]:02d}")
        print(f"Hasta: {hasta[0]}-{hasta[1]:02d}")

        procesar_rango(
            simbolo=simbolo,
            intervalo=intervalo,
            desde=desde,
            hasta=hasta,
        )

    print("\n" + "=" * 60)
    print("PLAN HISTÓRICO COMPLETADO CORRECTAMENTE")
    print("=" * 60)
    print(f"Procesos completados: {total_procesos}")


def main() -> None:
    try:
        procesar_plan_historico()

    except KeyboardInterrupt:
        print("\nEl procesamiento fue interrumpido por el usuario.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()