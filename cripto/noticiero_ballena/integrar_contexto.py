from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .configuracion import ARCHIVO_VARIABLES_MINUTO
from .utilidades import fecha_utc, guardar_tabla, leer_tabla, primera_columna


def _prefijo(ruta: Path) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", ruta.stem.lower()).strip("_")
    return f"nb_ctx_{base}_"


def integrar(
    variables: pd.DataFrame,
    contextos: list[tuple[Path, int]],
) -> pd.DataFrame:
    salida = variables.copy()
    salida["fecha_disponible"] = fecha_utc(salida["fecha_disponible"])
    salida = salida.sort_values("fecha_disponible")

    for ruta, retraso_minutos in contextos:
        contexto = leer_tabla(ruta).copy()
        fecha = primera_columna(contexto.columns, ("fecha_disponible", "fecha", "date", "timestamp", "t"))
        if fecha is None:
            raise ValueError(f"{ruta} no contiene columna temporal")
        if fecha == "t" and pd.api.types.is_numeric_dtype(contexto[fecha]):
            contexto["fecha_disponible"] = pd.to_datetime(contexto[fecha], unit="s", utc=True, errors="coerce")
        else:
            contexto["fecha_disponible"] = fecha_utc(contexto[fecha], permitir_nulos=True)
        contexto["fecha_disponible"] = contexto["fecha_disponible"] + pd.Timedelta(minutes=retraso_minutos)
        prefijo = _prefijo(ruta)
        columnas_numericas = [
            c for c in contexto.columns
            if c != "fecha_disponible" and pd.api.types.is_numeric_dtype(contexto[c])
        ]
        renombradas = {c: f"{prefijo}{c}" for c in columnas_numericas}
        contexto = (
            contexto[["fecha_disponible", *columnas_numericas]]
            .rename(columns=renombradas)
            .dropna(subset=["fecha_disponible"])
            .sort_values("fecha_disponible")
            .drop_duplicates("fecha_disponible", keep="last")
        )
        salida = pd.merge_asof(
            salida,
            contexto,
            on="fecha_disponible",
            direction="backward",
            allow_exact_matches=True,
        )
    return salida


def main() -> None:
    parser = argparse.ArgumentParser(description="Integra contexto agregado con alineación causal.")
    parser.add_argument("--variables", type=Path, default=ARCHIVO_VARIABLES_MINUTO)
    parser.add_argument("--contexto", action="append", required=True, type=Path)
    parser.add_argument(
        "--retraso-minutos",
        action="append",
        type=int,
        help="Uno por contexto. Si falta, se usa 0; configura el retraso real del proveedor.",
    )
    parser.add_argument("--salida", type=Path, default=ARCHIVO_VARIABLES_MINUTO)
    args = parser.parse_args()
    retrasos = args.retraso_minutos or []
    pares = [(ruta, retrasos[i] if i < len(retrasos) else 0) for i, ruta in enumerate(args.contexto)]
    salida = integrar(leer_tabla(args.variables), pares)
    guardar_tabla(salida, args.salida)
    print(f"Contextos integrados: {len(pares)}")
    print(f"Salida: {args.salida}")


if __name__ == "__main__":
    main()
