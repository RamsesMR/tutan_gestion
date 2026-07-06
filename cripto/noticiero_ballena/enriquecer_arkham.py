from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .configuracion import ARCHIVO_EVENTOS_NORMALIZADOS, RUTA_CRUDOS
from .proveedores.arkham import ClienteArkham
from .utilidades import guardar_tabla, leer_tabla, texto_seguro


def _buscar_entidad(respuesta: dict[str, Any]) -> dict[str, Any] | None:
    candidatos: list[dict[str, Any]] = []
    if isinstance(respuesta, dict):
        candidatos.append(respuesta)
        candidatos.extend(v for v in respuesta.values() if isinstance(v, dict))
    for candidato in candidatos:
        entidad = candidato.get("arkhamEntity") or candidato.get("entity")
        if not isinstance(entidad, dict):
            continue
        nombre = texto_seguro(entidad.get("name") or entidad.get("label") or entidad.get("id"))
        if not nombre:
            continue
        return {
            "entidad": nombre,
            "identificador": texto_seguro(entidad.get("id")) or None,
            "tipo": texto_seguro(entidad.get("type") or candidato.get("entityType")).upper() or "WALLET",
            "confianza": float(candidato.get("confidence") or entidad.get("confidence") or 0.85),
        }
    return None


def enriquecer(df: pd.DataFrame, cliente: ClienteArkham, cache_ruta: Path, limite: int | None = None) -> pd.DataFrame:
    salida = df.copy()
    cache: dict[str, dict[str, Any] | None] = {}
    if cache_ruta.exists():
        with cache_ruta.open("r", encoding="utf-8") as archivo:
            for linea in archivo:
                if not linea.strip():
                    continue
                fila = json.loads(linea)
                cache[str(fila["direccion"]).lower()] = fila.get("atribucion")

    direcciones = pd.concat([
        salida.get("direccion_origen", pd.Series(dtype="object")),
        salida.get("direccion_destino", pd.Series(dtype="object")),
    ]).dropna().map(texto_seguro)
    direcciones = [d for d in dict.fromkeys(direcciones) if d]
    if limite is not None:
        direcciones = direcciones[:limite]

    cache_ruta.parent.mkdir(parents=True, exist_ok=True)
    for direccion in direcciones:
        clave = direccion.lower()
        if clave in cache:
            continue
        respuesta = cliente.resolver_direccion(direccion)
        atribucion = _buscar_entidad(respuesta)
        cache[clave] = atribucion
        with cache_ruta.open("a", encoding="utf-8") as archivo:
            archivo.write(json.dumps({
                "direccion": direccion,
                "atribucion": atribucion,
                "respuesta": respuesta,
            }, ensure_ascii=False, default=str) + "\n")

    for lado in ("origen", "destino"):
        columna_direccion = f"direccion_{lado}"
        if columna_direccion not in salida:
            continue
        atribuciones = salida[columna_direccion].map(
            lambda x: cache.get(texto_seguro(x).lower()) if texto_seguro(x) else None
        )
        entidad = atribuciones.map(lambda x: x.get("entidad") if isinstance(x, dict) else None)
        identificador = atribuciones.map(lambda x: x.get("identificador") if isinstance(x, dict) else None)
        tipo = atribuciones.map(lambda x: x.get("tipo") if isinstance(x, dict) else None)
        confianza = atribuciones.map(lambda x: x.get("confianza") if isinstance(x, dict) else None)
        salida[f"entidad_{lado}"] = entidad.combine_first(salida.get(f"entidad_{lado}"))
        salida[f"identificador_entidad_{lado}"] = identificador.combine_first(
            salida.get(f"identificador_entidad_{lado}")
        )
        salida[f"tipo_{lado}"] = tipo.combine_first(salida.get(f"tipo_{lado}"))
        base_confianza = pd.to_numeric(salida.get("confianza_etiquetado", 0.0), errors="coerce")
        salida["confianza_etiquetado"] = pd.concat([base_confianza, confianza], axis=1).max(axis=1).fillna(0.0)
        salida["identidad_verificada"] = (
            pd.Series(salida.get("identidad_verificada", False), index=salida.index).fillna(False).astype(bool)
            | atribuciones.notna()
        )
    return salida


def main() -> None:
    parser = argparse.ArgumentParser(description="Enriquece direcciones con Arkham sin inventar identidades.")
    parser.add_argument("--entrada", type=Path, default=ARCHIVO_EVENTOS_NORMALIZADOS)
    parser.add_argument("--salida", type=Path, default=ARCHIVO_EVENTOS_NORMALIZADOS)
    parser.add_argument("--cache", type=Path, default=RUTA_CRUDOS / "arkham_cache.jsonl")
    parser.add_argument("--limite", type=int)
    args = parser.parse_args()
    salida = enriquecer(leer_tabla(args.entrada), ClienteArkham(), args.cache, args.limite)
    guardar_tabla(salida, args.salida)
    print(f"Eventos enriquecidos: {len(salida):,}")
    print(f"Salida: {args.salida}")


if __name__ == "__main__":
    main()
