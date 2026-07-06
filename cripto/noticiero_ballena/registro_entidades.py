from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .configuracion import TIPOS_ENTIDAD
from .utilidades import leer_json, texto_seguro


@dataclass(frozen=True)
class Atribucion:
    direccion: str
    entidad: str
    tipo_entidad: str
    fuente_verificacion: str
    confianza: float
    identificador_entidad: str | None = None


class RegistroEntidades:
    """Registro exacto de direcciones verificadas.

    No infiere identidades. Solo aplica coincidencias exactas de direcciones y
    conserva la fuente de verificación declarada.
    """

    def __init__(self, atribuciones: dict[str, Atribucion] | None = None):
        self._atribuciones = atribuciones or {}

    @classmethod
    def desde_json(cls, ruta: Path | None) -> "RegistroEntidades":
        if ruta is None or not ruta.exists():
            return cls()
        datos = leer_json(ruta)
        filas = datos.get("entidades", datos) if isinstance(datos, dict) else datos
        atribuciones: dict[str, Atribucion] = {}
        for fila in filas:
            direccion = texto_seguro(fila.get("direccion")).lower()
            if not direccion:
                continue
            tipo = texto_seguro(fila.get("tipo_entidad")).upper() or "DESCONOCIDO"
            if tipo not in TIPOS_ENTIDAD:
                raise ValueError(f"Tipo de entidad no soportado: {tipo}")
            confianza = float(fila.get("confianza", 0.0))
            if not 0.0 <= confianza <= 1.0:
                raise ValueError(f"Confianza fuera de rango para {direccion}")
            atribuciones[direccion] = Atribucion(
                direccion=direccion,
                entidad=texto_seguro(fila.get("entidad")),
                tipo_entidad=tipo,
                fuente_verificacion=texto_seguro(fila.get("fuente_verificacion")),
                confianza=confianza,
                identificador_entidad=texto_seguro(fila.get("identificador_entidad")) or None,
            )
        return cls(atribuciones)

    def buscar(self, direccion: Any) -> Atribucion | None:
        clave = texto_seguro(direccion).lower()
        return self._atribuciones.get(clave)

    def enriquecer(self, df: pd.DataFrame) -> pd.DataFrame:
        salida = df.copy()
        for lado in ("origen", "destino"):
            columna_direccion = f"direccion_{lado}"
            if columna_direccion not in salida:
                continue
            atribuciones = salida[columna_direccion].map(self.buscar)
            for columna, extractor in {
                f"entidad_{lado}": lambda x: x.entidad,
                f"tipo_{lado}": lambda x: x.tipo_entidad,
                f"identificador_entidad_{lado}": lambda x: x.identificador_entidad,
            }.items():
                valores = atribuciones.map(lambda x: extractor(x) if x else None)
                if columna in salida:
                    salida[columna] = valores.combine_first(salida[columna])
                else:
                    salida[columna] = valores
            confianza = atribuciones.map(lambda x: x.confianza if x else None)
            confianza_base = pd.Series(
                salida["confianza_etiquetado"] if "confianza_etiquetado" in salida else 0.0,
                index=salida.index,
            )
            confianza_base = pd.to_numeric(confianza_base, errors="coerce")
            salida["confianza_etiquetado"] = pd.concat(
                [confianza_base, confianza], axis=1
            ).max(axis=1).fillna(0.0)
            base_verificada = pd.Series(
                salida["identidad_verificada"] if "identidad_verificada" in salida else False,
                index=salida.index,
            ).fillna(False).astype(bool)
            salida["identidad_verificada"] = base_verificada | atribuciones.notna()
        return salida
