"""Puntaje y selección del Top 10 del E2 a partir de la lista larga.

Uso (desde la raíz del repo):
    python literatura/e2/puntaje_e2.py

Entrada: literatura/e2/lista_larga_E2.csv (un modelo por fila; citas, AP en COCO, implementación y resultado de la
prueba de humo de 6 GB). Salida: literatura/e2/puntaje_E2.csv y la selección impresa en pantalla.

Reglas, fijadas antes de calcular (2026-11-15):
- Puntaje = 0.25 C + 0.15 R + 0.25 D + 0.20 I + 0.15 F
  C citación: citas por año (citas / (2026 - año del registro citado + 1)) en percentil dentro de la lista larga
    (proporción de valores estrictamente menores / (n - 1)).
  R recencia: (año de lanzamiento - mínimo) / (máximo - mínimo).
  D desempeño: mask AP en COCO de la configuración de referencia, min-max.
  I implementación: 1 si está en MMDetection 3.3 o ultralytics, 0.5 si solo hay repo oficial.
  F representatividad: 1 si es el más citado por año de su familia, 0.5 si no.
- Filtros: F1 máscara por instancia sin prompts y con supervisión de máscaras, F2 arquitectura definida en el paper,
  F3 código y pesos COCO públicos (columna exclusion_F1_F3 vacía), F4 entrena en 6 GB a 1280x720 con AMP y lote >= 1.
- Los modelos base del E1 no compiten (PDF E2 §3.2: van además del Top 10).
- Selección: los 10 de mayor puntaje que pasan los filtros; si cubren menos de 4 familias, se sustituye el último de la
  familia más representada por el mejor de una familia que falta, hasta tener 4.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
AHORA = 2026


def puntuar(D: pd.DataFrame) -> pd.DataFrame:
    D = D.copy()
    D["citas_por_anio"] = D.citas / (AHORA - D.anio_registro_cita + 1)
    v = D.citas_por_anio.dropna().values
    D["C"] = D.citas_por_anio.map(lambda x: np.nan if np.isnan(x) else (v < x).sum() / (len(v) - 1))
    D["R"] = (D.anio_lanzamiento - D.anio_lanzamiento.min()) / (D.anio_lanzamiento.max() - D.anio_lanzamiento.min())
    D["D"] = (D.mask_AP_COCO - D.mask_AP_COCO.min()) / (D.mask_AP_COCO.max() - D.mask_AP_COCO.min())
    D["I"] = D.implementacion
    D["F"] = D.groupby("familia").citas_por_anio.transform(lambda s: (s == s.max()).map({True: 1.0, False: 0.5}))
    D.loc[D.citas_por_anio.isna(), "F"] = np.nan
    D["PUNTAJE"] = 0.25 * D.C + 0.15 * D.R + 0.25 * D.D + 0.20 * D.I + 0.15 * D.F
    return D


def seleccionar(D: pd.DataFrame, n: int = 10, familias_min: int = 4) -> tuple[pd.DataFrame, list[str]]:
    elegibles = D[(D.modelo_base_E1 == 0) & (D.exclusion_F1_F3 == "") & ~D.F4_6GB_prueba_humo.str.startswith("no")
                  & D.PUNTAJE.notna()].sort_values("PUNTAJE", ascending=False)
    top = elegibles.head(n).copy()
    notas = []
    while top.familia.nunique() < familias_min:
        mas = top.familia.value_counts().idxmax()
        sale = top[top.familia == mas].sort_values("PUNTAJE").iloc[0]
        faltan = elegibles[~elegibles.familia.isin(top.familia) & ~elegibles.modelo.isin(top.modelo)]
        if faltan.empty:
            break
        entra = faltan.iloc[0]
        notas.append(f"Regla de familias: sale {sale.modelo} ({mas}, {sale.PUNTAJE:.3f}) y entra {entra.modelo} "
                     f"({entra.familia}, {entra.PUNTAJE:.3f}).")
        top = pd.concat([top[top.modelo != sale.modelo], entra.to_frame().T]).sort_values("PUNTAJE", ascending=False)
    sin = top[~top.F4_6GB_prueba_humo.str.startswith("sí")]
    if len(sin):
        notas.append("Falta la prueba de 6 GB de: " + ", ".join(sin.modelo))
    return top, notas


def main():
    L = pd.read_csv(AQUI / "lista_larga_E2.csv").fillna({"exclusion_F1_F3": "", "fuente_citas": ""})
    D = puntuar(L)
    cols = ["modelo", "familia", "anio_lanzamiento", "citas_por_anio", "C", "R", "D", "I", "F", "PUNTAJE",
            "F4_6GB_prueba_humo", "AP50_mascara_4img_prueba_humo"]
    D.sort_values("PUNTAJE", ascending=False)[["modelo_base_E1", "exclusion_F1_F3"] + cols].round(3).to_csv(
        AQUI / "puntaje_E2.csv", index=False, encoding="utf-8-sig")
    top, notas = seleccionar(D)
    pd.set_option("display.width", 220)
    print("Top 10:")
    print(top[cols].round(3).to_string(index=False))
    for x in notas:
        print("-", x)
    resto = D[(D.modelo_base_E1 == 0) & ~D.modelo.isin(top.modelo) & D.F4_6GB_prueba_humo.str.startswith("sí")]
    print("Reservas (pasan F4, por puntaje):", ", ".join(resto.sort_values("PUNTAJE", ascending=False).modelo))


if __name__ == "__main__":
    main()
