"""Diagrama PRISMA 2020 del E2, dibujado con los conteos que escribe cribado_e2.py.

Uso (desde la raíz del repo, después de cribado_e2.py):
    python literatura/e2/prisma_e2.py

Entrada: literatura/e2/prisma_E2.json y literatura/e2/decisiones_lectura_E2.csv (desglose de la lectura R5).
Salida: literatura/e2/prisma_E2.png. Requiere matplotlib y pandas.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

AQUI = Path(__file__).resolve().parent
AZUL, ROJO, VERDE = "#dbe9f6", "#f6e0db", "#e3f1dc"


def main():
    P = json.loads((AQUI / "prisma_E2.json").read_text(encoding="utf8"))
    D = pd.read_csv(AQUI / "decisiones_lectura_E2.csv").fillna("")
    A = D[D.capa == "A"]
    ia_excl = int((A.decision_ia == "excluir").sum())
    dud = A[A.decision_ia == "dudoso"]
    no_recup = int(dud.evidencia_texto_completo.str.contains("no recuperado").sum())
    dud_excl = int((dud.decision_final == "excluir").sum()) - no_recup
    dud_incl = int((dud.decision_final == "incluir").sum())
    pa, pb, r = P["capa_A"], P["capa_B"], P["capa_A"]["por_regla"]
    assert pa["incluidos"] == int((A.decision_ia == "incluir").sum()) + dud_incl

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def caja(x, y, t, c):
        ax.text(x, y, t, ha="center", va="center", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.5", fc=c, ec="#555"))

    def flecha(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#333"))

    NOMBRE = {"scopus": "Scopus", "wos": "WoS", "ieee": "IEEE Xplore"}
    bases = ", ".join(f"{NOMBRE.get(k, k)} {v}" for k, v in P["por_base"].items())
    caja(0.30, 0.94, f"Registros identificados: {P['identificados']}\n{bases}\n(cadenas A1–A3 y B1–B4, 2026-11-15)", AZUL)
    caja(0.72, 0.94, f"Duplicados eliminados: {P['duplicados']}\n(DOI; sin DOI, título normalizado)", ROJO)
    caja(0.30, 0.82, f"Registros únicos: {P['unicos']}", AZUL)
    caja(0.72, 0.82, f"Excluidos por tipo de documento: {P['excluidos_tipo_documento']}\n(portadas de actas, erratas)", ROJO)
    caja(0.30, 0.71, f"Cribados por título y resumen: {P['cribados']}", AZUL)
    flecha(0.30, 0.91, 0.30, 0.845); flecha(0.30, 0.795, 0.30, 0.735)
    flecha(0.46, 0.94, 0.56, 0.94); flecha(0.40, 0.82, 0.53, 0.82)

    # Capa A (problema)
    caja(0.18, 0.59, f"Capa A (problema): {pa['cribados']}", AZUL)
    caja(0.18, 0.44, f"Leídos (R5, título y resumen): {r['R5']}", AZUL)
    caja(0.18, 0.29, f"Texto completo buscado (dudosos): {len(dud)}", AZUL)
    caja(0.18, 0.16, f"Texto completo evaluado: {len(dud) - no_recup}", AZUL)
    caja(0.18, 0.03, f"Incluidos capa A: {pa['incluidos']}\n({int((A.decision_ia == 'incluir').sum())} por resumen + {dud_incl} por texto completo)", VERDE)
    caja(0.47, 0.515, f"Excluidos por reglas: {r['R1'] + r['R2'] + r['R3'] + r['R4']}\nR1 3D / nube de puntos {r['R1']}\nR2 no menciona instancias {r['R2']}\n"
                      f"R3 sin métricas {r['R3']}\nR4 < 2 arquitecturas de instancias {r['R4']}", ROJO)
    caja(0.47, 0.365, f"Excluidos por lectura del resumen: {ia_excl}", ROJO)
    caja(0.47, 0.225, f"No recuperados: {no_recup}", ROJO)
    caja(0.47, 0.095, f"Excluidos por texto completo: {dud_excl}\n(una sola arquitectura, tarea semántica,\nsupervisión con cajas)", ROJO)
    for y0, y1 in [(0.565, 0.465), (0.415, 0.315), (0.265, 0.185), (0.135, 0.065)]:
        flecha(0.18, y0, 0.18, y1)
    for y in [0.515, 0.365, 0.225, 0.095]:
        flecha(0.18, y + 0.02, 0.36, y)

    # Capa B (mecanismos)
    caja(0.82, 0.59, f"Capa B (mecanismos): {pb['cribados']}", AZUL)
    caja(0.82, 0.44, f"Excluidos: {pb['cribados'] - pb['incluidos']}\n(no describen un mecanismo buscado,\n3D, audio, ECG, ataques adversarios)", ROJO)
    mec = "\n".join(f"{k} {v}" for k, v in pb["por_mecanismo"].items())
    caja(0.82, 0.20, f"Incluidos capa B: {pb['incluidos']}\n(tabla de mecanismos)\n{mec}", VERDE)
    flecha(0.82, 0.565, 0.82, 0.49); flecha(0.82, 0.39, 0.82, 0.30)
    flecha(0.30, 0.685, 0.18, 0.615); flecha(0.30, 0.685, 0.82, 0.615)

    for y, t in [(0.88, "Identificación"), (0.58, "Cribado"), (0.22, "Elegibilidad"), (0.03, "Inclusión")]:
        ax.text(0.0, y, t, rotation=90, va="center", fontsize=10, weight="bold", color="#335")
    ax.set_title("Diagrama PRISMA 2020 del E2: segmentación de instancias (capa A) y mecanismos transferibles (capa B)", fontsize=11)
    plt.tight_layout()
    plt.savefig(AQUI / "prisma_E2.png", dpi=200, bbox_inches="tight")
    print("Diagrama:", AQUI / "prisma_E2.png")


if __name__ == "__main__":
    main()
