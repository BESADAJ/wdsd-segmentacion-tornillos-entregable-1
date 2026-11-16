"""Búsqueda y cribado del E2: de las exportaciones de Scopus, WoS e IEEE a la tabla de registros únicos con decisiones.

Uso (desde la raíz del repo):
    python literatura/e2/cribado_e2.py

Entrada:
- literatura/e2/exportaciones/: los 11 archivos tal como salieron de las bases, con nombre "<cadena> - <base>.<ext>"
  (p. ej. "A1 - scopus.csv", "A1 - wos.xls", "A1 - ieee.csv"). No se versionan (términos de uso de Scopus/WoS);
  las cadenas exactas y la fecha están en cadenas_busqueda_E2.md y registro_busquedas_E2.md.
- literatura/e2/decisiones_lectura_E2.csv: las decisiones que no salen de una regla automática
  (lectura de título y resumen asistida por IA, revisión humana de los dudosos y asignación de mecanismos de la capa B).

Salida:
- literatura/e2/registros_unicos_E2.csv: un registro único por fila, con marcas automáticas, regla aplicada y decisión final.
- literatura/e2/prisma_E2.json: los conteos del diagrama PRISMA.

Reglas (en este orden; la primera que se cumple decide):
  R0 tipo de documento que no es un estudio (Conference review, Erratum)
  R1 3D / nube de puntos
  R2 el título o resumen no menciona segmentación de instancias
  R3 el resumen no reporta métricas
  R4 no nombra al menos 2 arquitecturas de segmentación de instancias (salvo cadena A3: surveys y benchmarks)
  R5 lectura del título y resumen (decisiones_lectura_E2.csv)
Requiere pandas, xlrd (para los .xls de WoS) y openpyxl.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
EXPORT = AQUI / "exportaciones"

# Columnas de cada base -> nombre común
MAPAS = {
    "wos": dict(title="Article Title", year="Publication Year", doi="DOI", abstract="Abstract", src="Source Title",
                cites="Times Cited, All Databases", dtype="Document Type", authors="Authors"),
    "ieee": dict(title="Document Title", year="Publication Year", doi="DOI", abstract="Abstract", src="Publication Title",
                 cites="Article Citation Count", dtype="Publisher", authors="Authors"),
    "scopus": dict(title="Title", year="Year", doi="DOI", abstract="Abstract", src="Source title", cites="Cited by",
                   dtype="Document Type", authors="Authors"),
}

# Arquitecturas buscadas en título + resumen (minúsculas)
ARQ = dict(
    mask_rcnn=r"mask[\s\-]?r[\s\-]?cnn", cascade=r"cascade mask|cascade r-?cnn", htc=r"hybrid task cascade|\bhtc\b",
    ms_rcnn=r"mask scoring", pointrend=r"pointrend", solo=r"\bsolo\b(?!v2)|\bsolov1\b", solov2=r"solo ?v2",
    condinst=r"condinst", boxinst=r"boxinst", blendmask=r"blendmask", yolact=r"yolact", centermask=r"centermask",
    sparseinst=r"sparseinst", queryinst=r"queryinst", maskformer=r"maskformer(?!2)", mask2former=r"mask2former|mask 2 former",
    oneformer=r"oneformer", mask_dino=r"mask ?dino", sam=r"segment anything|\bsam\b|\bsam2\b", rtmdet=r"rtmdet",
    yolov5seg=r"yolov5", yolov7=r"yolov7", yolov8=r"yolov8|yolo ?v8", yolov9=r"yolov9", yolo11=r"yolo ?11|yolov11",
    yolo12=r"yolov12|yolo12", detr=r"\bdetr\b", swin=r"swin", convnext=r"convnext", fastinst=r"fastinst",
    rtdetr=r"rt-?detr", tensormask=r"tensormask", polarmask=r"polarmask", panet=r"\bpanet\b|path aggregation",
    unet=r"u-?net", deeplab=r"deeplab", mamba=r"mamba|state space")
# Las que cuentan para R4 (segmentación de instancias; se excluyen backbones, DETR de detección, U-Net, etc.)
ARQ_INSTANCIAS = set("mask_rcnn cascade htc ms_rcnn pointrend solo solov2 condinst boxinst blendmask yolact centermask "
                     "sparseinst queryinst maskformer mask2former oneformer mask_dino sam rtmdet yolov5seg yolov7 yolov8 "
                     "yolov9 yolo11 yolo12 fastinst tensormask polarmask".split())


def cargar() -> pd.DataFrame:
    filas = []
    archivos = sorted(f for f in EXPORT.glob("*") if f.suffix in (".csv", ".xls"))
    if not archivos:
        raise SystemExit(f"No hay exportaciones en {EXPORT}. Copia ahí los 11 archivos de la búsqueda.")
    for f in archivos:
        cadena, base = f.name.split(" - ")[0], f.name.split(" - ")[1].split(".")[0]
        df = pd.read_excel(f) if f.suffix == ".xls" else pd.read_csv(f)
        m = MAPAS["wos" if f.suffix == ".xls" else base]
        out = pd.DataFrame({k: (df[v] if v in df.columns else None) for k, v in m.items()})
        out["cadena"], out["base"] = cadena, base
        filas.append(out)
    return pd.concat(filas, ignore_index=True)


def deduplicar(R: pd.DataFrame) -> pd.DataFrame:
    """Clave = DOI normalizado; sin DOI, título normalizado (y si ese título aparece con DOI en otro registro, ese DOI)."""
    def nd(x):
        if not isinstance(x, str):
            return None
        x = re.sub(r"^https?://(dx\.)?doi\.org/", "", x.strip().lower())
        return x or None

    R["doi_n"] = R.doi.map(nd)
    R["tit_n"] = R.title.map(lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower())[:120])
    t2k = {}
    for _, r in R.iterrows():
        if isinstance(r.doi_n, str):
            t2k.setdefault(r.tit_n, r.doi_n)
    R["key"] = [r.doi_n if isinstance(r.doi_n, str) else t2k.get(r.tit_n, "T:" + r.tit_n) for _, r in R.iterrows()]
    U = R.drop_duplicates("key").copy()
    g = R.groupby("key").agg(cadenas=("cadena", lambda s: ";".join(sorted(set(s)))),
                             bases=("base", lambda s: ";".join(sorted(set(s)))))
    return U.set_index("key").join(g).reset_index()


def marcar(U: pd.DataFrame) -> pd.DataFrame:
    t = (U.title.fillna("") + " . " + U.abstract.fillna("")).str.lower()
    for k, p in ARQ.items():
        U[k] = t.str.contains(p, regex=True)
    U["excl_auto"] = U.dtype.isin(["Conference review", "Erratum"]).map({True: "tipo (no es estudio)", False: ""})
    U["menciona_instancias"] = t.str.contains(r"instance[\s\-]?(?:segmentation|aware|level)|panoptic|mask[\s\-]?r[\s\-]?cnn|yolo\w*[\s\-]?seg")
    U["reporta_metrica"] = t.str.contains(r"\bm?ap\b|map@|map50|ap50|\bf1\b|f-score|\biou\b|precision|recall|accuracy|dice|\bpq\b|%")
    U["es_3d"] = t.str.contains(r"point cloud|lidar|3d instance|voxel")
    U["arquitecturas_mencionadas"] = U[list(ARQ)].apply(lambda r: ";".join(c for c in ARQ if r[c]), axis=1)
    U["n_arq_instancias"] = U.arquitecturas_mencionadas.map(lambda s: len([x for x in s.split(";") if x in ARQ_INSTANCIAS]))
    U = U.sort_values(["cadenas", "year"], ascending=[True, False]).reset_index(drop=True)
    U.insert(0, "id", ["R%04d" % i for i in range(1, len(U) + 1)])
    return U


def decidir(U: pd.DataFrame) -> pd.DataFrame:
    L = pd.read_csv(AQUI / "decisiones_lectura_E2.csv").fillna("").set_index("id")
    regla, final, mec = [], [], []
    for _, r in U.iterrows():
        capaA, capaB = "A" in r.cadenas, "B" in r.cadenas and "A" not in r.cadenas
        if r.excl_auto:
            regla.append("R0 tipo de documento (no es estudio)"); final.append("excluir"); mec.append(""); continue
        if capaB:
            d = L.loc[r.id]
            regla.append("lectura capa B" + (": " + d.razon_ia if d.razon_ia else "")); final.append(d.decision_final)
            mec.append(d.mecanismo); continue
        if r.es_3d:
            regla.append("R1 3D / nube de puntos"); final.append("excluir"); mec.append(""); continue
        if not r.menciona_instancias:
            regla.append("R2 no menciona segmentación de instancias"); final.append("excluir"); mec.append(""); continue
        if not r.reporta_metrica:
            regla.append("R3 no reporta métricas en el resumen"); final.append("excluir"); mec.append(""); continue
        if r.n_arq_instancias < 2 and "A3" not in r.cadenas:
            regla.append("R4 no compara >= 2 arquitecturas de segmentación de instancias nombradas en el resumen")
            final.append("excluir"); mec.append(""); continue
        d = L.loc[r.id]  # R5: lectura (falla si falta una decisión: así no queda ningún candidato sin revisar)
        regla.append("R5 lectura (IA: " + d.decision_ia + (", " + d.razon_ia if d.razon_ia else "") + ")")
        final.append(d.decision_final); mec.append("")
    U["regla"], U["decision_final"], U["mecanismo_capaB"] = regla, final, mec
    return U


def main():
    R = cargar()
    U = marcar(deduplicar(R))
    U = decidir(U)
    cols = ["id", "cadenas", "bases", "title", "authors", "year", "src", "dtype", "doi", "cites", "abstract",
            "excl_auto", "menciona_instancias", "reporta_metrica", "es_3d", "arquitecturas_mencionadas",
            "n_arq_instancias", "regla", "decision_final", "mecanismo_capaB"]
    U[cols].to_csv(AQUI / "registros_unicos_E2.csv", index=False, encoding="utf-8-sig")

    A = U[U.cadenas.str.contains("A")]
    B = U[U.cadenas.str.contains("B") & ~U.cadenas.str.contains("A")]
    cuenta = lambda D: {k: int(v) for k, v in D.regla.str[:2].value_counts().sort_index().items()}
    prisma = {
        "identificados": int(len(R)),
        "por_base": {k: int(v) for k, v in R.base.value_counts().items()},
        "por_cadena_y_base": {f"{c} {b}": int(n) for (c, b), n in R.groupby(["cadena", "base"]).size().items()},
        "duplicados": int(len(R) - len(U)),
        "unicos": int(len(U)),
        "excluidos_tipo_documento": int((U.excl_auto != "").sum()),
        "cribados": int((U.excl_auto == "").sum()),
        "capa_A": {"cribados": int((A.excl_auto == "").sum()), "por_regla": cuenta(A[A.excl_auto == ""]),
                   "incluidos": int((A.decision_final == "incluir").sum())},
        "capa_B": {"cribados": int((B.excl_auto == "").sum()), "incluidos": int((B.decision_final == "incluir").sum()),
                   "por_mecanismo": {k: int(v) for k, v in B[B.mecanismo_capaB != ""].mecanismo_capaB.value_counts().sort_index().items()}},
    }
    (AQUI / "prisma_E2.json").write_text(json.dumps(prisma, indent=1, ensure_ascii=False), encoding="utf8")
    print(json.dumps(prisma, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
