# Cadena para traer las citas de los papers originales de la lista larga (Scopus)

Uso: pegar en Scopus > Advanced search, exportar CSV con "Cited by", DOI, año y fuente. Registrar la fecha. Esto da el criterio "citación normalizada por antigüedad" desde una sola base (reproducible).

```
(TITLE("Mask R-CNN") AND AUTHLASTNAME(Girshick) AND PUBYEAR > 2016)
OR (TITLE("Cascade R-CNN") AND AUTHLASTNAME(Vasconcelos))
OR (TITLE("Hybrid Task Cascade for Instance Segmentation"))
OR (TITLE("Mask Scoring R-CNN"))
OR (TITLE("PointRend: Image Segmentation as Rendering"))
OR (TITLE("SOLO: Segmenting Objects by Locations"))
OR (TITLE("SOLOv2: Dynamic and Fast Instance Segmentation"))
OR (TITLE("Conditional Convolutions for Instance Segmentation"))
OR (TITLE("YOLACT") AND AUTHLASTNAME(Bolya))
OR (TITLE("BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation"))
OR (TITLE("CenterMask: Real-Time Anchor-Free Instance Segmentation"))
OR (TITLE("Instances as Queries"))
OR (TITLE("Masked-attention Mask Transformer for Universal Image Segmentation"))
OR (TITLE("Sparse Instance Activation for Real-Time Instance Segmentation"))
OR (TITLE("Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation"))
OR (TITLE("FastInst: A Simple Query-Based Model for Real-Time Instance Segmentation"))
OR (TITLE("PolarMask: Single Shot Instance Segmentation with Polar Representation"))
OR (TITLE("SCNet: Training Inference Sample Consistency for Instance Segmentation"))
OR (TITLE("YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"))
OR (TITLE("YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information"))
OR (TITLE("Segment Anything") AND AUTHLASTNAME(Kirillov))
```
Nota: Scopus evalúa OR antes que AND, por eso cada línea va entre paréntesis.

**Sin paper revisado por pares (no salen en Scopus):** YOLOv5-seg, YOLOv8-seg, YOLO11-seg (Ultralytics, software) y RTMDet (solo arXiv 2212.07784). Para estos la citación se toma de Google Scholar en la misma fecha y se declara la diferencia de fuente.
