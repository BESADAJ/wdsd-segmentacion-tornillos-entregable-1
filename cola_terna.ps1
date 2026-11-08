# Cola por rondas de semilla con los 3 modelos de la terna (Mask R-CNN, YOLOv8-seg, SOLOv2).
# Uso, en PowerShell, desde la carpeta wdsd:
#   powershell -ExecutionPolicy Bypass -File .\cola_terna.ps1 -Hasta 18:00 -EpMaskRcnn 12 -EpYolo 40 -EpSolo 12
# Para repartir entre máquinas, cada una corre semillas distintas, siempre con los 3 modelos:
#   -Semillas 42,43,44   (3060)      -Semillas 45,46   (otra máquina)
# Los lotes deben ser iguales en todas las máquinas (-LoteMaskRcnn 2 -LoteYolo 4 -LoteSolo 2 por defecto).
# Si se corta, al relanzar salta las corridas que ya tienen resumen.json.
param(
    [string]$Hasta = "18:00",
    [int]$EpMaskRcnn = 12,
    [int]$EpYolo = 40,
    [int]$EpSolo = 12,
    [int]$LoteMaskRcnn = 2,
    [int]$LoteYolo = 4,
    [int]$LoteSolo = 2,
    [int[]]$Semillas = @(42, 43, 44, 45, 46),
    [string]$PY = "C:\Users\ivanj\miniconda3\envs\wdsd_dl\python.exe",
    [string]$PYM = "C:\Users\ivanj\miniconda3\envs\wdsd_mmdet\python.exe"
)
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force -Path "results\instancias" | Out-Null
$log = "results\instancias\cola_terna.log"
foreach ($s in $Semillas) {
    "== $(Get-Date -Format 'HH:mm') ronda semilla $s" | Tee-Object -FilePath $log -Append
    & $PY entrenar_instancias.py cola --hasta $Hasta --semillas $s --epocas-maskrcnn $EpMaskRcnn --epocas-yolo $EpYolo --lote-maskrcnn $LoteMaskRcnn --lote-yolo $LoteYolo 2>&1 |
        Tee-Object -FilePath $log -Append
    & $PYM solov2_instancias.py cola --hasta $Hasta --semillas $s --epocas $EpSolo --lote $LoteSolo 2>&1 |
        Tee-Object -FilePath $log -Append
}
& $PY entrenar_instancias.py resumen 2>&1 | Tee-Object -FilePath $log -Append
"== $(Get-Date -Format 'HH:mm') cola terminada" | Tee-Object -FilePath $log -Append
