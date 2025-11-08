param(
    [int]$Port = 5000,
    [switch]$SkipBrowser
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Rutas del proyecto (este script vive en ./scripts/)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root      = Split-Path -Parent $ScriptDir

$VenvPy    = Join-Path $Root "venv\Scripts\python.exe"
$BackendPy = Join-Path $Root "backend\server.py"
$ModelDir  = Join-Path $Root "model"
$ModelPy   = Join-Path $ModelDir "inference_classifier_video.py"

if (-not (Test-Path $VenvPy))    { throw "No se encontró $VenvPy" }
if (-not (Test-Path $BackendPy)) { throw "No se encontró $BackendPy" }
if (-not (Test-Path $ModelPy))   { throw "No se encontró $ModelPy" }

# Logs (usar archivos separados para stdout y stderr)
$LogDir = Join-Path $Root "reports\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd-HHmmss"

$BackendOut = Join-Path $LogDir "backend-$ts.out.log"
$BackendErr = Join-Path $LogDir "backend-$ts.err.log"
$ModelOut   = Join-Path $LogDir "model-$ts.out.log"
$ModelErr   = Join-Path $LogDir "model-$ts.err.log"

$BackendProcess = $null
$ModelProcess   = $null

try {
    Write-Host "[HELEN] Lanzando backend..."
    $BackendProcess = Start-Process -FilePath $VenvPy `
        -ArgumentList @($BackendPy, "--port", "$Port") `
        -RedirectStandardOutput $BackendOut `
        -RedirectStandardError  $BackendErr `
        -WorkingDirectory $Root -PassThru

Start-Sleep -Seconds 2

Write-Host "[HELEN] Lanzando modelo..."
$ModelProcess = Start-Process -FilePath $VenvPy `
    -ArgumentList @($ModelPy) `
    -RedirectStandardOutput $ModelOut `
    -RedirectStandardError  $ModelErr `
    -WorkingDirectory $ModelDir -PassThru

Write-Host "[HELEN] Backend PID=$($BackendProcess.Id) | Modelo PID=$($ModelProcess.Id)"
Write-Host "[HELEN] Logs:"
Write-Host "  - Backend OUT: $BackendOut"
Write-Host "  - Backend ERR: $BackendErr"
Write-Host "  - Modelo  OUT: $ModelOut"
Write-Host "  - Modelo  ERR: $ModelErr"

if (-not $SkipBrowser) {
    Start-Process "http://localhost:$Port"
}

    Write-Host "`n=== Presiona Ctrl+C para detener ===`n"
    # Sigue el log de backend en vivo (opcional)
    Get-Content -Path $BackendOut -Wait -Tail 20 | ForEach-Object { "[backend] $_" }

} finally {
    Write-Host "`n[HELEN] Deteniendo procesos..."
    if ($BackendProcess -and -not $BackendProcess.HasExited) { Stop-Process -Id $BackendProcess.Id -Force }
    if ($ModelProcess   -and -not $ModelProcess.HasExited)   { Stop-Process -Id $ModelProcess.Id   -Force }
    Write-Host "[HELEN] Finalizado."
}
