$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Write-Host "[HELEN] Iniciando setup..."

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root      = Split-Path -Parent $ScriptDir

$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    throw "No se encontró Python en el PATH. Instálalo y asegúrate de poder ejecutar 'python' en la terminal."
}

$VenvDir   = Join-Path $Root "venv"
$VenvPy    = Join-Path $VenvDir "Scripts\python.exe"
$ReqFile   = Join-Path $Root "requirements.txt"

if (-not (Test-Path $VenvDir)) {
    Write-Host "[HELEN] Creando entorno virtual en 'venv'..."
    & $PythonExe -m venv $VenvDir
} else {
    Write-Host "[HELEN] Entorno virtual ya existe."
}

if (-not (Test-Path $VenvPy)) {
    throw "No se encontró $VenvPy. La creación del entorno falló."
}

Write-Host "[HELEN] Activando entorno virtual temporalmente..."
$env:VIRTUAL_ENV = $VenvDir
$env:PATH = "$($VenvDir)\Scripts;$env:PATH"

if (Test-Path $ReqFile) {
    Write-Host "[HELEN] Instalando dependencias desde requirements.txt..."
    & $VenvPy -m pip install --upgrade pip
    & $VenvPy -m pip install -r $ReqFile
} else {
    Write-Warning "[HELEN] No se encontró requirements.txt en $ReqFile"
}

Write-Host "`n[HELEN] Setup completado correctamente."
Write-Host "  Entorno: $VenvDir"
Write-Host "  Python:  $($VenvPy)"
Write-Host "  Usa '.\venv\Scripts\activate' para activarlo manualmente."
