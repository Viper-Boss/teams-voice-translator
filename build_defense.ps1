param([string]$PythonPath = '')
$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = if ($PythonPath) { $PythonPath } else { Join-Path $ProjectDir ".venv\Scripts\python.exe" }
if (-not (Test-Path $Python)) {
    throw "Run setup.ps1 first."
}
Push-Location $ProjectDir
try {
    & $Python -m PyInstaller --noconfirm --clean (Join-Path $ProjectDir "DefenseMode.spec")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed: $LASTEXITCODE" }
} finally {
    Pop-Location
}
Write-Host "Build complete: $ProjectDir\dist\DefenseMode\DefenseMode.exe" -ForegroundColor Green
