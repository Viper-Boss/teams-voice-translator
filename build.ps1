$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Run setup.ps1 first."
}
& $Python -m pip install --only-binary=:all: -i https://pypi.tuna.tsinghua.edu.cn/simple -r (Join-Path $ProjectDir "requirements-build.txt")
Push-Location $ProjectDir
try {
    & $Python -m PyInstaller --noconfirm --clean (Join-Path $ProjectDir "TeamsVoiceTranslator.spec")
} finally {
    Pop-Location
}
Write-Host "Build complete: $ProjectDir\dist\TeamsVoiceTranslator\TeamsVoiceTranslator.exe" -ForegroundColor Green
