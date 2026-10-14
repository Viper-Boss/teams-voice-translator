$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectDir ".venv"

if (-not (Test-Path $VenvDir)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m venv $VenvDir
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        python -m venv $VenvDir
    } elseif (Test-Path "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe") {
        & "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m venv $VenvDir
    } else {
        throw "Python was not found. Install 64-bit Python 3.11 or 3.12 and enable Add Python to PATH."
    }
}

$Python = Join-Path $VenvDir "Scripts\python.exe"
& $Python -m pip install --upgrade pip
& $Python -m pip install --only-binary=:all: -i https://pypi.tuna.tsinghua.edu.cn/simple -r (Join-Path $ProjectDir "requirements.txt")

Write-Host ""
Write-Host "Setup complete. Next steps:" -ForegroundColor Green
Write-Host "1. Install VB-CABLE: https://vb-audio.com/Cable/"
Write-Host "2. Run run.bat"
Write-Host "3. Select CABLE Input as app output and CABLE Output as the Teams microphone"
