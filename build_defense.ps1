param([string]$PythonPath = '')
$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = if ($PythonPath) { $PythonPath } else { Join-Path $ProjectDir ".venv\Scripts\python.exe" }
if (-not (Test-Path $Python)) {
    throw "Run setup.ps1 first."
}
Push-Location $ProjectDir
try {
    # Install every runtime dependency first so the EXE always bundles them
    # (for example pypdf for thesis PDFs); people who get the EXE need no setup.
    & $Python -m pip install -r (Join-Path $ProjectDir "requirements-build.txt") -i https://pypi.tuna.tsinghua.edu.cn/simple --only-binary=:all: --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) { throw "Installing build requirements failed: $LASTEXITCODE" }
    & $Python -c "import pypdf, sys; sys.exit(0 if int(pypdf.__version__.split('.')[0]) >= 5 else 1)"
    if ($LASTEXITCODE -ne 0) { throw "pypdf 5 or newer is required for thesis PDF import." }
    & $Python -m PyInstaller --noconfirm --clean (Join-Path $ProjectDir "DefenseMode.spec")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed: $LASTEXITCODE" }
} finally {
    Pop-Location
}
Write-Host "Build complete: $ProjectDir\dist\DefenseMode\DefenseMode.exe" -ForegroundColor Green
