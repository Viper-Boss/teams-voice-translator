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
        throw "未找到 Python。请先安装 Python 3.11 或 3.12（64 位），安装时勾选 Add Python to PATH。"
    }
}

$Python = Join-Path $VenvDir "Scripts\python.exe"
& $Python -m pip install --upgrade pip
& $Python -m pip install --only-binary=:all: -i https://pypi.tuna.tsinghua.edu.cn/simple -r (Join-Path $ProjectDir "requirements.txt")

Write-Host ""
Write-Host "安装完成。下一步：" -ForegroundColor Green
Write-Host "1. 安装 VB-CABLE：https://vb-audio.com/Cable/"
Write-Host "2. 运行 run.bat"
Write-Host "3. 设置 CABLE Input 为程序输出、CABLE Output 为 Teams 麦克风"
