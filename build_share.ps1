param([string]$PythonPath = '')
# 打包「分享版」：给同学用。不带个人主题图片、不带任何配置 / API Key / 问答库 / 讲稿 / 语音缓存
# （这些都在 %APPDATA%，本来就不会进包）。结果是一个 zip，同学解压后双击 DefenseMode.exe 即可。
$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = if ($PythonPath) { $PythonPath } else { Join-Path $ProjectDir ".venv\Scripts\python.exe" }
if (-not (Test-Path $Python)) { throw "找不到 .venv，请先运行 setup.ps1" }

Push-Location $ProjectDir
try {
    $Version = (& $Python -c "import teams_voice_translator as t; print(t.__version__.replace('-local',''))").Trim()
    $Stamp = Get-Date -Format "yyyy-MM-dd"
    $DistRoot = Join-Path $ProjectDir "dist"
    $OutDir = Join-Path $DistRoot "share-build"
    $WorkDir = Join-Path $ProjectDir "build\share"
    $ZipPath = Join-Path $DistRoot "DefenseMode-v$Version-分享版-$Stamp.zip"

    Write-Host "== 1/4 安装打包依赖 ==" -ForegroundColor Cyan
    & $Python -m pip install -r (Join-Path $ProjectDir "requirements-build.txt") -i https://pypi.tuna.tsinghua.edu.cn/simple --only-binary=:all: --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) { throw "安装打包依赖失败：$LASTEXITCODE" }

    Write-Host "== 2/4 打包（不带个人主题）==" -ForegroundColor Cyan
    $env:DEFENSE_SHARE_BUILD = "1"
    try {
        & $Python -m PyInstaller --noconfirm --clean --distpath $OutDir --workpath $WorkDir (Join-Path $ProjectDir "DefenseMode.spec")
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller 失败：$LASTEXITCODE" }
    } finally {
        Remove-Item Env:\DEFENSE_SHARE_BUILD -ErrorAction SilentlyContinue
    }
    $AppDir = Join-Path $OutDir "DefenseMode"
    if (-not (Test-Path (Join-Path $AppDir "DefenseMode.exe"))) { throw "没有生成 DefenseMode.exe" }

    Write-Host "== 3/4 检查包里没有个人内容 ==" -ForegroundColor Cyan
    $bad = @()
    $bad += Get-ChildItem -Path $AppDir -Recurse -Force -ErrorAction SilentlyContinue | Where-Object {
        $_.FullName -match 'local_assets\\.+' -or
        $_.Name -match '^(qa_bank|script_desk|quick_phrases|render_manifest|script_overrides|answer_overrides|thesis_index)\.json$' -or
        $_.Extension -eq '.pcm' -or
        $_.Name -match '\.env$'
    }
    if ($bad.Count -gt 0) {
        $bad | ForEach-Object { Write-Host ("  发现：" + $_.FullName) -ForegroundColor Red }
        throw "包里有个人文件，已停止（上面列出了是哪些）"
    }
    # 说明书放在 exe 旁边
    $Readme = (Get-Content (Join-Path $ProjectDir "share_readme_template.txt") -Raw -Encoding UTF8).Replace("{VERSION}", $Version)
    $ReadmePath = Join-Path $OutDir "使用说明-先看我.txt"
    [System.IO.File]::WriteAllText($ReadmePath, $Readme, (New-Object System.Text.UTF8Encoding($true)))

    Write-Host "== 4/4 压缩成 zip ==" -ForegroundColor Cyan
    if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
    Compress-Archive -Path $AppDir, $ReadmePath -DestinationPath $ZipPath -CompressionLevel Optimal
    $SizeMB = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
    Write-Host ""
    Write-Host "完成：$ZipPath（$SizeMB MB）" -ForegroundColor Green
    Write-Host "把这个 zip 发给同学：解压后双击 DefenseMode\DefenseMode.exe，第一次用先看「使用说明-先看我.txt」。" -ForegroundColor Green
    Start-Process explorer.exe "/select,`"$ZipPath`""
} finally {
    Pop-Location
}
