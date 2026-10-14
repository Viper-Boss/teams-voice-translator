@echo off
setlocal
cd /d "%~dp0"
echo 这个脚本会打包一个「分享版」zip，发给同学解压就能用。
echo 包里不带你的 API Key、音色、问答库、讲稿、语音缓存和个人主题图片。
echo.
echo 请先关掉正在运行的答辩程序，再按任意键开始（大约 3~8 分钟）。
pause >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_share.ps1"
if errorlevel 1 (
  echo.
  echo 打包失败，上面是错误信息，截图发给 Claude 就行。
  pause
  exit /b 1
)
echo.
pause
