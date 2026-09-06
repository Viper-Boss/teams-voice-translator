$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "dist\DefenseMode\DefenseMode.exe"
$p = Start-Process -FilePath $exe -ArgumentList @('--smoke-test', '--smoke-output', 'test-artifacts/packaged-smoke') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru
if (-not $p.WaitForExit(30000)) { throw 'Packaged smoke test timed out.' }
if ($p.ExitCode -ne 0) { throw "Packaged smoke test failed: $($p.ExitCode)" }
$result = Join-Path $PSScriptRoot 'test-artifacts\packaged-smoke\smoke-result.json'
if (-not (Test-Path -LiteralPath $result)) { throw 'Smoke result missing.' }
Write-Host 'SMOKE_OK: isolated Qt UI started and exited without network or recording.'
