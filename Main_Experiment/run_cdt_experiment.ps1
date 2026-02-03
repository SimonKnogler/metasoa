# Run CDT experiment using PsychoPy's Python (has psychopy, numpy, pandas)
$PsychoPyPython = "C:\Program Files\PsychoPy\python.exe"
$ScriptDir = $PSScriptRoot
$Script = Join-Path $ScriptDir "CDT_windows_blockwise_fast_response.py"

if (-not (Test-Path $PsychoPyPython)) {
    Write-Host "PsychoPy Python not found at: $PsychoPyPython"
    Write-Host "Trying default Python (script will auto-detect PsychoPy if import fails)..."
    Set-Location $ScriptDir
    & python $Script
    exit $LASTEXITCODE
}

Set-Location $ScriptDir
& $PsychoPyPython $Script
exit $LASTEXITCODE
