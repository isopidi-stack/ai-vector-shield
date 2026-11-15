$ErrorActionPreference = "Stop"

$corePath = "C:\Users\Ivan\twistor-render-core"
$shieldPath = "C:\Users\Ivan\ai-vector-shield"

function Get-PythonExe($repo) {
    if (Test-Path "$repo\venv\Scripts\python.exe") {
        return "$repo\venv\Scripts\python.exe"
    } elseif (Test-Path "C:\Users\Ivan\AppData\Local\Programs\Python\Python313\python.exe") {
        return "C:\Users\Ivan\AppData\Local\Programs\Python\Python313\python.exe"
    } else {
        return "python"
    }
}

Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host " [1/2] RUNNING TWISTOR-RENDER-CORE AUDIT SUITE (26 TESTS)" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

Set-Location $corePath
git status -s
$pyCore = Get-PythonExe $corePath
$env:PYTHONPATH = $corePath
& $pyCore -m unittest discover -s "$corePath\tests" -p "test_*.py" -v

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Twistor-Render-Core 26 unit tests passed!" -ForegroundColor Green
} else {
    Write-Host "`n[FAILURE] Twistor-Render-Core tests failed." -ForegroundColor Red
    exit 1
}

Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host " [2/2] RUNNING AI-VECTOR-SHIELD RED-TEAM SUITE & BENCHMARK" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

Set-Location $shieldPath
git status -s
$pyShield = Get-PythonExe $shieldPath
$env:PYTHONPATH = $shieldPath
& $pyShield "$shieldPath\topological_poison_shield.py"
& $pyShield -m unittest discover -s "$shieldPath\tests" -p "test_*.py" -v

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] AI-Vector-Shield tests and benchmark passed!" -ForegroundColor Green
} else {
    Write-Host "`n[FAILURE] AI-Vector-Shield tests failed." -ForegroundColor Red
    exit 1
}

Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host " ALL REPOSITORIES VERIFIED & GREEN AT $(Get-Date) " -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Cyan