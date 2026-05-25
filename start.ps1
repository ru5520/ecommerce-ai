# Start ecommerce AI app (public link enabled for remote friends)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "1"
$env:NO_PROXY = "127.0.0.1,localhost"
$env:GRADIO_SERVER_NAME = "0.0.0.0"
$env:GRADIO_SHARE = "1"

# 从 .env 加载敏感配置（ADMIN_PASSWORD / DEEPSEEK_API_KEY 等）
# 注意：.env 已在 .gitignore 中，不会被提交
if (Test-Path "$PSScriptRoot\.env") {
    Get-Content "$PSScriptRoot\.env" | ForEach-Object {
        if ($_ -match "^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.+?)\s*$") {
            $name = $matches[1]
            $value = $matches[2].Trim('"').Trim("'")
            Set-Item -Path "Env:$name" -Value $value
        }
    }
    Write-Host "[OK] Loaded secrets from .env"
} else {
    Write-Host "[!] No .env file found. Copy .env.example to .env and set ADMIN_PASSWORD."
}

# Close old Gradio on port 7860 (avoid "Cannot find empty port" error)
$old = Get-NetTCPConnection -LocalPort 7860 -ErrorAction SilentlyContinue
if ($old) {
    $old | ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Write-Host "[OK] Closed previous process on port 7860"
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "Starting... first launch may take ~1 min (CLIP index)"
Write-Host "When ready, copy the line:  Running on public URL: https://....gradio.live"
Write-Host "Send that link to your friends. Keep this window open."
Write-Host ""

& "$PSScriptRoot\ai_env\Scripts\python.exe" "$PSScriptRoot\frontend\gradio_app.py"
