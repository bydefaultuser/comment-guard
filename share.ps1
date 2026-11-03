# Builds the page and serves the whole app (page + API) on one port.
# Forward port 8000 to share it, then send the forwarded URL. Stop with Ctrl+C.
param([int]$Port = 8000)
$root = $PSScriptRoot

Write-Host "Building the frontend..."
Push-Location "$root\frontend"
npm run build | Out-Null
$built = $LASTEXITCODE -eq 0
Pop-Location
if (-not $built) { Write-Host "Frontend build failed - run 'npm run build' in frontend to see why."; exit 1 }

Write-Host "Serving http://localhost:$Port - forward port $Port to share it."
Set-Location "$root\backend"
# 0.0.0.0 so forwarded traffic and other machines on the network can reach it, not just localhost.
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port $Port
