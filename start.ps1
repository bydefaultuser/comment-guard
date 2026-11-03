# Starts the backend and the frontend in two new windows, then opens the app in the browser.
$root = $PSScriptRoot

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\backend'; .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\frontend'; npm run dev"

Write-Host "Starting... the page opens in 15 seconds (the first start takes longer while the models load)."
Start-Sleep -Seconds 15
Start-Process "http://localhost:5173"
