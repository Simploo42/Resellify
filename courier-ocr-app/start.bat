@echo off
REM Start the courier OCR server and expose it via Cloudflare Tunnel (Windows)

cd /d "%~dp0"

echo Starting server...
start /B python main.py

timeout /t 2 /nobreak >nul

echo.
echo Starting Cloudflare Tunnel...
echo.
cloudflared tunnel --url http://localhost:8000
