@echo off
title Jarvis Mk.X — Streamlit App
echo.
echo Starting Jarvis Mk.X Web Application...
echo.

cd /d "%~dp0"
call .venv\Scripts\activate

if "%VOYAGE_API_KEY%"=="" (
echo.
echo [WARNING] VOYAGE_API_KEY is not set.
echo The default embedding model is now "voyage-4-large" and requires a Voyage API key.
echo Set VOYAGE_API_KEY in your environment before running for best results.
echo.
)

streamlit run app.py --server.port 8501

echo.
echo Jarvis has exited. Press any key to close.
pause >nul
