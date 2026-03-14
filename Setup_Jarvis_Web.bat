@echo off
title Jarvis Mk.X — Streamlit Setup
echo.
echo ==========================================
echo   Jarvis Mk.X — Streamlit App Setup
echo ==========================================
echo.
echo This will install all required dependencies.
echo Make sure Python 3.10+ is installed.
echo.
pause

cd /d "%~dp0"

echo.
echo [1/4] Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate

echo.
echo [2/4] Installing PyTorch with CUDA...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo [3/4] Installing ML dependencies...
pip install transformers accelerate peft bitsandbytes sentence-transformers voyageai chromadb rank-bm25 PyMuPDF trl datasets

echo.
echo [4/4] Installing Streamlit and utilities...
pip install streamlit plotly matplotlib scikit-learn fpdf2 numpy pandas Pillow

echo.
echo ==========================================
echo   Setup complete!
echo   Run "Run_Jarvis_Web.bat" to start.
echo ==========================================
echo.
pause
