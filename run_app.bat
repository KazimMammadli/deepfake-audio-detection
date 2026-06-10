@echo off
echo ========================================
echo Deepfake Audio Detection - Streamlit App
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    echo Dependencies OK
)

echo.
echo [2/3] Checking model files...
if not exist "models\resnet50_v2.keras" (
    echo WARNING: Model files not found in models\ directory
    echo Please ensure you have the trained models
    echo.
)

echo [3/3] Starting Streamlit app...
echo.
echo App will open at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
streamlit run app.py

pause
