@echo off
echo ========================================
echo Deepfake Audio Detection - Streamlit App
echo ========================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo Dependencies OK
)

echo.
echo [2/3] Checking model files...
if not exist "models\resnet50_v3.keras" (
    if not exist "models\mobilenetv2.keras" (
        if not exist "models\baseline_cnn.keras" (
            echo WARNING: No model files found in models\ directory.
            echo Please ensure you have trained or downloaded the model weights.
            echo.
        )
    )
)

echo [3/3] Starting Streamlit app...
echo App will open at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
streamlit run app\\app.py
pause
