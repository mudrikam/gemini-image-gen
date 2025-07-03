@echo off
echo Starting Gemini Image Generator...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.12+ and try again
    pause
    exit /b 1
)

REM Install required packages if not already installed
echo Checking required packages...
pip install PySide6 qtawesome Pillow google-genai >nul 2>&1

REM Run the application
echo Launching Gemini Image Generator...
python gemini_image_gen.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo Application encountered an error
    pause
)
