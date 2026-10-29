@echo off
setlocal EnableDelayedExpansion

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
chcp 65001 >nul

echo =======================================================================
echo    EuroSAT Land Cover Classification - Master Training Batch Runner
echo =======================================================================
echo.

:: 1. Verify Python availability
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in your system PATH.
    pause
    exit /b 1
)

echo [OK] Python detected:
python -X utf8 --version
echo.

:: 2. Install dependencies
echo -----------------------------------------------------------------------
echo [STEP 1/6] Verifying Python Dependencies...
echo -----------------------------------------------------------------------
cd /d "%~dp0"
pip install -r requirements.txt --quiet
echo [OK] Dependencies verified.
echo.

:: 3. Execute multi-dataset verification loader
echo -----------------------------------------------------------------------
echo [STEP 2/6] Executing Multi-Dataset Harmonizer (AID + GID)...
echo Script: load_multi_datasets.py
echo -----------------------------------------------------------------------
python -X utf8 load_multi_datasets.py
echo.

:: 4. Execute Road & Linear Infrastructure Training Pipeline
echo -----------------------------------------------------------------------
echo [STEP 3/6] Executing Road Feature Training Pipeline...
echo Script: train_with_road_datasets.py
echo Output: road_optimized_resnet50.pth
echo -----------------------------------------------------------------------
python -X utf8 train_with_road_datasets.py
echo.

:: 5. Execute Mixed-Perspective Training Pipeline
echo -----------------------------------------------------------------------
echo [STEP 4/6] Executing Mixed-Perspective Training (Satellite + Drone)...
echo Script: train_mixed_perspective.py
echo Output: mixed_perspective_resnet50.pth
echo -----------------------------------------------------------------------
python -X utf8 train_mixed_perspective.py
echo.

:: 6. Execute Terrain & Arid Land Training Pipeline
echo -----------------------------------------------------------------------
echo [STEP 5/6] Executing Terrain Training (Mountains/Hills + Sand/Barren)...
echo Script: train_terrain_model.py
echo Output: terrain_optimized_resnet50.pth
echo -----------------------------------------------------------------------
python -X utf8 train_terrain_model.py
echo.

:: 7. Execute Master Mixed Training Pipeline (All Hugging Face Benchmarks)
echo -----------------------------------------------------------------------
echo [STEP 6/6] Executing Master Mixed Training Pipeline...
echo Script: train_master_mixed.py
echo Output: mixed_perspective_resnet50.pth, mixed_perspective_confusion_matrix.png
echo -----------------------------------------------------------------------
python -X utf8 train_master_mixed.py
echo.

echo =======================================================================
echo    All Training Pipelines Executed Successfully!
echo =======================================================================
pause
