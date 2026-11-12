@echo off
setlocal
set "CNN_PROJECT=%~dp0"
if exist "%CNN_PROJECT%app.py" goto found_project
set "CNN_PROJECT=E:\deeplearn\classic-cnn-digits\"
if exist "%CNN_PROJECT%app.py" goto found_project
set "CNN_PROJECT=%~dp0lenet5-mnist-innovations\"
if exist "%CNN_PROJECT%app.py" goto found_project
echo Project folder not found. Place this script beside app.py.
pause
exit /b 1
:found_project
cd /d "%CNN_PROJECT%"
set "CNN_PYTHON=%CNN_PROJECT%.venv\Scripts\python.exe"
if exist "%CNN_PYTHON%" goto run_visual
set "CNN_PYTHON=E:\yolo_course\.venv\Scripts\python.exe"
if exist "%CNN_PYTHON%" goto run_visual
python "%CNN_PROJECT%launch_visual.py"
if errorlevel 1 pause
exit /b
:run_visual
"%CNN_PYTHON%" "%CNN_PROJECT%launch_visual.py"
if errorlevel 1 pause
exit /b
