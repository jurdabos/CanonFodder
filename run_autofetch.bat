@echo off
echo Starting the batch script...
cd C:\Users\jurda\PycharmProjects\MyLifeInData
echo Changed directory to %cd%

echo Checking for leftover temporary files...
if exist "C:\Users\jurda\Downloads\jurda.csv" (
    echo Deleting temporary file: C:\Users\jurda\Downloads\jurda.csv
    del "C:\Users\jurda\Downloads\jurda.csv"
)

echo Activating virtual environment...
call .\.venv\Scripts\activate
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate the virtual environment. Exiting. >> run_error_log.txt
    pause
    exit /b 1
)

echo Running Python script...
call python csv_autofetch.py > run_log.txt 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python script failed to run. Check run_log.txt for details. >> run_error_log.txt
    pause
    exit /b 1
)

echo Deactivating virtual environment...
call deactivate
pause
