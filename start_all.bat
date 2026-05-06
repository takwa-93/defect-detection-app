@echo off
echo Starting Smart Quality Control System...

:: --- CONFIGURATION ---
set PYTHON_EXE="C:\Python314\python.exe"
set ROOT=C:\Users\takou\Desktop\PFE\defect-detection-app
set BACKEND=%ROOT%\backend

:: ── 1. Node Backend ───
echo [1/5] Starting Node.js backend...
start "Backend :5000" cmd /k "cd /d "%BACKEND%" && node server.js"
timeout /t 4 /nobreak > nul

:: ── 2. Niryo Camera ( ───
echo [3/5] Starting Niryo Live Stream...
start "Robot :5001" cmd /k "cd /d "%BACKEND%" && %PYTHON_EXE% niryo_stream.py"
timeout /t 2 /nobreak > nul

:: ── 3. AI Processor (Using Global 3.10) ───
echo [2/5] Starting AI Processor...
start "AI Processor" cmd /k "cd /d "%BACKEND%" && %PYTHON_EXE% ai_processor.py"
timeout /t 2 /nobreak > nul

:: ── 4. Angular Frontend ─────────────────────────────────────────────
start "Angular Frontend" cmd /k "cd /d "C:\Users\takou\Desktop\PFE\defect-detection-app" && npx ng serve"

:: ── 5. Niryo Controller (Using Global 3.10) ───
::echo [3/5] Starting Niryo controller...
::start "Robot :5002" cmd /k "cd /d "%BACKEND%" && %PYTHON_EXE% niryo_pick_place.py"
::timeout /t 2 /nobreak > nul

:: Wait for Angular to compile before opening browser
echo.
echo Waiting for Angular to compile...
timeout /t 10 /nobreak > nul

:: Open the app in the default browser
start "" http://localhost:4200

echo.
echo All services started!
echo  - Niryo Stream  : http://localhost:5001/stream
echo  - Node Backend  : http://localhost:5000
echo  - Angular App   : http://localhost:4200
echo.
echo Close Niryo Studio before connecting to the robot.
pause
