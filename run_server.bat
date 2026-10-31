@echo off
title Rose Leaf Disease Detection API Server
cd /d "%~dp0"
echo ========================================================
echo  Starting Rose Leaf Disease Detection Backend Server...
echo ========================================================
echo.
echo Local IP: 192.168.62.164 (Port: 5000)
echo Health Check: http://127.0.0.1:5000/ or http://192.168.62.164:5000/
echo.
call .\venv\Scripts\activate.bat
python app.py
pause
