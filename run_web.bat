@echo off
REM Web UI: nap cau hinh Gemini roi khoi dong server tai http://localhost:8000
call run_gemini.bat
python server.py
