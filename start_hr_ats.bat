@echo off
title HR Recruitment ATS Agent - IntelligenzIT
echo ============================================================
echo   HR Recruitment ATS Agent - IntelligenzIT
echo   Drag-and-drop resume screening powered by AI
echo ============================================================
echo.
echo Starting the app...
echo Once loaded, open: http://localhost:8501
echo.
echo To stop: Close this window or press Ctrl+C
echo.
echo ============================================================

cd /d "%~dp0"

.venv\Scripts\streamlit run app.py --server.port=8501 --server.headless=true

pause