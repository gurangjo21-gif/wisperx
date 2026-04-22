@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "ST=.venv\Scripts\streamlit.exe"

rem --- 의존성 자동 설치 (이미 설치돼 있으면 스킵) ---
"%PY%" -c "import streamlit, whisperx, pyannote.audio" 2>nul
if errorlevel 1 (
    echo [setup] 의존성 설치 중...
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r requirements.txt
)

rem --- ffmpeg 경로 보강 (winget 설치 위치) ---
for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_*") do (
    for /d %%E in ("%%D\ffmpeg-*-full_build") do (
        set "PATH=%%E\bin;%PATH%"
    )
)

"%ST%" run app.py
pause
endlocal
