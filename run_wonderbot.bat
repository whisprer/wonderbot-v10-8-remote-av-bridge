@echo off
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3.11 -m wonderbot.cli %*
) else (
  python -m wonderbot.cli %*
)
