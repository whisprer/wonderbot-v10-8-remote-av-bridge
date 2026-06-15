@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [bridge-client] ERROR: .venv\Scripts\python.exe not found.
  echo [bridge-client] Run this from the WonderBot Windows repo after creating the venv.
  pause
  exit /b 1
)

if "%WONDERBOT_BRIDGE_SERVER_URL%"=="" set "WONDERBOT_BRIDGE_SERVER_URL=http://192.168.1.191:8765"
if "%WONDERBOT_BRIDGE_TOKEN%"=="" set "WONDERBOT_BRIDGE_TOKEN=change-me"
if "%WONDERBOT_CAMERA_INDEX%"=="" set "WONDERBOT_CAMERA_INDEX=0"
if "%WONDERBOT_MIC_DEVICE%"=="" set "WONDERBOT_MIC_DEVICE=9"
if "%WONDERBOT_SAMPLE_RATE%"=="" set "WONDERBOT_SAMPLE_RATE=48000"
if "%WONDERBOT_CHANNELS%"=="" set "WONDERBOT_CHANNELS=1"
if "%WONDERBOT_SOURCE_NAME%"=="" set "WONDERBOT_SOURCE_NAME=desktop-bridge"
if "%WONDERBOT_AUDIO_METER_SECONDS%"=="" set "WONDERBOT_AUDIO_METER_SECONDS=3"

echo [bridge-client] server: %WONDERBOT_BRIDGE_SERVER_URL%
echo [bridge-client] camera index: %WONDERBOT_CAMERA_INDEX%
echo [bridge-client] mic device: %WONDERBOT_MIC_DEVICE%
echo [bridge-client] source: %WONDERBOT_SOURCE_NAME%

".venv\Scripts\python.exe" -m wonderbot.bridge client ^
  --server-url "%WONDERBOT_BRIDGE_SERVER_URL%" ^
  --token "%WONDERBOT_BRIDGE_TOKEN%" ^
  --camera-index "%WONDERBOT_CAMERA_INDEX%" ^
  --mic-device "%WONDERBOT_MIC_DEVICE%" ^
  --sample-rate "%WONDERBOT_SAMPLE_RATE%" ^
  --channels "%WONDERBOT_CHANNELS%" ^
  --source-name "%WONDERBOT_SOURCE_NAME%" ^
  --audio-meter ^
  --audio-meter-seconds "%WONDERBOT_AUDIO_METER_SECONDS%"

pause
