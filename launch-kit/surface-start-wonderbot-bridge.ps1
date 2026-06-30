param(
    [string]$Repo = "C:\github\wonderbot-v10_8-remote-av-bridge",
    [string]$ServerUrl = "http://192.168.1.190:8765",
    [string]$Token = "change-me",
    [int]$CameraIndex = 0,
    [int]$SampleRate = 48000,
    [int]$Channels = 1,
    [string]$SourceName = "surface-bridge",
    [switch]$NoTtsPlayback,
    [switch]$HealthOnly
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Text)
    Write-Host ""
    Write-Host "=== $Text ===" -ForegroundColor Cyan
}

function Test-PathOrThrow {
    param(
        [string]$Path,
        [string]$Message
    )

    if (-not (Test-Path $Path)) {
        throw "$Message`nMissing path: $Path"
    }
}

Write-Section "WonderBot Surface Bridge Launcher"

Test-PathOrThrow -Path $Repo -Message "WonderBot repo was not found."
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
Test-PathOrThrow -Path $Python -Message "WonderBot Python venv was not found."

Set-Location $Repo

Write-Host "Repo:       $Repo"
Write-Host "Python:     $Python"
Write-Host "Server:     $ServerUrl"
Write-Host "Source:     $SourceName"
Write-Host "Camera:     $CameraIndex"
Write-Host "Audio:      $SampleRate Hz, $Channels channel(s)"
Write-Host "TTS return: $(-not $NoTtsPlayback)"

Write-Section "Bridge health"

try {
    $health = Invoke-RestMethod "$ServerUrl/health" -TimeoutSec 5
    $health | ConvertTo-Json -Depth 8
}
catch {
    Write-Warning "Could not reach bridge health endpoint yet: $ServerUrl/health"
    Write-Warning $_.Exception.Message

    if ($HealthOnly) {
        exit 1
    }

    Write-Host ""
    Write-Host "Continuing anyway. Make sure the Linux bridge server is running:"
    Write-Host "  ./launch-kit/server-start-bridge.sh"
}

if ($HealthOnly) {
    exit 0
}

Write-Section "Dependency sanity check"

$PythonCheck = @'
import importlib.util
import sys

required = [
    "fastapi",
    "requests",
    "sounddevice",
    "soundfile",
    "cv2",
    "PIL",
    "numpy",
]

missing = [name for name in required if importlib.util.find_spec(name) is None]

if missing:
    print("Missing Python packages:", ", ".join(missing))
    print("Install with:")
    print(r".\.venv\Scripts\python.exe -m pip install fastapi uvicorn python-multipart requests httpx pillow numpy sounddevice soundfile opencv-python")
    sys.exit(1)

print("Dependency sanity check OK.")
'@

$PythonCheck | & $Python -

Write-Section "Starting Surface bridge client"

$bridgeArgs = @(
    "-m", "wonderbot.bridge", "client",
    "--server-url", $ServerUrl,
    "--token", $Token,
    "--camera-index", "$CameraIndex",
    "--sample-rate", "$SampleRate",
    "--channels", "$Channels",
    "--source-name", $SourceName
)

if (-not $NoTtsPlayback) {
    $bridgeArgs += "--tts-playback"
}

Write-Host "Command:"
Write-Host "$Python $($bridgeArgs -join ' ')"
Write-Host ""
Write-Host "Leave this window open. Stop with Ctrl+C."
Write-Host ""

& $Python @bridgeArgs
