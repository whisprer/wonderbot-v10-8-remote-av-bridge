
from fastapi import UploadFile
import argparse
import asyncio
import io
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


def _require_token(expected: str, received: str | None) -> None:
    if not expected:
        return
    if received != expected:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail='invalid bridge token')


def _coerce_sounddevice_device(value):
    """Return a sounddevice-compatible device selector.

    sounddevice treats integer indexes and string names differently. argparse gives us
    strings, so "9" must become integer 9 rather than a request for a device named
    literally "9".
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _describe_sounddevice_input(sd, device) -> str:
    """Return a human-readable description of the selected input device."""
    try:
        if device is None:
            default = sd.default.device
            if isinstance(default, (list, tuple)):
                input_index = default[0]
            else:
                input_index = default
            info = sd.query_devices(input_index, kind='input')
            return f"default input {input_index}: {info.get('name', '<unknown>')}"
        info = sd.query_devices(device, kind='input')
        return f"input {device!r}: {info.get('name', '<unknown>')}"
    except Exception as exc:
        selector = 'default input' if device is None else f"input {device!r}"
        return f"{selector}: <unable to describe: {exc}>"


@dataclass
class FrameState:
    jpeg: bytes | None = None
    timestamp_ms: int = 0
    width: int = 0
    height: int = 0
    source: str = ''


class AudioState:
    def __init__(self, max_seconds: float = 12.0) -> None:
        self.sample_rate = 16000
        self.channels = 1
        self.max_seconds = max_seconds
        self._chunks: deque = deque()
        self._frames = 0
        self.timestamp_ms = 0
        self.source = ''

    def append(self, array, sample_rate: int, channels: int, timestamp_ms: int, source: str) -> None:
        import numpy as np
        data = np.asarray(array, dtype='float32').reshape(-1)
        if data.size == 0:
            return
        if int(sample_rate) != int(self.sample_rate) or int(channels) != int(self.channels):
            self._chunks.clear()
            self._frames = 0
            self.sample_rate = int(sample_rate)
            self.channels = int(channels)
        self._chunks.append(data.copy())
        self._frames += int(data.size)
        self.timestamp_ms = int(timestamp_ms)
        self.source = source
        limit = int(self.max_seconds * max(1, self.sample_rate))
        while self._frames > limit and self._chunks:
            removed = self._chunks.popleft()
            self._frames -= int(removed.size)

    def latest(self, seconds: float):
        import numpy as np
        if not self._chunks:
            return np.zeros((0,), dtype='float32'), self.sample_rate, self.channels
        data = np.concatenate(list(self._chunks)).astype('float32', copy=False)
        wanted = max(1, int(seconds * max(1, self.sample_rate)))
        if data.size > wanted:
            data = data[-wanted:]
        return data, self.sample_rate, self.channels

    def buffered_seconds(self) -> float:
        return float(self._frames) / float(max(1, self.sample_rate))


class BridgeStore:
    def __init__(self) -> None:
        self.frame = FrameState()
        self.audio = AudioState()
        self.lock = threading.Lock()


STORE = BridgeStore()


def create_app(token: str = ''):
    from fastapi import FastAPI, File, Form, Header, UploadFile
    from fastapi.responses import JSONResponse, Response
    import soundfile as sf

    app = FastAPI(title='WonderBot AV Bridge', version='1.0.0')

    @app.get('/health')
    async def health():
        with STORE.lock:
            now = int(time.time() * 1000)
            camera_age = (now - STORE.frame.timestamp_ms) / 1000.0 if STORE.frame.timestamp_ms else None
            audio_age = (now - STORE.audio.timestamp_ms) / 1000.0 if STORE.audio.timestamp_ms else None
            return {
                'ok': True,
                'camera': {
                    'available': STORE.frame.jpeg is not None,
                    'age_seconds': camera_age,
                    'width': STORE.frame.width,
                    'height': STORE.frame.height,
                    'source': STORE.frame.source,
                },
                'audio': {
                    'available': STORE.audio.timestamp_ms > 0,
                    'age_seconds': audio_age,
                    'sample_rate': STORE.audio.sample_rate,
                    'channels': STORE.audio.channels,
                    'buffered_seconds': round(STORE.audio.buffered_seconds(), 3),
                    'source': STORE.audio.source,
                },
            }

    @app.post('/api/ingest/frame')
    async def ingest_frame(
        frame: UploadFile = File(...),
        timestamp_ms: int = Form(...),
        width: int = Form(...),
        height: int = Form(...),
        source: str = Form('desktop-client'),
        x_bridge_token: str | None = Header(default=None),
    ):
        _require_token(token, x_bridge_token)
        payload = await frame.read()
        with STORE.lock:
            STORE.frame.jpeg = payload
            STORE.frame.timestamp_ms = int(timestamp_ms)
            STORE.frame.width = int(width)
            STORE.frame.height = int(height)
            STORE.frame.source = source
        return JSONResponse({'ok': True})

    @app.post('/api/ingest/audio')
    async def ingest_audio(
        audio: UploadFile = File(...),
        timestamp_ms: int = Form(...),
        sample_rate: int = Form(...),
        channels: int = Form(...),
        source: str = Form('desktop-client'),
        x_bridge_token: str | None = Header(default=None),
    ):
        _require_token(token, x_bridge_token)
        payload = await audio.read()
        array, sr = sf.read(io.BytesIO(payload), dtype='float32', always_2d=False)
        if getattr(array, 'ndim', 1) > 1:
            import numpy as np
            array = np.asarray(array, dtype='float32').mean(axis=1)
        with STORE.lock:
            STORE.audio.append(array, sample_rate=int(sr or sample_rate), channels=1, timestamp_ms=int(timestamp_ms), source=source)
        return JSONResponse({'ok': True})

    @app.get('/api/camera/latest.jpg')
    async def latest_frame(x_bridge_token: str | None = Header(default=None)):
        _require_token(token, x_bridge_token)
        with STORE.lock:
            if STORE.frame.jpeg is None:
                return Response(status_code=404)
            return Response(content=STORE.frame.jpeg, media_type='image/jpeg', headers={
                'X-Timestamp-Ms': str(STORE.frame.timestamp_ms),
                'X-Width': str(STORE.frame.width),
                'X-Height': str(STORE.frame.height),
                'X-Source': STORE.frame.source,
            })

    @app.get('/api/audio/window.wav')
    async def audio_window(seconds: float = 3.0, x_bridge_token: str | None = Header(default=None)):
        _require_token(token, x_bridge_token)
        with STORE.lock:
            data, sample_rate, channels = STORE.audio.latest(seconds)
            timestamp_ms = STORE.audio.timestamp_ms
            source = STORE.audio.source
        if data.size == 0:
            return Response(status_code=404)
        buf = io.BytesIO()
        sf.write(buf, data, sample_rate, format='WAV', subtype='FLOAT')
        return Response(content=buf.getvalue(), media_type='audio/wav', headers={
            'X-Timestamp-Ms': str(timestamp_ms),
            'X-Sample-Rate': str(sample_rate),
            'X-Channels': str(channels),
            'X-Source': source,
        })

    return app


class DesktopBridgeClient:
    def __init__(
        self,
        server_url: str,
        token: str = '',
        camera_index: int = 0,
        width: int = 640,
        height: int = 360,
        fps: float = 3.0,
        jpeg_quality: int = 80,
        mic_device: str = '',
        sample_rate: int = 48000,
        channels: int = 1,
        audio_chunk_seconds: float = 0.75,
        source_name: str = 'desktop-client',
        audio_meter: bool = False,
        audio_meter_seconds: float = 5.0,
        warn_silence_dbfs: float = -70.0,
    ) -> None:
        import cv2  # type: ignore
        import httpx
        import numpy as np
        import sounddevice as sd  # type: ignore
        import soundfile as sf
        self._cv2 = cv2
        self._httpx = httpx
        self._np = np
        self._sd = sd
        self._sf = sf
        self.server_url = server_url.rstrip('/')
        self.token = token
        self.camera_index = int(camera_index)
        self.width = int(width)
        self.height = int(height)
        self.fps = max(0.5, float(fps))
        self.jpeg_quality = max(20, min(95, int(jpeg_quality)))
        self.mic_device = _coerce_sounddevice_device(mic_device)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.audio_meter = bool(audio_meter)
        self.audio_meter_seconds = max(0.5, float(audio_meter_seconds))
        self.warn_silence_dbfs = float(warn_silence_dbfs)
        self._last_audio_meter_at = 0.0
        self.audio_chunk_seconds = max(0.25, float(audio_chunk_seconds))
        self.source_name = source_name
        self._stop = threading.Event()
        self._audio_lock = threading.Lock()
        self._audio_chunks: deque = deque()
        self._audio_frames = 0
        self._audio_limit = int(self.sample_rate * max(4.0, self.audio_chunk_seconds * 4))
        self._stream = None
        self._client = httpx.Client(timeout=10.0)

    def _headers(self) -> dict[str, str]:
        return {'X-Bridge-Token': self.token} if self.token else {}

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        _ = frames, time_info, status
        mono = self._np.asarray(indata, dtype='float32')
        if mono.ndim > 1:
            mono = mono.mean(axis=1)
        mono = mono.reshape(-1).copy()
        if mono.size == 0:
            return
        with self._audio_lock:
            self._audio_chunks.append(mono)
            self._audio_frames += int(mono.size)
            while self._audio_frames > self._audio_limit and self._audio_chunks:
                removed = self._audio_chunks.popleft()
                self._audio_frames -= int(removed.size)

    def _latest_audio(self):
        wanted = max(1, int(self.audio_chunk_seconds * self.sample_rate))
        with self._audio_lock:
            if not self._audio_chunks:
                return self._np.zeros((0,), dtype='float32')
            data = self._np.concatenate(list(self._audio_chunks)).astype('float32', copy=False)
        if data.size > wanted:
            data = data[-wanted:]
        return data

    def _camera_loop(self) -> None:
        cap = self._cv2.VideoCapture(self.camera_index)
        if self.width > 0:
            cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        period = 1.0 / self.fps
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(period)
                    continue
                ok, enc = self._cv2.imencode('.jpg', frame, [int(self._cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                if ok:
                    payload = enc.tobytes()
                    files = {'frame': ('frame.jpg', payload, 'image/jpeg')}
                    data = {
                        'timestamp_ms': str(int(time.time() * 1000)),
                        'width': str(frame.shape[1]),
                        'height': str(frame.shape[0]),
                        'source': self.source_name,
                    }
                    try:
                        self._client.post(f'{self.server_url}/api/ingest/frame', files=files, data=data, headers=self._headers())
                    except Exception:
                        pass
                time.sleep(period)
        finally:
            cap.release()

    def _audio_loop(self) -> None:
        device = self.mic_device
        print(f'[bridge-client] audio input: {_describe_sounddevice_input(self._sd, device)}')
        self._stream = self._sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            device=device,
            latency='high',
            callback=self._audio_callback,
        )
        self._stream.start()
        try:
            while not self._stop.is_set():
                time.sleep(self.audio_chunk_seconds)
                chunk = self._latest_audio()
                if chunk.size == 0:
                    continue
                if self.audio_meter:
                    self._maybe_print_audio_meter(chunk)
                buf = io.BytesIO()
                self._sf.write(buf, chunk, self.sample_rate, format='WAV', subtype='FLOAT')
                files = {'audio': ('audio.wav', buf.getvalue(), 'audio/wav')}
                data = {
                    'timestamp_ms': str(int(time.time() * 1000)),
                    'sample_rate': str(self.sample_rate),
                    'channels': '1',
                    'source': self.source_name,
                }
                try:
                    self._client.post(f'{self.server_url}/api/ingest/audio', files=files, data=data, headers=self._headers())
                except Exception:
                    pass
        finally:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass

    def _maybe_print_audio_meter(self, chunk) -> None:
        now = time.time()
        if now - self._last_audio_meter_at < self.audio_meter_seconds:
            return
        self._last_audio_meter_at = now

        data = self._np.asarray(chunk, dtype='float32').reshape(-1)
        if data.size == 0:
            return

        peak = float(self._np.max(self._np.abs(data)))
        rms = float(self._np.sqrt(self._np.mean(data ** 2)))
        dbfs = 20.0 * math.log10(max(rms, 1.0e-12))
        status = 'LOW/SILENT' if dbfs <= self.warn_silence_dbfs else 'ok'
        print(
            f'[bridge-client] audio level: rms_dbfs={dbfs:.1f}, '
            f'peak={peak:.6f}, status={status}'
        )

    def run(self) -> None:
        threads = [
            threading.Thread(target=self._camera_loop, daemon=True),
            threading.Thread(target=self._audio_loop, daemon=True),
        ]
        for t in threads:
            t.start()
        print(f'[bridge-client] streaming to {self.server_url} as {self.source_name}')
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print('[bridge-client] stopping')
            self._stop.set()
            for t in threads:
                t.join(timeout=2.0)


def _server_main() -> int:
    parser = argparse.ArgumentParser(description='WonderBot AV bridge server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--token', default=os.environ.get('WONDERBOT_BRIDGE_TOKEN', ''))
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(token=args.token), host=args.host, port=args.port)
    return 0


def _client_main() -> int:
    parser = argparse.ArgumentParser(description='WonderBot desktop AV bridge client')
    parser.add_argument('--server-url', required=True)
    parser.add_argument('--token', default=os.environ.get('WONDERBOT_BRIDGE_TOKEN', ''))
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--fps', type=float, default=3.0)
    parser.add_argument('--jpeg-quality', type=int, default=80)
    parser.add_argument('--mic-device', default='')
    parser.add_argument('--sample-rate', type=int, default=48000)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--audio-chunk-seconds', type=float, default=0.75)
    parser.add_argument('--source-name', default='desktop-client')
    parser.add_argument('--audio-meter', action='store_true', help='print periodic microphone RMS/peak levels')
    parser.add_argument('--audio-meter-seconds', type=float, default=5.0)
    parser.add_argument('--warn-silence-dbfs', type=float, default=-70.0)
    args = parser.parse_args()
    DesktopBridgeClient(
        server_url=args.server_url,
        token=args.token,
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
        mic_device=args.mic_device,
        sample_rate=args.sample_rate,
        channels=args.channels,
        audio_chunk_seconds=args.audio_chunk_seconds,
        source_name=args.source_name,
        audio_meter=args.audio_meter,
        audio_meter_seconds=args.audio_meter_seconds,
        warn_silence_dbfs=args.warn_silence_dbfs,
    ).run()
    return 0


if __name__ == '__main__':
    import sys
    mode = os.environ.get('WONDERBOT_BRIDGE_MODE', '')
    if len(sys.argv) > 1 and sys.argv[1] in {'server', 'client'}:
        mode = sys.argv[1]
        del sys.argv[1]
    if mode == 'server':
        raise SystemExit(_server_main())
    if mode == 'client':
        raise SystemExit(_client_main())
    print('Set WONDERBOT_BRIDGE_MODE=server|client or pass server/client as first arg.')
