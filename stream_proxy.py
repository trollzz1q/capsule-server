"""
Прокси HLS -> MP3 для колонки.

SoundCloud всё чаще отдаёт только HLS (.m3u8). mpg123 его не понимает, поэтому сервер
перекодирует HLS через ffmpeg "на лету" и отдаёт колонке обычный MP3-поток по HTTP.
Колонка просто делает: curl -s -L <url> | mpg123 -a plughw:0,0 -q -
"""
import asyncio
import secrets
import shutil
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter()

# token -> (hls_url, headers, created_at)
STREAMS: dict = {}
TTL_SEC = 3 * 3600


def ffmpeg_path():
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg  # статический ffmpeg из pip, если системного нет
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def register_hls(src_url: str, headers: dict | None = None) -> str:
    """Запоминает HLS-ссылку и возвращает относительный путь вида /stream/<token>.mp3"""
    now = time.time()
    for t in [t for t, v in STREAMS.items() if now - v[2] > TTL_SEC]:
        STREAMS.pop(t, None)
    token = secrets.token_urlsafe(8)
    STREAMS[token] = (src_url, headers or {}, now)
    return f"/stream/{token}.mp3"


@router.get("/stream/{token}.mp3")
async def stream_mp3(token: str):
    item = STREAMS.get(token)
    if not item:
        raise HTTPException(status_code=404, detail="stream expired")

    ff = ffmpeg_path()
    if not ff:
        raise HTTPException(status_code=500, detail="ffmpeg not found")

    src, headers, _ = item
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-user_agent", "Mozilla/5.0"]
    if headers.get("Referer"):
        cmd += ["-referer", headers["Referer"]]
    cmd += ["-i", src, "-vn", "-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3", "pipe:1"]

    async def generate():
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            while True:
                chunk = await proc.stdout.read(16384)
                if not chunk:
                    break
                yield chunk
        finally:
            # колонка оборвала соединение (next/stop) -> гасим ffmpeg
            if proc.returncode is None:
                proc.kill()
            try:
                await proc.wait()
            except Exception:
                pass

    return StreamingResponse(generate(), media_type="audio/mpeg")