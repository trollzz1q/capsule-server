import os
import json
import time
import urllib.request
from dotenv import load_dotenv
import yt_dlp

from stream_proxy import register_hls

load_dotenv(override=True)

SOUNDCLOUD_AUTH_TOKEN = os.getenv("SOUNDCLOUD_AUTH_TOKEN", "").strip()

playback_queue = []
current_queue_index = 0
current_playing_track = None
my_user_id = None

# Жёсткого селектора формата здесь нет: при "format" без совпадений yt-dlp падает с ошибкой.
# Вместо этого берём полный список formats и сами выбираем прогрессивный MP3.
ydl_opts = {
    'noplaylist': True,
    'quiet': True,
    'no_warnings': True,
    'source_address': '0.0.0.0',
}


def _is_hls(fmt: dict) -> bool:
    proto = (fmt.get("protocol") or "").lower()
    return "m3u8" in proto or "hls" in proto or ".m3u8" in (fmt.get("url") or "")


def pick_stream(info: dict) -> str:
    """
    Возвращает URL, который колонка может проиграть через `curl | mpg123 -`.
    1) Прогрессивный MP3 (http/https, ext=mp3) -> прямая ссылка на CDN.
    2) Иначе любой HLS -> регистрируем в ffmpeg-прокси, получаем /stream/<token>.mp3
       (относительный путь, колонка достраивает его до SERVER_URL).
    """
    formats = info.get("formats") or []

    progressive = [
        f for f in formats
        if f.get("url")
        and not _is_hls(f)
        and (f.get("protocol") or "").lower() in ("http", "https")
        and (f.get("ext") == "mp3" or "mp3" in (f.get("format_id") or ""))
    ]
    if progressive:
        progressive.sort(key=lambda f: f.get("abr") or f.get("tbr") or 0, reverse=True)
        return progressive[0]["url"]

    hls = [f for f in formats if f.get("url") and _is_hls(f)]
    if hls:
        # mp3-HLS предпочтительнее: ffmpeg его просто ремуксит/перекодирует быстрее
        hls.sort(key=lambda f: (("mp3" in (f.get("format_id") or "")), f.get("abr") or f.get("tbr") or 0), reverse=True)
        best = hls[0]
        return register_hls(best["url"], best.get("http_headers"))

    # последний шанс: то, что yt-dlp выбрал сам
    top = info.get("url", "")
    if top:
        return register_hls(top, info.get("http_headers")) if _is_hls(info) else top
    return ""


def get_my_user_id() -> str:
    global my_user_id
    if my_user_id:
        return my_user_id
    if not SOUNDCLOUD_AUTH_TOKEN:
        return ""

    try:
        url = "https://api-v2.soundcloud.com/me"
        req = urllib.request.Request(url, headers={
            "Authorization": f"OAuth {SOUNDCLOUD_AUTH_TOKEN}",
            "User-Agent": "Mozilla/5.0"
        })
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode())
            my_user_id = str(data.get("id", ""))
            print(f"[+] Авторизован в SoundCloud: {data.get('username')} (ID: {my_user_id})", flush=True)
            return my_user_id
    except Exception as e:
        print(f"[!] Ошибка профиля SoundCloud: {e}", flush=True)
        return ""


def resolve_stream_url(web_url: str) -> str:
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(web_url, download=False)
            return pick_stream(info)
    except Exception as e:
        print(f"[!] Ошибка извлечения потока: {e}", flush=True)
        return ""


def _set_current(track_id: str, title: str, stream_url: str, web_url: str = ""):
    global current_playing_track
    current_playing_track = {
        "id": track_id,
        "title": title,
        "stream_url": stream_url,
        "web_url": web_url,
    }


def search_and_get_track(query: str) -> dict:
    t0 = time.perf_counter()
    print(f"[*] Ищу в SoundCloud: '{query}'...", flush=True)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"scsearch1:{query}", download=False)
            entries = [e for e in (info.get("entries") or []) if e]
            entry = entries[0] if entries else None

            if not entry:
                return {"success": False, "error": "Ничего не найдено"}

            # у результатов поиска formats иногда нет -> дорезолвим по ссылке на страницу трека
            if not entry.get("formats") and entry.get("webpage_url"):
                entry = ydl.extract_info(entry["webpage_url"], download=False)

            stream_url = pick_stream(entry)
            if not stream_url:
                return {"success": False, "error": "Нет доступного аудиопотока"}

            title = entry.get("title", query)
            artist = entry.get("uploader", "SoundCloud")
            track_id = str(entry.get("id"))
            full_title = f"{artist} — {title}"
            _set_current(track_id, full_title, stream_url, entry.get("webpage_url", ""))

            dt = time.perf_counter() - t0
            print(f"[+] Найден трек ({dt:.2f}с): {full_title} -> {stream_url[:60]}", flush=True)
            return {
                "success": True,
                "title": full_title,
                "stream_url": stream_url,
                "track_id": track_id
            }
    except Exception as e:
        print(f"[!] Ошибка поиска SoundCloud: {e}", flush=True)
        return {"success": False, "error": str(e)}


def _play_queue_item(index: int) -> dict:
    """Резолвит элемент очереди заново: прямые ссылки SoundCloud протухают."""
    item = playback_queue[index]
    stream = resolve_stream_url(item.get("web_url", ""))
    if not stream:
        return {"success": False, "error": "Не удалось получить поток"}
    _set_current(item["id"], item["title"], stream, item.get("web_url", ""))
    return {
        "success": True,
        "title": item["title"],
        "stream_url": stream,
        "track_id": item["id"]
    }


def load_my_liked_tracks() -> dict:
    global playback_queue, current_queue_index
    uid = get_my_user_id()
    if not uid:
        return {"success": False, "error": "Не удалось авторизоваться в SoundCloud"}

    try:
        url = f"https://api-v2.soundcloud.com/users/{uid}/likes?limit=30"
        req = urllib.request.Request(url, headers={
            "Authorization": f"OAuth {SOUNDCLOUD_AUTH_TOKEN}",
            "User-Agent": "Mozilla/5.0"
        })
        with urllib.request.urlopen(req, timeout=6.0) as resp:
            data = json.loads(resp.read().decode())
            collection = data.get("collection", [])

            queue = []
            for item in collection:
                t = item.get("track")
                if t and t.get("permalink_url"):
                    queue.append({
                        "id": str(t.get("id")),
                        "title": f"{t.get('user', {}).get('username')} — {t.get('title')}",
                        "web_url": t.get("permalink_url")
                    })

            if not queue:
                return {"success": False, "error": "Плейлист пуст"}

            playback_queue = queue
            current_queue_index = 0
            return _play_queue_item(0)
    except Exception as e:
        print(f"[!] Ошибка плейлиста: {e}", flush=True)
        return {"success": False, "error": str(e)}


def get_next_track() -> dict:
    global current_queue_index
    if not playback_queue:
        return {"success": False, "error": "Очередь пуста"}
    current_queue_index = (current_queue_index + 1) % len(playback_queue)
    return _play_queue_item(current_queue_index)


def get_prev_track() -> dict:
    global current_queue_index
    if not playback_queue:
        return {"success": False, "error": "Очередь пуста"}
    current_queue_index = (current_queue_index - 1) % len(playback_queue)
    return _play_queue_item(current_queue_index)


def like_current_playing_track() -> bool:
    if not current_playing_track or not current_playing_track.get("id"):
        return False

    uid = get_my_user_id()
    if not uid:
        return False

    track_id = current_playing_track["id"]
    try:
        url = f"https://api-v2.soundcloud.com/users/{uid}/track_likes/{track_id}"
        req = urllib.request.Request(url, method="PUT")
        req.add_header("Authorization", f"OAuth {SOUNDCLOUD_AUTH_TOKEN}")
        req.add_header("User-Agent", "Mozilla/5.0")

        with urllib.request.urlopen(req, timeout=4.0) as resp:
            return resp.status in [200, 201, 204]
    except Exception as e:
        print(f"[!] Ошибка лайка: {e}", flush=True)
        return False