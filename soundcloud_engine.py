import os
import json
import urllib.request
import urllib.parse
from dotenv import load_dotenv
import yt_dlp

load_dotenv(override=True)

SOUNDCLOUD_AUTH_TOKEN = os.getenv("SOUNDCLOUD_AUTH_TOKEN", "").strip()

playback_queue = []
current_queue_index = 0
current_playing_track = None
my_user_id = None

ydl_opts = {
    'format': 'bestaudio/best',
    'noplaylist': True,
    'quiet': True,
    'no_warnings': True,
    'default_search': 'scsearch',
    'source_address': '0.0.0.0'
}


def get_my_user_id() -> str:
    """Получает ID вашего профиля SoundCloud через токен из .env."""
    global my_user_id
    if my_user_id:
        return my_user_id
    if not SOUNDCLOUD_AUTH_TOKEN:
        print("[!] SOUNDCLOUD_AUTH_TOKEN не найден в .env", flush=True)
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
            print(f"[+] Авторизован в SoundCloud как: {data.get('username')} (ID: {my_user_id})", flush=True)
            return my_user_id
    except Exception as e:
        print(f"[!] Ошибка авторизации SoundCloud: {e}", flush=True)
        return ""


def resolve_stream_url(web_url: str) -> str:
    """Достает прямой MP3-поток из ссылки на трек через yt-dlp."""
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(web_url, download=False)
            return info.get("url", "")
    except Exception as e:
        print(f"[!] Ошибка извлечения потока yt-dlp: {e}", flush=True)
        return ""


def search_and_get_track(query: str) -> dict:
    """Ищет трек по запросу (жанр, исполнитель, песня)."""
    global current_playing_track, playback_queue, current_queue_index
    try:
        search_query = f"scsearch5:{query}"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_query, download=False)
            entries = info.get("entries", [])
            if not entries:
                return {"success": False, "error": "Ничего не найдено"}

            track = entries[0]
            stream_url = track.get("url")
            title = track.get("title", query)
            artist = track.get("uploader", "SoundCloud")
            track_id = str(track.get("id"))

            current_playing_track = {
                "id": track_id,
                "title": f"{artist} — {title}",
                "stream_url": stream_url
            }

            playback_queue = [
                {"id": str(t.get("id")), "title": f"{t.get('uploader')} — {t.get('title')}",
                 "web_url": t.get("webpage_url")}
                for t in entries
            ]
            current_queue_index = 0

            return {
                "success": True,
                "title": current_playing_track["title"],
                "stream_url": stream_url,
                "track_id": track_id
            }
    except Exception as e:
        print(f"[!] Ошибка поиска SoundCloud: {e}", flush=True)
        return {"success": False, "error": str(e)}


def load_my_liked_tracks() -> dict:
    """Загружает ваш личный плейлист 'Liked Tracks' из SoundCloud."""
    global playback_queue, current_queue_index, current_playing_track
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
                return {"success": False, "error": "Твой плейлист лайков пуст"}

            playback_queue = queue
            current_queue_index = 0

            first = playback_queue[0]
            stream = resolve_stream_url(first["web_url"])
            current_playing_track = {
                "id": first["id"],
                "title": first["title"],
                "stream_url": stream
            }

            return {
                "success": True,
                "title": first["title"],
                "stream_url": stream,
                "track_id": first["id"]
            }
    except Exception as e:
        print(f"[!] Ошибка загрузки плейлиста лайков: {e}", flush=True)
        return {"success": False, "error": str(e)}


def get_next_track() -> dict:
    """Переключает на следующий трек в очереди."""
    global current_queue_index, current_playing_track
    if not playback_queue:
        return {"success": False, "error": "Очередь воспроизведения пуста"}

    current_queue_index = (current_queue_index + 1) % len(playback_queue)
    item = playback_queue[current_queue_index]

    stream = item.get("stream_url") or resolve_stream_url(item.get("web_url", ""))
    current_playing_track = {
        "id": item["id"],
        "title": item["title"],
        "stream_url": stream
    }

    return {
        "success": True,
        "title": item["title"],
        "stream_url": stream,
        "track_id": item["id"]
    }


def get_prev_track() -> dict:
    """Переключает на предыдущий трек в очереди."""
    global current_queue_index, current_playing_track
    if not playback_queue:
        return {"success": False, "error": "Очередь воспроизведения пуста"}

    current_queue_index = (current_queue_index - 1) % len(playback_queue)
    item = playback_queue[current_queue_index]

    stream = item.get("stream_url") or resolve_stream_url(item.get("web_url", ""))
    current_playing_track = {
        "id": item["id"],
        "title": item["title"],
        "stream_url": stream
    }

    return {
        "success": True,
        "title": item["title"],
        "stream_url": stream,
        "track_id": item["id"]
    }


def like_current_playing_track() -> bool:
    """Ставит лайк текущему играющему треку в вашем аккаунте."""
    if not current_playing_track or not current_playing_track.get("id"):
        print("[!] Нет играющего трека для лайка", flush=True)
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
            success = resp.status in [200, 201, 204]
            print(f"[+] Лайк треку {track_id} успешно поставлен: {success}", flush=True)
            return success
    except Exception as e:
        print(f"[!] Ошибка постановки лайка: {e}", flush=True)
        return False