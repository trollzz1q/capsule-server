import os
import io
import json
import time
import asyncio
import urllib.request
import urllib.parse
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from faster_whisper import WhisperModel
import edge_tts

import soundcloud_engine as sc

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
print(f"[*] Старт сервера (SoundCloud + Диалог). API: {bool(GEMINI_API_KEY)}", flush=True)

AUDIO_DIR = Path("static_audio")
AUDIO_DIR.mkdir(exist_ok=True)

print("[*] Загрузка Faster-Whisper...", flush=True)
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("[+] STT готов!", flush=True)

try:
    client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
except Exception as e:
    print(f"[!] Ошибка GenAI: {e}", flush=True)
    client = None

app = FastAPI(title="Capsule Assistant Backend")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

SYSTEM_INSTRUCTION = (
    "Ты — Сима, умный голосовой ассистент смарт-колонки в Новороссийске. "
    "Отвечай строго на русском языке! "
    "Если пользователь просто общается или задает вопрос — отвечай полно, интересно и живо (2-3 предложения). "
    "ВАЖНО: Если пользователь отдает команду (музыка, громкость, лайк), верни ответ СТРОГО в формате JSON без кавычек и markdown вокруг: "
    "1. Включить личный плейлист (мой плейлист, моя музыка, мои треки, любимое): "
    "{\"action\": \"music\", \"cmd\": \"my_playlist\", \"reply\": \"Включаю твой плейлист из Саундклауда\"} "
    "2. Включить трек/жанр (включи фонк, поставь рок, включи песню X): "
    "{\"action\": \"music\", \"cmd\": \"play\", \"query\": \"название_или_жанр\", \"reply\": \"Включаю трек\"} "
    "3. Переключение и лайки: "
    "{\"action\": \"music\", \"cmd\": \"next\", \"reply\": \"Следующий трек\"} "
    "{\"action\": \"music\", \"cmd\": \"prev\", \"reply\": \"Предыдущий трек\"} "
    "{\"action\": \"music\", \"cmd\": \"like\", \"reply\": \"Добавила в твои лайки\"} "
    "{\"action\": \"music\", \"cmd\": \"pause\", \"reply\": \"Музыка на паузе\"} "
    "{\"action\": \"music\", \"cmd\": \"resume\", \"reply\": \"Продолжаю воспроизведение\"} "
    "4. Громкость: {\"action\": \"volume\", \"value\": 70, \"reply\": \"Громкость 70\"} или {\"action\": \"volume\", \"delta\": \"+15\", \"reply\": \"Прибавила\"} "
    "Если это обычный вопрос без команд — отвечай просто чистым текстом без JSON."
)

CASCADE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash"
]

file_counter = 0
chat_history: list[dict] = []
last_interaction_time: float = 0.0


def cleanup_old_audio():
    now = time.time()
    for f in AUDIO_DIR.glob("reply_*.mp3"):
        try:
            if now - f.stat().st_mtime > 300:
                f.unlink(missing_ok=True)
        except Exception:
            pass


def fetch_weather_report(city: str = "Novorossiysk") -> str:
    try:
        encoded = urllib.parse.quote(city.strip())
        url = f"https://wttr.in/{encoded}?format=%l:+%C,+температура+%t,+ветер+%w&lang=ru"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = resp.read().decode("utf-8").strip()
            return data
    except Exception as e:
        return ""


def get_history_with_query(user_query: str) -> list[dict]:
    global chat_history, last_interaction_time
    now = time.time()

    if now - last_interaction_time > 60.0:
        chat_history = []
        print("[*] Контекст сброшен по таймауту (60с).", flush=True)

    last_interaction_time = now
    q_lower = user_query.lower()
    prompt_with_data = user_query

    if any(w in q_lower for w in ["погод", "градус", "дожд", "температур", "ветер", "тепло", "холод"]):
        target_city = "Novorossiysk"
        if "москв" in q_lower: target_city = "Moscow"
        elif "питер" in q_lower or "петербург" in q_lower: target_city = "Saint Petersburg"
        w_data = fetch_weather_report(target_city)
        if w_data:
            prompt_with_data = f"{user_query}\n[АКТУАЛЬНЫЕ МЕТЕОДАННЫЕ: {w_data}]"

    chat_history.append({"role": "user", "parts": [{"text": prompt_with_data}]})
    if len(chat_history) > 8:
        chat_history = chat_history[-8:]
    return list(chat_history)


def append_assistant_reply(reply_text: str):
    global chat_history
    chat_history.append({"role": "model", "parts": [{"text": reply_text}]})


def local_stt(audio_bytes: bytes) -> str:
    try:
        t0 = time.perf_counter()
        segments, _ = whisper_model.transcribe(
            io.BytesIO(audio_bytes),
            language="ru",
            beam_size=2,
            initial_prompt="Разговорная русская речь, голосовой помощник Сима, музыка, саундклауд, плейлист, лайк, трек, погода.",
            vad_filter=False
        )
        text = " ".join([seg.text for seg in segments]).strip()
        dt = time.perf_counter() - t0
        print(f"[+] STT ({dt:.2f}с): '{text}'", flush=True)
        return text
    except Exception as e:
        print(f"[!] Ошибка STT: {e}", flush=True)
        return ""


async def query_gemini_with_context(user_query: str) -> tuple[str, dict]:
    """Генерация с разбором намерений (чат vs музыка vs громкость)."""
    if not client or not GEMINI_API_KEY:
        return "В настройках сервера не указан API ключ.", {}

    full_context = get_history_with_query(user_query)

    for model_name in CASCADE_MODELS:
        t0 = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model_name,
                    contents=full_context,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        max_output_tokens=300,
                        temperature=0.5,
                    )
                ),
                timeout=12.0
            )

            raw_text = getattr(response, "text", "")
            if raw_text:
                clean_text = " ".join(raw_text.split()).strip()
                dt = time.perf_counter() - t0
                print(f"[+] {model_name} ({dt:.2f}с): '{clean_text}'", flush=True)

                cmd_payload = {}
                speech_text = clean_text

                # Проверяем, вернула ли модель команду
                if "{" in clean_text and "}" in clean_text:
                    try:
                        start_idx = clean_text.find("{")
                        end_idx = clean_text.rfind("}") + 1
                        parsed = json.loads(clean_text[start_idx:end_idx])
                        cmd_payload = parsed
                        speech_text = parsed.get("reply", "Готово")
                    except Exception:
                        pass

                speech_text = speech_text.replace('"', '').replace("'", "").replace('\\', '').strip()
                append_assistant_reply(speech_text)
                return speech_text, cmd_payload

        except Exception as e:
            print(f"[!] Сбой {model_name}: {e}. Переключаем...", flush=True)
            continue

    return "Сервер задумался, спроси еще разок.", {}


async def synthesize_voice(text: str, filename: str) -> bool:
    filepath = AUDIO_DIR / filename
    if filepath.exists():
        filepath.unlink(missing_ok=True)

    for voice in ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"]:
        try:
            communicate = edge_tts.Communicate(text, voice=voice, rate="+20%")
            await asyncio.wait_for(communicate.save(str(filepath)), timeout=15.0)
            if filepath.exists() and filepath.stat().st_size > 0:
                return True
        except Exception:
            continue
    return False


@app.get("/")
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "capsule-server", "soundcloud": bool(sc.SOUNDCLOUD_AUTH_TOKEN)}


@app.post("/voice")
async def voice_endpoint(file: UploadFile = File(...)):
    global file_counter
    t_start = time.perf_counter()

    try:
        audio_bytes = await file.read()
        if not audio_bytes or len(audio_bytes) < 1000:
            return {"silent": True, "text": "", "audio_ready": False}

        loop = asyncio.get_running_loop()
        recognized_text = await loop.run_in_executor(None, local_stt, audio_bytes)

        if not recognized_text or recognized_text in [".", "...", "!", "?"]:
            dt = round(time.perf_counter() - t_start, 2)
            print(f"[+] Чистая тишина ({dt}с).", flush=True)
            return {"silent": True, "text": "", "audio_ready": False, "latency_sec": dt}

        reply_text, cmd_payload = await query_gemini_with_context(recognized_text)

        # --- ОБРАБОТКА МУЗЫКАЛЬНЫХ КОМАНД ЧЕРЕЗ SOUNDCLOUD ---
        music_response = {}
        if cmd_payload.get("action") == "music":
            cmd = cmd_payload.get("cmd")
            if cmd == "my_playlist":
                res = await loop.run_in_executor(None, sc.load_my_liked_tracks)
                if res.get("success"):
                    music_response = {"action": "play_stream", "stream_url": res["stream_url"], "title": res["title"]}
                    reply_text = f"Включаю твой плейлист: {res['title']}"
            elif cmd == "play":
                query = cmd_payload.get("query", "music")
                res = await loop.run_in_executor(None, sc.search_and_get_track, query)
                if res.get("success"):
                    music_response = {"action": "play_stream", "stream_url": res["stream_url"], "title": res["title"]}
                    reply_text = f"Включаю: {res['title']}"
            elif cmd == "next":
                res = await loop.run_in_executor(None, sc.get_next_track)
                if res.get("success"):
                    music_response = {"action": "play_stream", "stream_url": res["stream_url"], "title": res["title"]}
                    reply_text = f"Следующий трек: {res['title']}"
            elif cmd == "prev":
                res = await loop.run_in_executor(None, sc.get_prev_track)
                if res.get("success"):
                    music_response = {"action": "play_stream", "stream_url": res["stream_url"], "title": res["title"]}
                    reply_text = f"Предыдущий трек: {res['title']}"
            elif cmd == "like":
                liked = await loop.run_in_executor(None, sc.like_current_playing_track)
                reply_text = "Добавила в твои любимые треки" if liked else "Не удалось поставить лайк"
            elif cmd == "pause":
                music_response = {"action": "music_pause"}
            elif cmd == "resume":
                music_response = {"action": "music_resume"}

        file_counter = (file_counter % 10) + 1
        audio_filename = f"reply_{file_counter}.mp3"

        cleanup_old_audio()
        audio_ready = await synthesize_voice(reply_text, audio_filename)
        total_time = time.perf_counter() - t_start

        safe_user_said = " ".join(recognized_text.split()).replace('"', '').replace("'", "")
        safe_reply = " ".join(reply_text.split()).replace('"', '').replace("'", "")

        return {
            "silent": False,
            "text": safe_reply,
            "user_said": safe_user_said,
            "command": cmd_payload,
            "music": music_response,
            "audio_ready": audio_ready,
            "audio_url": f"/audio/{audio_filename}" if audio_ready else None,
            "latency_sec": round(total_time, 2),
        }

    except Exception as e:
        print(f"[!] Ошибка voice_endpoint: {e}", flush=True)
        return {"silent": False, "text": "Сбой сервера", "audio_ready": False, "audio_url": None}