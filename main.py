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
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from faster_whisper import WhisperModel
import edge_tts

import soundcloud_engine as sc

load_dotenv(override=True)

print("[*] Старт полностью автономного сервера (Qwen 2.5 0.5B Local).", flush=True)

AUDIO_DIR = Path("static_audio")
AUDIO_DIR.mkdir(exist_ok=True)

# 1. Загрузка STT Whisper
print("[*] Инициализация Faster-Whisper (base, int8)...", flush=True)
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("[+] STT готов!", flush=True)

# 2. Скачивание и инициализация локальной модели Qwen 2.5 0.5B
print("[*] Подготовка модели Qwen 2.5 0.5B Instruct GGUF...", flush=True)
model_path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    filename="qwen2.5-0.5b-instruct-q4_k_m.gguf"
)
print(f"[+] Модель скачана: {model_path}", flush=True)

print("[*] Загрузка Qwen 2.5 в оперативную память (2 потока CPU)...", flush=True)
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=2,
    verbose=False
)
print("[+] Qwen 2.5 0.5B успешно загружен в память! Никаких внешних API.", flush=True)

app = FastAPI(title="Capsule Assistant Local Backend")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

SYSTEM_INSTRUCTION = (
    "Ты — Сима, умный голосовой ассистент смарт-колонки. Твой хозяин живет в городе Новороссийск. "
    "Отвечай строго на русском языке кратко, живо и по делу (2-3 предложения). "
    "Если пользователь отдает команду управления (музыка или громкость), верни строго валидный JSON без лишнего текста вокруг: "
    'Громкость: {"action": "volume", "value": 70, "reply": "Громкость 70"} или {"action": "volume", "delta": "+15", "reply": "Прибавила"} '
    'Включить плейлист: {"action": "music", "cmd": "my_playlist", "reply": "Включаю твой плейлист"} '
    'Включить трек/жанр: {"action": "music", "cmd": "play", "query": "название_или_жанр", "reply": "Включаю"} '
    'Управление: {"action": "music", "cmd": "next|prev|like|pause|resume", "reply": "Готово"} '
    "Если это обычный вопрос или диалог — отвечай просто обычным текстом без JSON, кавычек и markdown."
)

file_counter = 0
dialogue_history = []
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
            print(f"[+] Сводка погоды ({city}): '{data}'", flush=True)
            return data
    except Exception as e:
        print(f"[!] Ошибка wttr.in: {e}", flush=True)
        return ""


def build_chat_messages(user_query: str) -> list[dict]:
    global dialogue_history, last_interaction_time
    now = time.time()

    if now - last_interaction_time > 60.0:
        dialogue_history = []
        print("[*] Контекст сброшен по таймауту неактивности (60с).", flush=True)

    last_interaction_time = now
    q_lower = user_query.lower()

    # RAG: подтягиваем живую погоду при вопросе о погоде
    weather_injection = ""
    if any(w in q_lower for w in ["погод", "градус", "дожд", "температур", "ветер", "тепло", "холод"]):
        target_city = "Novorossiysk"
        if "москв" in q_lower: target_city = "Moscow"
        elif "питер" in q_lower or "петербург" in q_lower: target_city = "Saint Petersburg"
        w_data = fetch_weather_report(target_city)
        if w_data:
            weather_injection = f"\n[ФАКТИЧЕСКИЕ ДАННЫЕ О ПОГОДЕ: {w_data}]"

    final_user_prompt = f"{user_query}{weather_injection}"

    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    for role, text in dialogue_history[-6:]:
        messages.append({"role": role, "content": text})

    messages.append({"role": "user", "content": final_user_prompt})
    dialogue_history.append(("user", user_query))

    return messages


def record_assistant_reply(reply_text: str):
    global dialogue_history
    dialogue_history.append(("assistant", reply_text))
    if len(dialogue_history) > 8:
        dialogue_history = dialogue_history[-8:]


def local_stt(audio_bytes: bytes) -> str:
    try:
        t0 = time.perf_counter()
        segments, _ = whisper_model.transcribe(
            io.BytesIO(audio_bytes),
            language="ru",
            beam_size=2,
            initial_prompt="Разговорная русская речь, голосовой помощник Сима, трек, включи, музыка, саундклауд, погода.",
            vad_filter=False
        )
        text = " ".join([seg.text for seg in segments]).strip()
        dt = time.perf_counter() - t0
        print(f"[+] STT ({dt:.2f}с): '{text}'", flush=True)
        return text
    except Exception as e:
        print(f"[!] Ошибка STT: {e}", flush=True)
        return ""


def run_qwen_inference(messages: list[dict]) -> tuple[str, dict]:
    """Синхронный запуск инференса Qwen 2.5 на процессоре."""
    t0 = time.perf_counter()
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=250,
        temperature=0.6,
        top_p=0.9
    )
    raw_text = response["choices"][0]["message"]["content"].strip()
    dt = time.perf_counter() - t0
    print(f"[+] Qwen 2.5 0.5B ({dt:.2f}с): '{raw_text}'", flush=True)

    cmd_payload = {}
    speech_text = raw_text

    # Парсим JSON-команду, если модель её вернула
    if "{" in raw_text and "}" in raw_text:
        try:
            start_idx = raw_text.find("{")
            end_idx = raw_text.rfind("}") + 1
            cmd_payload = json.loads(raw_text[start_idx:end_idx])
            speech_text = cmd_payload.get("reply", "Готово")
        except Exception:
            pass

    speech_text = " ".join(speech_text.split()).replace('"', '').replace("'", "").replace('\\', '').strip()
    record_assistant_reply(speech_text)
    return speech_text, cmd_payload


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
        except Exception as e:
            print(f"[!] Ошибка озвучки ({voice}): {e}", flush=True)
            continue
    return False


@app.get("/")
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "capsule-server", "engine": "qwen2.5-0.5b-local"}


@app.post("/voice")
async def voice_endpoint(file: UploadFile = File(...)):
    global file_counter
    t_start = time.perf_counter()

    try:
        audio_bytes = await file.read()
        if not audio_bytes or len(audio_bytes) < 1000:
            return {"silent": True, "text": "", "audio_ready": False}

        loop = asyncio.get_running_loop()

        # 1. Распознавание речи
        recognized_text = await loop.run_in_executor(None, local_stt, audio_bytes)

        if not recognized_text or recognized_text in [".", "...", "!", "?"]:
            dt = round(time.perf_counter() - t_start, 2)
            print(f"[+] Чистая тишина ({dt}с).", flush=True)
            return {"silent": True, "text": "", "audio_ready": False, "latency_sec": dt}

        # 2. Инференс локального Qwen 2.5 в пуле потоков
        messages = build_chat_messages(recognized_text)
        reply_text, cmd_payload = await loop.run_in_executor(None, run_qwen_inference, messages)

        # 3. Обработка SoundCloud
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

        # 4. Синтез речи
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