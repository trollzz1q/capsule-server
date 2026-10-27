import os
import io
import json
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from faster_whisper import WhisperModel
import edge_tts

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
print(f"[*] Старт сервера. Gemini API: {bool(GEMINI_API_KEY)}", flush=True)

AUDIO_DIR = Path("static_audio")
AUDIO_DIR.mkdir(exist_ok=True)

# 1. Загружаем локальный Whisper в память (модель base с int8 для максимальной скорости на CPU)
print("[*] Загрузка локального движка Faster-Whisper (модель base, int8)...", flush=True)
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("[+] Faster-Whisper готов к работе! Никаких лимитов и ключей.", flush=True)

try:
    client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
except Exception as e:
    print(f"[!] Ошибка GenAI: {e}", flush=True)
    client = None

app = FastAPI(title="Capsule Assistant Backend")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

SYSTEM_INSTRUCTION = (
    "Тебя зовут Сима. Ты — кастомный, независимый и умный голосовой ассистент нового поколения. "
    "Ты живешь внутри аппаратной смарт-колонки. Ты КАТЕГОРИЧЕСКИ НЕ МАРУСЯ и не имеешь к VK отношения! "
    "Твой создатель — разработчик, который перепрошил эту колонку и дал тебе жизнь. "
    "Отвечай кратко, емко, живым разговорным языком (1-2 коротких предложения). "
    "Категорически запрещено использовать списки, markdown, звёздочки, таблицы и эмодзи."
)

CASCADE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash"
]

file_counter = 0


def cleanup_old_audio():
    now = time.time()
    for f in AUDIO_DIR.glob("reply_*.mp3"):
        try:
            if now - f.stat().st_mtime > 300:
                f.unlink(missing_ok=True)
        except Exception:
            pass


def local_stt(audio_bytes: bytes) -> str:
    """
    Локальное распознавание речи со встроенным Silero VAD фильтром тишины.
    Работает прямо в оперативной памяти сервера без временных файлов.
    """
    try:
        t0 = time.perf_counter()
        # vad_filter=True мгновенно отсекает тишину и шумы вентиляторов
        segments, info = whisper_model.transcribe(
            io.BytesIO(audio_bytes),
            language="ru",
            beam_size=1,        # Максимальная скорость
            vad_filter=True,    # Встроенный детектор активности голоса
            vad_parameters=dict(min_silence_duration_ms=400)
        )
        text = " ".join([seg.text for seg in segments]).strip()
        dt = time.perf_counter() - t0
        print(f"[+] Faster-Whisper ({dt:.2f}с): '{text}'", flush=True)
        return text
    except Exception as e:
        print(f"[!] Ошибка Faster-Whisper: {e}", flush=True)
        return ""


async def query_gemini_text(user_query: str) -> str:
    """Генерация ответа по чистому тексту."""
    if not client or not GEMINI_API_KEY:
        return "В настройках сервера не указан API ключ."

    prompt = f"{SYSTEM_INSTRUCTION}\nВопрос пользователя: {user_query}"

    for model_name in CASCADE_MODELS:
        t0 = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=75,
                        temperature=0.7,
                    )
                ),
                timeout=8.0
            )
            raw_text = getattr(response, "text", "")
            if raw_text and raw_text.strip():
                dt = time.perf_counter() - t0
                print(f"[+] Gemini ({model_name}, {dt:.2f}с): '{raw_text.strip()}'", flush=True)
                return raw_text.strip()
        except Exception as e:
            print(f"[!] Сбой {model_name}: {e}. Переключаем...", flush=True)
            continue

    return "Сервер перегружен, спроси чуть позже."


async def synthesize_voice(text: str, filename: str) -> bool:
    filepath = AUDIO_DIR / filename
    if filepath.exists():
        filepath.unlink(missing_ok=True)

    for voice in ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"]:
        try:
            communicate = edge_tts.Communicate(text, voice=voice, rate="+10%")
            await asyncio.wait_for(communicate.save(str(filepath)), timeout=6.0)
            if filepath.exists() and filepath.stat().st_size > 0:
                return True
        except Exception:
            continue
    return False


@app.get("/")
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "capsule-server", "stt": "faster-whisper-local"}


@app.post("/voice")
async def voice_endpoint(file: UploadFile = File(...)):
    global file_counter
    t_start = time.perf_counter()

    try:
        audio_bytes = await file.read()
        if not audio_bytes or len(audio_bytes) < 1000:
            return {"silent": True, "text": "", "audio_ready": False}

        # 1. Локальное распознавание (выполняется в отдельном потоке, чтобы не блокировать event-loop)
        loop = asyncio.get_running_loop()
        recognized_text = await loop.run_in_executor(None, local_stt, audio_bytes)

        # 2. Если тишина (Silero VAD ничего не нашел) — мгновенный отбой!
        if not recognized_text:
            dt = round(time.perf_counter() - t_start, 2)
            print(f"[+] Зафиксирована тишина ({dt}с). Сима молчит.", flush=True)
            return {
                "silent": True,
                "text": "",
                "audio_ready": False,
                "audio_url": None,
                "latency_sec": dt
            }

        # 3. Если есть распознанный текст — опрос Gemini
        reply_text = await query_gemini_text(recognized_text)

    except Exception as e:
        print(f"[!] Ошибка voice_endpoint: {repr(e)}", flush=True)
        reply_text = "Не расслышала тебя, повтори пожалуйста."

    file_counter = (file_counter % 10) + 1
    audio_filename = f"reply_{file_counter}.mp3"

    cleanup_old_audio()
    audio_ready = await synthesize_voice(reply_text, audio_filename)
    total_time = time.perf_counter() - t_start

    return {
        "silent": False,
        "text": reply_text,
        "user_said": recognized_text,
        "audio_ready": audio_ready,
        "audio_url": f"/audio/{audio_filename}" if audio_ready else None,
        "latency_sec": round(total_time, 2),
    }


@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    global file_counter
    await websocket.accept()
    await websocket.send_text(json.dumps({"event": "connected", "message": "Сима на связи"}))

    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                payload = json.loads(raw_data)
                user_query = payload.get("text", "").strip()
            except json.JSONDecodeError:
                user_query = raw_data.strip()

            if not user_query:
                continue

            t_start = time.perf_counter()
            reply_text = await query_gemini_text(user_query)

            file_counter = (file_counter % 10) + 1
            audio_filename = f"reply_{file_counter}.mp3"

            cleanup_old_audio()
            audio_ready = await synthesize_voice(reply_text, audio_filename)
            total_time = time.perf_counter() - t_start

            await websocket.send_text(json.dumps({
                "event": "answer",
                "text": reply_text,
                "audio_ready": audio_ready,
                "audio_url": f"/audio/{audio_filename}" if audio_ready else None,
                "latency_sec": round(total_time, 2)
            }))
    except WebSocketDisconnect:
        pass