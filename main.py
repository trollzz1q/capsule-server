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
print(f"[*] Старт сервера. Gemini: {bool(GEMINI_API_KEY)}", flush=True)

AUDIO_DIR = Path("static_audio")
AUDIO_DIR.mkdir(exist_ok=True)

# Модель 'tiny' работает в 3 раза быстрее 'base' на CPU хостинга
print("[*] Загрузка ультрабыстрого Faster-Whisper (tiny, int8)...", flush=True)
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
print("[+] Движок распознавания готов!", flush=True)

try:
    client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
except Exception as e:
    print(f"[!] Ошибка GenAI: {e}", flush=True)
    client = None

app = FastAPI(title="Capsule Assistant Backend")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# Умный, живой характер без занудства
SYSTEM_INSTRUCTION = (
    "Твое имя — Сима. Ты — остроумный, эрудированный и живой голосовой помощник в смарт-колонке. "
    "Ты не Маруся и не Алиса, ты создана независимым инженером. "
    "Отвечай интересно, емко, умным разговорным языком (2-3 хороших предложения). "
    "Не строй из себя робота — проявляй характер, где уместно шути или подмечай детали. "
    "Запрещено использовать списки, markdown, звездочки и таблицы, так как ответ пойдет в голос."
)

# Первой идет самая умная модель
CASCADE_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash"
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
    """Быстрое распознавание с фильтром тишины."""
    try:
        t0 = time.perf_counter()
        segments, _ = whisper_model.transcribe(
            io.BytesIO(audio_bytes),
            language="ru",
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300)
        )
        text = " ".join([seg.text for seg in segments]).strip()
        dt = time.perf_counter() - t0
        print(f"[+] STT ({dt:.2f}с): '{text}'", flush=True)
        return text
    except Exception as e:
        print(f"[!] Ошибка STT: {e}", flush=True)
        return ""


async def query_gemini_text(user_query: str) -> str:
    """Генерация умного ответа."""
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
                        max_output_tokens=150,  # Даем модели развернуться мыслью
                        temperature=0.8,        # Больше креатива и живости
                    )
                ),
                timeout=7.0
            )
            raw_text = getattr(response, "text", "")
            if raw_text and raw_text.strip():
                dt = time.perf_counter() - t0
                print(f"[+] {model_name} ({dt:.2f}с): '{raw_text.strip()}'", flush=True)
                return raw_text.strip()
        except Exception as e:
            print(f"[!] Сбой {model_name}: {e}. Переключаем...", flush=True)
            continue

    return "Сервер задумался, спроси еще разок."


async def synthesize_voice(text: str, filename: str) -> bool:
    filepath = AUDIO_DIR / filename
    if filepath.exists():
        filepath.unlink(missing_ok=True)

    for voice in ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"]:
        try:
            communicate = edge_tts.Communicate(text, voice=voice, rate="+15%")  # +15% скорости речи (звучит бодрее!)
            await asyncio.wait_for(communicate.save(str(filepath)), timeout=5.0)
            if filepath.exists() and filepath.stat().st_size > 0:
                return True
        except Exception:
            continue
    return False


@app.get("/")
@app.get("/health")
async def health_check():
    return {"status": "ok", "model": CASCADE_MODELS[0]}


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

        if not recognized_text:
            dt = round(time.perf_counter() - t_start, 2)
            print(f"[+] Тишина ({dt}с).", flush=True)
            return {"silent": True, "text": "", "audio_ready": False, "latency_sec": dt}

        reply_text = await query_gemini_text(recognized_text)

    except Exception as e:
        print(f"[!] Ошибка: {e}", flush=True)
        reply_text = "Что-то пошло не так, повтори пожалуйста."

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
    await websocket.accept()
    await websocket.send_text(json.dumps({"event": "connected", "message": "Сима на связи"}))

    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                user_query = json.loads(raw_data).get("text", "").strip()
            except Exception:
                user_query = raw_data.strip()

            if not user_query:
                continue

            reply_text = await query_gemini_text(user_query)
            audio_filename = f"reply_ws.mp3"
            audio_ready = await synthesize_voice(reply_text, audio_filename)

            await websocket.send_text(json.dumps({
                "event": "answer",
                "text": reply_text,
                "audio_ready": audio_ready,
                "audio_url": f"/audio/{audio_filename}" if audio_ready else None
            }))
    except WebSocketDisconnect:
        pass