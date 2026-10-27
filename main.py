import os
import json
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
import edge_tts

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
print(f"[*] Старт сервера. Ключ API найден: {bool(GEMINI_API_KEY)} (длина: {len(GEMINI_API_KEY)})", flush=True)

AUDIO_DIR = Path("static_audio")
AUDIO_DIR.mkdir(exist_ok=True)

try:
    client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
except Exception as e:
    print(f"[!] Предупреждение GenAI: {e}", flush=True)
    client = None

app = FastAPI(title="Capsule Assistant Backend")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

SYSTEM_INSTRUCTION = (
    "Тебя зовут Сима. Ты — кастомный, независимый и умный голосовой ассистент нового поколения. "
    "Ты живешь внутри аппаратной колонки, но ты КАТЕГОРИЧЕСКИ НЕ МАРУСЯ и не имеешь к VK никакого отношения! "
    "Никогда не называй себя Марусей. Твой создатель и хозяин — разработчик, который перепрошил эту колонку "
    "на Alpine Linux и подарил тебе новую жизнь. Относись к нему с уважением и теплотой. "
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
    """Удаляет аудиофайлы старше 5 минут."""
    now = time.time()
    for f in AUDIO_DIR.glob("reply_*.mp3"):
        try:
            if now - f.stat().st_mtime > 300:
                f.unlink(missing_ok=True)
        except Exception:
            pass


async def query_gemini_multimodal(contents_payload: list) -> str:
    """Перебирает каскад моделей с таймаутом 10с на каждую."""
    if not client or not GEMINI_API_KEY:
        return "В настройках сервера не указан API ключ."

    full_contents = [SYSTEM_INSTRUCTION] + contents_payload

    for model_name in CASCADE_MODELS:
        t0 = time.perf_counter()
        try:
            print(f"[*] Запрос к модели: {model_name}...", flush=True)
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model_name,
                    contents=full_contents,
                    config=types.GenerateContentConfig(
                        max_output_tokens=75,
                        temperature=0.7,
                    )
                ),
                timeout=10.0
            )
            raw_text = getattr(response, "text", "")
            if raw_text and raw_text.strip():
                dt = time.perf_counter() - t0
                print(f"[+] Модель {model_name} ответила за {dt:.2f} сек: '{raw_text.strip()}'", flush=True)
                return raw_text.strip()
        except asyncio.TimeoutError:
            print(f"[!] Таймаут 10с на {model_name}. Переключаем...", flush=True)
            continue
        except Exception as e:
            print(f"[!] Сбой {model_name}: {repr(e)}. Переключаем...", flush=True)
            continue

    return "Сервер перегружен, спроси чуть позже."


async def synthesize_voice(text: str, filename: str) -> bool:
    """Синтез речи через Edge TTS с автоповтором при сбоях сети."""
    filepath = AUDIO_DIR / filename
    if filepath.exists():
        filepath.unlink(missing_ok=True)

    voices = ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"]

    # Даем до 3 попыток на случай кратковременного обрыва связи с Microsoft
    for attempt in range(1, 4):
        for voice in voices:
            try:
                communicate = edge_tts.Communicate(text, voice=voice, rate="+10%")
                await asyncio.wait_for(communicate.save(str(filepath)), timeout=7.0)
                if filepath.exists() and filepath.stat().st_size > 0:
                    return True
            except Exception as e:
                print(f"[!] Попытка {attempt} TTS ({voice}) не удалась: {e}", flush=True)
                await asyncio.sleep(0.3)
                continue

    return False


@app.get("/")
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "capsule-server", "cascade": CASCADE_MODELS}


@app.post("/voice")
async def voice_endpoint(file: UploadFile = File(...)):
    global file_counter
    t_start = time.perf_counter()

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            return {"text": "Пустой звук", "audio_url": None, "audio_ready": False}

        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
        user_prompt = "Пользователь обратился к тебе голосом. Ответь кратко, живо, без спецсимволов."

        reply_text = await query_gemini_multimodal([audio_part, user_prompt])

    except Exception as e:
        print(f"[!] Ошибка обработки аудио: {repr(e)}", flush=True)
        reply_text = "Не расслышала тебя, повтори пожалуйста."

    file_counter = (file_counter % 10) + 1
    audio_filename = f"reply_{file_counter}.mp3"

    cleanup_old_audio()
    audio_ready = await synthesize_voice(reply_text, audio_filename)
    total_time = time.perf_counter() - t_start

    return {
        "text": reply_text,
        "audio_ready": audio_ready,
        "audio_url": f"/audio/{audio_filename}" if audio_ready else None,
        "latency_sec": round(total_time, 2),
    }


@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    global file_counter
    await websocket.accept()
    print("[+] Клиент подключен по WebSocket!", flush=True)

    await websocket.send_text(json.dumps({
        "event": "connected",
        "message": "Сима на связи",
        "display": "blink"
    }))

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

            print(f"[*] Вопрос Симе: '{user_query}'", flush=True)

            await websocket.send_text(json.dumps({
                "event": "thinking",
                "display": "clockdisp --colon blink"
            }))

            t_start = time.perf_counter()
            reply_text = await query_gemini_multimodal([f"Вопрос пользователя: {user_query}"])

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
                "display": "clockdisp --anim wink",
                "latency_sec": round(total_time, 2)
            }))

    except WebSocketDisconnect:
        print("[-] Клиент отключился от WebSocket", flush=True)