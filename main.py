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

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app = FastAPI(title="Capsule Assistant Backend")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

SYSTEM_INSTRUCTION = (
    "Ты — компактный умный голосовой ассистент по имени Сима внутри колонки VK Капсула Мини. "
    "Отвечай кратко, емко, живым разговорным языком (1-2 коротких предложения). "
    "Категорически запрещено использовать списки, markdown, звёздочки, таблицы и эмодзи, "
    "так как текст напрямую идёт в голосовой синтезатор."
)

MODEL_NAME = "gemini-3.8-flash"

# Глобальный чат для удержания памяти/контекста диалога
global_chat = None
file_counter = 0


def get_chat():
    """Возвращает или создает активный чат с Симой для сохранения памяти диалога."""
    global global_chat
    if client and global_chat is None:
        global_chat = client.aio.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                max_output_tokens=75,
                temperature=0.7,
            ),
        )
    return global_chat


def cleanup_old_audio():
    """Удаляет аудиофайлы старше 5 минут, чтобы диск сервера не забивался."""
    now = time.time()
    for f in AUDIO_DIR.glob("reply_*.mp3"):
        try:
            if now - f.stat().st_mtime > 300:
                f.unlink(missing_ok=True)
        except Exception:
            pass


async def synthesize_voice(text: str, filename: str) -> bool:
    """Синтезирует речь в MP3 через Microsoft Edge TTS."""
    filepath = AUDIO_DIR / filename
    if filepath.exists():
        filepath.unlink(missing_ok=True)

    for voice in ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"]:
        try:
            communicate = edge_tts.Communicate(text, voice=voice, rate="+10%")
            await communicate.save(str(filepath))
            return True
        except Exception as e:
            print(f"[!] Ошибка TTS ({voice}): {e}", flush=True)
            continue
    return False


@app.get("/")
async def root():
    return {"status": "ok", "service": "capsule-server", "model": MODEL_NAME}


@app.post("/reset")
async def reset_context():
    """Сбрасывает историю диалога."""
    global global_chat
    global_chat = None
    print("[*] Память диалога сброшена.", flush=True)
    return {"status": "cleared", "message": "Память Симы очищена"}


@app.post("/voice")
async def voice_endpoint(file: UploadFile = File(...)):
    """Принимает голос с микрофона Капсулы через curl, обрабатывает в Gemini и возвращает MP3."""
    global file_counter
    t_start = time.perf_counter()

    try:
        # Читаем аудиобайты прямо из потока без временных файлов на диске
        audio_bytes = await file.read()
        if not audio_bytes:
            return {"text": "Пустой звук", "audio_url": None}

        chat = get_chat()
        if not chat:
            reply_text = "В настройках сервера не указан API-ключ."
        else:
            # Передаем аудиофайл как инлайн-часть напрямую в модель
            audio_part = types.Part.from_bytes(
                data=audio_bytes,
                mime_type="audio/wav"
            )
            prompt = (
                "Пользователь задал вопрос голосом в аудиозаписи. "
                "Прослушай его и ответь кратко (1-2 предложения), живо, без спецсимволов."
            )

            response = await asyncio.wait_for(
                chat.send_message([audio_part, prompt]),
                timeout=10.0
            )
            reply_text = response.text.strip()
            print(f"[+] Сима услышала вопрос и ответила: '{reply_text}'", flush=True)

    except asyncio.TimeoutError:
        print("[!] Таймаут ответа Gemini по голосу", flush=True)
        reply_text = "Сервер перегружен, спроси чуть позже."
    except Exception as e:
        print(f"[!] Ошибка Gemini по голосу: {repr(e)}", flush=True)
        reply_text = "Не расслышала тебя, повтори пожалуйста."

    # Ротация файлов (reply_1.mp3 ... reply_10.mp3)
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
    """Подключение по WebSocket (для текстовых тестов и эмуляторов)."""
    global file_counter
    await websocket.accept()
    print("[+] Клиент подключен по WebSocket!", flush=True)

    chat = get_chat()

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

            try:
                if chat:
                    response = await asyncio.wait_for(
                        chat.send_message(user_query),
                        timeout=10.0
                    )
                    reply_text = response.text.strip()
                else:
                    reply_text = "В настройках сервера не указан API-ключ."
            except Exception as e:
                print(f"[!] Ошибка Gemini: {repr(e)}", flush=True)
                reply_text = "Сервер перегружен, спроси чуть позже."

            file_counter = (file_counter % 10) + 1
            audio_filename = f"reply_{file_counter}.mp3"

            cleanup_old_audio()
            audio_ready = await synthesize_voice(reply_text, audio_filename)
            total_time = time.perf_counter() - t_start

            print(f"[+] Итог: '{reply_text}' ({audio_filename}, время: {total_time:.2f}с)", flush=True)

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