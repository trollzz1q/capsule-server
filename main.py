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
print(f"[*] Старт сервера (Контекстный диалог). Gemini: {bool(GEMINI_API_KEY)}", flush=True)

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
    "Ты — Сима, умный, дружелюбный и эрудированный голосовой ассистент смарт-колонки. "
    "Отвечай строго на русском языке! "
    "Давай полные, интересные, умные и живые ответы (2-4 предложения). "
    "Не используй markdown, звездочки, списки и таблицы, так как ответ озвучивается вслух. "
    "Помни контекст текущего диалога с пользователем."
)

CASCADE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash"
]

file_counter = 0

# --- СИСТЕМА ПАМЯТИ ДИАЛОГА ---
chat_history: list[types.Content] = []
last_interaction_time: float = 0.0


def cleanup_old_audio():
    now = time.time()
    for f in AUDIO_DIR.glob("reply_*.mp3"):
        try:
            if now - f.stat().st_mtime > 300:
                f.unlink(missing_ok=True)
        except Exception:
            pass


def get_history_with_query(user_query: str) -> list[types.Content]:
    """Удерживает контекст диалога. Сбрасывает при паузе > 60 секунд."""
    global chat_history, last_interaction_time
    now = time.time()

    # Если прошло больше минуты — начинаем диалог с чистого листа
    if now - last_interaction_time > 60.0:
        chat_history = []
        print("[*] Контекст сброшен по таймауту неактивности (60с).", flush=True)

    last_interaction_time = now

    chat_history.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_query)])
    )
    # Держим в памяти до 8 последних реплик
    if len(chat_history) > 8:
        chat_history = chat_history[-8:]

    return list(chat_history)


def append_assistant_reply(reply_text: str):
    """Сохраняет ответ Симы в память."""
    global chat_history
    chat_history.append(
        types.Content(role="model", parts=[types.Part.from_text(text=reply_text)])
    )


def local_stt(audio_bytes: bytes) -> str:
    try:
        t0 = time.perf_counter()
        segments, _ = whisper_model.transcribe(
            io.BytesIO(audio_bytes),
            language="ru",
            beam_size=2,
            initial_prompt="Разговорная русская речь, голосовой помощник Сима, города, погода, музыка.",
            vad_filter=False
        )
        text = " ".join([seg.text for seg in segments]).strip()
        dt = time.perf_counter() - t0
        print(f"[+] STT ({dt:.2f}с): '{text}'", flush=True)
        return text
    except Exception as e:
        print(f"[!] Ошибка STT: {e}", flush=True)
        return ""


async def query_gemini_with_context(user_query: str) -> str:
    """Генерация с учетом контекста беседы."""
    if not client or not GEMINI_API_KEY:
        return "В настройках сервера не указан API ключ."

    full_context_contents = get_history_with_query(user_query)

    for model_name in CASCADE_MODELS:
        t0 = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model_name,
                    contents=full_context_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        max_output_tokens=300,
                        temperature=0.7,
                    )
                ),
                timeout=12.0
            )

            raw_text = getattr(response, "text", "")
            if raw_text:
                clean_text = " ".join(raw_text.split()).replace('"', '').replace("'", "").replace('\\', '').strip()
                dt = time.perf_counter() - t0
                print(f"[+] {model_name} ({dt:.2f}с): '{clean_text}'", flush=True)
                append_assistant_reply(clean_text)
                return clean_text

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
    return {"status": "ok", "service": "capsule-server", "context_turns": len(chat_history)}


@app.post("/reset")
async def reset_context():
    global chat_history
    chat_history = []
    return {"status": "ok", "message": "Контекст диалога очищен"}


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

        reply_text = await query_gemini_with_context(recognized_text)

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
            "audio_ready": audio_ready,
            "audio_url": f"/audio/{audio_filename}" if audio_ready else None,
            "latency_sec": round(total_time, 2),
        }

    except Exception as e:
        print(f"[!] Ошибка: {e}", flush=True)
        return {"silent": False, "text": "Сбой сервера", "audio_ready": False, "audio_url": None}