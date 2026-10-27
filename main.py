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

print("[*] Загрузка Faster-Whisper (base, int8)...", flush=True)
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("[+] Движок распознавания готов!", flush=True)

try:
    client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
except Exception as e:
    print(f"[!] Ошибка GenAI: {e}", flush=True)
    client = None

app = FastAPI(title="Capsule Assistant Backend")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# ЧИСТЫЙ МИНИМАЛЬНЫЙ ПРОМПТ
SYSTEM_INSTRUCTION = (
    "Отвечай всегда строго на русском языке. "
    "Давай полные, интересные, грамотные и развернутые ответы для смарт-колонки. "
    "Запрещено использовать markdown, списки, звездочки, кавычки и таблицы, так как текст идет в синтезатор речи."
)

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
    """Распознавание речи через Whisper Base."""
    try:
        t0 = time.perf_counter()
        segments, _ = whisper_model.transcribe(
            io.BytesIO(audio_bytes),
            language="ru",
            beam_size=2,
            initial_prompt="Разговорная русская речь, вопросы голосовому ассистенту Сима.",
            vad_filter=False
        )
        text = " ".join([seg.text for seg in segments]).strip()
        dt = time.perf_counter() - t0
        print(f"[+] STT ({dt:.2f}с): '{text}'", flush=True)
        return text
    except Exception as e:
        print(f"[!] Ошибка STT: {e}", flush=True)
        return ""


async def query_gemini_text(user_query: str) -> str:
    """Генерация с автоматическим мышлением и фильтрацией английских мыслей."""
    if not client or not GEMINI_API_KEY:
        return "В настройках сервера не указан API ключ."

    for model_name in CASCADE_MODELS:
        t0 = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model_name,
                    contents=user_query,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        max_output_tokens=1000,  # Сняли лимит — дает полные ответы!
                        temperature=0.7,
                    )
                ),
                timeout=12.0
            )

            # ОТДЕЛЯЕМ РУССКИЙ ОТВЕТ ОТ АНГЛИЙСКИХ МЫСЛЕЙ (THOUGHTS)
            answer_parts = []
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    # Если блок помечен как мысль (thought) — пропускаем!
                    if getattr(part, "thought", False):
                        continue
                    if part.text:
                        answer_parts.append(part.text)

            raw_text = "".join(answer_parts).strip()

            # Если частей не нашлось — берем fallback text
            if not raw_text:
                raw_text = getattr(response, "text", "").strip()

            if raw_text:
                # Очищаем спецсимволы
                clean_text = " ".join(raw_text.split())
                clean_text = clean_text.replace('"', '').replace("'", "").replace('\\', '').strip()
                dt = time.perf_counter() - t0
                print(f"[+] {model_name} ({dt:.2f}с): '{clean_text}'", flush=True)
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
            communicate = edge_tts.Communicate(text, voice=voice, rate="+15%")
            await asyncio.wait_for(communicate.save(str(filepath)), timeout=8.0)
            if filepath.exists() and filepath.stat().st_size > 0:
                return True
        except Exception:
            continue
    return False


@app.get("/")
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "capsule-server", "stt": "faster-whisper-base"}


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

        reply_text = await query_gemini_text(recognized_text)

    except Exception as e:
        print(f"[!] Ошибка: {e}", flush=True)
        reply_text = "Что-то пошло не так, повтори пожалуйста."

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
            audio_filename = "reply_ws.mp3"
            audio_ready = await synthesize_voice(reply_text, audio_filename)

            await websocket.send_text(json.dumps({
                "event": "answer",
                "text": reply_text,
                "audio_ready": audio_ready,
                "audio_url": f"/audio/{audio_filename}" if audio_ready else None
            }))
    except WebSocketDisconnect:
        pass