import os
import io
import json
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from faster_whisper import WhisperModel
import edge_tts

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
print(f"[*] Старт быстрого сервера. API: {bool(GEMINI_API_KEY)}", flush=True)

AUDIO_DIR = Path("static_audio")
AUDIO_DIR.mkdir(exist_ok=True)

# Модель tiny на 2 потоках CPU отрабатывает за 0.25-0.35 сек!
print("[*] Загрузка ультрабыстрого Whisper tiny...", flush=True)
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=2)
print("[+] STT готов!", flush=True)

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app = FastAPI(title="Capsule Fast Backend")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# КОРОТКИЙ ПРОМПТ В 1 СТРОКУ
SYSTEM_INSTRUCTION = "Отвечай строго на русском языке, кратко, в 1-2 предложения, живым разговорным языком, без markdown, звездочек и спецсимволов."

file_counter = 0


def local_stt(audio_bytes: bytes) -> str:
    """Мгновенное распознавание без фейковых подсказок."""
    try:
        t0 = time.perf_counter()
        segments, _ = whisper_model.transcribe(
            io.BytesIO(audio_bytes),
            language="ru",
            beam_size=1,        # 1 поток = максимальная скорость
            vad_filter=False
        )
        text = " ".join([seg.text for seg in segments]).strip()
        dt = time.perf_counter() - t0
        print(f"[+] STT ({dt:.2f}с): '{text}'", flush=True)
        return text
    except Exception as e:
        print(f"[!] Ошибка STT: {e}", flush=True)
        return ""


async def query_gemini_fast(user_query: str) -> str:
    """Быстрый запрос к Gemini Lite (0.3-0.4 сек)."""
    if not client or not GEMINI_API_KEY:
        return "В настройках сервера не указан API ключ."

    t0 = time.perf_counter()
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=user_query,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    max_output_tokens=100,  # Короткий емкий ответ
                    temperature=0.7,
                )
            ),
            timeout=6.0
        )
        raw_text = getattr(response, "text", "")
        clean_text = " ".join(raw_text.split()).replace('"', '').replace("'", "").replace('\\', '').strip()
        dt = time.perf_counter() - t0
        print(f"[+] Gemini ({dt:.2f}с): '{clean_text}'", flush=True)
        return clean_text or "Я тебя услышала."
    except Exception as e:
        print(f"[!] Ошибка Gemini: {e}", flush=True)
        return "Сервер задумался, спроси еще разок."


async def synthesize_voice(text: str, filename: str) -> bool:
    """Быстрый синтез речи (+25% к скорости)."""
    filepath = AUDIO_DIR / filename
    try:
        communicate = edge_tts.Communicate(text, voice="ru-RU-SvetlanaNeural", rate="+25%")
        await asyncio.wait_for(communicate.save(str(filepath)), timeout=6.0)
        return filepath.exists() and filepath.stat().st_size > 0
    except Exception as e:
        print(f"[!] Ошибка TTS: {e}", flush=True)
        return False


@app.get("/health")
async def health_check():
    return {"status": "ok", "speed": "fast"}


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
            return {"silent": True, "text": "", "audio_ready": False}

        reply_text = await query_gemini_fast(recognized_text)

        file_counter = (file_counter % 10) + 1
        audio_filename = f"reply_{file_counter}.mp3"

        audio_ready = await synthesize_voice(reply_text, audio_filename)
        total_time = time.perf_counter() - t_start
        print(f"[+] ИТОГ за {total_time:.2f} сек: '{reply_text}'", flush=True)

        return {
            "silent": False,
            "text": reply_text,
            "user_said": recognized_text,
            "audio_ready": audio_ready,
            "audio_url": f"/audio/{audio_filename}" if audio_ready else None,
            "latency_sec": round(total_time, 2),
        }

    except Exception as e:
        print(f"[!] Ошибка voice: {e}", flush=True)
        return {"silent": False, "text": "Сбой сервера", "audio_ready": False, "audio_url": None}