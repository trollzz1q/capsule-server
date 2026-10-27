import os
import io
import wave
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

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
print(f"[*] Старт сервера с нативным голосом Gemini (Lyra). API: {bool(GEMINI_API_KEY)}", flush=True)

AUDIO_DIR = Path("static_audio")
AUDIO_DIR.mkdir(exist_ok=True)

print("[*] Загрузка Faster-Whisper (base, int8)...", flush=True)
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
    "Говори живым, естественным разговорным языком, развернуто и интересно (2-4 предложения). "
    "Не используй markdown, звездочки, списки и таблицы, так как твой ответ напрямую воспроизводится вслух."
)

# Модели Gemini с поддержкой прямого аудиовыхода
CASCADE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-3.8-flash"
]

file_counter = 0


def cleanup_old_audio():
    now = time.time()
    for f in AUDIO_DIR.glob("reply_*.wav"):
        try:
            if now - f.stat().st_mtime > 300:
                f.unlink(missing_ok=True)
        except Exception:
            pass


def local_stt(audio_bytes: bytes) -> str:
    """Локальное распознавание речи."""
    try:
        t0 = time.perf_counter()
        segments, _ = whisper_model.transcribe(
            io.BytesIO(audio_bytes),
            language="ru",
            beam_size=2,
            initial_prompt="Разговорная русская речь, голосовой помощник Сима, умная колонка Сима.",
            vad_filter=False
        )
        text = " ".join([seg.text for seg in segments]).strip()
        dt = time.perf_counter() - t0
        print(f"[+] STT ({dt:.2f}с): '{text}'", flush=True)
        return text
    except Exception as e:
        print(f"[!] Ошибка STT: {e}", flush=True)
        return ""


def save_as_wav(raw_data: bytes, target_path: Path, sample_rate: int = 24000):
    """Сохраняет аудиоданные (готовый WAV или 24кГц PCM) в стандартный WAV-файл."""
    if raw_data.startswith(b"RIFF"):
        target_path.write_bytes(raw_data)
    else:
        # Упаковываем сырые PCM-байты в чистый WAV (24 кГц, 16-бит моно)
        with wave.open(str(target_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(raw_data)


async def query_gemini_native_voice(user_query: str, target_filename: str) -> tuple[bool, str]:
    """Запрашивает ответ у Gemini с прямым синтезом голоса Lyra."""
    if not client or not GEMINI_API_KEY:
        return False, "В настройках сервера не указан API ключ."

    filepath = AUDIO_DIR / target_filename
    if filepath.exists():
        filepath.unlink(missing_ok=True)

    for model_name in CASCADE_MODELS:
        t0 = time.perf_counter()
        try:
            print(f"[*] Запрос к {model_name} (голос Lyra)...", flush=True)
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model_name,
                    contents=user_query,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_modalities=["AUDIO"],  # ПРЯМОЙ ЗВУК ИЗ GEMINI
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name="Lyra"   # Голос Lyra из Gemini Live
                                )
                            )
                        ),
                        temperature=0.7,
                    )
                ),
                timeout=15.0
            )

            # Ищем аудиочасть в ответе
            audio_data = None
            text_transcript = ""

            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.data:
                        audio_data = part.inline_data.data
                    if part.text:
                        text_transcript = part.text

            if audio_data:
                save_as_wav(audio_data, filepath)
                dt = time.perf_counter() - t0
                print(f"[+] {model_name} вернул звук Lyra за {dt:.2f}с! (Размер: {len(audio_data)} байт)", flush=True)
                return True, text_transcript or "Ответ озвучен голосом Lyra"

        except Exception as e:
            print(f"[!] Сбой {model_name}: {e}. Переключаем...", flush=True)
            continue

    return False, "Сервер задумался, спроси еще разок."


@app.get("/")
@app.get("/health")
async def health_check():
    return {"status": "ok", "voice": "gemini-native-lyra"}


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

        file_counter = (file_counter % 10) + 1
        audio_filename = f"reply_{file_counter}.wav"

        # Прямой синтез голоса Lyra из нейросети
        audio_ready, reply_text = await query_gemini_native_voice(recognized_text, audio_filename)
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