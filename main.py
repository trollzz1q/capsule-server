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
import torch

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
print(f"[*] Старт сервера (Gemini + Local Silero TTS). API: {bool(GEMINI_API_KEY)}", flush=True)

AUDIO_DIR = Path("static_audio")
AUDIO_DIR.mkdir(exist_ok=True)

# 1. Загрузка STT (Whisper)
print("[*] Загрузка Faster-Whisper (base, int8)...", flush=True)
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("[+] STT готов!", flush=True)

# 2. Загрузка сверхбыстрого Silero TTS (V4 Russian, 0.08с)
print("[*] Загрузка локального Silero TTS (голос Kseniya)...", flush=True)
device = torch.device('cpu')
torch.set_num_threads(4)

silero_model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='ru',
    speaker='v4_ru'
)
silero_model.to(device)
print("[+] Silero TTS загружен и готов к мгновенному синтезу!", flush=True)

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
    "Давай полные, интересные, умные и развернутые ответы (2-4 предложения). "
    "Не используй markdown, звездочки, списки и таблицы, так как ответ озвучивается вслух."
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
    for f in AUDIO_DIR.glob("reply_*.wav"):
        try:
            if now - f.stat().st_mtime > 300:
                f.unlink(missing_ok=True)
        except Exception:
            pass


def local_stt(audio_bytes: bytes) -> str:
    """Распознавание речи."""
    try:
        t0 = time.perf_counter()
        segments, _ = whisper_model.transcribe(
            io.BytesIO(audio_bytes),
            language="ru",
            beam_size=2,
            initial_prompt="Разговорная русская речь, голосовой помощник Сима, погода, космос.",
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
    """Генерация ответа через каскад моделей."""
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
                        max_output_tokens=300,
                        temperature=0.7,
                    )
                ),
                timeout=10.0
            )

            answer_parts = []
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if getattr(part, "thought", False):
                        continue
                    if part.text:
                        answer_parts.append(part.text)

            raw_text = "".join(answer_parts).strip()
            if not raw_text:
                raw_text = getattr(response, "text", "").strip()

            if raw_text:
                clean_text = " ".join(raw_text.split()).replace('"', '').replace("'", "").replace('\\', '').strip()
                dt = time.perf_counter() - t0
                print(f"[+] {model_name} ({dt:.2f}с): '{clean_text}'", flush=True)
                return clean_text

        except Exception as e:
            print(f"[!] Сбой {model_name}: {e}. Переключаем...", flush=True)
            continue

    return "Сервер задумался, спроси еще разок."


def synthesize_silero(text: str, filename: str) -> bool:
    """Мгновенный локальный синтез речи (0.08с) в чистый WAV (24 кГц)."""
    try:
        t0 = time.perf_counter()
        filepath = AUDIO_DIR / filename

        # Генерация аудио тензора (голос kseniya, частота 24000 Гц)
        audio = silero_model.apply_tts(
            text=text,
            speaker='kseniya',
            sample_rate=24000
        )

        # Сохранение в WAV через torchaudio/io
        import soundfile as sf
        sf.write(str(filepath), audio.numpy(), 24000)

        dt = time.perf_counter() - t0
        print(f"[+] Silero TTS ({dt:.2f}с) сохранил {filename}", flush=True)
        return True
    except Exception as e:
        # Fallback на wave если soundfile не установлен
        try:
            import wave
            filepath = AUDIO_DIR / filename
            audio_int16 = (audio * 32767).clamp(-32768, 32767).to(torch.int16).numpy()
            with wave.open(str(filepath), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_int16.tobytes())
            return True
        except Exception as err:
            print(f"[!] Ошибка Silero TTS: {e} / {err}", flush=True)
            return False


@app.get("/")
@app.get("/health")
async def health_check():
    return {"status": "ok", "tts": "silero-local-kseniya"}


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

        file_counter = (file_counter % 10) + 1
        audio_filename = f"reply_{file_counter}.wav"

        cleanup_old_audio()
        # Мгновенный синтез в отдельном потоке
        audio_ready = await loop.run_in_executor(None, synthesize_silero, reply_text, audio_filename)
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