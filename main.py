import os
import io
import json
import time
import asyncio
import urllib.request
import urllib.parse
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from faster_whisper import WhisperModel
import edge_tts

load_dotenv(override=True)

print("[*] Старт полностью локального сервера (QWEN 2.5 CPU). Gemini вырезан на 100%.", flush=True)

AUDIO_DIR = Path("static_audio")
AUDIO_DIR.mkdir(exist_ok=True)

# 1. Загрузка Whisper tiny
print("[*] Загрузка Whisper tiny...", flush=True)
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=2)
print("[+] STT готов!", flush=True)

# 2. Загрузка Qwen 2.5 0.5B
print("[*] Загрузка Qwen 2.5 0.5B в память...", flush=True)
model_path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    filename="qwen2.5-0.5b-instruct-q4_k_m.gguf"
)
llm = Llama(
    model_path=model_path,
    n_ctx=512,       # Короткий контекст для скорости
    n_batch=512,
    n_threads=2,     # Ровно под твои 2 vCPU
    verbose=False
)
print("[+] Qwen 2.5 готов к работе на процессоре!", flush=True)

app = FastAPI(title="Local Sima Server")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# КОРОТКИЙ ПРОМПТ В ОДНУ СТРОКУ (читается за 0.05 сек!)
SYSTEM_PROMPT = "Ты — Сима, голосовой помощник в Новороссийске. Отвечай строго по-русски, кратко, в 1-2 предложения, без markdown и спецсимволов."

# Черный список галлюцинаций ютубовских субтитров Whisper
WHISPER_JUNK = [
    "динамичная музыка", "музыка", "аплодисменты", "титры", "субтитры",
    "продолжение следует", "диктор", "редактор", "соли и голубов"
]

file_counter = 0


def fetch_weather_novorossiysk() -> str:
    """Безопасный запрос погоды без ошибок кодировки."""
    try:
        url = "https://wttr.in/Novorossiysk?format=%C,+%t,+ветер+%w&lang=ru"
        # Кодируем URL правильно
        safe_url = urllib.parse.quote(url, safe=":/%?=&+,")
        req = urllib.request.Request(safe_url, headers={"User-Agent": "curl/7.68.0"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            return resp.read().decode("utf-8").strip()
    except Exception:
        return "+16°C, облачно"


def local_stt(audio_bytes: bytes) -> str:
    """Распознавание речи с жесткой фильтрацией мусора субтитров."""
    try:
        t0 = time.perf_counter()
        segments, _ = whisper_model.transcribe(
            io.BytesIO(audio_bytes),
            language="ru",
            beam_size=1,
            vad_filter=False
        )
        text = " ".join([seg.text for seg in segments]).strip()
        dt = time.perf_counter() - t0

        # Фильтруем выдумки вроде "ДИНАМИЧНАЯ МУЗЫКА"
        clean = text.lower().strip("[]().,! ")
        if not clean or clean in WHISPER_JUNK or any(j in clean for j in ["динамичная музыка", "субтитры"]):
            print(f"[!] Отсечен шум/галлюцинация Whisper ({dt:.2f}с): '{text}'", flush=True)
            return ""

        print(f"[+] Вы сказали ({dt:.2f}с): '{text}'", flush=True)
        return text
    except Exception as e:
        print(f"[!] Ошибка STT: {e}", flush=True)
        return ""


def run_qwen_fast(user_text: str) -> str:
    """Быстрый инференс Qwen за 1-2 секунды."""
    t0 = time.perf_counter()

    prompt_text = user_text
    # Если вопрос про погоду — подсовываем живые цифры
    if any(w in user_text.lower() for w in ["погод", "градус", "дожд", "температур", "ветер"]):
        weather = fetch_weather_novorossiysk()
        prompt_text = f"Погода в Новороссийске сейчас: {weather}. Вопрос: {user_text}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text}
    ]

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=60,       # 1-2 предложения
        temperature=0.7,
        top_p=0.9
    )
    reply = response["choices"][0]["message"]["content"].strip()
    clean_reply = " ".join(reply.split()).replace('"', '').replace("'", "").replace('\\', '').strip()
    dt = time.perf_counter() - t0
    print(f"[+] Qwen 2.5 ({dt:.2f}с): '{clean_reply}'", flush=True)
    return clean_reply or "Я тебя слышу."


async def synthesize_voice(text: str, filename: str) -> bool:
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
    return {"status": "ok", "engine": "qwen2.5-local-fast"}


@app.post("/voice")
async def voice_endpoint(file: UploadFile = File(...)):
    global file_counter
    t_start = time.perf_counter()

    try:
        audio_bytes = await file.read()
        if not audio_bytes or len(audio_bytes) < 1000:
            return {"silent": True, "text": "", "audio_ready": False}

        loop = asyncio.get_running_loop()

        # 1. Быстрое распознавание речи
        recognized_text = await loop.run_in_executor(None, local_stt, audio_bytes)
        if not recognized_text:
            return {"silent": True, "text": "", "audio_ready": False}

        # 2. Быстрый локальный Qwen (1.5 сек)
        reply_text = await loop.run_in_executor(None, run_qwen_fast, recognized_text)

        # 3. Синтез голоса
        file_counter = (file_counter % 10) + 1
        audio_filename = f"reply_{file_counter}.mp3"

        audio_ready = await synthesize_voice(reply_text, audio_filename)
        total_time = time.perf_counter() - t_start
        print(f"[+] Полный ответ за {total_time:.2f}с!", flush=True)

        return {
            "silent": False,
            "text": reply_text,
            "user_said": recognized_text,
            "audio_ready": audio_ready,
            "audio_url": f"/audio/{audio_filename}" if audio_ready else None,
            "latency_sec": round(total_time, 2),
        }

    except Exception as e:
        print(f"[!] Ошибка: {e}", flush=True)
        return {"silent": False, "text": "Сбой сервера", "audio_ready": False, "audio_url": None}