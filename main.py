import os
import json
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from google import genai
import edge_tts

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PORT = int(os.getenv("PORT", 8080))

# Папка для отдачи аудиофайлов колонке
AUDIO_DIR = Path("static_audio")
AUDIO_DIR.mkdir(exist_ok=True)

# Инициализируем клиент Google GenAI
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Capsule Assistant Backend")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

SYSTEM_PROMPT = (
    "Ты — компактный умный голосовой ассистент внутри колонки VK Капсула Мини. "
    "Отвечай кратко, емко, живым разговорным языком (1-2 коротких предложения). "
    "Категорически запрещено использовать списки, markdown, звёздочки, таблицы и эмодзи, "
    "так как текст напрямую идёт в голосовой синтезатор."
)

FAST_MODELS = [
    "gemini-3.8-flash",
]

async def ask_gemini(user_text: str) -> str:
    global client
    if not GEMINI_API_KEY:
        print("[!] GEMINI_API_KEY не задан", flush=True)
        return "В настройках сервера не указан API-ключ."

    if client is None:
        client = genai.Client(api_key=GEMINI_API_KEY)

    full_prompt = f"{SYSTEM_PROMPT}\nВопрос пользователя: {user_text}"

    for model_name in FAST_MODELS:
        t0 = time.perf_counter()
        try:
            print(f"[*] Запрос к {model_name}...", flush=True)
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                ),
                timeout=7.0
            )
            dt = time.perf_counter() - t0
            if response and response.text:
                print(f"[+] {model_name} ответила за {dt:.2f} сек", flush=True)
                return response.text.strip()
        except Exception as e:
            print(f"[!] Сбой вызова {model_name}: {repr(e)}", flush=True)
            continue

    return "Сервер перегружен, спроси чуть позже."


async def synthesize_voice(text: str, filename: str) -> bool:
    """Генерация аудио через Edge TTS с запасным голосом."""
    filepath = AUDIO_DIR / filename
    for voice in ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"]:
        try:
            communicate = edge_tts.Communicate(text, voice=voice)
            await communicate.save(str(filepath))
            return True
        except Exception as e:
            print(f"[!] Ошибка TTS ({voice}): {e}")
            continue
    return False


@app.get("/")
async def health_check():
    return {"status": "ok", "service": "capsule-server", "models": FAST_MODELS}


@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    await websocket.accept()
    print("[+] Колонка подключилась по WebSocket!")

    # Отправляем приветствие и команду для экрана
    await websocket.send_text(json.dumps({
        "event": "connected",
        "message": "Капсула на связи",
        "display": "blink"
    }, ensure_ascii=False))

    try:
        while True:
            raw_data = await websocket.receive_text()
            print(f"[>] Получено: {raw_data}")

            try:
                payload = json.loads(raw_data)
                user_query = payload.get("text", "")
            except json.JSONDecodeError:
                user_query = raw_data

            if not user_query.strip():
                continue

            # Уведомляем колонку, что запрос принят
            await websocket.send_text(json.dumps({
                "event": "thinking",
                "display": "clockdisp --colon blink"
            }, ensure_ascii=False))

            # 1. Запрос к Gemini
            t_total_start = time.perf_counter()
            reply_text = await ask_gemini(user_query)
            print(f"[<] Gemini: {reply_text}")

            # 2. Синтез голоса в уникальный файл
            audio_filename = f"reply_{int(time.time())}.mp3"
            t_tts = time.perf_counter()
            audio_ready = await synthesize_voice(reply_text, audio_filename)
            print(f"[+] TTS готов за {time.perf_counter() - t_tts:.2f} сек")

            # 3. Отправляем ответ и ссылку на аудиофайл
            total_time = time.perf_counter() - t_total_start
            await websocket.send_text(json.dumps({
                "event": "answer",
                "text": reply_text,
                "audio_ready": audio_ready,
                "audio_url": f"/audio/{audio_filename}" if audio_ready else None,
                "display": "clockdisp --anim wink",
                "latency_sec": round(total_time, 2)
            }, ensure_ascii=False))

    except WebSocketDisconnect:
        print("[-] Колонка отключилась")
    except Exception as e:
        print(f"[-] Ошибка в WS-цикле: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT)