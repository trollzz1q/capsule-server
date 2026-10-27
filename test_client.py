import asyncio
import json
import websockets

SERVER_URI = "ws://31.77.128.129:40000/ws"

async def test():
    print(f"[*] Подключаемся к {SERVER_URI}...")
    async with websockets.connect(SERVER_URI) as ws:
        # Приветственное сообщение
        greeting = await ws.recv()
        print(f"[<] Ответ сервера: {greeting}")

        # Отправляем тестовый вопрос
        query = {"text": "Сколько лететь до Марса?"}
        print(f"[>] Отправляем: {query}")
        await ws.send(json.dumps(query))

        # Слушаем ответы (thinking и answer)
        while True:
            response = await ws.recv()
            data = json.loads(response)
            print(f"[<] Получено: {response}")
            if data.get("event") == "answer":
                print(f"\n[OK] Готово! Задержка: {data.get('latency_sec')} сек")
                print(f"Текст: {data.get('text')}")
                print(f"Аудио: {data.get('audio_url')}")
                break

if __name__ == "__main__":
    asyncio.run(test())