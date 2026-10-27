import os
import asyncio
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY", "").strip()
print(f"[*] Проверка ключа (длина: {len(api_key)}, первые 6 символов: {api_key[:6]}...)")

async def test():
    try:
        client = genai.Client(api_key=api_key)
        print("[*] Отправляем запрос к gemini-2.0-flash...")
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents="Скажи одно слово: тест"
        )
        print(f"[+] УСПЕХ! Ответ модели: {response.text}")
    except Exception as e:
        print(f"\n[!] ВОТ ТОЧНАЯ ОШИБКА GOOGLE API:\n{repr(e)}\n")

if __name__ == "__main__":
    asyncio.run(test())