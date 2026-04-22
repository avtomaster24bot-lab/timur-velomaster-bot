import os
import requests
import asyncio
import logging
# ... (остальные ваши импорты)
from config import BOT_LINK

# ... (ваши настройки, список SERVICES и SYSTEM_PROMPT_POST остаются без изменений) ...

async def generate_post(service: str) -> str:
    """
    Генерирует пост для канала через Grok (OpenRouter) API.
    """
    prompt = f"""
    {SYSTEM_PROMPT_POST}
    Сегодняшний пост посвящён услуге/проблеме: **{service}**.
    """

    # Получаем API-ключ из переменных окружения
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        logger.error("❌ Не найден OPENROUTER_API_KEY в переменных окружения!")
        # Возвращаем заранее заготовленный текст, чтобы скрипт не упал
        return f"Сам Самыч на связи! 🔧 Сегодня говорим про **{service}**. Не пренебрегайте обслуживанием велосипеда! Переходите в бот: {BOT_LINK}"

    # Адрес API OpenRouter
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
        # Необязательно, но рекомендуется для статистики на сайте OpenRouter
        "HTTP-Referer": "https://your-site.com",
        "X-Title": "Velomaster Bot"
    }
    data = {
        "model": "xai/grok-2-1212", # Указываем модель Grok
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 3072,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()  # Проверяем, не вернул ли сервер ошибку
        result = response.json()
        post_text = result['choices'][0]['message']['content'].strip()

        if not post_text:
            logger.error("OpenRouter вернул пустой ответ")
            return f"Сам Самыч на связи! 🔧 Сегодня говорим про **{service}**. Не пренебрегайте обслуживанием велосипеда! Переходите в бот: {BOT_LINK}"

        logger.info(f"Пост успешно сгенерирован через Grok, длина {len(post_text)} символов")
        return post_text

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Ошибка при запросе к OpenRouter API: {e}")
        # В случае ошибки также возвращаем fallback-текст
        return f"Сам Самыч на связи! 🔧 Сегодня говорим про **{service}**. Не пренебрегайте обслуживанием велосипеда! Переходите в бот: {BOT_LINK}"
