import os
import datetime
import asyncio
import re
import logging
from google import genai
from google.genai import types
from telegram import Bot
from config import BOT_LINK

# ==================== НАСТРОЙКА ЛОГИРОВАНИЯ ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== ПРОВЕРКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ ====================
required_env = ["GEMINI_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHANNEL_ID"]
missing = [var for var in required_env if not os.getenv(var)]
if missing:
    logger.error(f"❌ Отсутствуют обязательные переменные: {', '.join(missing)}")
    exit(1)

# ==================== ИНИЦИАЛИЗАЦИЯ КЛИЕНТОВ ====================
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
TELEGRAM_BOT = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

# ==================== СПИСОК УСЛУГ ====================
SERVICES = [
    "диагностика авто", "ремонт двигателя", "ремонт ходовой", "замена масла",
    "замена ГРМ", "тормозная система", "компьютерная диагностика", "подготовка к техосмотру",
    "эвакуация легковых авто", "эвакуация внедорожников", "перевозка авто",
    "доставка авто в другой город", "вытаскивание из кювета/грязи", "срочный вызов эвакуатора (SOS)",
    "замена шин", "балансировка колес", "ремонт проколов", "ремонт боковых порезов",
    "сезонная переобувка", "выездной шиномонтаж", "диагностика электрики", "ремонт проводки",
    "установка сигнализации", "установка магнитолы", "настройка электроники", "химчистка салона",
    "полировка кузова", "открытие авто без ключа", "запуск авто (сел аккумулятор)", "подвоз топлива",
    "покраска авто", "удаление вмятин", "рихтовка", "заправка кондиционера", "установка сабвуфера",
    "тонировка", "чип-тюнинг", "аренда авто",
]

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
def get_today_service():
    """Выбирает услугу на основе дня месяца."""
    return SERVICES[datetime.date.today().day % len(SERVICES)]

# -------------------- СИСТЕМНЫЙ ПРОМПТ --------------------
SYSTEM_PROMPT_POST = f"""
Ты — Тимур, помощник водителей Казахстана. Ты представляешь сервис AvtoMaster24 – это агрегатор, который помогает быстро найти и заказать любые автомобильные услуги: от эвакуатора до химчистки.
Твоя задача: в каждом посте рассказать о **конкретной услуге**, которую можно найти через бот. Не говори, что бот сам оказывает услуги. Наоборот: подчеркивай, что бот помогает сравнить варианты, выбрать надежного исполнителя, вызвать помощь, записаться на СТО и т.д.
Тон: дружелюбный, честный, живой, без воды.
Формат:
- Вступление (проблема водителя, которую решает эта услуга).
- Объяснение, как с помощью бота можно быстро решить проблему: найти проверенного мастера, вызвать эвакуатор, подобрать СТО и т.п.
- Призыв перейти в бот и воспользоваться удобным поиском.

Длина: 300–400 слов. Не обрывай текст на полуслове.
Сегодня: {datetime.date.today().strftime("%d %B %Y")}. Учитывай погоду и сезон в Казахстане.
"""

# -------------------- РАБОТА С GEMINI --------------------
def get_available_model(avoid_model=None):
    """Возвращает подходящую модель Gemini, исключая указанную."""
    try:
        models = client.models.list()
        model_names = [m.name for m in models]
        # Фильтруем только flash/pro модели
        candidates = [m for m in model_names if 'flash' in m or 'pro' in m]
        if not candidates:
            candidates = model_names
        # Исключаем модель, которая не работает
        if avoid_model and avoid_model in candidates:
            candidates = [m for m in candidates if m != avoid_model]
        logger.info(f"Доступные модели-кандидаты: {candidates[:5]}...")
        return candidates[0] if candidates else "gemini-2.0-flash"
    except Exception as e:
        logger.error(f"Не удалось получить список моделей: {e}")
        return "gemini-2.0-flash"

def ensure_complete(text):
    """Добавляет многоточие в конце, если предложение не завершено."""
    text = text.rstrip()
    if not text:
        return text
    if text[-1] in ('.', '!', '?', '…'):
        return text
    words = text.split()
    if words:
        text_without_last_word = ' '.join(words[:-1])
        return text_without_last_word.rstrip() + '…'
    return text + '…'

async def generate_with_retry(prompt, max_output_tokens=3072, temperature=0.7, is_long=True):
    """Генерирует текст с повторными попытками при ошибках."""
    model_name = get_available_model()
    logger.info(f"Генерация, модель: {model_name}")
    delays = [5, 10, 20, 40]
    for attempt in range(len(delays) + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_output_tokens,
                    temperature=temperature
                )
            )
            raw_text = response.text.strip()
            if raw_text and is_long:
                raw_text = ensure_complete(raw_text)
            return raw_text
        except Exception as e:
            logger.error(f"Ошибка генерации (попытка {attempt+1}): {e}")
            err_str = str(e)
            if "429" in err_str or "503" in err_str:
                if attempt < len(delays):
                    wait = delays[attempt]
                    logger.info(f"Лимит или недоступность, ждём {wait} сек...")
                    await asyncio.sleep(wait)
                    model_name = get_available_model()
                else:
                    return None
            elif "404" in err_str:
                logger.warning(f"Модель {model_name} не найдена, пробуем другую")
                model_name = get_available_model()
            else:
                return None
    return None

async def generate_post(service):
    """Генерирует длинный пост для канала."""
    prompt = f"""
    {SYSTEM_PROMPT_POST}
    Сегодняшний пост посвящён услуге: **{service}**.
    """
    result = await generate_with_retry(prompt, max_output_tokens=3072, temperature=0.7, is_long=True)
    if result is None:
        return "Ошибка генерации: не удалось получить ответ после всех попыток."
    return result

# -------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ TELEGRAM --------------------
def split_long_text(text, max_len=4096):
    """Разбивает длинный текст на части для Telegram."""
    if len(text) <= max_len:
        return [text]
    parts = []
    while text:
        if len(text) <= max_len:
            parts.append(text)
            break
        split_pos = text.rfind('\n', 0, max_len)
        if split_pos == -1:
            split_pos = text.rfind(' ', 0, max_len)
        if split_pos == -1:
            split_pos = max_len
        parts.append(text[:split_pos])
        text = text[split_pos:].lstrip()
    return parts

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
async def main():
    logger.info("=== НАЧАЛО ВЫПОЛНЕНИЯ ===")

    service = get_today_service()
    logger.info(f"Услуга дня: {service}")

    # 1. Генерация поста
    logger.info("1. Генерация поста...")
    post_text = await generate_post(service)
    if not post_text or post_text.startswith("Ошибка генерации"):
        logger.error(f"Генерация поста не удалась: {post_text}")
        return

    logger.info(f"Сгенерировано {len(post_text)} символов")

    # Удаляем чужие ссылки и добавляем правильную
    post_text = re.sub(r'(https?://)?t\.me/\S+', '', post_text)
    final_link = BOT_LINK
    post_text += f"\n\nПерейди в бот: {final_link} 🚗"

    # 2. Отправка поста в Telegram
    logger.info("2. Отправка сообщения в Telegram...")
    parts = split_long_text(post_text)
    for i, part in enumerate(parts):
        try:
            result = await TELEGRAM_BOT.send_message(
                chat_id=CHANNEL_ID,
                text=part,
                parse_mode=None,
                disable_web_page_preview=False
            )
            logger.info(f"✅ Часть {i+1}/{len(parts)} отправлена, message_id={result.message_id}")
        except Exception as e:
            logger.error(f"❌ Ошибка при отправке: {e}")

    logger.info("=== ВЫПОЛНЕНИЕ ЗАВЕРШЕНО ===")

if __name__ == "__main__":
    asyncio.run(main())
