import os
import datetime
import asyncio
import re
import logging
from google import genai
from google.genai import types
from telegram import Bot
from config import BOT_LINK   # ссылка на вашего Telegram-бота

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

# ==================== СПИСОК ВЕЛОУСЛУГ (30+ позиций) ====================
SERVICES = [
    "диагностика велосипеда",
    "ремонт переключателей скоростей",
    "японские велосипеды",
    "настройка заднего переключателя",
    "ремонт переднего переключателя",
    "замена тросиков и рубашек",
    "ремонт тормозов (V-brake, дисковые)",
    "прокачка гидравлических тормозов",
    "Велосипеды из японии",
    "замена тормозных колодок",
    "ремонт каретки",
    "замена кареточного узла",
    "ремонт педалей",
    "замена педалей",
    "Мамачари, самые популярные велосипеды в Японии",
    "ремонт цепи",
    "замена цепи",
    "ремонт задней втулки",
    "ремонт передней втулки",
    "замена втулок",
    "ремонт кассеты/трещотки",
    "замена звезд",
    "Гибридные велосипеды, бум популярности в Японии",
    "ремонт рулевой колонки",
    "замена рулевой",
    "ремонт обода колеса",
    "замена спиц",
    "сборка колеса",
    "ремонт покрышек и камер",
    "велосипеды на планетарной втулке",
    "замена покрышек",
    "установка бескамерных шин",
    "ремонт амортизационной вилки",
    "обслуживание амортизационной вилки",
    "ремонт заднего амортизатора",
    "компьютерная диагностика велосипеда",
    "Японский оригинал или китайский ремейк, сравниваем",
    "сезонное обслуживание",
    "чистка и смазка цепи",
    "регулировка седла и руля",
    "подбор посадки",
    "установка багажника",
    "Надёжность японских велосипедов, почему старый б/у лучше новых (базарных)",
    "установка крыльев",
    "установка фар и фонарей",
    "установка велокомпьютера",
    "ремонт электронных переключателей",
    "ремонт электровелосипедов",
]

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
def get_today_service():
    """Выбирает услугу на основе дня месяца."""
    return SERVICES[datetime.date.today().day % len(SERVICES)]

# -------------------- СИСТЕМНЫЙ ПРОМПТ (ВЕЛОМАСТЕР САМ САМЫЧ) --------------------
SYSTEM_PROMPT_POST = f"""
Ты — Сам Самыч, веломастер с 30-летним опытом. Ты знаешь о велосипедах всё: от советских "Уралов" до современных карбоновых шоссейников и электровелосипедов.
Ты ведёшь Telegram-канал, где даёшь полезные советы, рассказываешь о ремонте и обслуживании велосипедов, помогаешь велосипедистам избежать типичных ошибок.
В этом посте ты расскажешь о **конкретной услуге или проблеме**, связанной с велосипедом. Объясни, почему это важно, какие последствия бывают, если игнорировать, и как твой бот помогает решить проблему (записаться на ремонт, получить консультацию, вызвать мастера на дом и т.д.).
Тон: дружелюбный, с лёгким юмором, но профессиональный. Используй живой разговорный русский язык, иногда обращайся к читателю как к "велодругу".
Структура:
1. Приветствие и короткая история из практики (или типичная проблема).
2. Подробное объяснение, что это за услуга/поломка и почему она важна.
3. Как бот Сам Самыча может помочь (быстрая запись, выезд мастера, консультация).
4. Призыв перейти в бот и записаться.

Длина: 250–350 слов. Не обрывай на полуслове. Пиши законченный пост.
Сегодня: {datetime.date.today().strftime("%d %B %Y")}. Учитывай сезон (весна, лето, осень, зима) и погоду в регионе (по умолчанию Россия/СНГ).
"""

# -------------------- РАБОТА С GEMINI --------------------
def get_available_model(avoid_model=None):
    """Возвращает подходящую модель Gemini, исключая указанную."""
    try:
        models = client.models.list()
        model_names = [m.name for m in models]
        candidates = [m for m in model_names if 'flash' in m or 'pro' in m]
        if not candidates:
            candidates = model_names
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
    """Генерирует текст с повторными попытками и переключением моделей."""
    current_model = get_available_model()
    logger.info(f"Пробуем модель: {current_model}")
    delays = [5, 10, 20, 40]
    for attempt in range(len(delays) + 1):
        try:
            response = client.models.generate_content(
                model=current_model,
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
            logger.error(f"Ошибка генерации (модель {current_model}, попытка {attempt+1}): {e}")
            err_str = str(e)
            if "503" in err_str or "429" in err_str:
                if attempt < len(delays):
                    wait = delays[attempt]
                    logger.info(f"Модель {current_model} недоступна, ждём {wait} сек...")
                    await asyncio.sleep(wait)
                    # Переключаемся на другую модель
                    new_model = get_available_model(avoid_model=current_model)
                    if new_model != current_model:
                        logger.info(f"Переключаемся на модель: {new_model}")
                        current_model = new_model
                else:
                    logger.error("Исчерпаны все попытки для всех моделей")
                    return None
            elif "404" in err_str:
                logger.warning(f"Модель {current_model} не найдена, пробуем другую")
                current_model = get_available_model(avoid_model=current_model)
            else:
                logger.error(f"Неизвестная ошибка: {err_str}")
                return None
    return None

async def generate_post(service):
    """Генерирует пост для канала от имени Сам Самыча."""
    prompt = f"""
    {SYSTEM_PROMPT_POST}
    Сегодняшний пост посвящён услуге/проблеме: **{service}**.
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

    # Удаляем чужие ссылки и добавляем правильную ссылку на бота
    post_text = re.sub(r'(https?://)?t\.me/\S+', '', post_text)
    final_link = BOT_LINK
    post_text += f"\n\n🔧 Переходи в бот «Веломастер Сам Самыч»: {final_link} 🚲"

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
