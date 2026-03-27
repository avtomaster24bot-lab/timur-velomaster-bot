import os
import datetime
import requests
import asyncio
import re
import logging
from google import genai
from google.genai import types
from telegram import Bot
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip
from config import BOT_LINK

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Инициализация клиентов
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

TELEGRAM_BOT = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
PEXELS_KEY = os.getenv("PEXELS_API_KEY")
DEV_CHAT_ID = os.getenv("DEV_CHAT_ID")  # опционально

# ----------------------------------------------------------------------
# Список услуг
SERVICES = [
    "диагностика авто",
    "ремонт двигателя",
    "ремонт ходовой",
    "замена масла",
    "замена ГРМ",
    "тормозная система",
    "компьютерная диагностика",
    "подготовка к техосмотру",
    "эвакуация легковых авто",
    "эвакуация внедорожников",
    "перевозка авто",
    "доставка авто в другой город",
    "вытаскивание из кювета/грязи",
    "срочный вызов эвакуатора (SOS)",
    "замена шин",
    "балансировка колес",
    "ремонт проколов",
    "ремонт боковых порезов",
    "сезонная переобувка",
    "выездной шиномонтаж",
    "диагностика электрики",
    "ремонт проводки",
    "установка сигнализации",
    "установка магнитолы",
    "настройка электроники",
    "химчистка салона",
    "полировка кузова",
    "открытие авто без ключа",
    "запуск авто (сел аккумулятор)",
    "подвоз топлива",
    "покраска авто",
    "удаление вмятин",
    "рихтовка",
    "заправка кондиционера",
    "установка сабвуфера",
    "тонировка",
    "чип-тюнинг",
    "аренда авто",
]

# ----------------------------------------------------------------------
def get_today_service():
    """Выбирает услугу дня на основе текущей даты."""
    return SERVICES[datetime.date.today().day % len(SERVICES)]

SYSTEM_PROMPT = f"""
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

# ----------------------------------------------------------------------
async def notify_dev(message):
    """Отправляет сообщение разработчику, если указан DEV_CHAT_ID."""
    if DEV_CHAT_ID:
        try:
            await TELEGRAM_BOT.send_message(chat_id=DEV_CHAT_ID, text=message[:4096])
        except Exception as e:
            logger.error(f"Не удалось отправить уведомление разработчику: {e}")

def ensure_complete(text):
    """Обеспечивает завершённость текста (знак препинания в конце)."""
    text = text.rstrip()
    if not text:
        return text
    last_char = text[-1]
    if last_char in ('.', '!', '?', '…'):
        return text
    words = text.split()
    if words:
        text_without_last_word = ' '.join(words[:-1])
        return text_without_last_word.rstrip() + '…'
    return text + '…'

def select_model():
    """Динамически выбирает подходящую модель Gemini."""
    try:
        models = client.models.list()
        model_list = list(models)
        logger.info(f"Найдено моделей: {len(model_list)}")
        for m in model_list[:5]:
            logger.info(f"  {m.name}")
        # Ищем модели с flash или pro, исключая embed
        candidates = [m.name for m in model_list if "flash" in m.name and "exp" not in m.name and "preview" not in m.name]
        if candidates:
            selected = candidates[0]
            logger.info(f"Выбрана модель: {selected}")
            return selected
        for m in model_list:
            if "embed" not in m.name:
                logger.info(f"Выбрана модель: {m.name}")
                return m.name
        return "gemini-1.5-flash"
    except Exception as e:
        logger.error(f"Ошибка получения списка моделей: {e}")
        return "gemini-1.5-flash"

async def generate_post(service):
    """Генерирует пост с помощью Gemini, выбирая модель динамически."""
    prompt = f"""
    {SYSTEM_PROMPT}
    Сегодняшний пост посвящён услуге: **{service}**.
    Расскажи, с какой проблемой сталкиваются водители, и как с помощью бота AvtoMaster24 они могут легко и быстро найти проверенного исполнителя (СТО, эвакуатор, шиномонтаж, автоэлектрика и т.д.).
    Используй конкретные преимущества: экономия времени, возможность сравнить цены и отзывы, удобный заказ в один клик.
    Закончи призывом перейти в бот.
    """
    model_name = select_model()
    max_retries = 2
    for attempt in range(max_retries):
        try:
            logger.info(f"Попытка генерации с моделью {model_name}, попытка {attempt+1}")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=3072,
                    temperature=0.7
                )
            )
            raw_text = response.text.strip()
            if raw_text:
                raw_text = ensure_complete(raw_text)
            return raw_text
        except Exception as e:
            logger.error(f"Ошибка генерации (попытка {attempt+1}): {e}")
            if "429" in str(e):
                wait = 60
                logger.info(f"Превышена квота, ждём {wait} сек...")
                await asyncio.sleep(wait)
            else:
                await notify_dev(f"❌ Ошибка генерации Gemini: {e}")
                return f"Ошибка генерации: {str(e)}"
    return "Ошибка генерации: не удалось получить ответ после нескольких попыток."

def download_pexels_video(query):
    """Скачивает одно видео с Pexels по запросу."""
    if not PEXELS_KEY:
        return False
    headers = {"Authorization": PEXELS_KEY}
    try:
        r = requests.get(
            f"https://api.pexels.com/videos/search?query={query}+car+Kazakhstan&per_page=1",
            headers=headers,
            timeout=15
        )
        r.raise_for_status()
        data = r.json()
        if data.get("videos"):
            video_url = data["videos"][0]["video_files"][0]["link"]
            with open("stock.mp4", "wb") as f:
                f.write(requests.get(video_url, timeout=30).content)
            return True
    except Exception as e:
        logger.warning(f"Не удалось скачать видео с Pexels: {e}")
        return False
    return False

def create_short(text, trend):
    """Создаёт короткое видео (Shorts) из текста и фонового видео."""
    temp_files = ["voice.mp3", "stock.mp4", "short.mp4"]
    try:
        # Озвучка
        tts = gTTS(text[:500], lang='ru')
        tts.save("voice.mp3")

        # Видеофон
        if not download_pexels_video(trend):
            logger.warning("Не удалось скачать видео, Shorts не будет создан")
            return None

        video = VideoFileClip("stock.mp4")
        if video.duration < 1:
            logger.warning("Скачанное видео слишком короткое")
            video.close()
            return None

        duration = min(45, video.duration)
        video = video.subclip(0, duration)

        audio = AudioFileClip("voice.mp3")
        if audio.duration > duration:
            audio = audio.subclip(0, duration)

        final = video.set_audio(audio)
        final.write_videofile("short.mp4", fps=24, codec="libx264", audio_codec="aac")
        final.close()
        video.close()
        audio.close()

        if os.path.exists("short.mp4"):
            return "short.mp4"
        else:
            return None
    except Exception as e:
        logger.error(f"Ошибка при создании Shorts: {e}")
        return None
    finally:
        # Удаляем временные файлы (кроме short.mp4, его удалим в основном коде)
        for f in temp_files:
            if f != "short.mp4" and os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

def split_long_text(text, max_len=4096):
    """Разбивает длинный текст на части по max_len символов."""
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

# ----------------------------------------------------------------------
async def main():
    logger.info("=== НАЧАЛО ВЫПОЛНЕНИЯ ===")

    # Уведомление разработчику о старте (опционально)
    await notify_dev("🚀 Бот Тимур запущен и начинает генерацию поста.")

    service = get_today_service()
    logger.info(f"Услуга дня: {service}")

    # 1. Генерация поста
    logger.info("1. Генерация поста...")
    post_text = await generate_post(service)

    # Проверка на ошибку генерации
    if post_text.startswith("Ошибка генерации"):
        await notify_dev(f"❌ Бот не смог сгенерировать пост: {post_text}")
        logger.error(f"Генерация не удалась: {post_text}")
        return

    logger.info(f"Сгенерировано {len(post_text)} символов")

    # 2. Очистка от чужих ссылок и добавление правильной
    post_text = re.sub(r'(https?://)?t\.me/\S+', '', post_text)
    final_link = BOT_LINK
    post_text += f"\n\nПерейди в бот: {final_link} 🚗"

    # 3. Отправка сообщения в Telegram (с разбивкой, если длинное)
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
            await notify_dev(f"❌ Ошибка отправки сообщения: {e}")

    # 4. Создание и отправка видео
    logger.info("3. Создание Shorts...")
    short_path = create_short(post_text, "car service useful tips")
    if short_path and os.path.exists(short_path):
        logger.info("Видео создано, отправляем...")
        try:
            with open(short_path, "rb") as video_file:
                await TELEGRAM_BOT.send_video(
                    chat_id=CHANNEL_ID,
                    video=video_file,
                    caption=f"AvtoMaster24 — полезные сервисы для автовладельцев 🚗\n{final_link}"
                )
            logger.info("✅ Видео отправлено")
        except Exception as e:
            logger.error(f"❌ Ошибка при отправке видео: {e}")
            await notify_dev(f"❌ Ошибка отправки видео: {e}")
        finally:
            if os.path.exists(short_path):
                os.remove(short_path)
    else:
        logger.warning("Видео не создано (пропущено)")

    logger.info("=== ВЫПОЛНЕНИЕ ЗАВЕРШЕНО ===")
    await notify_dev("✅ Работа бота Тимур завершена успешно.")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
