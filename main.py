import os
import datetime
import requests
import asyncio
import re
import logging
import subprocess
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
def get_available_model():
    """Получает список доступных моделей и возвращает первую подходящую."""
    try:
        models = client.models.list()
        model_names = [m.name for m in models]
        logger.info(f"Доступные модели: {model_names}")
        for name in model_names:
            if 'flash' in name or 'pro' in name:
                return name
        for name in model_names:
            if 'embed' not in name:
                return name
        return "gemini-1.5-flash"
    except Exception as e:
        logger.error(f"Не удалось получить список моделей: {e}")
        return "gemini-1.5-flash"

def ensure_complete(text):
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

def prepare_voice_text(full_text):
    """Извлекает короткую выжимку для озвучки, убирая имя 'Тимур'."""
    lines = full_text.split('\n')
    voice_lines = []
    found_start = False
    for line in lines:
        if not found_start:
            if "Тимур" in line and ("С вами" in line or "привет" in line.lower()):
                continue
            else:
                found_start = True
        voice_lines.append(line)
    voice_text = '\n'.join(voice_lines).strip()
    voice_text = re.sub(r'\bТимур\b', 'AvtoMaster24', voice_text)
    if len(voice_text) > 500:
        cut = voice_text[:500]
        last_period = cut.rfind('.')
        last_newline = cut.rfind('\n')
        cut_pos = max(last_period, last_newline)
        if cut_pos > 0:
            voice_text = voice_text[:cut_pos+1]
        else:
            voice_text = cut + '…'
    return voice_text

async def generate_post(service):
    """Генерирует пост, автоматически выбирая модель."""
    prompt = f"""
    {SYSTEM_PROMPT}
    Сегодняшний пост посвящён услуге: **{service}**.
    Расскажи, с какой проблемой сталкиваются водители, и как с помощью бота AvtoMaster24 они могут легко и быстро найти проверенного исполнителя (СТО, эвакуатор, шиномонтаж, автоэлектрика и т.д.).
    Используй конкретные преимущества: экономия времени, возможность сравнить цены и отзывы, удобный заказ в один клик.
    Закончи призывом перейти в бот.
    """
    model_name = get_available_model()
    logger.info(f"Используемая модель: {model_name}")
    max_retries = 2
    for attempt in range(max_retries):
        try:
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
            elif "404" in str(e):
                logger.warning(f"Модель {model_name} не найдена, пробуем другую")
                model_name = get_available_model()
            else:
                return f"Ошибка генерации: {str(e)}"
    return "Ошибка генерации: не удалось получить ответ после нескольких попыток."

def download_pexels_video(query):
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

def speed_up_audio(input_file, output_file, factor=1.25):
    """Ускоряет аудиофайл с помощью ffmpeg."""
    try:
        subprocess.run([
            'ffmpeg', '-i', input_file,
            '-filter:a', f'atempo={factor}',
            '-y', output_file
        ], check=True, capture_output=True)
        return True
    except Exception as e:
        logger.error(f"Ошибка ускорения аудио: {e}")
        return False

def create_short(voice_text, trend):
    temp_files = ["voice.mp3", "voice_sped.mp3", "stock.mp4", "short.mp4"]
    try:
        tts = gTTS(voice_text, lang='ru')
        tts.save("voice.mp3")
        
        # Ускоряем аудио
        if speed_up_audio("voice.mp3", "voice_sped.mp3", factor=1.25):
            audio_file = "voice_sped.mp3"
        else:
            logger.warning("Ускорение не удалось, использую исходное аудио")
            audio_file = "voice.mp3"

        if not download_pexels_video(trend):
            return None

        video = VideoFileClip("stock.mp4")
        if video.duration < 1:
            video.close()
            return None

        duration = min(45, video.duration)
        video = video.subclip(0, duration)

        audio = AudioFileClip(audio_file)
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
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

def split_long_text(text, max_len=4096):
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

    service = get_today_service()
    logger.info(f"Услуга дня: {service}")

    # 1. Генерация поста
    logger.info("1. Генерация поста...")
    post_text = await generate_post(service)

    if post_text.startswith("Ошибка генерации"):
        logger.error(f"Генерация не удалась: {post_text}")
        return

    logger.info(f"Сгенерировано {len(post_text)} символов")

    # 2. Очистка от чужих ссылок и добавление правильной
    post_text = re.sub(r'(https?://)?t\.me/\S+', '', post_text)
    final_link = BOT_LINK
    post_text += f"\n\nПерейди в бот: {final_link} 🚗"

    # 3. Отправка сообщения в Telegram
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

    # 4. Создание и отправка видео
    logger.info("3. Создание Shorts...")
    voice_text = prepare_voice_text(post_text)
    logger.info(f"Короткий текст для видео ({len(voice_text)} символов): {voice_text[:100]}...")
    short_path = create_short(voice_text, "car service useful tips")
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
        finally:
            if os.path.exists(short_path):
                os.remove(short_path)
    else:
        logger.warning("Видео не создано (пропущено)")

    logger.info("=== ВЫПОЛНЕНИЕ ЗАВЕРШЕНО ===")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
