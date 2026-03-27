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
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from pydub import AudioSegment
from config import BOT_LINK

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

TELEGRAM_BOT = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
PEXELS_KEY = os.getenv("PEXELS_API_KEY")

# ----------------------------------------------------------------------
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

# Системный промпт для основного поста
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

# Промпт для короткого видео-текста
SYSTEM_PROMPT_VOICE = """
Ты — голос сервиса AvtoMaster24. Задача: создать короткий текст для озвучки видео (30–50 слов).
Текст должен состоять из трёх частей:
1. Название услуги.
2. Проблема, которую решает услуга.
3. Призыв: «Найди проверенных исполнителей в боте AvtoMaster24 и закажи помощь за минуту!»
Не используй имя «Тимур», говори от лица сервиса.
Текст должен быть законченным, заканчиваться восклицательным знаком.
Пример: «Нужно открыть авто без ключа? Захлопнулась дверь, а ключи остались внутри. Найди проверенных исполнителей в боте AvtoMaster24 и закажи помощь за минуту!»
"""

# ----------------------------------------------------------------------
def get_available_model():
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

async def generate_with_retry(prompt, max_output_tokens=3072, temperature=0.7, is_long=True):
    """Генерирует текст, повторяя попытки при ошибках квоты (429) и недоступности (503)."""
    model_name = get_available_model()
    logger.info(f"Генерация, модель: {model_name}")
    # Экспоненциальная задержка: 5, 10, 20, 40 секунд
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
                    # Меняем модель, чтобы сбросить квоту на конкретную модель
                    model_name = get_available_model()
                else:
                    return None
            elif "404" in err_str:
                logger.warning(f"Модель {model_name} не найдена, пробуем другую")
                model_name = get_available_model()
            else:
                # Неизвестная ошибка – возвращаем текст ошибки
                return f"Ошибка генерации: {err_str}"
    return None

async def generate_post(service):
    """Генерирует длинный пост."""
    prompt = f"""
    {SYSTEM_PROMPT_POST}
    Сегодняшний пост посвящён услуге: **{service}**.
    """
    result = await generate_with_retry(prompt, max_output_tokens=3072, temperature=0.7, is_long=True)
    if result is None:
        return "Ошибка генерации: не удалось получить ответ после всех попыток."
    return result

async def generate_voice_text(service):
    """Генерирует короткий текст для видео с обязательным призывом."""
    prompt = f"""
    {SYSTEM_PROMPT_VOICE}
    Услуга: **{service}**.
    """
    result = await generate_with_retry(prompt, max_output_tokens=300, temperature=0.5, is_long=False)
    if result is None:
        # Запасной вариант
        return f"Нужна {service}? AvtoMaster24 поможет найти мастера рядом. Переходи в бот и решай проблему за минуту!"
    # Пост-обработка
    result = result.strip('"“”')
    if result.endswith('…'):
        result = result[:-1] + '.'
    # Проверка длины
    if len(result) < 80:
        logger.warning(f"Текст слишком короткий ({len(result)} символов). Использую запасной.")
        return f"Нужна {service}? AvtoMaster24 поможет найти мастера рядом. Переходи в бот и решай проблему за минуту!"
    return result

def download_pexels_video(query):
    if not PEXELS_KEY:
        logger.warning("PEXELS_API_KEY не задан")
        return False
    headers = {"Authorization": PEXELS_KEY}
    try:
        url = f"https://api.pexels.com/videos/search?query={query}+car+Kazakhstan&per_page=1"
        logger.info(f"Запрос к Pexels: {url}")
        r = requests.get(url, headers=headers, timeout=15)
        logger.info(f"Статус ответа Pexels: {r.status_code}")
        if r.status_code != 200:
            logger.error(f"Ошибка Pexels: {r.text}")
            return False
        data = r.json()
        if data.get("videos"):
            video_url = data["videos"][0]["video_files"][0]["link"]
            logger.info(f"Скачиваем видео: {video_url}")
            with open("stock.mp4", "wb") as f:
                f.write(requests.get(video_url, timeout=30).content)
            return True
        else:
            logger.warning("Pexels не вернул видео")
            return False
    except Exception as e:
        logger.error(f"Исключение при скачивании видео: {e}")
        return False

def create_short(voice_text, trend, speed_factor=1.25):
    temp_files = ["voice.mp3", "voice_speed.mp3", "stock.mp4"]
    short_path = "short.mp4"
    try:
        # Синтез речи
        tts = gTTS(voice_text, lang='ru')
        tts.save("voice.mp3")

        # Ускорение голоса
        audio = AudioSegment.from_mp3("voice.mp3")
        audio = audio.speedup(playback_speed=speed_factor)
        audio.export("voice_speed.mp3", format="mp3")

        # Фоновое видео
        if not download_pexels_video(trend):
            return None

        video = VideoFileClip("stock.mp4")
        if video.duration < 1:
            video.close()
            return None

        duration = min(45, video.duration)
        video = video.subclip(0, duration)

        # Речь
        audio_clip = AudioFileClip("voice_speed.mp3")
        if audio_clip.duration > duration:
            audio_clip = audio_clip.subclip(0, duration)

        # --- Фоновая музыка (только если файл существует и читается) ---
        final_audio = audio_clip
        if os.path.exists("background.mp3"):
            try:
                # Проверим, что файл не пустой и читается
                if os.path.getsize("background.mp3") > 1024:  # хотя бы 1 КБ
                    bg_music = AudioFileClip("background.mp3")
                    if bg_music.duration < 0.1:
                        raise ValueError("Music file too short")
                    # Зацикливаем на всю длительность видео
                    if bg_music.duration < duration:
                        bg_music = bg_music.loop(duration=duration)
                    else:
                        bg_music = bg_music.subclip(0, duration)
                    # Уменьшаем громкость музыки
                    bg_music = bg_music.volumex(0.06)  # ~ -25 dB
                    # Смешиваем с речью
                    final_audio = CompositeAudioClip([audio_clip, bg_music])
            except Exception as e:
                logger.warning(f"Не удалось добавить музыку: {e}. Продолжаем без неё.")

        final = video.set_audio(final_audio)
        final.write_videofile(short_path, fps=24, codec="libx264", audio_codec="aac")
        final.close()
        video.close()
        audio_clip.close()
        if 'bg_music' in locals():
            bg_music.close()

        if os.path.exists(short_path):
            logger.info(f"Видео успешно создано: {short_path}")
            return short_path
        else:
            logger.error(f"Файл {short_path} не найден после записи")
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

    # 1. Генерация длинного поста
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

    # 4. Генерация короткого текста для видео
    logger.info("3. Генерация текста для видео...")
    voice_text = await generate_voice_text(service)
    logger.info(f"Текст для видео ({len(voice_text)} символов): {voice_text}")

    # 5. Создание и отправка видео
    logger.info("4. Создание Shorts...")
    short_path = create_short(voice_text, "car service useful tips", speed_factor=1.25)
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
