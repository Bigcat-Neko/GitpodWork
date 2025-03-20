TELEGRAM_BOT_TOKEN = "950251170:AAEHwpGH4SKQIG8KgRS6EoHupBX-lZeknlQ"
TELEGRAM_CHAT_ID = "734698844"


import os
from dotenv import load_dotenv

load_dotenv()  # This loads the .env file

FOREX_API_KEY = os.getenv("FOREX_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ECONOMIC_API_KEY = os.getenv("ECONOMIC_API_KEY")
