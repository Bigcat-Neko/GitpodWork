import requests
import telebot  # Install using: pip install pyTelegramBotAPI

TELEGRAM_BOT_TOKEN = "950251170:AAEHwpGH4SKQIG8KgRS6EoHupBX-lZeknlQ"
TELEGRAM_CHAT_ID = "734698844"

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

def send_signal_to_telegram(signal):
    """Sends a trade signal to Telegram"""
    message = f"🔹 Live Trade Signal: {signal}"
    bot.send_message(TELEGRAM_CHAT_ID, message)

# Test the function
send_signal_to_telegram("BUY EUR/USD at 1.1234")  # Replace with AI-generated signal
