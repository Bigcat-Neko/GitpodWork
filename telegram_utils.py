# telegram_utils.py
import os
import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def send_telegram_message(message: str, to_channel: bool = True) -> bool:
    """
    Send message to Telegram using environment variables
    """
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
    
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set in environment")
        return False
        
    chat_id = TELEGRAM_CHANNEL_ID if to_channel else TELEGRAM_CHAT_ID
    if not chat_id:
        logger.error(f"No chat ID configured for {'channel' if to_channel else 'chat'} messages")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            logger.error(f"Telegram API error: {response.status_code} - {response.text}")
            return False
        logger.info(f"Message sent to Telegram {'channel' if to_channel else 'chat'}")
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {str(e)}")
        return False