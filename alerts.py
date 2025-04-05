# alerts.py
import requests
import os

def send_telegram_alert(message):
    bot_token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('CHAT_ID')
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    requests.post(url, json={
        'chat_id': chat_id,
        'text': f"🚨 Trade Alert: {message}",
        'parse_mode': 'Markdown'
    })