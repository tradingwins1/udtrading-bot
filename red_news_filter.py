# red_news_filter.py
from datetime import datetime
import pytz

# Simulated red-folder events (replace with real API logic later)
RED_EVENTS = [
    {"date": "2025-04-01", "event": "FOMC Statement"},
    {"date": "2025-04-03", "event": "NFP Data Release"},
]

def is_red_folder_event_today():
    central_now = datetime.now(pytz.timezone("US/Central"))
    today = central_now.date()
    for event in RED_EVENTS:
        if datetime.strptime(event["date"], "%Y-%m-%d").date() == today:
            print(f"🚨 Red-folder event detected: {event['event']} (Today)")
            return True
    return False
