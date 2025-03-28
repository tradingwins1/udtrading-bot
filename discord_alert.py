import requests
import json

# Load config
with open('config.json') as f:
    config = json.load(f)

webhooks = config['discord_webhooks']

def send_discord_alert(message, trade_type='scalp'):
    url = webhooks['scalp'] if trade_type == 'scalp' else webhooks['swing']
    payload = {'content': message}
    try:
        r = requests.post(url, json=payload)
        if r.status_code == 429:
            print(f"⚠️ Discord alert failed (429): {r.text}")
        else:
            print(f"✅ {trade_type.capitalize()} alert sent to Discord.")
    except Exception as e:
        print("❌ Error sending Discord alert:", e)

def send_image_to_discord(image_path, webhook_url):
    with open(image_path, 'rb') as f:
        image_file = {'file': f}
        response = requests.post(webhook_url, files=image_file)
        if response.status_code == 429:
            print(f"⚠️ Discord image alert failed (429): {response.text}")
        elif response.status_code == 204:
            print("✅ Image sent to Discord.")
        else:
            print("⚠️ Unexpected response:", response.status_code, response.text)
