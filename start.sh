
#!/bin/bash

echo "🔧 Starting Xvfb..."
Xvfb :1 -screen 0 1024x768x16 &

echo "📈 Launching TWS..."
/root/Jts/twsstart.sh &

echo "🤖 Starting AI trading bot..."
sleep 20  # Give TWS time to start
python3 ai_scheduler.py
