#!/bin/bash
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x16 &

# Start IB Gateway with IBC
/opt/ibc/scripts/displaybannerandlaunch.sh \
    --tws-path=/opt/ibgateway \
    --ibc-path=/opt/ibc \
    --ibc-ini=/home/ibuser/ibc/config.ini \
    --mode=paper \
    --java-path=/opt/ibgateway/jre/bin &

# Wait for IB Gateway to initialize
sleep 10

# Check if IB Gateway is running (basic PID check)
if ! pgrep -f "ibgateway" > /dev/null; then
    echo "Warning: IB Gateway may not have started correctly"
fi

# Start the trading bot
python3 /home/ibuser/bot/ai_scheduler.py