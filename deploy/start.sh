#!/bin/bash
export DISPLAY=:99

# Clean up any existing Xvfb lock files
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99

# Start Xvfb as the ibuser user
Xvfb :99 -screen 0 1024x768x16 &

# Wait for Xvfb to start
sleep 2

# Start IB Gateway with IBC using absolute paths
echo "Running: /opt/ibc/scripts/displaybannerandlaunch.sh 1031 -g --tws-path=/opt/ibgateway --ibc-path=/opt/ibc --ibc-ini=/home/ibuser/ibc/config.ini --mode=$TRADING_MODE --java-path=/opt/i4j_jres/Oda-jK0QgTEmVssfllLP/1.8.0_202/bin"
/opt/ibc/scripts/displaybannerandlaunch.sh 1031 -g \
    --tws-path=/opt/ibgateway \
    --ibc-path=/opt/ibc \
    --ibc-ini=/home/ibuser/ibc/config.ini \
    --mode=$TRADING_MODE \
    --java-path=/opt/i4j_jres/Oda-jK0QgTEmVssfllLP/1.8.0_202/bin &

# Wait for IB Gateway to initialize
sleep 10

# Check if IB Gateway is running (basic PID check)
if ! pgrep -f "ibgateway" > /dev/null; then
    echo "Warning: IB Gateway may not have started correctly"
fi

# Start the trading bot
python3 /home/ibuser/bot/ai_scheduler.py
