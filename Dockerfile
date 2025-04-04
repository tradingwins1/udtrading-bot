
FROM ubuntu:22.04

# Disable interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    xvfb \
    curl \
    unzip \
    openjdk-11-jre \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy bot files
COPY . /app

# Install Python dependencies
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Download and install TWS
RUN mkdir -p /ibkr && \
    wget -q https://download2.interactivebrokers.com/installers/tws/latest/tws-latest-standalone-linux-x64.sh -O /ibkr/tws.sh && \
    chmod +x /ibkr/tws.sh && \
    /ibkr/tws.sh -q -dir /root/Jts

# Set DISPLAY environment for Xvfb
ENV DISPLAY=:1

# Start TWS + Xvfb + bot
CMD ["bash", "/app/start.sh"]
