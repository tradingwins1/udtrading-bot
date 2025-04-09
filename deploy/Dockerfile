FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    xvfb \
    libxtst6 \
    libxrender1 \
    libxi6 \
    socat \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Define versions
ARG IB_GATEWAY_VERSION=10.31.1i
ARG IBC_VERSION=3.18.0

# Install IB Gateway (includes its own JRE under /opt/ibgateway/jre)
RUN mkdir -p /opt/ibgateway && \
    curl -sSL "https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh" -o ibgateway-install.sh && \
    chmod +x ibgateway-install.sh && \
    ./ibgateway-install.sh -q -dir /opt/ibgateway || { echo "IB Gateway install failed"; exit 1; } && \
    rm ibgateway-install.sh && \
    ln -sf /opt/ibgateway/ibgateway/${IB_GATEWAY_VERSION} /opt/ibgateway/ibgateway && \
    ls -l /opt/ibgateway/ibgateway || { echo "Symbolic link failed"; exit 1; }

# Install IBC
RUN curl -sSL "https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip" -o ibc.zip && \
    unzip ibc.zip -d /opt/ibc && \
    chmod -R +x /opt/ibc/*.sh /opt/ibc/scripts/*.sh && \
    rm ibc.zip

# Install Python dependencies
COPY bot/requirements.txt /home/ibuser/bot/requirements.txt
RUN pip3 install --no-cache-dir -r /home/ibuser/bot/requirements.txt

# Create non-root user
RUN useradd -m -s /bin/bash ibuser && \
    mkdir -p /home/ibuser/Jts /home/ibuser/logs /home/ibuser/bot && \
    chown -R ibuser:ibuser /home/ibuser /opt/ibgateway /opt/ibc

# Copy project files
COPY ibc/config.ini /home/ibuser/ibc/config.ini
COPY run.sh /home/ibuser/run.sh
COPY bot/ /home/ibuser/bot/

# Set permissions
RUN chmod +x /home/ibuser/run.sh && \
    chown -R ibuser:ibuser /home/ibuser/ibc /home/ibuser/run.sh /home/ibuser/bot

USER ibuser
WORKDIR /home/ibuser

EXPOSE 4004

CMD socat TCP-LISTEN:4004,fork TCP:127.0.0.1:4002 & /home/ibuser/run.sh