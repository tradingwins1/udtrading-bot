# Stage 1: Base stage for dependencies, IB Gateway, and IBC
FROM ubuntu:22.04 AS base

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
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
    sudo \
    libc6 \
    libx11-6 \
    libxext6 \
    libxft2 \
    && rm -rf /var/lib/apt/lists/*

# Define version arguments
ARG IB_GATEWAY_VERSION=10.31.1i
ARG IBC_VERSION=3.20.0

# Install IB Gateway
RUN mkdir -p /opt/ibgateway && \
    curl -sSL "https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh" -o ibgateway-install.sh && \
    chmod +x ibgateway-install.sh && \
    ./ibgateway-install.sh -q -dir /opt/ibgateway || { echo "IB Gateway install failed"; exit 1; } && \
    rm ibgateway-install.sh && \
    # Remove any existing ibgateway symlink or directory to avoid loops
    rm -rf /opt/ibgateway/ibgateway && \
    # Create a symlink pointing directly to the versioned directory
    ln -sf /opt/ibgateway/${IB_GATEWAY_VERSION} /opt/ibgateway/ibgateway && \
    ls -l /opt/ibgateway/ibgateway || { echo "Symbolic link failed"; exit 1; }

# Debug: Check for JRE after installation
RUN ls -l /opt/ibgateway/ibgateway/jre/bin/java || echo "JRE not found in expected location" && \
    # Check the actual JRE location
    find /opt -name "java" -type f

# Install IBC from pre-downloaded ZIP
COPY ibc.zip /ibc.zip
RUN ls -l /ibc.zip && \
    unzip /ibc.zip -d /opt/ibc && \
    ls -l /opt/ibgateway && \
    ls -l /opt/ibgateway/ibgateway && \
    ls -l /opt/ibc && \
    ls -l /opt/ibc/scripts && \
    chmod -R +x /opt/ibc/*.sh /opt/ibc/scripts/*.sh && \
    chmod 755 /opt/ibc/scripts/ibcstart.sh && \
    chown -R root:root /opt/ibc && \
    rm /ibc.zip

# Debug: Check the script before copying the updated version
RUN echo "Before copying updated script:" && \
    grep "read IBC_VRSN" /opt/ibc/scripts/displaybannerandlaunch.sh || echo "Script not found"

# Copy the updated displaybannerandlaunch.sh after extracting ibc.zip
COPY deploy/displaybannerandlaunch.sh /opt/ibc/scripts/displaybannerandlaunch.sh
RUN ls -l /opt/ibc/scripts/displaybannerandlaunch.sh && \
    echo "After copying updated script:" && \
    grep "read IBC_VRSN" /opt/ibc/scripts/displaybannerandlaunch.sh || echo "Script not found"

# Create non-root user and app directories
RUN useradd -m -s /bin/bash ibuser && \
    mkdir -p /home/ibuser/Jts /home/ibuser/logs /home/ibuser/bot && \
    mkdir -p /tmp/.X11-unix && \
    chown root:root /tmp/.X11-unix && \
    chmod 1777 /tmp/.X11-unix && \
    chown -R ibuser:ibuser /home/ibuser /opt/ibgateway /opt/ibc

# Install Python dependencies
COPY requirements.txt /home/ibuser/bot/requirements.txt
RUN ls -l /home/ibuser/bot/requirements.txt && \
    pip3 install --no-cache-dir -r /home/ibuser/bot/requirements.txt

# Stage 2: App stage for application files and runtime
FROM base AS app

# Copy all Python + config files into the container
COPY deploy/config.ini /home/ibuser/ibc/config.ini
COPY deploy/start.sh /home/ibuser/run.sh
COPY deploy/utils.zip /home/ibuser/utils.zip
RUN ls -l /home/ibuser/ibc/config.ini && \
    ls -l /home/ibuser/run.sh && \
    ls -l /home/ibuser/utils.zip

# Extract utils and clean up
RUN unzip -o /home/ibuser/utils.zip -d /home/ibuser/bot && \
    rm /home/ibuser/utils.zip && \
    chmod +x /home/ibuser/run.sh && \
    chown -R ibuser:ibuser /home/ibuser

# Set user context
USER ibuser
WORKDIR /home/ibuser

# Expose IB Gateway port
EXPOSE 4004

# Start the bot and forward port
CMD socat TCP-LISTEN:4004,fork TCP:127.0.0.1:4002 & /home/ibuser/run.sh
