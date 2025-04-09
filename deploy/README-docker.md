# UGTradingBot Docker Deployment Guide

## Prerequisites
- IBKR Paper Account
- Docker installed on Ubuntu 22.04 (Vultr VPS)

## Setup
1. **Edit `.env`:**
   - Copy `.env.template` to `.env` and update `TWS_USERID` and `TWS_PASSWORD` with your IBKR paper credentials.

2. **Build and Run:**
   ```bash
   docker build -t ib-trading-bot .
   docker run -d --name ib-bot -p 4004:4004 --env-file .env ib-trading-bot