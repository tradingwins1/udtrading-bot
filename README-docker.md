
# 🚀 ugtrading-bot Docker Deployment

## 📦 Build the Docker Image
```bash
docker build -t ugtrading-bot .
```

## ▶️ Run the Bot in Docker
```bash
docker run -it --env-file .env ugtrading-bot
```

- Make sure your `.env` file contains proper IBKR paper credentials and tokens.
- TWS will auto-launch inside the container using Xvfb (headless GUI).
