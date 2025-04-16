# config.py

CONFIG = {
    "assets": [
        {
            "symbol": "MNQ",
            "asset_type": "futures",
            "timeframe": "5m"
        },
        {
            "symbol": "MGC",
            "asset_type": "futures",
            "timeframe": "1h"
        },
        {
            "symbol": "USDJPY",
            "asset_type": "forex",
            "timeframe": "4h"
        },
        {
            "symbol": "EURUSD",
            "asset_type": "forex",
            "timeframe": "4h"
        },
        {
            "symbol": "ETHUSD",
            "asset_type": "crypto",
            "timeframe": "15m"
        },
        {
            "symbol": "BTCUSD",
            "asset_type": "crypto",
            "timeframe": "15m"
        },
        {
            "symbol": "AAPL",
            "asset_type": "stock",
            "timeframe": "5m"
        },
        {
            "symbol": "NVDA",
            "asset_type": "stock",
            "timeframe": "5m"
        }
    ],
    "finnhub_api_key": "cvho01pr01qgkck54di0cvho01pr01qgkck54dig",
    "filters": {
        "wick_body_ratio_threshold": 2.5,
        "volume_spike_multiplier": 1.5
    },
    "strategy": {
        "risk_per_trade": 60,
        "min_win_rate": 60,
        "sl_buffer_points": 60,
        "tp_buffer_points": 120
    },
    "time_filters": {
        "stocks": ["09:30", "11:30"],
        "futures": ["09:30", "11:30"],
        "forex": ["03:00", "16:00"],
        "crypto": ["00:00", "23:59"]
    },
    "discord_webhooks": {
        "scalp": "https://discord.com/api/webhooks/1353245464866066442/ZmarfW4Tm2wgAuzgmrJV8MR-GdqcZzrXNCtnHCEYvf0ePmn3ZHSVp5uJEVbrdje6C3uh",
        "swing": "https://discord.com/api/webhooks/1353549628703903815/J51yoXWld4_m8G2nGOuKoZZPLgog7lpDVOfKh0ZTXdAZz6bMyl8-EbGeQ_GfqABXKKQd"
    },
    "flags": {
        "enable_news_filter": True,
        "enable_htf_trend": True,
        "avoid_pdh_pdl_liquidity": True
    },
    "take_profit_levels": [1.5, 2.5],
    "trailing_stop": {
        "enabled": True,
        "activation_rr": 1.5,
        "trail_by": 0.5
    },
    "plot_live_dashboard": True
}

def get_config():
    return CONFIG
