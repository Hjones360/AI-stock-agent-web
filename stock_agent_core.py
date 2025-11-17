import math
import json
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()  # reads OPENAI_API_KEY from .env


def get_recent_candles(symbol="SPY", interval="5", lookback_minutes=60) -> pd.DataFrame:
    """
    Fetch recent intraday candles using yfinance in a way that does NOT rely
    on the local system clock (uses Yahoo's 'period' argument instead of start/end).
    """
    interval_map = {
        "1": "1m",
        "2": "2m",
        "5": "5m",
        "15": "15m",
        "30": "30m",
        "60": "60m"
    }
    yf_interval = interval_map.get(interval, "5m")

    df = yf.download(
        symbol,
        period="1d",
        interval=yf_interval,
        progress=False,
    )

    # Handle MultiIndex like ('Close','SPY')
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, axis=1, level=1)

    if df.empty:
        raise RuntimeError("No data returned from yfinance. Try a different symbol or interval.")

    df = df.reset_index().rename(
        columns={
            "Datetime": "time",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    if "time" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "time"})

    return df[["time", "open", "high", "low", "close", "volume"]]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, EMA, RSI columns to the dataframe."""
    df = df.copy()

    # SMA
    df["SMA_10"] = df["close"].rolling(window=10).mean()
    df["SMA_20"] = df["close"].rolling(window=20).mean()

    # EMA
    df["EMA_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    return df


def build_market_stats(df: pd.DataFrame, symbol: str, interval: str, lookback_minutes: int) -> dict:
    """Build compact stats for AI or rule-based analysis."""
    recent = df.tail(20).copy()
    current = recent.iloc[-1]
    prev = recent.iloc[-2] if len(recent) > 1 else current

    if prev["close"] != 0:
        pct_change = ((current["close"] - prev["close"]) / prev["close"]) * 100
    else:
        pct_change = 0.0

    stats = {
        "symbol": symbol,
        "interval_minutes": interval,
        "bars_considered": len(recent),
        "lookback_minutes": lookback_minutes,
        "current_price": float(current["close"]),
        "previous_close": float(prev["close"]),
        "pct_change_vs_prev_bar": float(round(pct_change, 4)),
        "session_high_recent": float(recent["high"].max()),
        "session_low_recent": float(recent["low"].min()),
        "latest_volume": int(current["volume"]),
        "average_volume_recent": float(round(recent["volume"].mean(), 2)),
        "sma_10": None if pd.isna(current["SMA_10"]) else float(round(current["SMA_10"], 4)),
        "sma_20": None if pd.isna(current["SMA_20"]) else float(round(current["SMA_20"], 4)),
        "ema_10": None if pd.isna(current["EMA_10"]) else float(round(current["EMA_10"], 4)),
        "ema_20": None if pd.isna(current["EMA_20"]) else float(round(current["EMA_20"], 4)),
        "rsi_14": None if pd.isna(current["RSI_14"]) else float(round(current["RSI_14"], 2)),
    }

    return stats


def rule_based_analysis(stats: dict) -> str:
    """Fallback analysis that does NOT call the API (offline mode)."""
    symbol = stats["symbol"]
    price = stats["current_price"]
    rsi = stats["rsi_14"]
    sma10 = stats["sma_10"]
    sma20 = stats["sma_20"]
    ema10 = stats["ema_10"]
    ema20 = stats["ema_20"]
    vol = stats["latest_volume"]
    avg_vol = stats["average_volume_recent"]
    high = stats["session_high_recent"]
    low = stats["session_low_recent"]

    bullets = []

    if sma10 and sma20:
        if sma10 > sma20:
            bullets.append("Short-term trend appears upward, with SMA-10 above SMA-20.")
        elif sma10 < sma20:
            bullets.append("Short-term trend appears downward, with SMA-10 below SMA-20.")
        else:
            bullets.append("SMA-10 and SMA-20 are equal, indicating sideways price action.")

    if rsi is not None:
        if rsi < 30:
            bullets.append("RSI is below 30, suggesting recent oversold conditions.")
        elif rsi > 70:
            bullets.append("RSI is above 70, suggesting recent overbought conditions.")
        else:
            bullets.append(f"RSI is {rsi:.1f}, reflecting neutral momentum.")

    if ema10 and ema20:
        if ema10 > ema20:
            bullets.append("EMA-10 above EMA-20 indicates improving short-term momentum.")
        else:
            bullets.append("EMA-10 below EMA-20 indicates weakening short-term momentum.")

    if vol and avg_vol:
        if vol > avg_vol:
            bullets.append("Recent volume is above average, indicating elevated trading activity.")
        else:
            bullets.append("Recent volume is below its recent average.")

    if price:
        if price >= high:
            bullets.append("Price is near the recent session high.")
        elif price <= low:
            bullets.append("Price is near the recent session low.")
        else:
            bullets.append("Price is trading in the middle of the recent session range.")

    return "\n".join(f"- {b}" for b in bullets)


def get_ai_analysis(stats: dict) -> str:
    """
    Try to call OpenAI for rich commentary.
    If quota or model error occurs, fall back to local rule-based analysis.
    """
    stats_json = json.dumps(stats, indent=2)

    prompt = f"""
You are an experienced, calm market analyst.

Here are recent intraday stats for {stats.get("symbol")} on a {stats.get("interval_minutes")} minute chart:

{stats_json}

In 4–7 short bullet points, provide a neutral analysis that covers:
- the short-term trend (bullish, bearish, or sideways),
- how price is behaving relative to SMA_10 and SMA_20,
- what RSI_14 suggests about momentum / overbought / oversold conditions,
- any notable changes in volume,
- key levels or zones to watch based on the recent high/low range.

Important rules:
- Do NOT give trading advice or tell anyone to buy/sell/hold.
- Focus on describing the situation and risks in plain English.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You provide concise, neutral market commentary without trade advice."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # Fallback: offline rule-based commentary
        return rule_based_analysis(stats)
