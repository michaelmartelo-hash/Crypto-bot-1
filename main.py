# main.py
import os
import asyncio
import datetime
from zoneinfo import ZoneInfo
import requests
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from telegram import Bot
from fastapi import FastAPI

# ============================
# CONFIG - leer de Secrets
# ============================
TOKEN = os.getenv("TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID"))
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

bot = Bot(token=TOKEN)
app = FastAPI()
TZ = ZoneInfo("America/Bogota")

COINBASE_SYMBOL = {"bitcoin": "BTC", "ethereum": "ETH", "ripple": "XRP"}
COINGECKO_ID = {"bitcoin": "bitcoin", "ethereum": "ethereum", "ripple": "ripple"}
SEND_ORDERBOOK_FALLBACK_ZERO = (0.0, 0.0, 0.0, 0.0)

# ============================
# UTIL
# ============================
def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return {}

def get_coinbase_price(coin_id):
    symbol = COINBASE_SYMBOL.get(coin_id, coin_id).upper()
    url = f"https://api.exchange.coinbase.com/products/{symbol}-USD/ticker"
    try:
        r = requests.get(url, timeout=8)
        data = safe_json(r)
        if "price" in data:
            return float(data["price"])
    except Exception:
        pass
    # fallback CoinGecko
    try:
        cg_id = COINGECKO_ID.get(coin_id, coin_id)
        r2 = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd", timeout=8).json()
        return float(r2[cg_id]["usd"])
    except Exception:
        return None

def get_coinbase_orderbook(coin_id):
    symbol = COINBASE_SYMBOL.get(coin_id, coin_id).upper()
    url = f"https://api.exchange.coinbase.com/products/{symbol}-USD/book?level=1"
    try:
        r = requests.get(url, timeout=8)
        data = safe_json(r)
        bid_price = float(data["bids"][0][0])
        bid_qty = float(data["bids"][0][1])
        ask_price = float(data["asks"][0][0])
        ask_qty = float(data["asks"][0][1])
        return bid_price, bid_qty, ask_price, ask_qty
    except Exception:
        return SEND_ORDERBOOK_FALLBACK_ZERO

def get_history_coingecko(coin_id, days=2):
    cg_id = COINGECKO_ID.get(coin_id, coin_id)
    try:
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart",
                         params={"vs_currency":"usd", "days":str(days)}, timeout=10).json()
        prices = r.get("prices", [])
        df = pd.DataFrame(prices, columns=["timestamp","price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["price"] = df["price"].astype(float)
        return df
    except Exception:
        return pd.DataFrame(columns=["timestamp","price"])

def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def get_news_for_symbol(symbol, max_articles=3):
    if NEWS_API_KEY:
        try:
            r = requests.get("https://newsapi.org/v2/everything",
                             params={"q":f"{symbol} OR crypto OR cryptocurrency OR blockchain",
                                     "language":"en", "pageSize":max_articles, "sortBy":"publishedAt",
                                     "apiKey":NEWS_API_KEY}, timeout=8).json()
            articles = r.get("articles", [])[:max_articles]
            return "📰 *Noticias relevantes:*\n" + "\n".join([f"• {a.get('title')} ({a.get('source',{}).get('name')})\n  {a.get('url')}" for a in articles]) if articles else ""
        except Exception:
            pass
    if GNEWS_API_KEY:
        try:
            r = requests.get("https://gnews.io/api/v4/search",
                             params={"q":symbol,"lang":"en","max":max_articles,"token":GNEWS_API_KEY}, timeout=8).json()
            articles = r.get("articles", [])[:max_articles]
            return "📰 *Noticias relevantes:*\n" + "\n".join([f"• {a.get('title')} ({a.get('source',{}).get('name')})\n  {a.get('url')}" for a in articles]) if articles else ""
        except Exception:
            pass
    return "📰 No hay noticias relevantes disponibles."

def create_chart_image(df, symbol_label):
    try:
        plt.figure(figsize=(8,3.6))
        plt.plot(df["timestamp"], df["price"], label="Precio", linewidth=1.4)
        if "SMA20" in df.columns:
            plt.plot(df["timestamp"], df["SMA20"], label="SMA20", linewidth=1.2)
        plt.title(f"{symbol_label} - últimas 72h")
        plt.xlabel("Hora")
        plt.ylabel("USD")
        plt.legend()
        plt.grid(alpha=0.3)
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close()
        return buf
    except Exception:
        return None

# ============================
# ANALYSIS
# ============================
async def analyze_coin(coin_id):
    label = COINBASE_SYMBOL.get(coin_id, coin_id).upper()
    now = datetime.datetime.now(TZ)
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

    price = get_coinbase_price(coin_id)
    if price is None:
        print(f"{label}: precio no disponible")
        return

    df = get_history_coingecko(coin_id, days=2)
    if df.empty:
        print(f"{label}: historial no disponible")
        return

    # Obtener precio exacto de "ayer a la misma hora"
    yesterday_same_hour = now - datetime.timedelta(days=1)
    df_prev = df.iloc[(df["timestamp"] - pd.Timestamp(yesterday_same_hour)).abs().argsort()[:1]]
    prev_price = df_prev["price"].values[0] if not df_prev.empty else None
    if prev_price is None:
        print(f"{label}: precio de ayer no disponible")
        return

    change_pct = (price - prev_price) / prev_price * 100
    if abs(change_pct) < 3:
        print(f"{label}: cambio {change_pct:.2f}% < 3%, no se envía mensaje")
        return

    # Indicators
    df["SMA20"] = df["price"].rolling(20).mean()
    df["RSI14"] = calc_rsi(df["price"], 14)
    sma_val = df["SMA20"].iloc[-1]
    rsi_val = df["RSI14"].iloc[-1]

    # Build message
    lines = [
        f"📊 *ANÁLISIS EDUCATIVO — {label}*",
        f"⏱ {timestamp_str} (hora Colombia)",
        f"💵 *Precio actual:* ${price:,.2f}",
        f"📈 Precio {'por encima' if price > sma_val else 'por debajo'} de SMA20 (${sma_val:,.2f})" if pd.notna(sma_val) else "",
        f"📉 RSI14: {rsi_val:.2f}" if pd.notna(rsi_val) else "",
        f"📊 Cambio respecto ayer misma hora: {change_pct:.2f}%",
        "",
        get_news_for_symbol(label),
        "",
        "_Este análisis es educativo, no es asesoramiento financiero._"
    ]

    message = "\n".join([l for l in lines if l])

    # Send
    try:
        bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")
        chart_buf = create_chart_image(df, label)
        if chart_buf:
            bot.send_photo(chat_id=CHAT_ID, photo=chart_buf)
        print(f"✅ {label} enviado, cambio {change_pct:.2f}%")
    except Exception as e:
        print(f"Error enviando {label}:", e)

# ============================
# LOOP
# ============================
async def loop_crypto():
    try:
        bot.send_message(chat_id=CHAT_ID, text="🤖 Crypto Bot iniciado (educativo). Notificaciones solo si cambio ≥3% vs ayer misma hora.")
    except Exception:
        pass

    while True:
        now = datetime.datetime.now(TZ)
        if 6 <= now.hour < 21 or (now.hour == 21 and now.minute <= 30):
            for coin in ["bitcoin","ethereum","ripple"]:
                await analyze_coin(coin)
        next_run = (now + datetime.timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
        wait_seconds = max((next_run - now).total_seconds(), 60)
        await asyncio.sleep(wait_seconds)

# ============================
# FastAPI
# ============================
@app.get("/")
def home():
    return {"status":"alive"}

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(loop_crypto())



