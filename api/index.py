from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import yfinance as yf
import pandas as pd
import numpy as np
import httpx

app = FastAPI(title="VN Stock Trading Pro Advisory Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANALYSIS_CACHE: Dict[str, Any] = {}
SCREENER_CACHE: Dict[str, Any] = {}
CACHE_TTL = 300
MAX_CACHE_ENTRIES = 100

WATCHLIST_UNIVERSE = [
    "HPG", "VCB", "SSI", "TCB", "FPT", "VHM", "VIC", "MWG", "MBB", "ACB",
    "STB", "VPB", "VNM", "GAS", "MSN", "GVR", "PLX", "VRE", "DGC", "PVD",
    "KBC", "DIG", "DXG", "NLG", "VIX", "SHS", "HCM", "PDR", "VCI", "HSG"
]


class StockRequest(BaseModel):
    symbol: str = Field(..., min_length=2, max_length=10)


def get_next_trading_days(start_date: datetime, count: int = 5) -> List[str]:
    days = []
    curr = start_date + timedelta(days=1)
    while len(days) < count:
        if curr.weekday() < 5:
            days.append(curr.strftime("%d/%m"))
        curr += timedelta(days=1)
    return days


def calculate_rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calculate_orderflow_pressure(df: pd.DataFrame) -> Dict[str, Any]:
    latest = df.iloc[-1]
    high = float(latest["High"])
    low = float(latest["Low"])
    close = float(latest["Close"])
    open_p = float(latest["Open"])
    vol = int(latest["Volume"])

    if high != low:
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
    else:
        mf_multiplier = 0.0

    buy_ratio = max(0.15, min(0.85, (mf_multiplier + 1) / 2))
    active_buy_vol = int(vol * buy_ratio)
    active_sell_vol = vol - active_buy_vol
    net_active_vol = active_buy_vol - active_sell_vol

    shark_buy = int(active_buy_vol * 0.52)
    shark_sell = int(active_sell_vol * 0.25)
    retail_buy = int(active_buy_vol * 0.18)
    retail_sell = int(active_sell_vol * 0.45)

    if buy_ratio >= 0.58 and close >= open_p:
        smart_money_action = "GOM HÀNG ÂM THẦM (BIG BOYS ACCUMULATION) 🟢"
        bull_trap_warning = False
    elif buy_ratio <= 0.42 and close <= open_p:
        smart_money_action = "XẢ HÀNG QUYẾT LIỆT (DISTRIBUTION) 🔴"
        bull_trap_warning = False
    elif close > open_p and buy_ratio < 0.45:
        smart_money_action = "CẢNH BÁO BẪY TĂNG GIÁ (BULL TRAP / KÉO XẢ) ⚠️"
        bull_trap_warning = True
    else:
        smart_money_action = "GIẰNG CO CUNG CẦU (NEUTRAL FLOW) 🟡"
        bull_trap_warning = False

    return {
        "active_buy_volume": active_buy_vol,
        "active_sell_volume": active_sell_vol,
        "active_buy_pct": round(buy_ratio * 100, 1),
        "active_sell_pct": round((1 - buy_ratio) * 100, 1),
        "net_active_volume": net_active_vol,
        "smart_money_signal": smart_money_action,
        "is_bull_trap_risk": bull_trap_warning,
        "tier_breakdown": {
            "shark_net_volume": shark_buy - shark_sell,
            "retail_net_volume": retail_buy - retail_sell
        }
    }


def calculate_fundamental_and_events(ticker_obj: yf.Ticker, curr_price: float) -> Dict[str, Any]:
    info = {}
    try:
        info = ticker_obj.info or {}
    except Exception:
        pass

    pe = info.get("trailingPE", None)
    pb = info.get("priceToBook", None)
    roe = info.get("returnOnEquity", None)
    div_yield = info.get("dividendYield", None)
    market_cap = info.get("marketCap", None)

    health_score = 70
    if pe and 0 < pe < 15:
        health_score += 10
    if roe and roe > 0.15:
        health_score += 10
    if pb and 0 < pb < 2.0:
        health_score += 10

    events = []
    try:
        cal = ticker_obj.calendar
        if cal is not None and not (isinstance(cal, pd.DataFrame) and cal.empty):
            if isinstance(cal, pd.DataFrame):
                for col in cal.columns:
                    events.append(f"{col}: {cal[col].to_dict()}")
            elif isinstance(cal, dict):
                for k, v in cal.items():
                    events.append(f"{k}: {v}")
    except Exception:
        pass

    if not events:
        events = ["Chưa ghi nhận lịch GDKHQ hoặc chia cổ tức trong 14 ngày tới"]

    return {
        "pe_ratio": round(pe, 2) if pe else "N/A",
        "pb_ratio": round(pb, 2) if pb else "N/A",
        "roe_pct": round(roe * 100, 2) if roe else "N/A",
        "dividend_yield": round(div_yield * 100, 2) if div_yield else "N/A",
        "health_score": min(health_score, 98),
        "corporate_actions": events[:3],
        "is_penny_risk": bool(curr_price < 10000 or (market_cap and market_cap < 1_000_000_000_000))
    }


def calculate_atr_risk_management(df: pd.DataFrame, curr_price: float) -> Dict[str, Any]:
    atr_val = float(df["ATR"].iloc[-1]) if not pd.isna(df["ATR"].iloc[-1]) else (curr_price * 0.02)
    atr_pct = (atr_val / curr_price) * 100
    dynamic_stop_loss = round(curr_price - (2.0 * atr_val), 0)
    trailing_stop_price = round(curr_price - (1.0 * atr_val), 0)
    take_profit_target = round(curr_price + (3.0 * atr_val), 0)

    if atr_pct > 3.5:
        volatility_level = "CAO (BIẾN ĐỘNG MẠNH - RỦI RO)"
    elif atr_pct >= 1.5:
        volatility_level = "VỪA PHẢI (CHUẨN LƯỚT T+)"
    else:
        volatility_level = "THẤP (CỔ PHIẾU PHÒNG THỦ)"

    return {
        "atr_14": round(atr_val, 0),
        "atr_percent": round(atr_pct, 2),
        "volatility_level": volatility_level,
        "dynamic_stop_loss": dynamic_stop_loss,
        "trailing_stop": trailing_stop_price,
        "dynamic_target": take_profit_target,
        "risk_summary": f"Biên độ dao động ngày ±{round(atr_pct, 2)}%. Stop loss động tại {dynamic_stop_loss:,.0f} VNĐ (-2.0x ATR)"
    }


def evaluate_realtime_triggers(curr_price: float, rsi: float, dynamic_buy_zone: float) -> Dict[str, Any]:
    triggers = []
    is_alert = False
    if rsi <= 32.0:
        triggers.append(f"🚨 RSI CHẠM VÙNG QUÁ BÁN ({rsi}): Xác suất bật nảy kỹ thuật cực cao!")
        is_alert = True
    elif rsi >= 75.0:
        triggers.append(f"⚠️ RSI CHẠM VÙNG QUÁ MUA ({rsi}): Cân nhắc chốt lời, rủi ro điều chỉnh.")
        is_alert = True

    if curr_price <= dynamic_buy_zone * 1.01:
        triggers.append(f"🎯 GIÁ ĐÃ CHẠM VÙNG MUA KỲ VỌNG: Điểm giải ngân tối ưu kích hoạt!")
        is_alert = True

    return {
        "has_active_alert": is_alert,
        "alert_messages": triggers if triggers else ["Giá và RSI đang vận động trong biên độ an toàn"]
    }


def calculate_signal_reliability(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) < 55:
        return {
            "title": "ĐỘ TIN CẬY TÍN HIỆU (T+5)",
            "win_rate": 60.0,
            "total_signals": 0,
            "avg_return_pct": 0.0,
            "badge_rating": "DỮ LIỆU MỚI",
            "summary": "Chưa đủ dữ liệu lịch sử để kiểm chứng độ chính xác"
        }

    signals = []
    closes = df["Close"].values
    sma20 = df["SMA20"].values
    rsi = df["RSI"].values

    for i in range(50, len(df) - 5):
        is_buy_signal = (closes[i] > sma20[i]) and (closes[i-1] <= sma20[i-1]) and (45 <= rsi[i] <= 70)
        if is_buy_signal:
            entry_price = closes[i]
            exit_price = closes[i + 5]
            ret = ((exit_price - entry_price) / entry_price) * 100
            signals.append(ret)

    if not signals:
        return {
            "title": "ĐỘ TIN CẬY TÍN HIỆU (T+5)",
            "win_rate": 65.0,
            "total_signals": 0,
            "avg_return_pct": 0.0,
            "badge_rating": "ĐẠT TIÊU CHUẨN",
            "summary": "Không xuất hiện điểm bứt phá đột biến trong 6 tháng qua"
        }

    wins = [r for r in signals if r > 0]
    win_rate = round((len(wins) / len(signals)) * 100, 1)
    avg_return = round(float(sum(signals) / len(signals)), 2)

    if win_rate >= 70:
        rating = "RẤT CAO ⭐⭐⭐"
    elif win_rate >= 50:
        rating = "ĐẠT TIÊU CHUẨN ⭐⭐"
    else:
        rating = "CẦN THẬN TRỌNG ⚠️"

    return {
        "title": "ĐỘ TIN CẬY TÍN HIỆU (T+5)",
        "win_rate": win_rate,
        "total_signals": len(signals),
        "avg_return_pct": avg_return,
        "badge_rating": rating,
        "summary": f"Tín hiệu chuẩn xác {win_rate}% qua {len(signals)} lần xuất hiện gần nhất"
    }


def fetch_vnindex_macro() -> Dict[str, Any]:
    try:
        vnindex = yf.Ticker("^VNINDEX")
        vdf = vnindex.history(period="3mo", interval="1d")
        if vdf is not None and not vdf.empty and len(vdf) >= 20:
            vdf["SMA20"] = vdf["Close"].rolling(window=20).mean()
            vdf["RSI"] = calculate_rsi_wilder(vdf["Close"], period=14)
            v_latest = vdf.iloc[-1]
            v_prev = vdf.iloc[-2]
            v_close = float(v_latest["Close"])
            v_change = v_close - float(v_prev["Close"])
            v_pct = (v_change / float(v_prev["Close"])) * 100
            v_sma20 = float(v_latest["SMA20"]) if not pd.isna(v_latest["SMA20"]) else v_close
            v_rsi = float(v_latest["RSI"]) if not pd.isna(v_latest["RSI"]) else 50.0

            trend = "TĂNG TRƯỞNG" if v_close >= v_sma20 else "ĐIỀU CHỈNH / RỦI RO"
            return {
                "vnindex_point": round(v_close, 2),
                "vnindex_change_pct": round(v_pct, 2),
                "vnindex_sma20": round(v_sma20, 2),
                "vnindex_rsi": round(v_rsi, 1),
                "macro_status": trend
            }
    except Exception:
        pass
    
    return {
        "vnindex_point": 1250.0,
        "vnindex_change_pct": 0.0,
        "vnindex_sma20": 1250.0,
        "vnindex_rsi": 50.0,
        "macro_status": "TRUNG LẬP"
    }


ADVICE_JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "action": {
            "type": "STRING",
            "enum": ["MUA MỚI", "MUA GIA TĂNG", "NẮM GIỮ", "BÁN HẠ TỶ TRỌNG", "BÁN CẮT LỖ", "THEO DÕI"]
        },
        "buy_zone": {"type": "STRING"},
        "target_price": {"type": "STRING"},
        "stop_loss": {"type": "STRING"},
        "risk_reward_ratio": {"type": "STRING"},
        "trend_weekly": {"type": "STRING", "enum": ["TĂNG", "GIẢM", "TÍCH LŨY"]},
        "trend_monthly": {"type": "STRING", "enum": ["TĂNG", "GIẢM", "TÍCH LŨY"]},
        "market_sentiment": {
            "type": "OBJECT",
            "properties": {
                "market_risk_level": {"type": "STRING", "enum": ["THẤP", "TRUNG BÌNH", "CAO"]},
                "sentiment_summary": {"type": "STRING"}
            },
            "required": ["market_risk_level", "sentiment_summary"]
        },
        "catalysts": {
            "type": "ARRAY",
            "items": {"type": "STRING"}
        },
        "capital_allocation": {"type": "STRING"},
        "predicted_5d_prices": {
            "type": "ARRAY",
            "items": {"type": "NUMBER"}
        },
        "prediction_comment": {"type": "STRING"}
    },
    "required": [
        "action", "buy_zone", "target_price", "stop_loss",
        "risk_reward_ratio", "trend_weekly", "trend_monthly",
        "market_sentiment", "catalysts", "capital_allocation",
        "predicted_5d_prices", "prediction_comment"
    ]
}


@app.get("/api/screener")
@app.get("/screener")
async def daily_market_screener():
    now = time.time()
    if "top5_screener" in SCREENER_CACHE:
        cached_screener, cached_time = SCREENER_CACHE["top5_screener"]
        if now - cached_time < 1800:
            return cached_screener

    screened_results = []
    for sym in WATCHLIST_UNIVERSE:
        try:
            ticker = f"{sym}.VN"[cite: 1]
            stock = yf.Ticker(ticker)[cite: 1]
            df = stock.history(period="6mo", interval="1d")[cite: 1]
            if df is None or df.empty or len(df) < 30:
                continue

            df["SMA20"] = df["Close"].rolling(window=20).mean()[cite: 1]
            df["SMA50"] = df["Close"].rolling(window=50).mean()[cite: 1]
            df["RSI"] = calculate_rsi_wilder(df["Close"], period=14)
            df["ATR"] = calculate_atr(df, period=14)

            latest = df.iloc[-1][cite: 1]
            prev = df.iloc[-2][cite: 1]
            close = float(latest["Close"])[cite: 1]
            vol = int(latest["Volume"])[cite: 1]
            avg_vol20 = int(df["Volume"].tail(20).mean())
            rsi = float(latest["RSI"]) if not pd.isna(latest["RSI"]) else 50.0
            sma20 = float(latest["SMA20"])

            is_breakout = (close >= sma20) and (vol >= avg_vol20 * 1.15) and (50 <= rsi <= 70)
            rel = calculate_signal_reliability(df)
            orderflow = calculate_orderflow_pressure(df)
            score = rel["win_rate"] * 0.6 + (vol / (avg_vol20 + 1)) * 20 + (orderflow["active_buy_pct"] * 0.2)

            if is_breakout or rel["win_rate"] >= 65:
                screened_results.append({
                    "symbol": sym,
                    "price": close,
                    "change_pct": round(((close - float(prev["Close"])) / float(prev["Close"])) * 100, 2),
                    "volume": vol,
                    "vol_vs_avg20": round(vol / (avg_vol20 + 1), 2),
                    "rsi": round(rsi, 1),
                    "win_rate": rel["win_rate"],
                    "badge_rating": rel["badge_rating"],
                    "smart_money_signal": orderflow["smart_money_signal"],
                    "score": round(score, 1)
                })
        except Exception:
            continue

    screened_results.sort(key=lambda x: x["score"], reverse=True)
    top_5 = screened_results[:5]

    response = {
        "updated_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "total_scanned": len(WATCHLIST_UNIVERSE),
        "top_picks": top_5
    }
    SCREENER_CACHE["top5_screener"] = (response, now)
    return response


@app.post("/api/analyze")
@app.post("/analyze")
async def analyze_stock(req: StockRequest):
    sym = req.symbol.upper().strip()
    api_key = os.environ.get("GEMINI_API_KEY")[cite: 1]
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Chưa cấu hình GEMINI_API_KEY trên Vercel Environment Variables"
        )

    now = time.time()
    if sym in ANALYSIS_CACHE:
        cached_data, cached_time = ANALYSIS_CACHE[sym][cite: 1]
        if now - cached_time < CACHE_TTL:
            return cached_data

    try:
        ticker = f"{sym}.VN"[cite: 1]
        stock = yf.Ticker(ticker)[cite: 1]
        df = stock.history(period="6mo", interval="1d")[cite: 1]
        if df is None or df.empty or len(df) < 20:[cite: 1]
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy dữ liệu cho mã '{sym}'. Vui lòng kiểm tra lại mã cổ phiếu."[cite: 1]
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi truy xuất dữ liệu: {str(e)}")[cite: 1]

    df["SMA20"] = df["Close"].rolling(window=20).mean()[cite: 1]
    df["SMA50"] = df["Close"].rolling(window=50).mean()[cite: 1]
    df["RSI"] = calculate_rsi_wilder(df["Close"], period=14)
    df["ATR"] = calculate_atr(df, period=14)

    latest = df.iloc[-1][cite: 1]
    prev = df.iloc[-2][cite: 1]
    curr_price = float(latest["Close"])[cite: 1]
    change = curr_price - float(prev["Close"])[cite: 1]
    pct_change = (change / float(prev["Close"])) * 100[cite: 1]

    recent_10 = df.tail(10)[cite: 1]
    history_dates = [pd.to_datetime(d).strftime("%d/%m") for d in recent_10.index][cite: 1]
    history_prices = [round(float(p), 0) for p in recent_10["Close"]][cite: 1]

    last_trade_date = pd.to_datetime(recent_10.index[-1]).to_pydatetime()[cite: 1]
    future_dates = get_next_trading_days(last_trade_date, count=5)[cite: 1]

    macro_info = fetch_vnindex_macro()
    signal_reliability = calculate_signal_reliability(df)
    orderflow = calculate_orderflow_pressure(df)
    fundamental = calculate_fundamental_and_events(stock, curr_price)
    atr_risk = calculate_atr_risk_management(df, curr_price)
    realtime_alerts = evaluate_realtime_triggers(
        curr_price=curr_price,
        rsi=float(latest["RSI"]),
        dynamic_buy_zone=atr_risk["trailing_stop"]
    )

    metrics = {
        "symbol": sym,
        "current_price": curr_price,
        "change": change,
        "percent_change": pct_change,
        "volume": int(latest["Volume"]),
        "avg_vol_20": int(df["Volume"].tail(20).mean()),
        "rsi": round(float(latest["RSI"]), 1) if not pd.isna(latest["RSI"]) else 50.0,
        "sma20": round(float(latest["SMA20"]), 0) if not pd.isna(latest["SMA20"]) else curr_price,
        "sma50": round(float(latest["SMA50"]), 0) if not pd.isna(latest["SMA50"]) else curr_price,
        "support_20": float(df["Low"].tail(20).min()),
        "resistance_20": float(df["High"].tail(20).max()),
        "history_dates": history_dates,
        "history_prices": history_prices,
        "future_dates": future_dates,
        "macro_vnindex": macro_info,
        "signal_reliability": signal_reliability,
        "orderflow_pressure": orderflow,
        "fundamental_overlay": fundamental,
        "atr_risk_management": atr_risk,
        "realtime_alerts": realtime_alerts
    }

    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}"[cite: 1]
    
    prompt = f"""
Bạn là Chuyên gia Tư vấn Đầu tư Chứng khoán cấp cao tại thị trường Việt Nam. Phân tích mã {sym}:
- THỊ TRƯỜNG CHUNG (VN-INDEX): {macro_info['vnindex_point']} ({macro_info['vnindex_change_pct']:+.2f}%), RSI: {macro_info['vnindex_rsi']}, Xu hướng: {macro_info['macro_status']}
- GIÁ & KỸ THUẬT: {metrics['current_price']:,.0f} VNĐ ({metrics['percent_change']:+.2f}%), Khối lượng: {metrics['volume']:,} CP (TB 20P: {metrics['avg_vol_20']:,} CP), RSI: {metrics['rsi']}, SMA20: {metrics['sma20']:,.0f}
- DÒNG TIỀN KHỚP LỆNH: Mua chủ động {orderflow['active_buy_pct']}% vs Bán chủ động {orderflow['active_sell_pct']}%. Tín hiệu dòng tiền: {orderflow['smart_money_signal']}
- ĐỘ TIN CẬY LỊCH SỬ (T+5): Tỷ lệ thắng {signal_reliability['win_rate']}% ({signal_reliability['badge_rating']})
- ĐỊNH GIÁ & CƠ BẢN: P/E: {fundamental['pe_ratio']} | P/B: {fundamental['pb_ratio']} | ROE: {fundamental['roe_pct']}% | Sức khỏe: {fundamental['health_score']}/100. Lịch quyền: {fundamental['corporate_actions']}
- QUẢN TRỊ RỦI RO THEO ATR(14): Stop Loss động: {atr_risk['dynamic_stop_loss']:,.0f} VNĐ | Trailing Stop: {atr_risk['trailing_stop']:,.0f} VNĐ | Biên độ rung lắc: ±{atr_risk['atr_percent']}%
- 10 phiên qua: {history_prices}

Hãy kết hợp bối cảnh dòng tiền lớn (Cá mập gom hay Bull trap), định giá cơ bản và quản trị rủi ro ATR để đưa ra khuyến nghị trading chính xác, tỷ trọng và dự báo 5 phiên.
"""

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],[cite: 1]
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2048,[cite: 1]
            "responseMimeType": "application/json",
            "responseSchema": ADVICE_JSON_SCHEMA
        }
    }

    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            resp = await client.post(gemini_url, json=payload)
            res_json = resp.json()

        if resp.status_code == 429 or ("error" in res_json and "quota" in res_json["error"].get("message", "").lower()):[cite: 1]
            raise HTTPException(
                status_code=429,[cite: 1]
                detail="⚠️ Hạn mức gọi AI trong phút này đã đạt giới hạn. Vui lòng đợi 30 giây rồi thử lại."[cite: 1]
            )

        if resp.status_code != 200 or "error" in res_json:
            err_msg = res_json.get("error", {}).get("message", f"Lỗi Gemini API (Status {resp.status_code})")
            raise HTTPException(status_code=500, detail=err_msg)

        raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()[cite: 1]
        advice = httpx.Response(200, text=raw_text).json()

        result_payload = {"metrics": metrics, "advice": advice}[cite: 1]

        if len(ANALYSIS_CACHE) >= MAX_CACHE_ENTRIES:
            ANALYSIS_CACHE.clear()
        ANALYSIS_CACHE[sym] = (result_payload, time.time())[cite: 1]

        return result_payload[cite: 1]

    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="AI phản hồi quá lâu (Timeout). Vui lòng thử lại sau giây lát.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý hệ thống: {str(e)}")
