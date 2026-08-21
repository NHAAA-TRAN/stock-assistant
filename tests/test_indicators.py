import pytest
import pandas as pd
import numpy as np

# Import trực tiếp các hàm logic từ api/index.py
from api.index import (
    calculate_rsi_wilder,
    calculate_atr,
    calculate_orderflow_pressure,
    calculate_signal_reliability,
    evaluate_realtime_triggers
)

@pytest.fixture
def sample_stock_df():
    """Tạo tập dữ liệu OHLCV 60 phiên giả lập cho việc test chỉ báo"""
    np.random.seed(42)
    n = 60
    base_price = 28000.0
    returns = np.random.normal(0.001, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)

    data = {
        "Open": prices * (1 - np.random.uniform(0.001, 0.005, n)),
        "High": prices * (1 + np.random.uniform(0.005, 0.015, n)),
        "Low": prices * (1 - np.random.uniform(0.005, 0.015, n)),
        "Close": prices,
        "Volume": np.random.randint(5_000_000, 25_000_000, n)
    }
    df = pd.DataFrame(data)
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    return df


def test_rsi_wilder_bounds(sample_stock_df):
    """Kiểm tra RSI luôn nằm trong khoảng [0, 100] và không trả về NaN ở đuôi dữ liệu"""
    rsi = calculate_rsi_wilder(sample_stock_df["Close"], period=14)
    valid_rsi = rsi.dropna()

    assert not valid_rsi.empty
    assert (valid_rsi >= 0).all()
    assert (valid_rsi <= 100).all()


def test_atr_positive(sample_stock_df):
    """Kiểm tra ATR(14) luôn mang giá trị dương"""
    atr = calculate_atr(sample_stock_df, period=14)
    valid_atr = atr.dropna()

    assert not valid_atr.empty
    assert (valid_atr > 0).all()


def test_orderflow_volume_consistency(sample_stock_df):
    """Kiểm tra tổng khối lượng Mua + Bán chủ động bằng đúng tổng khối lượng khớp lệnh"""
    of = calculate_orderflow_pressure(sample_stock_df)
    
    total_vol = int(sample_stock_df["Volume"].iloc[-1])
    assert of["active_buy_volume"] + of["active_sell_volume"] == total_vol
    assert 0 <= of["active_buy_pct"] <= 100
    assert 0 <= of["active_sell_pct"] <= 100


def test_realtime_triggers():
    """Kiểm tra hệ thống trigger cảnh báo khi RSI quá bán hoặc giá chạm vùng mua"""
    # Trường hợp 1: RSI quá bán
    oversold_alert = evaluate_realtime_triggers(curr_price=28000, rsi=28.5, dynamic_buy_zone=26000)
    assert oversold_alert["has_active_alert"] is True
    assert any("QUÁ BÁN" in msg for msg in oversold_alert["alert_messages"])

    # Trường hợp 2: Trạng thái bình thường
    normal_alert = evaluate_realtime_triggers(curr_price=28000, rsi=55.0, dynamic_buy_zone=25000)
    assert normal_alert["has_active_alert"] is False
