# 📈 VN Stock AI Advisor & Pro Trading Engine

Hệ thống phân tích kỹ thuật, sàng lọc cổ phiếu và khuyến nghị đầu tư tự động cho thị trường chứng khoán Việt Nam sử dụng FastAPI và Google Gemini 2.5 Flash.

## ✨ Tính năng chính
- **Macro VN-Index & Market Sentiment:** Tự động định giá bối cảnh vĩ mô toàn thị trường.
- **Microstructure Orderflow:** Phân tích tỷ lệ Mua/Bán chủ động, phát hiện dòng tiền Cá mập và Bull trap.
- **RSI Wilder & ATR Dynamic Risk:** Tính toán điểm dừng lỗ động (-2x ATR) và Trailing Stop.
- **Độ Tin Cậy Tín Hiệu (T+5):** Thuật toán backtest lịch sử đo lường Win Rate trên từng mã.
- **Daily Market Screener:** Bộ lọc Top 5 mã Breakout sau phiên ATC.

## 🚀 Hướng dẫn cài đặt & Chạy cục bộ

1. **Cài đặt môi trường:**
   ```bash
   pip install -r requirements.txt
