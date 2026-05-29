import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()  
    # Tạo bản sao — không sửa df gốc
    # Nếu không copy, df bên ngoài cũng bị thay đổi theo

    result["ma_7"] = result["close"].rolling(window=7).mean()
    # rolling(window=7) = "cửa sổ trượt 7 nến"
    # .mean() = tính trung bình trong cửa sổ đó
    # Kết quả gán vào cột mới "ma_7"

    result["ma_25"] = result["close"].rolling(window=25).mean()
    result["ma_99"] = result["close"].rolling(window=99).mean()
    return result  # trả về df đã có thêm 3 cột mới


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    result = df.copy()

    result["rsi"] = RSIIndicator(
        close=result["close"],  # truyền cột giá close vào
        window=period           # period=14 theo mặc định
    ).rsi()                     # .rsi() để lấy ra giá trị RSI
    return result


def add_bollinger_bands(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    result = df.copy()

    bands = BollingerBands(
        close=result["close"],  # truyền giá close vào
        window=period           # period=20 theo mặc định
    )
    result["bb_upper"] = bands.bollinger_hband()   # dải trên
    result["bb_middle"] = bands.bollinger_mavg()   # đường giữa (SMA20)
    result["bb_lower"] = bands.bollinger_lband()   # dải dưới
    return result


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = add_moving_averages(df)   # bước 1: thêm MA
    result = add_rsi(result)           # bước 2: thêm RSI
    result = add_bollinger_bands(result) # bước 3: thêm BB
    return result
    # Mỗi bước nhận df từ bước trước
    # df cuối cùng có đủ tất cả 7 cột mới
