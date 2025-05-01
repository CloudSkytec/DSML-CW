from pathlib import Path
import numpy as np
import pandas as pd
import talib


def _zscore(series: pd.Series, window: int) -> pd.Series:
    return (series - series.rolling(window).mean()) / (series.rolling(window).std(ddof=0) + 1e-6)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.replace(0, np.nan)
    return a / b


WINS = [8]

FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
    'pct_change_price', 'percent_change_volume_over_last_wk',
    'sma_3', 'sma_5',
    'ema_3', 'ema_5',
    'price_ma5_deviation',
    'rsi_3', 'rsi_5',
    'macd_hist_3_7_3',
    'momentum_5', 'roc_5',
    'atr_3', 'atr_5', 'atr_7', 'atr_chg_7_3',
    'volatility_5',
    'ln_high_low_ratio',
    'vol_change', 'volume_z_5',
    'price_volume_ratio', 'vpt', 'close_z_5',
    'support3w_distance',

    'cumulative_return_5w',
    'max_drawdown_5w',
    'bollinger_band_width_5w',
    'stochastic_k_5w',
    'stochastic_d_5w'
]
TIME_FEATURES = ['is_month_end', 'week_of_month', 'month', 'quarter']


# 不完整窗口
def make_sliding_allow_incomplete(df_feat: pd.DataFrame, feature_cols, win: int = 8, horizon: int = 1,
                                  step: int = 1) -> pd.DataFrame:
    samples = []
    for stock, group in df_feat.groupby("stock", sort=False):
        group = group.reset_index(drop=True)
        n = len(group)
        max_start = n - horizon
        if max_start <= 0:
            continue
        for start in range(0, max_start, step):
            end = min(start + win, n)
            window = group.iloc[start:end]
            target = group.iloc[min(start + win + horizon - 1, n - 1)]
            rec = target[feature_cols].to_dict()
            rec.update({
                'win_close_mean': window['close'].mean(),
                'win_close_std': window['close'].std(ddof=0),
                'win_pctchg_sum': window['pct_change_price'].sum(),
                'win_volatility': window['pct_change_price'].std(ddof=0),
                'win_rsi_mean': window['rsi_3'].mean(),
                'actual_window_len': len(window)
            })
            rec['stock'] = stock
            rec['date'] = target['date']
            rec['fall_risk'] = target['fall_risk']
            samples.append(rec)
    return pd.DataFrame(samples)


# 主函数
def load_and_preprocess(path: str | Path, win: int = 8, horizon: int = 1, step: int = 1) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 清洗数据，替换符号并转为浮动数据类型
    for col in ["open", "high", "low", "close", "volume",
                "next_weeks_open", "next_weeks_close", "days_to_next_dividend", "percent_return_next_dividend"]:
        if col in df.columns:
            df[col] = df[col].replace("[$,]", "", regex=True).astype(float)

    # 处理日期列
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock', 'date']).reset_index(drop=True)

    enriched = []
    for _, group in df.groupby('stock', sort=False):
        g = group.copy()
        g["pct_change_price"] = g["close"].pct_change()
        g["percent_change_volume_over_last_wk"] = g["volume"].pct_change(periods=5)
        g["sma_3"] = talib.SMA(g["close"], 3)
        g["sma_5"] = talib.SMA(g["close"], 5)
        g["sma_3_5_diff"] = g["sma_3"] - g["sma_5"]
        g["ema_3"] = talib.EMA(g["close"], 3)
        g["ema_5"] = talib.EMA(g["close"], 5)
        g["ema_diff_3_5"] = g["ema_3"] - g["ema_5"]
        g["price_ma5_deviation"] = (g["close"] / g["sma_5"]) - 1
        g["rsi_3"] = talib.RSI(g["close"], 3)
        g["rsi_5"] = talib.RSI(g["close"], 5)
        g["rsi_diff_3_5"] = g["rsi_3"] - g["rsi_5"]
        _, _, macd_hist = talib.MACD(g["close"], 3, 7, 3)
        g["macd_hist_3_7_3"] = macd_hist
        g["momentum_5"] = g["close"].pct_change(5)
        g["roc_5"] = talib.ROC(g["close"], 5)
        g["atr_3"] = talib.ATR(g["high"], g["low"], g["close"], 3)
        g["atr_5"] = talib.ATR(g["high"], g["low"], g["close"], 5)
        g["atr_7"] = talib.ATR(g["high"], g["low"], g["close"], 7)
        g["atr_chg_7_3"] = (g["atr_7"] - g["atr_3"]) / g["atr_7"]
        g["volatility_5"] = g["pct_change_price"].rolling(5).std()
        g["high_low_ratio"] = g["high"] / g["low"]
        g["ln_high_low_ratio"] = np.log(g["high_low_ratio"])
        g["vol_change"] = g["volume"].pct_change()
        g["volume_z_5"] = _zscore(g["volume"], 5)
        g["price_volume_ratio"] = _safe_div(g["pct_change_price"], g["percent_change_volume_over_last_wk"])
        g["vpt"] = (g["pct_change_price"] * g["volume"]).cumsum()
        g["close_z_5"] = _zscore(g["close"], 5)
        g["support3w_distance"] = (g["close"] / g["low"].rolling(15).min()) - 1
        g["is_month_end"] = g["date"].dt.is_month_end.astype(int)
        g["week_of_month"] = (g["date"].dt.day - 1) // 7 + 1
        g["month"] = g["date"].dt.month
        g["quarter"] = g["date"].dt.quarter
        g["cumulative_return_5w"] = g["close"].pct_change().add(1).rolling(5).apply(np.prod, raw=True) - 1
        g["max_drawdown_5w"] = (g["close"].rolling(5, min_periods=1).max() - g["close"]) / g["close"].rolling(5,
                                                                                                              min_periods=1).max()
        # 计算布林带宽度
        rolling_mean = g["close"].rolling(5)
        g["bollinger_band_width_5w"] = 4 * rolling_mean.std() / rolling_mean.mean()

        # 随机指标 K 和 D
        lowest_low = g["low"].rolling(5).min()
        highest_high = g["high"].rolling(5).max()
        g["stochastic_k_5w"] = 100 * (g["close"] - lowest_low) / (highest_high - lowest_low + 1e-6)
        g["stochastic_d_5w"] = g["stochastic_k_5w"].rolling(3).mean()

        g[FEATURES] = g[FEATURES].ffill()
        for c in FEATURES:
            g[c] = g[c].fillna(g[c].mean())
        g[FEATURES] = g[FEATURES].fillna(0)
        enriched.append(g)

    df_feat = pd.concat(enriched, ignore_index=True)
    df_feat['weekly_ret'] = (df_feat['close'].shift(-1) - df_feat['close']) / df_feat['close']
    df_feat['fall_risk'] = (df_feat['weekly_ret'] < -0.05).astype(int)

    feature_cols = FEATURES + TIME_FEATURES

    # 使用不完整窗口
    df_slid = make_sliding_allow_incomplete(df_feat, feature_cols, win=win, horizon=horizon, step=step)

    return df_slid.reset_index(drop=True)
