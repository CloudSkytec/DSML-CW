from pathlib import Path
import numpy as np
import pandas as pd
import talib

def _zscore(series: pd.Series, window: int) -> pd.Series:
    return (series - series.rolling(window).mean()) / series.rolling(window).std(ddof=0)

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.replace(0, np.nan)
    return a / b

FEATURES = [
    "sma_5","sma_3_5_diff","sma_3",
    "ema_5","ema_diff_3_5","ema_3","price_ma5_deviation",
    "rsi_5","rsi_diff_3_5","rsi_3",
    "macd_hist_3_7_3","momentum_5","roc_5",
    "atr_5","atr_chg_7_3","atr_3",
    "volatility_5","high_low_ratio","ln_high_low_ratio",
    "vol_change","volume_z_5","price_volume_ratio",
    "vpt","close_z_5","support3w_distance"
]
TIME_FEATURES = ["is_month_end", "week_of_month", "month", "quarter"]

def make_sliding(df_feat: pd.DataFrame, feature_cols: list[str], win: int = 8, horizon: int = 1) -> pd.DataFrame:
    samples = []
    for stock, group in df_feat.groupby("stock", sort=False):
        group = group.reset_index(drop=True)
        total = len(group) - win - horizon + 1
        if total <= 0:
            continue
        for start in range(total):
            window = group.iloc[start : start + win]
            target_row = group.iloc[start + win + horizon - 1]
            record = target_row[feature_cols].to_dict()
            record.update({
                "win_close_mean": window["close"].mean(),
                "win_close_std": window["close"].std(ddof=0),
                "win_pctchg_sum": window["pct_change_price"].sum(),
                "win_volatility": window["pct_change_price"].std(ddof=0),
                "win_rsi_mean": window["rsi_5"].mean(),
            })
            record["stock"] = stock
            record["date"] = target_row["date"]
            record["fall_risk"] = target_row["fall_risk"]
            samples.append(record)
    return pd.DataFrame(samples)

def load_and_preprocess(path: str | Path, win: int = 8, horizon: int = 1) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["open", "high", "low", "close", "volume",
                "next_weeks_open", "next_weeks_close", "days_to_next_dividend"]:
        if col in df.columns:
            df[col] = df[col].replace("[\$,]", "", regex=True).astype(float)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock", "date"]).reset_index(drop=True)

    enriched = []
    for _, group in df.groupby("stock", sort=False):
        g = group.copy()
        g["pct_change_price"] = g["close"].pct_change()
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
        g["price_volume_ratio"] = _safe_div(
            g["pct_change_price"],
            g.get("percent_change_volume_over_last_wk", pd.Series(np.nan, index=g.index))
        )
        g["vpt"] = (g["pct_change_price"] * g["volume"]).cumsum()
        g["close_z_5"] = _zscore(g["close"], 5)
        g["support3w_distance"] = (g["close"] / g["low"].rolling(15).min()) - 1
        g["is_month_end"] = g["date"].dt.is_month_end.astype(int)
        g["week_of_month"] = (g["date"].dt.day - 1) // 7 + 1
        g["month"] = g["date"].dt.month
        g["quarter"] = g["date"].dt.quarter

        num_cols = g.select_dtypes(include=[np.number]).columns
        g[num_cols] = g[num_cols].replace([np.inf, -np.inf], np.nan)
        g[num_cols] = g[num_cols].fillna(g[num_cols].mean())

        enriched.append(g)

    df_feat = pd.concat(enriched, ignore_index=True)
    df_feat["weekly_ret"] = (df_feat["close"].shift(-1) - df_feat["close"]) / df_feat["close"]
    df_feat["fall_risk"] = (df_feat["weekly_ret"] < -0.05).astype(int)

    df_slid = make_sliding(df_feat, FEATURES + TIME_FEATURES, win=win, horizon=horizon)
    return df_slid.reset_index(drop=True)
