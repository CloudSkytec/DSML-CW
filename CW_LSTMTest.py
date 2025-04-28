from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, roc_curve, \
    precision_recall_curve
import warnings

warnings.filterwarnings("ignore")


# ========== 工具函数 ==========
def _zscore(series: pd.Series, window: int) -> pd.Series:
    return (series - series.rolling(window).mean()) / series.rolling(window).std(ddof=0)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.replace(0, np.nan)
    return a / b


FEATURES = ["sma_5", "sma_3_5_diff", "sma_3", "ema_5", "ema_diff_3_5", "ema_3", "price_ma5_deviation",
            "rsi_5", "rsi_diff_3_5", "rsi_3", "macd_hist_3_7_3", "momentum_5", "roc_5",
            "atr_5", "atr_chg_7_3", "atr_3", "volatility_5", "high_low_ratio", "ln_high_low_ratio",
            "vol_change", "volume_z_5", "price_volume_ratio", "vpt", "close_z_5", "support3w_distance"]
TIME_FEATURES = ["is_month_end", "week_of_month", "month", "quarter"]


# ========== 滑动窗口 ==========
def make_sliding(df_feat: pd.DataFrame, feature_cols: list[str], win: int = 8, horizon: int = 1) -> pd.DataFrame:
    samples = []
    for stock, group in df_feat.groupby("stock", sort=False):
        group = group.reset_index(drop=True)
        total = len(group) - win - horizon + 1
        if total <= 0:
            continue
        for start in range(total):
            window = group.iloc[start: start + win]
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


# ========== 数据预处理 ==========
def load_and_preprocess(path: str | Path, win: int = 8, horizon: int = 1) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["open", "high", "low", "close", "volume", "next_weeks_open", "next_weeks_close",
                "days_to_next_dividend"]:
        if col in df.columns:
            df[col] = df[col].replace(r"[\\$,]", "", regex=True).astype(float)
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
        g["price_volume_ratio"] = _safe_div(g["pct_change_price"], g.get("percent_change_volume_over_last_wk",
                                                                         pd.Series(np.nan, index=g.index)))
        g["vpt"] = (g["pct_change_price"] * g["volume"]).cumsum()
        g["close_z_5"] = _zscore(g["close"], 5)
        g["support3w_distance"] = (g["close"] / g["low"].rolling(15).min()) - 1
        g["is_month_end"] = g["date"].dt.is_month_end.astype(int)
        g["week_of_month"] = (g["date"].dt.day - 1) // 7 + 1
        g["month"] = g["date"].dt.month
        g["quarter"] = g["date"].dt.quarter

        g = g.replace([np.inf, -np.inf], np.nan)
        num_cols = g.select_dtypes(include=[np.number]).columns
        g[num_cols] = g[num_cols].fillna(g[num_cols].mean())

        enriched.append(g)

    df_feat = pd.concat(enriched, ignore_index=True)
    df_feat["weekly_ret"] = (df_feat["close"].shift(-1) - df_feat["close"]) / df_feat["close"]
    df_feat["fall_risk"] = (df_feat["weekly_ret"] < -0.05).astype(int)

    return make_sliding(df_feat, FEATURES + TIME_FEATURES, win, horizon)


# ========== 模型定义 ==========
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X[:, None, :], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = self.drop(h[-1])
        return torch.sigmoid(self.fc(h)).squeeze()


# ========== 阈值搜索 ==========
def tune_threshold(y_true, y_prob):
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.01, 0.99, 99):
        preds = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_thr = thr
            best_f1 = f1
    return best_thr


# ==========================================
# 主程序
# ==========================================
DATA_PATH = "./dow_jones_index.data"
df = load_and_preprocess(DATA_PATH)
X = df.drop(columns=["fall_risk", "stock", "date"]).values
y = df["fall_risk"].values

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tscv = TimeSeriesSplit(n_splits=5)

all_fpr, all_tpr, all_prec, all_recall = [], [], [], []

results = {"thr": [], "roc": [], "pr": [], "f1": [], "acc": []}

for fold, (idx_tr, idx_te) in enumerate(tscv.split(X), 1):
    X_tr_raw, y_tr = X[idx_tr], y[idx_tr]
    X_te_raw, y_te = X[idx_te], y[idx_te]

    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    imputer = SimpleImputer(strategy="mean").fit(X_tr)
    X_tr = imputer.transform(X_tr)
    X_te = imputer.transform(X_te)

    minority = np.sum(y_tr == 1)
    k = max(1, min(5, minority - 1))
    X_tr, y_tr = SMOTE(random_state=42, k_neighbors=k).fit_resample(X_tr, y_tr)

    train_loader = DataLoader(TabularDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    test_dataset = TabularDataset(X_te, y_te)

    model = LSTMClassifier(input_dim=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    # EarlyStopping机制
    best_val_loss, patience, PATIENCE = np.inf, 0, 5
    best_state = None

    for epoch in range(50):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_probs = model(test_dataset.X.to(device)).cpu()
            val_loss = loss_fn(val_probs, test_dataset.y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        probs = model(test_dataset.X.to(device)).cpu().numpy()
    best_thr = tune_threshold(y_te, probs)
    preds = (probs >= best_thr).astype(int)

    fpr, tpr, _ = roc_curve(y_te, probs)
    prec, recall, _ = precision_recall_curve(y_te, probs)

    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_prec.append(prec)
    all_recall.append(recall)

    roc = roc_auc_score(y_te, probs)
    pr = average_precision_score(y_te, probs)
    f1 = f1_score(y_te, preds, zero_division=0)
    acc = accuracy_score(y_te, preds)

    results["thr"].append(best_thr)
    results["roc"].append(roc)
    results["pr"].append(pr)
    results["f1"].append(f1)
    results["acc"].append(acc)

    print(f"Fold {fold}: thr={best_thr:.2f} | ROC={roc:.3f} | PR={pr:.3f} | F1={f1:.3f} | ACC={acc:.3f}")

# ========== 汇总 ==========
print("\n===== LSTM交叉验证平均结果 =====")
print(f"thr={np.mean(results['thr']):.2f} |"
      f" ROC={np.mean(results['roc']):.4f} |"
      f" PR={np.mean(results['pr']):.4f} |"
      f" F1={np.mean(results['f1']):.4f} |"
      f" Acc={np.mean(results['acc']):.4f}")

# ========== 绘制 ROC 和 PR 曲线 ==========
plt.figure(figsize=(12, 5))

# ROC
plt.subplot(1, 2, 1)
for i in range(5):
    plt.plot(all_fpr[i], all_tpr[i], label=f"Fold {i + 1}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title("ROC Curve (5 Folds)")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()

# PR
plt.subplot(1, 2, 2)
for i in range(5):
    plt.plot(all_recall[i], all_prec[i], label=f"Fold {i + 1}")
plt.title("PR Curve (5 Folds)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

plt.tight_layout()
plt.show()
