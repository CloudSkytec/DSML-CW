import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')


# ==============================
#  数据读取 + 预处理 + 特征工程
# ==============================

def load_and_preprocess(path: str) -> pd.DataFrame:
    """
    读取 Dow Jones Index CSV → 清洗 → 技术指标特征 → 生成目标变量 fall_risk
    ---------------------------------------------------------------
    返回：已去 NaN 并按日期排序的 DataFrame
    """
    df = pd.read_csv(path)

    # ---------- 1) 基本清洗 ----------
    money_cols = ['open', 'high', 'low', 'close', 'volume']
    for c in money_cols:
        if c in df.columns:
            df[c] = df[c].replace('[\\$,]', '', regex=True).astype(float)

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # ---------- 2) 技术指标 ----------
    # EMA / SMA
    df['EMA3'] = talib.EMA(df['close'], timeperiod=3)
    df['EMA5'] = talib.EMA(df['close'], timeperiod=5)
    df['EMA_diff'] = df['EMA3'] - df['EMA5']

    df['SMA3'] = talib.SMA(df['close'], timeperiod=3)
    df['SMA5'] = talib.SMA(df['close'], timeperiod=5)
    df['SMA_diff'] = df['SMA3'] - df['SMA5']

    # RSI
    df['RSI7'] = talib.RSI(df['close'], timeperiod=7)
    df['RSI3'] = talib.RSI(df['close'], timeperiod=3)
    df['RSI_diff'] = df['RSI7'] - df['RSI3']

    # ATR
    df['ATR7'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=7)
    df['ATR3'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=3)
    df['ATR_chg'] = (df['ATR7'] - df['ATR3']) / df['ATR7']

    # ROC
    df['ROC3'] = talib.ROC(df['close'], timeperiod=3)
    df['ROC5'] = talib.ROC(df['close'], timeperiod=5)

    # 成交量派生特征
    df['vol_change'] = df['volume'].pct_change()
    df['vol_price_div'] = (
            (df['close'] > df['close'].shift(1)) &
            (df['volume'] < df['volume'].shift(1))
    ).astype(int)

    # 支撑 / 阻力
    df['support3w'] = df['low'].rolling(3).min()
    df['resistance3w'] = df['high'].rolling(3).max()

    # 高低价比 & 对数变换
    df['high_low_ratio'] = df['high'] / df['low']
    df['ln_high_low_ratio'] = np.log(df['high_low_ratio'])

    # 月末周标记
    df['is_month_end_week'] = (df['date'].dt.month != df['date'].shift(-1).dt.month).astype(int)

    # ---------- 3) 目标变量 ----------
    df['weekly_ret'] = df['close'].shift(-1).sub(df['close']).div(df['close'])
    df['fall_risk'] = (df['weekly_ret'] < -0.05).astype(int)  # 下周跌幅 >5% 判为 1

    # 去除所有 NaN（主要来自技术指标首尾缺口）
    return df.dropna().reset_index(drop=True)


DATA_PATH = "./dow_jones_index.data"
df = load_and_preprocess(DATA_PATH)

feature_cols = [
    'EMA3', 'EMA5', 'SMA3', 'SMA5', 'RSI7', 'RSI3',
    'ATR7', 'ATR3', 'ROC3', 'ROC5', 'vol_change', 'vol_price_div',
    'support3w', 'resistance3w', 'high_low_ratio', 'ln_high_low_ratio',
    'is_month_end_week'
]

X = df[feature_cols].values  # 特征矩阵
y = df['fall_risk'].values  # 标签


# ---------- Dataset ---------- #
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X[:, None, :], dtype=torch.float32)  # [batch, 1, features]
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, i): return self.X[i], self.y[i]


# ---------- LSTM + Attention 分类器 ---------- #
class LSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(hidden_dim, 1)  # attention打分
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: [batch, seq_len=1, feature]
        output, _ = self.lstm(x)  # output: [batch, seq_len, hidden]
        attn_weights = torch.softmax(self.attention(output), dim=1)  # [batch, seq_len, 1]
        attn_applied = torch.sum(output * attn_weights, dim=1)  # 加权求和 [batch, hidden]
        out = self.dropout(attn_applied)
        return torch.sigmoid(self.fc(out)).squeeze()


# ---------- 阈值搜索 ---------- #
def find_best_threshold(y_true, probas):
    best_thr, best_f1 = 0.5, 0
    for thr in np.linspace(0.01, 0.99, 99):
        f1 = f1_score(y_true, (probas >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_thr = thr
            best_f1 = f1
    return best_thr


# ========== 初始化 ROC / PR 收集器 ==========
roc_data_list = []
pr_data_list = []
auc_list = []
ap_list = []

# ---------- Training + Evaluation ---------- #
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tscv = TimeSeriesSplit(n_splits=5)

for fold, (i_tr, i_te) in enumerate(tscv.split(X), 1):
    X_tr_raw, X_te_raw = X[i_tr], X[i_te]
    y_tr, y_te = y[i_tr], y[i_te]

    # 标准化
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    # SMOTE
    minority = min(np.bincount(y_tr))
    k = max(1, min(5, minority - 1))
    X_tr, y_tr = SMOTE(random_state=42, k_neighbors=k).fit_resample(X_tr, y_tr)
    print(f"Fold {fold}: 少数类={minority}, k={k}")

    # Dataloader
    train_loader = DataLoader(TabularDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    test_loader = DataLoader(TabularDataset(X_te, y_te), batch_size=256)

    # 模型
    model = LSTM_Attention(input_dim=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # 训练
    model.train()
    for epoch in range(20):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

    # 测试
    model.eval()
    all_probs = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            all_probs.append(model(xb).cpu())
    probs = torch.cat(all_probs).numpy()
    thr = find_best_threshold(y_te, probs)
    preds = (probs >= thr).astype(int)

    # 评估
    roc_auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else float('nan')
    f1 = f1_score(y_te, preds, zero_division=0)
    acc = accuracy_score(y_te, preds)

    print(f"Fold {fold}: thr={thr:.2f} | AUC={roc_auc:.3f} | F1={f1:.3f} | ACC={acc:.3f}")

    # ========== 每折收集 ROC / PR 点 ==========
    fpr, tpr, _ = roc_curve(y_te, probs)
    precision, recall, _ = precision_recall_curve(y_te, probs)
    pr_auc = auc(recall, precision)

    roc_data_list.append((fpr, tpr))
    pr_data_list.append((recall, precision))
    auc_list.append(roc_auc)
    ap_list.append(pr_auc)

# ========== 画平均 ROC & PR 曲线 ==========
plt.figure(figsize=(12, 5))
mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)
tprs, prs = [], []

# ROC
plt.subplot(1, 2, 1)
for i, (fpr, tpr) in enumerate(roc_data_list):
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    tprs.append(interp_tpr)
    plt.plot(fpr, tpr, alpha=0.5, label=f"Fold {i + 1} AUC={auc_list[i]:.3f}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.plot(mean_fpr, np.mean(tprs, axis=0), color='black', lw=2, label=f"Mean AUC={np.mean(auc_list):.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (All Folds)")
plt.legend()
plt.grid(True)

# PR
plt.subplot(1, 2, 2)
for i, (recall, precision) in enumerate(pr_data_list):
    interp_prec = np.interp(mean_recall, recall[::-1], precision[::-1])
    prs.append(interp_prec)
    plt.plot(recall, precision, alpha=0.5, label=f"Fold {i + 1} AP={ap_list[i]:.3f}")
plt.plot(mean_recall, np.mean(prs, axis=0), color='black', lw=2, label=f"Mean AP={np.mean(ap_list):.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve (All Folds)")
plt.legend()
plt.grid(True)

plt.suptitle("Cross-Validation ROC & PR Curves", fontsize=14)
plt.tight_layout()
plt.show()
