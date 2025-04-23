from collections import Counter
import talib, pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def load_and_preprocess(path):
    # 读 CSV，清掉价格和成交量里的符号
    df = pd.read_csv(path)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = df[c].replace('[\$,]', '', regex=True).astype(float)

    # 转时间戳并按时间排一下
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 特征
    df['EMA3'] = talib.EMA(df['close'], timeperiod=3)
    df['EMA5'] = talib.EMA(df['close'], timeperiod=5)
    df['EMA_diff'] = df['EMA3'] - df['EMA5']

    df['SMA3'] = talib.SMA(df['close'], timeperiod=3)
    df['SMA5'] = talib.SMA(df['close'], timeperiod=5)
    df['SMA_diff'] = df['SMA3'] - df['SMA5']

    df['RSI7'] = talib.RSI(df['close'], timeperiod=7)
    df['RSI3'] = talib.RSI(df['close'], timeperiod=3)
    df['RSI_diff'] = df['RSI7'] - df['RSI3']

    df['ATR7'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=7)
    df['ATR3'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=3)
    df['ATR_chg'] = (df['ATR7'] - df['ATR3']) / df['ATR7']

    df['ROC3'] = talib.ROC(df['close'], timeperiod=3)
    df['ROC5'] = talib.ROC(df['close'], timeperiod=5)

    df['vol_change'] = df['volume'].pct_change()
    df['vol_price_div'] = (
        (df['close'] > df['close'].shift(1)) &
        (df['volume'] < df['volume'].shift(1))
    ).astype(int)

    df['support3w'] = df['low'].rolling(3).min()
    df['resistance3w'] = df['high'].rolling(3).max()

    df['high_low_ratio'] = df['high'] / df['low']
    df['ln_high_low_ratio'] = np.log(df['high_low_ratio'])

    df['is_month_end_week'] = (
        df['date'].dt.month != df['date'].shift(-1).dt.month
    ).astype(int)

    # 下周跌超5%就标1，否则0
    df['weekly_ret'] = df['close'].shift(-1).sub(df['close']).div(df['close'])
    df['fall_risk'] = (df['weekly_ret'] < -0.05).astype(int)

    return df.dropna().reset_index(drop=True)


def tune_threshold(y_true, proba):
    # 找到训练集上 F1 最高的阈值
    best_thr, best_f1 = 0.5, 0
    for thr in np.linspace(0.01, 0.99, 99):
        preds = (proba >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr


def evaluate_with_threshold(model, X_tr, y_tr, X_te, y_te):
    # 训练 + 在训练集上找阈值，再测测试集
    model.fit(X_tr, y_tr)
    p_tr = model.predict_proba(X_tr)[:, 1]
    thr = tune_threshold(y_tr, p_tr)

    p_te = model.predict_proba(X_te)[:, 1]
    preds = (p_te >= thr).astype(int)

    roc = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) > 1 else float('nan')
    f1 = f1_score(y_te, preds, zero_division=0)
    acc = accuracy_score(y_te, preds)
    return thr, roc, f1, acc

DATA_PATH = r'G:\assignment\DSML\data\dow_jones_index.csv'

# 读数据算特征
df = load_and_preprocess(DATA_PATH)
features = [
    'EMA3', 'EMA5', 'SMA3', 'SMA5', 'RSI7', 'RSI3',
    'ATR7', 'ATR3', 'ROC3', 'ROC5',
    'vol_change', 'vol_price_div',
    'support3w', 'resistance3w',
    'high_low_ratio', 'ln_high_low_ratio',
    'is_month_end_week'
]
X = df[features].values
y = df['fall_risk'].values

# 时间序列分 5 折
tscv = TimeSeriesSplit(n_splits=5)
results = {m: {'thr': [], 'roc': [], 'f1': [], 'acc': []}
           for m in ['XGBoost', 'MLP', 'LogReg']}

for fold, (i_tr, i_te) in enumerate(tscv.split(X), 1):
    X_tr_raw, X_te_raw = X[i_tr], X[i_te]
    y_tr, y_te = y[i_tr], y[i_te]

    # 特征标准化
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    # SMOTE 过采样 少数类样本少就调小 k
    minority = min(Counter(y_tr).values())
    k = max(1, min(5, minority - 1))
    X_tr, y_tr = SMOTE(random_state=42, k_neighbors=k).fit_resample(X_tr, y_tr)
    print(f"Fold {fold}: 少数类={minority}, k={k}")

    # 三个模型依次测
    for name, Model in [('XGBoost', XGBClassifier),
                        ('MLP', MLPClassifier),
                        ('LogReg', LogisticRegression)]:
        if name == 'XGBoost':
            model = Model(eval_metric='logloss', random_state=42,
                          n_estimators=100, max_depth=9,
                          learning_rate=0.2, subsample=0.8,
                          colsample_bytree=0.8, scale_pos_weight=1)
        elif name == 'MLP':
            model = Model(hidden_layer_sizes=(100,), activation='relu',
                          solver='adam', max_iter=200, random_state=42)
        else:
            model = Model(solver='liblinear', class_weight='balanced',
                          max_iter=200, random_state=42)

        thr, roc, f1, acc = evaluate_with_threshold(model, X_tr, y_tr, X_te, y_te)
        results[name]['thr'].append(thr)
        results[name]['roc'].append(roc)
        results[name]['f1'].append(f1)
        results[name]['acc'].append(acc)
        print(f"  {name}: thr={thr:.2f} | ROC={roc:.4f} | F1={f1:.4f} | Acc={acc:.4f}")

# 打印平均成绩
print("Average")
for name, m in results.items():
    print(f"{name}: thr={np.mean(m['thr']):.2f}, ROC={np.nanmean(m['roc']):.4f}, "
          f"F1={np.mean(m['f1']):.4f}, Acc={np.mean(m['acc']):.4f}")
