import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

from data_processing import load_and_preprocess

def tune_threshold(y_true, proba):
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.01, 0.99, 99):
        f1 = f1_score(y_true, (proba >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1
    return best_thr

def evaluate_with_threshold(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    thr = tune_threshold(y_te, model.predict_proba(X_te)[:, 1])
    p_te = model.predict_proba(X_te)[:, 1]
    preds = (p_te >= thr).astype(int)
    return (
        thr,
        roc_auc_score(y_te, p_te),
        average_precision_score(y_te, p_te),
        f1_score(y_te, preds, zero_division=0),
        accuracy_score(y_te, preds)
    )

if __name__ == "__main__":
    DATA_PATH = r"G:\assignment\DSML\data\dow_jones_index.csv"  # 请替换成你的数据路径
    df = load_and_preprocess(DATA_PATH, win=8, horizon=1)
    print(f"滑窗后样本数: {len(df)}, 正例数量: {df['fall_risk'].sum()}")

    X = df.drop(columns=["fall_risk", "stock", "date"]).values
    y = df["fall_risk"].values

    tscv = TimeSeriesSplit(n_splits=5)

    base_models = {
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42,
                                 n_estimators=120, max_depth=5, learning_rate=0.15,
                                 subsample=0.8, colsample_bytree=0.8),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                             solver="adam", max_iter=300, random_state=42),
        "LogReg": LogisticRegression(solver="liblinear", class_weight="balanced",
                                     max_iter=500, random_state=42)
    }

    voting = VotingClassifier(
        estimators=[
            ("xgb", base_models["XGBoost"]),
            ("mlp", base_models["MLP"]),
            ("logreg", base_models["LogReg"])
        ],
        voting="soft",
        weights=[1.5, 2.0, 1.0]
    )

    models = {**base_models, "Voting": voting}
    results = {name: {"thr": [], "roc": [], "pr": [], "f1": [], "acc": []} for name in models}

    for fold, (idx_tr, idx_te) in enumerate(tscv.split(X), 1):
        X_tr_raw, y_tr = X[idx_tr], y[idx_tr]
        X_te_raw, y_te = X[idx_te], y[idx_te]

        scaler = StandardScaler().fit(X_tr_raw)
        X_tr = scaler.transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)
        imp = SimpleImputer(strategy="mean").fit(X_tr)
        X_tr = imp.transform(X_tr)
        X_te = imp.transform(X_te)

        minority = np.sum(y_tr == 1)
        k = max(1, min(5, minority - 1))
        X_tr, y_tr = SMOTE(random_state=42, k_neighbors=k).fit_resample(X_tr, y_tr)
        print(f"\nFold {fold} 训练集正例:{y_tr.sum()}/{len(y_tr)} 测试集正例:{y_te.sum()}/{len(y_te)}  k={k}")

        for name, model in models.items():
            thr, roc, pr, f1, acc = evaluate_with_threshold(model, X_tr, y_tr, X_te, y_te)
            results[name]["thr"].append(thr)
            results[name]["roc"].append(roc)
            results[name]["pr"].append(pr)
            results[name]["f1"].append(f1)
            results[name]["acc"].append(acc)
            print(f"  {name:<8} 阈值={thr:.2f} | ROC={roc:.4f} | PR={pr:.4f} | F1={f1:.4f} | Acc={acc:.4f}")

    print("\n===== 交叉验证平均结果 =====")
    for name, m in results.items():
        print(f"{name:<8} | thr={np.mean(m['thr']):.2f} | ROC={np.nanmean(m['roc']):.4f} | PR={np.nanmean(m['pr']):.4f} | F1={np.mean(m['f1']):.4f}")
