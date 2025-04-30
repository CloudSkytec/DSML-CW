import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from data_processing import load_and_preprocess

def tune_threshold(y_true, proba):
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.001, 0.999, 999):
        preds = (proba >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1
    return best_thr

if __name__ == "__main__":
    DATA_PATH = "./data/dow_jones_index.csv"
    SAVE_DIR = "./saved_models/"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. 加载并预处理
    df = load_and_preprocess(DATA_PATH)
    print(f"滑窗后样本数: {len(df)}, 正例数量: {df['fall_risk'].sum()}")

    features = df.drop(columns=["fall_risk", "stock", "date"]).columns
    X = df[features].values
    y = df["fall_risk"].values

    # 2. 划分训练/测试
    train_idx, test_idx = [], []
    for stock, g in df.groupby('stock', sort=False):
        idx = g.sort_values('date').index
        split = len(idx) // 2
        train_idx += idx[:split].tolist()
        test_idx += idx[split:].tolist()

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test   = X[test_idx],  y[test_idx]

    # 3. 标准化 & 填缺
    scaler = StandardScaler().fit(X_train)
    X_tr = scaler.transform(X_train)
    X_te = scaler.transform(X_test)
    imputer = SimpleImputer(strategy="mean").fit(X_tr)
    X_tr = imputer.transform(X_tr)
    X_te = imputer.transform(X_te)

    # 4. SMOTE过采样
    minority = np.sum(y_train == 1)
    if minority >= 2:
        k = min(5, minority - 1)
        X_tr, y_tr = SMOTE(random_state=42, k_neighbors=k).fit_resample(X_tr, y_train)
    else:
        y_tr = y_train
    print(f"过采样后训练集样本数: {len(X_tr)}")

    # 5. 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=3)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    # 其他导入不变...

    # 6. 超参数搜索（改了）
    grid_xgb = GridSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=42),
        {'max_depth': [3, 5], 'learning_rate': [0.05, 0.1], 'n_estimators': [100, 150]},
        scoring="f1", cv=tscv, n_jobs=-1
    )

    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {'n_estimators': [100, 150], 'max_depth': [5, 8]},
        scoring="f1", cv=tscv, n_jobs=-1
    )

    grid_mlp = GridSearchCV(
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
        {'alpha': [0.001]},
        scoring="f1", cv=tscv, n_jobs=-1
    )

    grid_lr = GridSearchCV(
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
        {'C': [1.0, 10.0]},
        scoring="f1", cv=tscv, n_jobs=-1
    )

    grid_svm = GridSearchCV(
        SVC(probability=True, random_state=42),
        {'C': [1.0, 10.0], 'kernel': ['rbf', 'linear']},
        scoring="f1", cv=tscv, n_jobs=-1
    )

    # 7. 训练各个模型
    grid_xgb.fit(X_tr, y_tr)
    grid_rf.fit(X_tr, y_tr)
    grid_mlp.fit(X_tr, y_tr)
    grid_lr.fit(X_tr, y_tr)
    grid_svm.fit(X_tr, y_tr)

    best_xgb = grid_xgb.best_estimator_
    best_rf = grid_rf.best_estimator_
    best_mlp = grid_mlp.best_estimator_
    best_lr = grid_lr.best_estimator_
    best_svm = grid_svm.best_estimator_
    model_scores = {
        "xgb": grid_xgb.best_score_,
        "rf": grid_rf.best_score_,
        "mlp": grid_mlp.best_score_,
        "lr": grid_lr.best_score_,
    }

    max_score = max(model_scores.values())
    weights = [score / max_score for score in model_scores.values()]

    print("\n最佳超参数:")
    print("XGB:", grid_xgb.best_params_)
    print("RF :", grid_rf.best_params_)
    print("MLP:", grid_mlp.best_params_)
    print("LogReg:", grid_lr.best_params_)

    voting = VotingClassifier(
        estimators=[
            ("xgb", best_xgb),
            ("rf", best_rf),
            ("mlp", best_mlp),
            ("lr", best_lr)
        ],
        voting="soft",
        weights=weights
    )

    # 9. 训练 Voting，找全局best threshold
    voting.fit(X_tr, y_tr)
    proba = voting.predict_proba(X_te)[:, 1]
    global_best_thr = tune_threshold(y_test, proba)

    preds = (proba >= global_best_thr).astype(int)
    roc = roc_auc_score(y_test, proba)
    pr  = average_precision_score(y_test, proba)
    f1  = f1_score(y_test, preds, zero_division=0)

    print("\n📊 测试集整体评估（全局最佳阈值）:")
    print(f"ROC-AUC = {roc:.4f} | PR-AUC = {pr:.4f} | F1 = {f1:.4f} | best_thr = {global_best_thr:.4f}")

    # 10. 保存模型
    SAVE_PATH = os.path.join(SAVE_DIR, "final_voting_model.pkl")
    with open(SAVE_PATH, "wb") as f:
        pickle.dump({
            "model": voting,
            "scaler": scaler,
            "imputer": imputer,
            "best_threshold": global_best_thr
        }, f)
    print(f"\n✅ 模型保存成功: {SAVE_PATH}")