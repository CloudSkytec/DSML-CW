import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from data_processing import load_and_preprocess
import os
from keras.api.models import load_model

model_list = ["lstm", "gru", "transformer", "lstm_attn"]
SAVE_DIR = "./saved_models/"
VOTING_PATH = os.path.join(SAVE_DIR, "final_voting_model.pkl")
DATA_PATH = "./data/dow_jones_index.csv"

if __name__ == "__main__":
    df = load_and_preprocess(DATA_PATH, win=8, horizon=1)
    feature_cols = df.drop(columns=["fall_risk", "stock", "date"]).columns
    X = df[feature_cols].values
    y = df["fall_risk"].values

    test_idx = []
    for stock, group in df.groupby("stock", sort=False):
        idx = group.sort_values("date").index
        split = len(idx) // 2
        test_idx += idx[split:].tolist()
    test_mask = np.isin(df.index.values, test_idx)
    X_test = X[test_mask]
    y_test = y[test_mask]

    with open(VOTING_PATH, "rb") as f:
        obj = pickle.load(f)
    voting_model = obj["model"]
    scaler = obj["scaler"]
    best_thr = obj["best_threshold"]

    results = []

    for model_name in model_list:
        MODEL_PATH = f"{SAVE_DIR}/{model_name}_weekly_forecast.h5"

        if not os.path.exists(MODEL_PATH):
            print(f"模型文件不存在: {MODEL_PATH}")
            continue

        try:
            if model_name == "lstm_attn":
                from lstm_attention import SelfAttention
                model = load_model(MODEL_PATH, custom_objects={"SelfAttention": SelfAttention})
            else:
                model = load_model(MODEL_PATH)
            print(f"加载模型: {model_name}")
        except Exception as e:
            print(f"加载模型失败 ({model_name}): {e}")
            continue

        n_sample = X_test.shape[0] - 8
        if n_sample <= 0:
            print(f"测试集太小，无法构造输入序列")
            continue

        seq_X = np.array([X_test[i:i+8] for i in range(n_sample)])
        input_scaler = StandardScaler().fit(seq_X.reshape(seq_X.shape[0], -1))
        seq_X_scaled = input_scaler.transform(seq_X.reshape(seq_X.shape[0], -1)).reshape(seq_X.shape)

        y_pred_feat = model.predict(seq_X_scaled)
        X_pred_scaled = scaler.transform(y_pred_feat)
        y_proba = voting_model.predict_proba(X_pred_scaled)[:, 1]
        y_pred = (y_proba >= best_thr).astype(int)

        aligned_y_true = y_test[8:8+len(y_pred)]

        f1 = f1_score(aligned_y_true, y_pred)
        roc = roc_auc_score(aligned_y_true, y_proba)
        pr = average_precision_score(aligned_y_true, y_proba)

        results.append({
            "Model": model_name,
            "F1-score": f"{f1:.4f}",
            "ROC-AUC": f"{roc:.4f}",
            "PR-AUC": f"{pr:.4f}"
        })

    print("\n四个模型在测试集上的预测分数：")
    print(pd.DataFrame(results).to_string(index=False))
# 绘制模型性能可视化对比图
import matplotlib.pyplot as plt

# 将结果转为 DataFrame 并确保数值为 float
df = pd.DataFrame(results)
df["Model"] = df["Model"].str.upper().str.replace("_", " + ").str.replace("LSTM + ATTN", "LSTM + Attention")
df["F1-score"] = df["F1-score"].astype(float)
df["ROC-AUC"] = df["ROC-AUC"].astype(float)
df["PR-AUC"] = df["PR-AUC"].astype(float)

# 设置图表参数
x = range(len(df))
bar_width = 0.25

plt.figure(figsize=(10, 6))
plt.bar([i - bar_width for i in x], df["F1-score"], width=bar_width, label="F1-score")
plt.bar(x, df["ROC-AUC"], width=bar_width, label="ROC-AUC")
plt.bar([i + bar_width for i in x], df["PR-AUC"], width=bar_width, label="PR-AUC")

plt.xticks(x, df["Model"])
plt.ylabel("Score")
plt.title("Performance Comparison of Forecasting Models (Voting Classifier)")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

