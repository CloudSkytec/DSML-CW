import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import os

model_list = ["lstm", "gru", "transformer","lstm_attn"]

if __name__ == "__main__":
    SAVE_DIR = "./saved_models/"
    MODEL_PATH = os.path.join(SAVE_DIR, "final_voting_model.pkl")

    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    voting_model = obj["model"]
    scaler = obj["scaler"]
    best_thr = obj["best_threshold"]

    results = []

    for model_name in model_list:
        CSV_PATH = f"{SAVE_DIR}/{model_name}_predicted_next_week.csv"

        if not os.path.exists(CSV_PATH):
            print(f"预测特征文件不存在: {CSV_PATH}")
            continue

        pred_df = pd.read_csv(CSV_PATH)
        X = scaler.transform(pred_df.values)

        proba = voting_model.predict_proba(X)[:, 1]
        label = (proba >= best_thr).astype(int)

        results.append({
            "Model": model_name,
            "Fall Risk Proba": f"{proba[0]:.4f}",
            "Fall Risk Label": int(label[0])
        })

    # 打印对比表格
    print("\n不同模型fall risk预测对比:")
    print(pd.DataFrame(results).to_string(index=False))
