import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api.models import load_model
from sklearn.preprocessing import StandardScaler
from data_processing import load_and_preprocess
import os

model_list = ["lstm", "gru", "transformer", "lstm_attn"]

def create_latest_input(X_all, input_weeks=8):
    return np.expand_dims(X_all[-input_weeks:], axis=0)

if __name__ == "__main__":
    DATA_PATH = "./data/dow_jones_index.csv"
    df = load_and_preprocess(DATA_PATH, win=8, horizon=1)

    feature_cols = df.drop(columns=["fall_risk", "stock", "date"]).columns
    X_all = df[feature_cols].values

    scaler = StandardScaler().fit(X_all)
    X_all = scaler.transform(X_all)

    for model_name in model_list:
        MODEL_PATH = f"./saved_models/{model_name}_weekly_forecast.h5"
        CSV_PATH = f"./saved_models/{model_name}_predicted_next_week.csv"

        if not os.path.exists(MODEL_PATH):
            print(f"模型文件不存在: {MODEL_PATH}")
            continue

        # 针对 lstm_attn 模型使用 custom_objects
        if model_name == "lstm_attn":
            try:
                from lstm_attention import SelfAttention
                model = load_model(MODEL_PATH, custom_objects={"SelfAttention": SelfAttention})
            except Exception as e:
                print(f"加载 lstm_attn 模型失败: {e}")
                continue
        else:
            model = load_model(MODEL_PATH)

        print(f"模型加载成功: {model_name}")

        X_input = create_latest_input(X_all, input_weeks=8)
        y_pred = model.predict(X_input)
        y_pred = y_pred.reshape(-1)

        pred_df = pd.DataFrame([y_pred], columns=feature_cols)
        pred_df.to_csv(CSV_PATH, index=False)
        print(f"保存未来一周预测特征到 {CSV_PATH}")
