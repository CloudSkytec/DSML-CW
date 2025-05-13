import numpy as np
import tensorflow as tf
import os
import random
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from keras.api.losses import MeanSquaredError
from data_processing import load_and_preprocess
# 设置全局随机种子
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 确保TensorFlow使用单线程（可选）
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
if __name__ == "__main__":
    DATA_PATH = "./data/dow_jones_index.csv"
    SAVE_DIR = "./saved_models/"
    os.makedirs(SAVE_DIR, exist_ok=True)

    df = load_and_preprocess(DATA_PATH)
    feature_cols = df.drop(columns=["fall_risk", "stock", "date"]).columns

    all_X, all_y = [], []
    all_idx = []

    for stock, group in df.groupby('stock', sort=False):
        g = group.sort_values('date')
        features = g[feature_cols].values
        idx = g.index.values  # 这一组原始的index

        for i in range(len(features) - 8):
            all_X.append(features[i:i + 8])
            all_y.append(features[i + 8])
            all_idx.append(idx[i + 8])  # 把y对应的原始index存下来

    # 转成数组
    all_X = np.array(all_X)
    all_y = np.array(all_y)
    all_idx = np.array(all_idx)

    # 划分train/test
    train_idx, test_idx = [], []
    for stock, group in df.groupby('stock', sort=False):
        g = group.sort_values('date')
        idx = g.index.values
        split = len(idx) // 2
        train_idx += idx[:split].tolist()
        test_idx += idx[split:].tolist()

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    # 然后根据all_idx来选取
    train_mask = np.isin(all_idx, train_idx)
    test_mask = np.isin(all_idx, test_idx)

    X_train = all_X[train_mask]
    y_train = all_y[train_mask]
    X_test = all_X[test_mask]
    y_test = all_y[test_mask]

    print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

    scaler = StandardScaler().fit(X_train.reshape(X_train.shape[0], -1))
    X_tr = scaler.transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_te = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

    tscv = TimeSeriesSplit(n_splits=3)

    best_model = None
    best_loss = np.inf

    for fold, (train_idx_cv, val_idx_cv) in enumerate(tscv.split(X_tr)):
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=(X_tr.shape[1], X_tr.shape[2])),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(X_tr.shape[2])
        ])
        model.compile(optimizer='adam', loss=MeanSquaredError())

        model.fit(X_tr[train_idx_cv], y_train[train_idx_cv], validation_data=(X_tr[val_idx_cv], y_train[val_idx_cv]), epochs=30, batch_size=16, verbose=0)

        val_loss = model.evaluate(X_tr[val_idx_cv], y_train[val_idx_cv], verbose=0)
        print(f"Fold {fold+1} 验证集Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model

    # 使用最佳模型在完整训练集上再训练一次
    final_model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(X_tr.shape[1], X_tr.shape[2])),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(X_tr.shape[2])
    ])
    final_model.compile(optimizer='adam', loss=MeanSquaredError())
    final_model.fit(X_tr, y_train, epochs=30, batch_size=16, verbose=0)

    final_model.save(os.path.join(SAVE_DIR, "LSTM_weekly_forecast.h5"))
    print(" LSTM模型保存成功")

    # 測試集上評估
    test_loss = final_model.evaluate(X_te, y_test, verbose=0)
    print(f"测试集最终Loss: {test_loss:.4f}")
