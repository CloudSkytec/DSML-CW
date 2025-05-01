import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from keras.api.models import Model
from keras.api.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, Add, GlobalAveragePooling1D
)
from keras.api.losses import MeanSquaredError
from data_processing import load_and_preprocess

# 设置随机种子保证可复现
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

def transformer_encoder(inputs, head_size=64, num_heads=2, ff_dim=128, dropout=0.1):
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x_ff = Dropout(dropout)(x_ff)
    x = Add()([x, x_ff])
    x = LayerNormalization(epsilon=1e-6)(x)

    return x

if __name__ == "__main__":
    DATA_PATH = "./data/dow_jones_index.csv"
    SAVE_DIR = "./saved_models/"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. 读取数据并滑窗
    df = load_and_preprocess(DATA_PATH)
    feature_cols = df.drop(columns=["fall_risk", "stock", "date"]).columns

    all_X, all_y, all_idx = [], [], []

    for stock, group in df.groupby('stock', sort=False):
        g = group.sort_values('date')
        features = g[feature_cols].values
        idx = g.index.values

        for i in range(len(features) - 8):
            all_X.append(features[i:i + 8])
            all_y.append(features[i + 8])
            all_idx.append(idx[i + 8])

    all_X = np.array(all_X)
    all_y = np.array(all_y)
    all_idx = np.array(all_idx)

    # 2. 划分训练测试集
    train_idx, test_idx = [], []
    for stock, group in df.groupby('stock', sort=False):
        g = group.sort_values('date')
        idx = g.index.values
        split = len(idx) // 2
        train_idx += idx[:split].tolist()
        test_idx += idx[split:].tolist()

    train_mask = np.isin(all_idx, train_idx)
    test_mask = np.isin(all_idx, test_idx)

    X_train, y_train = all_X[train_mask], all_y[train_mask]
    X_test, y_test = all_X[test_mask], all_y[test_mask]

    print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

    # 3. 标准化
    scaler = StandardScaler().fit(X_train.reshape(X_train.shape[0], -1))
    X_tr = scaler.transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_te = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

    # 4. 交叉验证选择最佳模型结构
    tscv = TimeSeriesSplit(n_splits=3)
    best_model = None
    best_loss = np.inf

    for fold, (train_idx_cv, val_idx_cv) in enumerate(tscv.split(X_tr)):
        input_layer = Input(shape=(X_tr.shape[1], X_tr.shape[2]))
        x = transformer_encoder(input_layer, head_size=64, num_heads=2, ff_dim=128)
        x = transformer_encoder(x, head_size=64, num_heads=2, ff_dim=128)
        x = GlobalAveragePooling1D()(x)
        output_layer = Dense(X_tr.shape[2])(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss=MeanSquaredError())

        model.fit(X_tr[train_idx_cv], y_train[train_idx_cv],
                  validation_data=(X_tr[val_idx_cv], y_train[val_idx_cv]),
                  epochs=30, batch_size=16, verbose=0)

        val_loss = model.evaluate(X_tr[val_idx_cv], y_train[val_idx_cv], verbose=0)
        print(f"Fold {fold+1} 验证集Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model

    # 5. 最终训练模型（在完整训练集上）
    input_layer = Input(shape=(X_tr.shape[1], X_tr.shape[2]))
    x = transformer_encoder(input_layer, head_size=64, num_heads=2, ff_dim=128)
    x = transformer_encoder(x, head_size=64, num_heads=2, ff_dim=128)
    x = GlobalAveragePooling1D()(x)
    output_layer = Dense(X_tr.shape[2])(x)

    final_model = Model(inputs=input_layer, outputs=output_layer)
    final_model.compile(optimizer='adam', loss=MeanSquaredError())
    final_model.fit(X_tr, y_train, epochs=30, batch_size=16, verbose=0)

    # 保存模型
    final_model.save(os.path.join(SAVE_DIR, "transformer_weekly_forecast.h5"))
    print("✅ Transformer 模型保存成功")

    # 测试评估
    test_loss = final_model.evaluate(X_te, y_test, verbose=0)
    print(f"测试集最终Loss: {test_loss:.4f}")
