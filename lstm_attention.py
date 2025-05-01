import numpy as np
import tensorflow as tf
import os
import random
from keras.api.models import Model
from keras.api.layers import Input, LSTM, Dense, Dropout, Layer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from keras.api.losses import MeanSquaredError
from keras.api.optimizers import Adam
# 自定义注意力机制
import tensorflow.python.keras.backend as K  # ✅ 正确版本
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

# 自定义注意力机制
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, return_sequences=True, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        self.V = self.add_weight(name="att_var", shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        score = K.tanh(K.dot(inputs, self.W) + self.b)
        attention_weights = K.softmax(K.dot(score, self.V), axis=1)
        context_vector = attention_weights * inputs
        context_vector = K.sum(context_vector, axis=1)
        return context_vector

# 构建模型
def build_lstm_attention_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = SelfAttention()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(input_shape[1])(x)  # 输出为特征维度
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss=MeanSquaredError())
    return model

if __name__ == "__main__":
    os.makedirs("./saved_models", exist_ok=True)

    # 加载并处理真实数据
    DATA_PATH = "./data/dow_jones_index.csv"
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

    # 划分测试集
    train_idx, test_idx = [], []
    for stock, group in df.groupby('stock', sort=False):
        g = group.sort_values('date')
        idx = g.index.values
        split = len(idx) // 2
        train_idx += idx[:split].tolist()
        test_idx += idx[split:].tolist()

    train_mask = np.isin(all_idx, train_idx)
    test_mask = np.isin(all_idx, test_idx)
    X_train = all_X[train_mask]
    y_train = all_y[train_mask]
    X_test = all_X[test_mask]
    y_test = all_y[test_mask]

    # 标准化
    scaler = StandardScaler().fit(X_train.reshape(X_train.shape[0], -1))
    X_tr = scaler.transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_te = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

    # 交叉验证选择最优模型
    tscv = TimeSeriesSplit(n_splits=3)
    best_model = None
    best_loss = np.inf

    for fold, (train_idx_cv, val_idx_cv) in enumerate(tscv.split(X_tr)):
        model = build_lstm_attention_model(input_shape=(X_tr.shape[1], X_tr.shape[2]))
        model.fit(X_tr[train_idx_cv], y_train[train_idx_cv],
                  validation_data=(X_tr[val_idx_cv], y_train[val_idx_cv]),
                  epochs=30, batch_size=16, verbose=0)
        val_loss = model.evaluate(X_tr[val_idx_cv], y_train[val_idx_cv], verbose=0)
        print(f"Fold {fold+1} 验证集 Loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model

    # 最终模型训练
    final_model = build_lstm_attention_model(input_shape=(X_tr.shape[1], X_tr.shape[2]))
    final_model.fit(X_tr, y_train, epochs=30, batch_size=16, verbose=0)
    final_model.save("./saved_models/lstm_attn_weekly_forecast.h5")
    print("LSTM + Attention 最终模型已保存！")

    # 测试集评估
    test_loss = final_model.evaluate(X_te, y_test, verbose=0)
    print(f"测试集 Loss: {test_loss:.4f}")