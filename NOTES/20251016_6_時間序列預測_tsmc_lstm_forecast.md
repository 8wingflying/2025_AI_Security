# 📈 使用 LSTM 預測台積電（2330.TW / TSM）股價走勢

> 本教學展示如何使用 **長短期記憶神經網路（LSTM）** 進行台積電股票的時間序列預測，包含資料下載、特徵處理、模型建立與未來股價預測。

⚠️ **注意**：本範例僅供學術與教學使用，**不構成任何投資建議**。

---

## 🧰 一、安裝所需套件

```bash
pip install yfinance numpy pandas scikit-learn matplotlib seaborn tensorflow==2.* tqdm
```

---

## 💾 二、下載台積電股價資料

```python
import yfinance as yf
import pandas as pd

df = yf.download("2330.TW", start="2015-01-01", auto_adjust=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
print(df.tail())
```

---

## ⚙️ 三、資料前處理與特徵工程

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 加入技術指標
df['MA5'] = df['Close'].rolling(5).mean()
df['MA20'] = df['Close'].rolling(20).mean()
df['STD20'] = df['Close'].rolling(20).std()
df['RSI14'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() /
                                 (-df['Close'].diff().clip(upper=0).rolling(14).mean()).abs())))
df = df.fillna(method='bfill').fillna(method='ffill')

# 特徵選擇
feature_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA5', 'MA20', 'STD20', 'RSI14']
data = df[feature_cols].values

# 正規化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data)

scaler_y = MinMaxScaler(feature_range=(0, 1))
scaled_y = scaler_y.fit_transform(df[['Close']])
```

---

## 🪜 四、建立 LSTM 序列資料

```python
def make_sequences(feat_arr, target_arr, window=60):
    X, y = [], []
    for i in range(window, len(feat_arr)):
        X.append(feat_arr[i-window:i, :])
        y.append(target_arr[i, 0])
    return np.array(X), np.array(y)

X_all, y_all = make_sequences(scaled, scaled_y, 60)
```

---

## 📊 五、切分訓練 / 驗證 / 測試集

```python
TEST_RATIO = 0.15
VAL_RATIO = 0.10

N = len(X_all)
test_size = int(N * TEST_RATIO)
trainval_size = N - test_size
val_size = int(trainval_size * VAL_RATIO)

X_train = X_all[:trainval_size - val_size]
y_train = y_all[:trainval_size - val_size]

X_val = X_all[trainval_size - val_size:trainval_size]
y_val = y_all[trainval_size - val_size:trainval_size]

X_test = X_all[trainval_size:]
y_test = y_all[trainval_size:]
```

---

## 🧠 六、建立與訓練 LSTM 模型

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(60, X_all.shape[-1])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)
```

---

## 🧾 七、模型評估與視覺化

```python
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import math
import matplotlib.pyplot as plt

pred_test_scaled = model.predict(X_test)
pred_test = scaler_y.inverse_transform(pred_test_scaled)
true_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))

rmse = math.sqrt(mean_squared_error(true_test, pred_test))
mape = mean_absolute_percentage_error(true_test, pred_test) * 100
print(f"[TEST] RMSE={rmse:,.2f}, MAPE={mape:.2f}%")

plt.figure(figsize=(12,5))
plt.plot(df.index[-len(true_test):], true_test, label='真實收盤價')
plt.plot(df.index[-len(pred_test):], pred_test, label='LSTM預測')
plt.legend()
plt.title(f'2330.TW LSTM 股價預測 (RMSE={rmse:.2f}, MAPE={mape:.2f}%)')
plt.show()
```

---

## 🔮 八、未來 5 日遞迴預測

```python
FORECAST_STEPS = 5
current_window = X_all[-1]
future_preds = []

for _ in range(FORECAST_STEPS):
    yhat_scaled = model.predict(current_window[np.newaxis, ...], verbose=0)
    yhat = scaler_y.inverse_transform(yhat_scaled)[0, 0]
    future_preds.append(yhat)

    next_row = current_window[-1].copy()
    next_row[0] = yhat_scaled[0, 0]
    current_window = np.vstack([current_window[1:], next_row])

print(future_preds)
```

---

## 🧩 九、延伸應用方向

| 應用方向 | 方法建議 |
|------------|------------|
| 提升預測準確度 | 雙向 LSTM、GRU、Seq2Seq、Attention |
| 特徵擴充 | 財報、新聞、情緒指標（Sentiment） |
| 多步預測 | Encoder-Decoder LSTM 或 Dense(多輸出) |
| 模型組合 | LSTM + XGBoost / LightGBM 混合修正 |

---

> ✅ 本教學展示如何以 LSTM 建立時間序列預測流程，從資料下載到未來股價推估皆可自動化實作。

