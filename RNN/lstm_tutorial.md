# 📘 LSTM（Long Short-Term Memory）教學文件

---

## 一、LSTM 是什麼？

**LSTM（長短期記憶網路）** 是一種改良型的 **循環神經網路（RNN）**，能夠有效解決傳統 RNN 在長序列中出現的 **梯度消失（vanishing gradient）問題**。  
它特別適合處理時間序列、語音、文本與股價預測等需要「記住上下文」的任務。

---

## 二、背景：為何需要 LSTM？

### 🔹 傳統 RNN 的問題
RNN 在每個時間步會將前一步的輸出傳遞給下一步，但：
- 當序列太長時，誤差在反向傳播過程中會逐漸消失（梯度消失），  
  導致模型難以記住長期依賴關係。

### 🔹 LSTM 的解決方案
LSTM 透過設計「**記憶單元（Cell State）**」與三個「**閥門（Gates）**」，  
能夠選擇性地 **記住或忘記資訊**，使模型能學習長距離依賴關係。

---

## 三、LSTM 架構總覽

LSTM 每個單元包含下列三個閥門：

| 閥門 | 功能 | 公式說明 |
|------|------|----------|
| **遺忘閥（Forget Gate）** | 決定要丟棄多少舊資訊 | \( f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \) |
| **輸入閥（Input Gate）** | 決定要新增多少資訊 | \( i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \) |
| **輸入候選（Candidate）** | 新的候選記憶內容 | \( \tilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \) |
| **記憶更新（Cell State）** | 更新整體記憶 | \( C_t = f_t * C_{t-1} + i_t * \tilde{C}_t \) |
| **輸出閥（Output Gate）** | 決定輸出多少資訊 | \( o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \) |
| **最終輸出（Hidden State）** | 輸出值 | \( h_t = o_t * \tanh(C_t) \) |

---

## 四、LSTM 的工作流程圖

```
輸入 x_t ─┬───────────────────────────┬──────────────────────────────────────┬──────────────────────────────────────┐
           ↓            ↓              ↓
       [Forget Gate] [Input Gate] [Output Gate]
           ↓            ↓              ↓
     決定遺忘多少     新增哪些資訊    輸出哪些結果
           ↓            ↓              ↓
        更新 Cell State →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→
```

---

## 五、Python 實作範例

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 產生假資料
x = np.random.rand(100, 10, 1)   # 100 筆資料，每筆序列長度10
y = np.random.rand(100, 1)       # 對應標籤

# 建立 LSTM 模型
model = Sequential([
    LSTM(64, input_shape=(10, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# 訓練模型
model.fit(x, y, epochs=20, batch_size=8)
```

---

## 六、應用範例

| 應用領域 | 範例 |
|-----------|------|
| 🕒 **時間序列分析** | 股價、氣象、能源需求預測 |
| 💬 **自然語言處理** | 文本生成、情緒分析、語音辨識 |
| 🧠 **醫學信號分析** | ECG、腦波序列異常偵測 |
| 📈 **IoT / OT 安全** | 感測器異常偵測（與 Autoencoder 結合） |

---

## 七、視覺化示範（時間序列預測）

```python
import matplotlib.pyplot as plt

# 假設 y_true 為真實值，y_pred 為模型預測
y_true = np.sin(np.linspace(0, 10, 100))
y_pred = y_true + np.random.normal(0, 0.1, 100)

plt.plot(y_true, label='True')
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.legend()
plt.title("LSTM 時間序列預測示意圖")
plt.show()
```

---

## 八、LSTM 與其他模型比較

| 模型 | 優點 | 缺點 |
|------|------|------|
| **RNN** | 結構簡單 | 容易梯度消失 |
| **LSTM** | 能記長期依賴、穩定訓練 | 較耗資源 |
| **GRU** | 簡化 LSTM 結構、訓練快 | 表現略低於 LSTM |
| **Transformer** | 可並行、全局依賴 | 需大量資料與運算資源 |

---

## 九、延伸主題

1. **GRU（Gated Recurrent Unit）**：LSTM 的簡化版本，僅兩個閥門。  
2. **Bi-LSTM（雙向 LSTM）**：同時考慮過去與未來語境。  
3. **Attention-LSTM**：在序列中加減不同時間步的重要性。  
4. **Hybrid CNN-LSTM**：結合 CNN 特徵抽取與 LSTM 時序學習。  

---

## 十、練習題（含答案）

### 🧩 單選題 1
LSTM 的哪個部分責任「保留前一狀態的重要資訊」？
- (A) 輸入閥  
- (B) 遺忘閥 ✅  
- (C) 輸出閥  
- (D) 候選記憶層  
**解析：** 遺忘閥（Forget Gate）控制前一時刻記憶被保留的比例。

---

### 🧠 情境題 2
若你要使用 LSTM 預測

