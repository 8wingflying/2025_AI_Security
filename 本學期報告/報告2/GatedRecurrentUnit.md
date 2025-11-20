# Gated Recurrent Unit (GRU) 深度解析：從數學原理到程式實作

本文件詳細解析 Gated Recurrent Unit (GRU) 的運作機制。內容涵蓋根據架構圖的數學公式推導、使用 TensorFlow 的高階實作，以及不依賴框架的純 Python (NumPy) 底層實作。

---

## 一、數學原理與架構解析

GRU 是循環神經網路 (RNN) 的一種變體，透過「門控機制」來解決梯度消失問題，能有效捕捉序列數據中的長短期依賴關係。

### 1. 符號定義
* **$x_t$**：當前時間步的輸入向量。
* **$h_{t-1}$**：上一個時間步的隱藏狀態 (Hidden State)，代表過去記憶。
* **$h_t$**：當前時間步計算出的新隱藏狀態。
* **$\sigma$**：Sigmoid 函數，將數值壓縮在 0~1 (作為開關)。
* **$\tanh$**：雙曲正切函數，將數值壓縮在 -1~1 (數據正規化)。
* **$*$**：Hadamard Product (矩陣元素對應相乘)。
* **$[\cdot, \cdot]$**：向量串接 (Concatenation)。

### 2. 運作流程 (四個關鍵公式)

#### **Step 1: 更新門 (Update Gate, $z_t$)**
> $$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$

* **功能**：決定要保留多少舊記憶 ($h_{t-1}$) 以及寫入多少新資訊。
* **意義**：$z_t \approx 1$ 代表傾向保留舊資訊；$z_t \approx 0$ 代表傾向使用新資訊。

#### **Step 2: 重置門 (Reset Gate, $r_t$)**
> $$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$

* **功能**：決定在計算新內容時，要「遺忘」多少過去的資訊。
* **意義**：若 $r_t \approx 0$，則在計算候選狀態時會忽略過去的 $h_{t-1}$，專注於當前輸入。

#### **Step 3: 候選隱藏狀態 (Candidate Hidden State, $\tilde{h}_t$)**
> $$\tilde{h}_t = \tanh(W \cdot [r_t * h_{t-1}, x_t])$$

* **功能**：當前時間步產生的新資訊內容。
* **機制**：透過 $r_t * h_{t-1}$ 先過濾舊記憶，再與 $x_t$ 結合。

#### **Step 4: 最終隱藏狀態 (Final Hidden State, $h_t$)**
> $$h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t$$

* **功能**：透過線性插值計算最終輸出。
* **機制**：根據更新門 $z_t$ 的比例，融合舊記憶與新資訊。

---

## 二、TensorFlow (Keras) 實作範例

這是業界常用的高階 API 實作方式，快速且經過最佳化。

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# --- 1. 參數設定 ---
TIME_STEPS = 10       # 序列長度 (例如過去 10 天)
INPUT_DIM = 1         # 輸入特徵維度 (例如股價)
HIDDEN_UNITS = 32     # 隱藏狀態 h_t 的維度

# --- 2. 建構模型 ---
model = models.Sequential()

# 加入 GRU 層
# units: 對應 h_t 的大小
# activation='tanh': 對應候選狀態的激活函數
# recurrent_activation='sigmoid': 對應門控 (z_t, r_t) 的激活函數
model.add(layers.GRU(units=HIDDEN_UNITS, 
                     input_shape=(TIME_STEPS, INPUT_DIM),
                     activation='tanh',
                     recurrent_activation='sigmoid',
                     return_sequences=False)) # False 代表只輸出最後一個時間步的 h_t

# 輸出層
model.add(layers.Dense(1))

model.summary()
# model.compile(optimizer='adam', loss='mse')
# model.fit(X, y, epochs=10)
