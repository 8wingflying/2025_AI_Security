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

# 三、Pure Python (NumPy) 物件導向實作
# 此實作不依賴深度學習框架，完全使用 NumPy 矩陣運算來還原數學公式，適合理解底層邏輯。

import numpy as np

class GRU:
    def __init__(self, input_dim, hidden_dim):
        """
        初始化 GRU 權重
        :param input_dim: 輸入特徵維度 (x_t)
        :param hidden_dim: 隱藏狀態維度 (h_t)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 初始化權重 (W) 與 偏差 (b)
        # 將公式中的 W · [h, x] 拆解為 W_x · x + W_h · h
        
        # 1. Update Gate (z_t) 參數
        self.W_xz = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W_hz = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_z = np.zeros((1, hidden_dim))
        
        # 2. Reset Gate (r_t) 參數
        self.W_xr = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W_hr = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_r = np.zeros((1, hidden_dim))
        
        # 3. Candidate State (h_tilde) 參數
        self.W_xh = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_h = np.zeros((1, hidden_dim))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward_step(self, x_t, h_prev):
        """
        執行單一時間步計算
        對應架構圖中的綠色方塊
        """
        # Step 1: Update Gate (z_t)
        # z_t = σ(W_z · [h_{t-1}, x_t])
        z_t = self.sigmoid(np.dot(x_t, self.W_xz) + np.dot(h_prev, self.W_hz) + self.b_z)
        
        # Step 2: Reset Gate (r_t)
        # r_t = σ(W_r · [h_{t-1}, x_t])
        r_t = self.sigmoid(np.dot(x_t, self.W_xr) + np.dot(h_prev, self.W_hr) + self.b_r)
        
        # Step 3: Candidate Hidden State (h_tilde)
        # h_tilde = tanh(W · [r_t * h_{t-1}, x_t])
        reset_hidden = r_t * h_prev  # 重置舊記憶
        h_tilde = self.tanh(np.dot(x_t, self.W_xh) + np.dot(reset_hidden, self.W_hh) + self.b_h)
        
        # Step 4: Final Hidden State (h_t)
        # h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t

    def forward(self, X):
        """處理完整序列輸入 [batch, time_steps, features]"""
        batch_size, time_steps, _ = X.shape
        h_t = np.zeros((batch_size, self.hidden_dim)) # 初始狀態 h_0
        outputs = []
        
        for t in range(time_steps):
            x_t = X[:, t, :]
            h_t = self.forward_step(x_t, h_t)
            outputs.append(h_t)
            
        return np.array(outputs).transpose(1, 0, 2)

# --- 測試範例 ---
if __name__ == "__main__":
    # 模擬數據: Batch=2, TimeSteps=5, Features=3
    x_input = np.random.randn(2, 5, 3)
    
    # 初始化 GRU
    gru_layer = GRU(input_dim=3, hidden_dim=4)
    
    # 前向傳播
    output = gru_layer.forward(x_input)
    
    print(f"輸入形狀: {x_input.shape}")
    print(f"輸出形狀: {output.shape} (Batch, TimeSteps, HiddenDim)")
    print("最後一個時間步的輸出 h_t:\n", output[:, -1, :])

```
