# LSTM 深度解析：純 Python (NumPy) 物件導向實作

本文件提供 Long Short-Term Memory (LSTM) 的詳細實作指南。不依賴 TensorFlow 或 PyTorch 等深度學習框架，而是使用 Python 與 NumPy 從底層建構，以幫助深入理解 LSTM 的內部運作機制。

---

## 一、核心概念與數學原理

LSTM 與標準 RNN 或 GRU 最大的不同在於它擁有兩個狀態變數：
1.  **隱藏狀態 (Hidden State, $h_t$)**：用於短期記憶與輸出。
2.  **細胞狀態 (Cell State, $C_t$)**：類似於傳送帶，貫穿整個鏈條，負責長期記憶的保存與傳遞。

### 關鍵公式

LSTM 透過三個「門控 (Gates)」來保護和控制細胞狀態：

1.  **遺忘門 (Forget Gate, $f_t$)**
    * 決定要從舊細胞狀態中丟棄什麼資訊。
    * $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

2.  **輸入門 (Input Gate, $i_t$)**
    * 決定要更新什麼新資訊到細胞狀態中。
    * $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

3.  **候選細胞狀態 (Candidate Cell State, $\tilde{C}_t$)**
    * 建立一個新的候選值向量，準備被加入狀態中。
    * $$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

4.  **更新細胞狀態 (Cell State Update, $C_t$)**
    * 結合舊記憶與新資訊：舊狀態乘以遺忘因子，加上新狀態乘以輸入因子。
    * $$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$

5.  **輸出門 (Output Gate, $o_t$)**
    * 決定基於當前的細胞狀態，要輸出什麼值作為隱藏狀態。
    * $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

6.  **最終隱藏狀態 (Final Hidden State, $h_t$)**
    * $$h_t = o_t * \tanh(C_t)$$

---

## 二、Python 物件導向實作 (Pure NumPy)

此程式碼完全模擬上述數學過程，適合用於教學與理解底層矩陣運算。

### 完整程式碼

```python
import numpy as np

class LSTM:
    def __init__(self, input_dim, hidden_dim):
        """
        初始化 LSTM 層
        :param input_dim: 輸入特徵的維度 (x_t)
        :param hidden_dim: 隱藏狀態/細胞狀態的維度 (h_t, C_t)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # --- 初始化權重 (Weights) ---
        # LSTM 需要 4 組權重矩陣，分別對應：遺忘門、輸入門、候選狀態、輸出門
        # 這裡使用簡單的高斯分佈初始化 (乘以 0.01 保持數值微小)
        
        # 1. Forget Gate (遺忘門 f_t)
        self.W_xf = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W_hf = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_f = np.zeros((1, hidden_dim))
        
        # 2. Input Gate (輸入門 i_t)
        self.W_xi = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W_hi = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_i = np.zeros((1, hidden_dim))
        
        # 3. Candidate Cell State (候選細胞狀態 C_tilde)
        self.W_xC = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W_hC = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_C = np.zeros((1, hidden_dim))
        
        # 4. Output Gate (輸出門 o_t)
        self.W_xo = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W_ho = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_o = np.zeros((1, hidden_dim))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward_step(self, x_t, h_prev, C_prev):
        """
        執行單一時間步的 LSTM 計算
        :param x_t: 當前輸入 [batch, input_dim]
        :param h_prev: 上一時刻隱藏狀態 [batch, hidden_dim]
        :param C_prev: 上一時刻細胞狀態 [batch, hidden_dim]
        """
        
        # Step 1: Forget Gate (遺忘門) - 決定丟棄多少舊記憶
        f_t = self.sigmoid(np.dot(x_t, self.W_xf) + np.dot(h_prev, self.W_hf) + self.b_f)
        
        # Step 2: Input Gate (輸入門) - 決定寫入多少新資訊
        i_t = self.sigmoid(np.dot(x_t, self.W_xi) + np.dot(h_prev, self.W_hi) + self.b_i)
        
        # Step 3: Candidate Cell State - 產生候選記憶
        C_tilde = self.tanh(np.dot(x_t, self.W_xC) + np.dot(h_prev, self.W_hC) + self.b_C)
        
        # Step 4: Update Cell State - 更新長期記憶 C_t
        # 這是 LSTM 的核心：f_t 控制遺忘，i_t 控制寫入
        C_t = f_t * C_prev + i_t * C_tilde
        
        # Step 5: Output Gate (輸出門) - 決定輸出比例
        o_t = self.sigmoid(np.dot(x_t, self.W_xo) + np.dot(h_prev, self.W_ho) + self.b_o)
        
        # Step 6: Final Hidden State - 計算短期狀態 h_t
        h_t = o_t * self.tanh(C_t)
        
        return h_t, C_t

    def forward(self, X):
        """
        處理整個序列數據
        :param X: 輸入張量，形狀為 [batch_size, time_steps, input_dim]
        """
        batch_size, time_steps, _ = X.shape
        
        # 初始化 h_0 和 C_0 為全零向量
        h_t = np.zeros((batch_size, self.hidden_dim))
        C_t = np.zeros((batch_size, self.hidden_dim))
        
        # 用來儲存每個時間步的輸出
        outputs = []
        
        for t in range(time_steps):
            x_t = X[:, t, :] # 取出當前時間步的輸入
            
            # 遞迴更新隱藏狀態與細胞狀態
            h_t, C_t = self.forward_step(x_t, h_t, C_t)
            
            outputs.append(h_t)
            
        # 將列表轉回 NumPy 陣列，形狀變為 [batch, time, hidden]
        return np.array(outputs).transpose(1, 0, 2)

# --- 測試區塊 ---
if __name__ == "__main__":
    # 1. 設定參數
    BATCH_SIZE = 2
    TIME_STEPS = 5
    INPUT_DIM = 4
    HIDDEN_DIM = 8
    
    # 2. 產生隨機輸入資料
    input_data = np.random.randn(BATCH_SIZE, TIME_STEPS, INPUT_DIM)
    print(f"輸入資料形狀: {input_data.shape}")
    
    # 3. 建立 LSTM 模型
    lstm_layer = LSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
    
    # 4. 執行前向傳播
    output_sequence = lstm_layer.forward(input_data)
    
    print("\n--- 運算完成 ---")
    print(f"輸出資料形狀: {output_sequence.shape} (Batch, TimeSteps, HiddenDim)")
    print("\n最後一個時間步的隱藏狀態 (h_t Sample):")
    print(output_sequence[0, -1, :])
```

```
這是一份完整的 Markdown 文件內容，包含了 LSTM 的數學原理 與 純 Python (NumPy) 的物件導向實作。您可以點擊代碼區塊右上角的 Copy 按鈕，將內容複製並儲存為 LSTM_Implementation.md。Markdown# LSTM 深度解析：純 Python (NumPy) 物件導向實作

本文件提供 Long Short-Term Memory (LSTM) 的詳細實作指南。不依賴 TensorFlow 或 PyTorch 等深度學習框架，而是使用 Python 與 NumPy 從底層建構，以幫助深入理解 LSTM 的內部運作機制。

---

## 一、核心概念與數學原理

LSTM 與標準 RNN 或 GRU 最大的不同在於它擁有兩個狀態變數：
1.  **隱藏狀態 (Hidden State, $h_t$)**：用於短期記憶與輸出。
2.  **細胞狀態 (Cell State, $C_t$)**：類似於傳送帶，貫穿整個鏈條，負責長期記憶的保存與傳遞。

### 關鍵公式

LSTM 透過三個「門控 (Gates)」來保護和控制細胞狀態：

1.  **遺忘門 (Forget Gate, $f_t$)**
    * 決定要從舊細胞狀態中丟棄什麼資訊。
    * $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

2.  **輸入門 (Input Gate, $i_t$)**
    * 決定要更新什麼新資訊到細胞狀態中。
    * $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

3.  **候選細胞狀態 (Candidate Cell State, $\tilde{C}_t$)**
    * 建立一個新的候選值向量，準備被加入狀態中。
    * $$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

4.  **更新細胞狀態 (Cell State Update, $C_t$)**
    * 結合舊記憶與新資訊：舊狀態乘以遺忘因子，加上新狀態乘以輸入因子。
    * $$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$

5.  **輸出門 (Output Gate, $o_t$)**
    * 決定基於當前的細胞狀態，要輸出什麼值作為隱藏狀態。
    * $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

6.  **最終隱藏狀態 (Final Hidden State, $h_t$)**
    * $$h_t = o_t * \tanh(C_t)$$

---

## 二、Python 物件導向實作 (Pure NumPy)

此程式碼完全模擬上述數學過程，適合用於教學與理解底層矩陣運算。

### 完整程式碼

```python
import numpy as np

class LSTM:
    def __init__(self, input_dim, hidden_dim):
        """
        初始化 LSTM 層
        :param input_dim: 輸入特徵的維度 (x_t)
        :param hidden_dim: 隱藏狀態/細胞狀態的維度 (h_t, C_t)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # --- 初始化權重 (Weights) ---
        # LSTM 需要 4 組權重矩陣，分別對應：遺忘門、輸入門、候選狀態、輸出門
        # 這裡使用簡單的高斯分佈初始化 (乘以 0.01 保持數值微小)
        
        # 1. Forget Gate (遺忘門 f_t)
        self.W_xf = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W_hf = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_f = np.zeros((1, hidden_dim))
        
        # 2. Input Gate (輸入門 i_t)
        self.W_xi = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W_hi = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_i = np.zeros((1, hidden_dim))
        
        # 3. Candidate Cell State (候選細胞狀態 C_tilde)
        self.W_xC = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W_hC = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_C = np.zeros((1, hidden_dim))
        
        # 4. Output Gate (輸出門 o_t)
        self.W_xo = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W_ho = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_o = np.zeros((1, hidden_dim))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward_step(self, x_t, h_prev, C_prev):
        """
        執行單一時間步的 LSTM 計算
        :param x_t: 當前輸入 [batch, input_dim]
        :param h_prev: 上一時刻隱藏狀態 [batch, hidden_dim]
        :param C_prev: 上一時刻細胞狀態 [batch, hidden_dim]
        """
        
        # Step 1: Forget Gate (遺忘門) - 決定丟棄多少舊記憶
        f_t = self.sigmoid(np.dot(x_t, self.W_xf) + np.dot(h_prev, self.W_hf) + self.b_f)
        
        # Step 2: Input Gate (輸入門) - 決定寫入多少新資訊
        i_t = self.sigmoid(np.dot(x_t, self.W_xi) + np.dot(h_prev, self.W_hi) + self.b_i)
        
        # Step 3: Candidate Cell State - 產生候選記憶
        C_tilde = self.tanh(np.dot(x_t, self.W_xC) + np.dot(h_prev, self.W_hC) + self.b_C)
        
        # Step 4: Update Cell State - 更新長期記憶 C_t
        # 這是 LSTM 的核心：f_t 控制遺忘，i_t 控制寫入
        C_t = f_t * C_prev + i_t * C_tilde
        
        # Step 5: Output Gate (輸出門) - 決定輸出比例
        o_t = self.sigmoid(np.dot(x_t, self.W_xo) + np.dot(h_prev, self.W_ho) + self.b_o)
        
        # Step 6: Final Hidden State - 計算短期狀態 h_t
        h_t = o_t * self.tanh(C_t)
        
        return h_t, C_t

    def forward(self, X):
        """
        處理整個序列數據
        :param X: 輸入張量，形狀為 [batch_size, time_steps, input_dim]
        """
        batch_size, time_steps, _ = X.shape
        
        # 初始化 h_0 和 C_0 為全零向量
        h_t = np.zeros((batch_size, self.hidden_dim))
        C_t = np.zeros((batch_size, self.hidden_dim))
        
        # 用來儲存每個時間步的輸出
        outputs = []
        
        for t in range(time_steps):
            x_t = X[:, t, :] # 取出當前時間步的輸入
            
            # 遞迴更新隱藏狀態與細胞狀態
            h_t, C_t = self.forward_step(x_t, h_t, C_t)
            
            outputs.append(h_t)
            
        # 將列表轉回 NumPy 陣列，形狀變為 [batch, time, hidden]
        return np.array(outputs).transpose(1, 0, 2)

# --- 測試區塊 ---
if __name__ == "__main__":
    # 1. 設定參數
    BATCH_SIZE = 2
    TIME_STEPS = 5
    INPUT_DIM = 4
    HIDDEN_DIM = 8
    
    # 2. 產生隨機輸入資料
    input_data = np.random.randn(BATCH_SIZE, TIME_STEPS, INPUT_DIM)
    print(f"輸入資料形狀: {input_data.shape}")
    
    # 3. 建立 LSTM 模型
    lstm_layer = LSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
    
    # 4. 執行前向傳播
    output_sequence = lstm_layer.forward(input_data)
    
    print("\n--- 運算完成 ---")
    print(f"輸出資料形狀: {output_sequence.shape} (Batch, TimeSteps, HiddenDim)")
    print("\n最後一個時間步的隱藏狀態 (h_t Sample):")
    print(output_sequence[0, -1, :])
三、實作重點總結參數量對比：LSTM：有 4 個操作區塊 (Forget, Input, Cell, Output)，因此參數矩陣有 4 組。GRU：只有 3 個操作區塊 (Update, Reset, Candidate)，因此 GRU 的參數較少，訓練通常較快。狀態傳遞：程式碼中的 forward 迴圈清楚展示了 $h_t$ 與 $C_t$ 是如何隨時間步 (Time Steps) 遞迴傳遞的。$C_t$ 的更新公式 (f_t * C_prev + i_t * C_tilde) 僅包含加法與乘法，這使得誤差梯度在反向傳播時不易消失，這是 LSTM 能處理長序列的關鍵。Hadamard Product：程式碼中使用 * 運算符 (例如 f_t * C_prev) 代表元素對應相乘 (Element-wise multiplication)，這是控制資訊流量的關鍵數學操作。
```

# 10 種 LSTM 變體 (Variants) 詳解

Long Short-Term Memory (LSTM) 自 1997 年提出以來，為了適應不同的數據類型與任務需求，衍生出了許多變體。以下整理十種具代表性的架構，涵蓋了從結構簡化到多維數據處理的各種改進。

---

## 1. Peephole LSTM (窺孔 LSTM)
* **特點**：標準 LSTM 的門控（Input/Forget/Output）只能看到 $h_{t-1}$ 和 $x_t$。Peephole 連接讓門控層能夠直接「窺視」當前的細胞狀態 $C_{t-1}$。
* **優勢**：讓門控機制更精準地掌握細胞內的記憶存量，特別有助於學習需要精確計時或計數的任務。

## 2. Coupled Input and Forget Gate (CIFG)
* **特點**：將「輸入門 ($i_t$)」和「遺忘門 ($f_t$)」連動。標準 LSTM 中這兩個門是獨立的，但 CIFG 強制設定 $f_t = 1 - i_t$。
* **優勢**：
    * 只有在遺忘舊資訊時，才能寫入新資訊。
    * 減少了一組權重參數，計算效率比標準 LSTM 更高，且效果通常相當。

## 3. GRU (Gated Recurrent Unit)
* **特點**：最激進且成功的變體之一。它移除了細胞狀態 ($C_t$)，將其與隱藏狀態 ($h_t$) 合併，並將三個門簡化為兩個（更新門 Update & 重置門 Reset）。
* **優勢**：結構簡單、參數少、訓練速度快，是目前最流行的 LSTM 替代方案。

## 4. BiLSTM (Bidirectional LSTM - 雙向 LSTM)
* **特點**：包含兩個獨立的 LSTM 層，一個按正向時間順序處理，另一個按逆向時間順序處理。最終輸出是兩個方向結果的結合。
* **優勢**：模型能同時參考「過去」與「未來」的資訊。在自然語言處理（如翻譯、詞性標註）中極為重要，因為理解一個詞往往需要參考上下文。

## 5. ConvLSTM (Convolutional LSTM - 卷積 LSTM)
* **特點**：將標準 LSTM 內部的「矩陣乘法 (Matrix Multiplication)」全部替換為「卷積運算 (Convolution)」。
* **優勢**：專為 **時空數據 (Spatiotemporal data)** 設計。適用於影片預測、雷達回波降雨預測等任務，能同時捕捉時間依賴性與圖片的空間特徵。

## 6. Tree-LSTM (樹狀 LSTM)
* **特點**：標準 LSTM 是鏈狀結構 (Chain)，Tree-LSTM 則是樹狀結構。一個單元可以接收多個前一時刻的隱藏狀態（來自多個子節點）。
* **優勢**：適合處理具有層次結構的數據，例如 NLP 中的句法分析樹 (Syntax Parse Trees) 或程式碼的抽象語法樹 (AST)。

## 7. Phased LSTM (相位 LSTM)
* **特點**：引入了一個新的「時間門 (Time Gate)」，由週期性的震盪函數控制。只有在時間門開啟時，LSTM 才會更新狀態。
* **優勢**：專門解決 **不規則採樣 (Irregularly sampled)** 數據的問題。對於感測器數據（時間間隔忽長忽短）有很強的適應性，能比標準 LSTM 收斂得更快。

## 8. Nested LSTM (巢狀 LSTM)
* **特點**：這是一種「層中層」設計。將標準 LSTM 內部的細胞狀態 ($C_t$) 計算過程，替換成另一個內部的 LSTM 單元（記憶單元管理記憶單元）。
* **優勢**：增加了模型的深度與表現力，能捕捉更長期的時間依賴關係，但計算成本相對較高。

## 9. Multiplicative LSTM (mLSTM)
* **特點**：針對輸入與隱藏狀態的交互方式進行修改，引入乘法交互 (Multiplicative interaction)，使權重矩陣可根據當前輸入動態變化。
* **優勢**：在某些複雜的序列生成任務（如文本生成）中表現優於標準 LSTM，能預測更靈活的機率分佈。

## 10. Grid LSTM (網格 LSTM)
* **特點**：將 LSTM 概念推廣到多維度。標準 LSTM 只有一個時間維度，Grid LSTM 則在多個維度（如深度維度、空間維度）上都設有記憶細胞和門控。
* **優勢**：提供統一框架處理多維數據，例如同時在「網路深度」和「時間步長」上應用 LSTM 機制，有助於解決極深層網路的梯度消失問題。

---

## 總結與選用建議

| 需求場景 | 推薦變體 |
| :--- | :--- |
| **追求速度與輕量化** | GRU, CIFG |
| **自然語言處理 (NLP)** | BiLSTM, Transformer (現今主流) |
| **影片、氣象圖 (時空數據)** | ConvLSTM |
| **程式碼分析、語法樹** | Tree-LSTM |
| **不規則時間間隔數據** | Phased LSTM |
| **複雜長期依賴** | Nested LSTM |
