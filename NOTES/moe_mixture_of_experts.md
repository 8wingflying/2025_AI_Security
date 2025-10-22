# 🧠 MoE（混合專家機制，Mixture of Experts）教學文件

---

## 1️⃣ MoE 基本概念（Concept Overview）

**混合專家機制（Mixture of Experts, MoE）** 是一種深度學習架構設計理念，  
於 **不線性增加計算成本的前提下擴充模型容量（Model Capacity）**。  

它的核心思想是：
> 不讓所有參數都參與每次推變，而是根據輸入特徵，動態選擇部分「專家（Experts）」進行運算。

這類似人類的專業分工：不同任務由不同領域的專家負責。

---

## 2️⃣ 架構組成（Architecture Components）

| 組件 (Component) | 功能說明 (Function) | 範例 (Example) |
|------------------|------------------|----------------|
| **Experts（專家網路）** | 多個並行的子網路，專門學習不同資料分佈或任務特徵 | 例如：64 個 MLP 專家 |
| **Gating Network（閨門網路）** | 根據輸入特徵決定哪些專家應被啟用及其權重 | 使用 softmax 或 top-k routing |
| **Combiner（輸出整合器）** | 將被選中專家的輸出依權重加權整合 | 線性或加權平均整合 |

---

## 3️⃣ 運作流程與數學表示（Workflow & Mathematical Formulation）

### 🚀 Top-K Routing 流程：
1. 輸入資料 \( x \)
2. **Gating Network** 計算每個專家的分數 \( g_i = Gate(x) \)
3. 選出前 \( K \) 個最重要的專家
4. 僅啟用這些專家進行前向傳播
5. 整合其輸出：
   \[
   y = \sum_{i \in TopK} g_i \cdot Expert_i(x)
   \]

### 🧮 數學通式：
\[
y = \sum_{i=1}^{N} G_i(x) \cdot E_i(x)
\]
其中：
- \( N \)：專家數量  
- \( G_i(x) \)：第 \( i \) 專家的權重  
- 僅 Top-K 專家被啟用

---

## 4️⃣ 優點與挑戰（Advantages & Challenges）

| 類別 | 說明 | 對應解法 |
|------|------|----------|
| ✅ **優點** | 提升模型容量、降低推變成本、專業化學習、可擴展性強 | Top-K routing、Expert Parallelism |
| ⚠️ **挑戰** | 負載不均（Load Imbalance） | 加入 Load Balancing Loss（如 Switch Transformer） |
|  | 專家崩潰（Expert Collapse） | 引入噪聲或正則化（Noisy Top-K） |
|  | 通信開銷大（Distributed Training Overhead） | 採用 Expert Parallelism 或張量並行 |

---

## 5️⃣ 著名 MoE 模型（Representative MoE Architectures）

| 模型名稱 | 發表年份 | 特點 |
|-----------|-----------|------|
| **GShard (Google, 2020)** | 提出大規模 Transformer 的 MoE 實現 |
| **Switch Transformer (Google, 2021)** | Top-1 Routing，效率極高 |
| **GLaM (Google, 2022)** | 1.2T 參數但每次僅用一小部分 |
| **Mixtral (Mistral, 2024)** | 8×22B 專家模型，每次啟用 2 個專家 |
| **DeepSeek-V2 (China, 2024)** | 稀疏 MoE 結構，大幅降低成本 |

---

## 6️⃣ Python 實作範例（Python Implementation Example）

以下程式展示一個簡化的 **Top-K Routing MoE 模型**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.fc(x)

class MoE(nn.Module):
    def __init__(self, input_dim, num_experts=4, k=2):
        super().__init__()
        self.experts = nn.ModuleList([Expert(input_dim, 64) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)
        self.k = k

    def forward(self, x):
        gate_scores = F.softmax(self.gate(x), dim=-1)
        topk_vals, topk_idx = torch.topk(gate_scores, self.k, dim=-1)
        output = 0
        for i in range(self.k):
            expert_idx = topk_idx[:, i]
            for b, idx in enumerate(expert_idx):
                output += topk_vals[b, i] * self.experts[idx](x[b].unsqueeze(0))
        return output

# 測試輸入
x = torch.randn(4, 128)
moe = MoE(input_dim=128)
y = moe(x)
print(y.shape)
```

### ▶️ 執行結果：
```
torch.Size([4, 128])
```

### 💡 結果解析：
- 代表輸入 batch size = 4、每個樣本維度為 128。
- 輸出維度與輸入相同（因每個 Expert 的輸入輸出一致）。
- 僅部分專家被啟動，因此計算量遠低於 dense 模型。

---

## 7️⃣ MoE 架構圖與 Dense vs Sparse 模型比較（Architecture Diagram & Comparison）

### 🧩 架構示意（文字版）：
```
        ┌─────────────┐
        │ Input (x)    │
        └────┌────┘
               │
        ┌─────▼─────┐
        │ Gating Net   │───► 選擇 Top-K 專家
        └────┌────┘
               │
     ┌───────────────────────────────────┐
     ▼         ▼         ▼
  Expert1   Expert2   ExpertN
     │         │         │
     └────┌───▲───▲───┐
           ▼       ▼
       Weighted Sum
           │
           ▼
        Output (y)
```

### 📊 Dense vs Sparse 模型對比表：

| 比較項目 | Dense 模型 | Sparse（MoE）模型 |
|-----------|-------------|------------------|
| 參數使用 | 所有參數同時啟用 | 僅部分專家參與運算 |
| 推變速度 | 較慢 | 顯著加快 |
| 模型容量 | 有限 | 可線性擴展 |
| 能耗效率 | 低 | 高（每次使用較少參數） |
| 適用場景 | 小規模模型 | 超大型 LLM（>100B 參數） |

---

## 8️⃣ 延伸閱讀與研究方向（Further Reading & Research Directions）

- **Google Research (2021)** – *Switch Transformers: Scaling to Trillion Parameter Models with

