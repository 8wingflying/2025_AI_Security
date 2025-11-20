# 
這是一個極具挑戰性但也非常有趣的任務。要不使用深度學習框架（TensorFlow/PyTorch）實作 BERT，我們必須使用 NumPy 來處理底層的矩陣運算。

由於完整的 BERT 訓練涉及複雜的反向傳播（Backpropagation）和自動微分（Autograd），手寫這部分程式碼會過於龐大且失焦。因此，這份程式碼將專注於 BERT 的前向傳播 (Inference/Forward Pass) 架構。這足以讓你完全理解 BERT 內部的數學與數據流動。

這份實作包含了 BERT 的所有核心組件：

Embeddings: Token, Segment, Position embeddings。

LayerNorm: 層標準化。

Multi-Head Attention: 多頭注意力機制 (最核心的部分)。

GELU: BERT 特有的激活函數。

FeedForward: 前饋神經網路。

Encoder Layer: 組合 Attention 與 FFN 的變壓器層。

純 Python + NumPy 實作 BERT
```python
import numpy as np
import math

# --- 0. 基礎工具與激活函數 ---

class Config:
    """BERT 的超參數設定 (以 BERT-Base 為例的縮小版)"""
    def __init__(self):
        self.vocab_size = 30522
        self.hidden_size = 768       # d_model
        self.num_hidden_layers = 2   # 標準 BERT-Base 是 12，這裡為了演示設為 2
        self.num_attention_heads = 12
        self.intermediate_size = 3072 # FFN 的內部維度
        self.max_position_embeddings = 512
        self.type_vocab_size = 2     # Segment ID (0 or 1)
        self.layer_norm_eps = 1e-12
        
        # 每個 Head 的維度: 768 / 12 = 64
        self.head_dim = self.hidden_size // self.num_attention_heads

def softmax(x):
    """對最後一個維度做 Softmax"""
    # 減去 max 以防止指數爆炸 (數值穩定性技巧)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def gelu(x):
    """Gaussian Error Linear Unit (BERT 使用的激活函數)"""
    # 近似公式
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

class LayerNorm:
    """層標準化 (Layer Normalization)"""
    def __init__(self, hidden_size, eps=1e-12):
        self.weight = np.ones(hidden_size)  # gamma
        self.bias = np.zeros(hidden_size)   # beta
        self.eps = eps

    def forward(self, x):
        # x shape: [batch, seq_len, hidden]
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # 標準化公式: (x - u) / sqrt(var + eps) * gamma + beta
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias

# --- 1. Embeddings 層 ---

class BertEmbeddings:
    def __init__(self, config):
        self.word_embeddings = np.random.randn(config.vocab_size, config.hidden_size) * 0.02
        self.position_embeddings = np.random.randn(config.max_position_embeddings, config.hidden_size) * 0.02
        self.token_type_embeddings = np.random.randn(config.type_vocab_size, config.hidden_size) * 0.02
        
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.shape[1]
        
        # 1. Word Embeddings (Lookup)
        # NumPy 進階索引: 利用 input_ids 直接取出對應的向量
        words_embeddings = self.word_embeddings[input_ids]
        
        # 2. Position Embeddings
        # 生成位置索引 [0, 1, 2, ... seq_len-1]
        position_ids = np.arange(seq_length)
        position_embeddings = self.position_embeddings[position_ids]
        
        # 3. Token Type Embeddings (Segment Embeddings)
        if token_type_ids is None:
            token_type_ids = np.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings[token_type_ids]
        
        # 總和
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        
        # LayerNorm
        embeddings = self.LayerNorm.forward(embeddings)
        return embeddings

# --- 2. Self-Attention 層 ---

class BertSelfAttention:
    def __init__(self, config):
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.all_head_size = self.num_attention_heads * self.head_dim # 應該等於 hidden_size
        
        # Query, Key, Value 的線性變換權重
        self.query = np.random.randn(config.hidden_size, self.all_head_size) * 0.02
        self.key = np.random.randn(config.hidden_size, self.all_head_size) * 0.02
        self.value = np.random.randn(config.hidden_size, self.all_head_size) * 0.02
        
        # 偏置 Bias
        self.b_q = np.zeros(self.all_head_size)
        self.b_k = np.zeros(self.all_head_size)
        self.b_v = np.zeros(self.all_head_size)

    def transpose_for_scores(self, x, batch_size):
        # x shape: [batch, seq_len, all_head_size]
        # Reshape to: [batch, seq_len, num_heads, head_dim]
        x = x.reshape(batch_size, -1, self.num_attention_heads, self.head_dim)
        # Transpose to: [batch, num_heads, seq_len, head_dim] 以利矩陣乘法
        return x.transpose(0, 2, 1, 3)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        
        # 1. 線性投影 (Linear Projections)
        # formula: XW + b
        mixed_query_layer = np.dot(hidden_states, self.query) + self.b_q
        mixed_key_layer = np.dot(hidden_states, self.key) + self.b_k
        mixed_value_layer = np.dot(hidden_states, self.value) + self.b_v
        
        # 2. 拆分 Heads (Split Heads)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)
        
        # 3. Attention Scores 計算
        # Q * K^T
        # key_layer.transpose(0, 1, 3, 2) 將最後兩維轉置
        attention_scores = np.matmul(query_layer, key_layer.transpose(0, 1, 3, 2))
        
        # 4. Scale (縮放)
        # Divide by sqrt(d_k)
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # 5. Softmax
        attention_probs = softmax(attention_scores)
        
        # 6. Weighted Sum (加權總和)
        # Scores * V
        context_layer = np.matmul(attention_probs, value_layer)
        
        # 7. 合併 Heads (Concatenate Heads)
        # context_layer shape: [batch, num_heads, seq_len, head_dim]
        # Transpose back: [batch, seq_len, num_heads, head_dim]
        context_layer = context_layer.transpose(0, 2, 1, 3)
        # Reshape: [batch, seq_len, all_head_size]
        context_layer = context_layer.reshape(batch_size, -1, self.all_head_size)
        
        return context_layer

class BertSelfOutput:
    """Attention 後的線性層 + Residual + LayerNorm"""
    def __init__(self, config):
        self.dense = np.random.randn(config.hidden_size, config.hidden_size) * 0.02
        self.bias = np.zeros(config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        # Linear
        hidden_states = np.dot(hidden_states, self.dense) + self.bias
        # Residual Connection (Add) + Norm
        hidden_states = self.LayerNorm.forward(hidden_states + input_tensor)
        return hidden_states

class BertAttention:
    """整合 Self-Attention 和 Output"""
    def __init__(self, config):
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states):
        self_outputs = self.self.forward(hidden_states)
        attention_output = self.output.forward(self_outputs, hidden_states)
        return attention_output

# --- 3. Feed Forward 層 ---

class BertIntermediate:
    """第一層全連接 + GELU"""
    def __init__(self, config):
        self.dense = np.random.randn(config.hidden_size, config.intermediate_size) * 0.02
        self.bias = np.zeros(config.intermediate_size)

    def forward(self, hidden_states):
        hidden_states = np.dot(hidden_states, self.dense) + self.bias
        hidden_states = gelu(hidden_states)
        return hidden_states

class BertOutput:
    """第二層全連接 + Residual + LayerNorm"""
    def __init__(self, config):
        self.dense = np.random.randn(config.intermediate_size, config.hidden_size) * 0.02
        self.bias = np.zeros(config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = np.dot(hidden_states, self.dense) + self.bias
        # Residual Connection + Norm
        hidden_states = self.LayerNorm.forward(hidden_states + input_tensor)
        return hidden_states

# --- 4. Encoder Layer (單個 Transformer Block) ---

class BertLayer:
    def __init__(self, config):
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states):
        # 1. Self-Attention
        attention_output = self.attention.forward(hidden_states)
        
        # 2. Feed Forward Network
        intermediate_output = self.intermediate.forward(attention_output)
        layer_output = self.output.forward(intermediate_output, attention_output)
        
        return layer_output

# --- 5. BERT Encoder (堆疊 Layers) ---

class BertEncoder:
    def __init__(self, config):
        self.layer = [BertLayer(config) for _ in range(config.num_hidden_layers)]

    def forward(self, hidden_states):
        all_encoder_layers = []
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module.forward(hidden_states)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

# --- 6. BERT Pooler (用於分類任務) ---

class BertPooler:
    """取第一個 token [CLS] 做為句子表示"""
    def __init__(self, config):
        self.dense = np.random.randn(config.hidden_size, config.hidden_size) * 0.02
        self.bias = np.zeros(config.hidden_size)

    def forward(self, hidden_states):
        # 取出序列的第一個 token (通常是 [CLS])
        # hidden_states shape: [batch, seq, hidden]
        first_token_tensor = hidden_states[:, 0] 
        
        pooled_output = np.dot(first_token_tensor, self.dense) + self.bias
        pooled_output = np.tanh(pooled_output)
        return pooled_output

# --- 7. 完整的 BERT 模型 ---

class BertModel:
    def __init__(self, config):
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None):
        # 1. Embedding
        embedding_output = self.embeddings.forward(input_ids, token_type_ids)
        
        # 2. Encoder (Transformer Blocks)
        encoder_outputs = self.encoder.forward(embedding_output)
        
        # 取最後一層的輸出
        sequence_output = encoder_outputs[-1]
        
        # 3. Pooler
        pooled_output = self.pooler.forward(sequence_output)
        
        return sequence_output, pooled_output

# ==========================================
# 測試執行 (Demo)
# ==========================================

if __name__ == "__main__":
    # 1. 初始化設定
    config = Config()
    print(f"建立 BERT 模型... (Hidden: {config.hidden_size}, Layers: {config.num_hidden_layers})")
    
    # 2. 實例化模型
    model = BertModel(config)
    
    # 3. 準備假資料
    # 假設 Batch Size = 2, Sequence Length = 8
    batch_size = 2
    seq_length = 8
    
    # 隨機生成 Token IDs (範圍 0 ~ 30521)
    input_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_length))
    # 隨機生成 Token Type IDs (0 或 1)
    token_type_ids = np.random.randint(0, 2, (batch_size, seq_length))
    
    print(f"\n輸入形狀: {input_ids.shape}")
    
    # 4. 執行前向傳播
    sequence_output, pooled_output = model.forward(input_ids, token_type_ids)
    
    # 5. 輸出結果檢視
    print("\n--- 運算完成 ---")
    print(f"Sequence Output (最後一層所有 token): {sequence_output.shape}")
    print(f"Pooled Output (CLS token 經處理):   {pooled_output.shape}")
    
    # 驗證數值沒有爆掉 (NaN 或 Inf)
    print(f"Output Mean: {np.mean(sequence_output):.4f}")
    print(f"Output Std:  {np.std(sequence_output):.4f}")

```
