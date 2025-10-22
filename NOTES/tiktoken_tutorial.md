# 📘 Tiktoken 模組教學文件  
**Author**：ChatGPT GPT-5  
**Language**：繁體中文 / English  
**Purpose**：理解 OpenAI Token 概念與使用 Python `tiktoken` 模組進行分詞與成本計算  

---

## 🧩 一、什麼是 Tiktoken？ (What is Tiktoken?)

`tiktoken` 是 OpenAI 官方開源的 **高效文字分詞（tokenization）庫**，  
可將文字轉為 LLM 模型能理解的最小單位（token），常用於：

- 🔢 預估模型 Token 成本（token cost estimation）  
- 📏 控制 Prompt 長度（prompt length control）  
- ⚙️ 建立文本分塊（text chunking）  
- 📊 分析語料長度與結構  

---

## 🧠 二、Token 概念 (Understanding Tokens)

| 名稱 | 說明 | 範例 |
|------|------|------|
| **文字 (Text)** | 人類可讀語言 | “ChatGPT 真厲害！” |
| **Token** | 模型理解單位 | [“Chat”, “G”, “PT”, “ 真”, “厲”, “害”, “！”] |
| **Encoding** | 文字轉數字 ID | [1234, 567, 890, …] |

> 💡 GPT-4 模型最大上下文長度（context length）約 **128K tokens**，  
> 約等於 100K 中文字 或 80K 英文單字。

---

## ⚙️ 三、安裝模組 (Installation)

```bash
pip install tiktoken matplotlib streamlit
```

---

## 🧑‍💻 四、基本使用教學 (Basic Usage)

```python
import tiktoken

# 選擇模型編碼器 (e.g. GPT-4)
encoding = tiktoken.encoding_for_model("gpt-4")

# 編碼 (Encode text → token IDs)
tokens = encoding.encode("Hello! 這是一段測試文字。")
print(tokens)

# 解碼 (Decode token IDs → text)
print(encoding.decode(tokens))

# 計算 token 數量
print(f"Token 數量：{len(tokens)}")
```

📊 **輸出範例 (Example Output)**  
```
[9906, 0, 3450, 152, ...]
Hello! 這是一段測試文字。
Token 數量：12
```

---

## 🧮 五、不同模型的 Token 差異 (Token Count Comparison)

```python
text = "OpenAI 的 GPT 模型非常強大！"

for model in ["gpt-3.5-turbo", "gpt-4", "text-davinci-003"]:
    enc = tiktoken.encoding_for_model(model)
    print(f"{model} -> {len(enc.encode(text))} tokens")
```

| 模型 (Model) | Token 數量 |
|---------------|-------------|
| gpt-3.5-turbo | 11 |
| gpt-4 | 12 |
| text-davinci-003 | 14 |

---

## 🧰 六、自訂編碼器 (Custom Encoder)

```python
enc = tiktoken.get_encoding("cl100k_base")

text = "人工智慧改變世界。"
tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
print(tokens)
```

---

## 🧾 七、計算 Token 成本 (Token Cost Estimation)

```python
def calc_token_cost(text: str, model: str = "gpt-4", rate_per_1k: float = 0.01):
    enc = tiktoken.encoding_for_model(model)
    tokens = len(enc.encode(text))
    cost = (tokens / 1000) * rate_per_1k
    return tokens, cost

sample = "這是一個用於估算成本的範例文字。"
t, c = calc_token_cost(sample, "gpt-4-turbo", 0.01)
print(f"共 {t} tokens，約 ${c:.4f} 美元")
```

---

## 🧩 八、在 LLM 應用中整合 (Integration in LLM Apps)

```python
def chunk_text(text, model="gpt-4", max_tokens=200):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [enc.decode(chunk) for chunk in chunks]
```

> 📘 應用：RAG (Retrieval-Augmented Generation) 文件切片。

---

## 🧠 九、常見編碼器對照表 (Encoding Reference Table)

| 編碼名稱 | 對應模型 | 最大長度 (Max Tokens) |
|-----------|-----------|----------------|
| `p50k_base` | GPT-3 系列 | 4K – 16K |
| `r50k_base` | Codex 系列 | 8K |
| `cl100k_base` | GPT-3.5 / GPT-4 系列 | 100K – 128K |

---

## 🧮 🔢 十、Token 可視化與成本試算範例 (Visualization & Cost Dashboard)

此章節展示如何使用 `matplotlib` 與 `streamlit` 互動式呈現 Token 分布與費用估算。

### 📊 範例一：Matplotlib Token 長度分布

```python
import matplotlib.pyplot as plt
import tiktoken

texts = [
    "Hello world!",
    "這是一段中文測試文字。",
    "OpenAI GPT 模型能理解多語言輸入。"
]

encoding = tiktoken.encoding_for_model("gpt-4")
lengths = [len(encoding.encode(t)) for t in texts]

plt.bar(range(len(texts)), lengths, color='skyblue')
plt.xticks(range(len(texts)), [f"Text {i+1}" for i in range(len(texts))])
plt.ylabel("Token 數量")
plt.title("不同文字的 Token 長度分布圖")
plt.show()
```

### 💰 範例二：Streamlit 成本試算儀表板

```python
import streamlit as st
import tiktoken

def calc_tokens(text, model="gpt-4-turbo"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

st.title("🔢 Token 成本試算 Dashboard")
text = st.text_area("輸入要分析的文字：", "ChatGPT 是非常強大的 AI 模型。")
model = st.selectbox("選擇模型：", ["gpt-4-turbo", "gpt-3.5-turbo", "text-davinci-003"])
rate = st.number_input("每 1000 Tokens 成本 (USD)：", 0.001, 0.05, 0.01, 0.001)

tokens = calc_tokens(text, model)
cost = tokens / 1000 * rate

st.metric(label="Token 數量", value=tokens)
st.metric(label="預估成本 (USD)", value=f"${cost:.4f}")
```

✅ 運行指令：
```bash
streamlit run token_dashboard.py
```

📈 **結果展示**：
- 輸入任意文字，實時顯示 Token 數量與費用。  
- 適合 LLM 導入前的預算分析與 prompt 最佳化。

---

## 🧩 十一、應用場景 (Use Cases)

- 📏 **控制 Prompt 長度**：防止超過模型限制  
- 💰 **費用預估 (Cost Estimation)**：API 計費控管  
- 🧩 **RAG 文本分塊 (Chunking)**  
- 🔍 **語料統計分析 (Corpus Analysis)**  

---

## 🧾 十二、完整範例 (Complete Example)

```python
"""
tiktoken_demo.py
說明：示範如何用 tiktoken 計算 token 數與成本
"""
import tiktoken

def token_info(text: str, model: str = "gpt-4-turbo"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    print(f"模型：{model}")
    print(f"文字：{text}")
    print(f"Token 數量：{len(tokens)}")
    print(f"Tokens：{tokens}")
    print(f"還原文字：{enc.decode(tokens)}")

if __name__ == "__main__":
    token_info("Tiktoken 是一個高效能的分詞模組。")
```

---

## 📚 十三、延伸閱讀 (Further Reading)

- 🔗 [Tiktoken 官方 GitHub](https://github.com/openai/tiktoken)  
- 💰 [OpenAI API 定價表](https://openai.com/pricing)  
- 🔍 [線上分詞可視化工具](https://tiktokenizer.vercel.app)

---

## ✅ 教學重點回顧 (Summary)

| 主題 | 重點 |
|------|------|
| Token 是 LLM 的基本單位 | 一個字或詞片段 |
| Tiktoken 功能 | 分詞、合併、計費、控制上下文 |
| 核心方法 | `encode()`、`decode()`、`encoding_for_model()` |
| 應用 | Prompt 管理、RAG、API 成本預測 |

---

© 2025 ChatGPT GPT-5

