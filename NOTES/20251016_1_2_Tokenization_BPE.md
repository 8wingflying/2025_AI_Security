# BPE（Byte Pair Encoding）子詞編碼
- https://zhuanlan.zhihu.com/p/424631681
- https://www.geeksforgeeks.org/nlp/byte-pair-encoding-bpe-in-nlp/
- https://ml-digest.com/byte-pair-encoding-bpe/
- https://www.youtube.com/watch?v=NrT5kmnTFCk

## 一、BPE 的核心概念
BPE（Byte Pair Encoding，字節對編碼）是一種常用於**自然語言處理（NLP）**與**大型語言模型（LLM）**的**子詞分詞法**。其目標是在「字詞級」與「字元級」之間取得平衡，透過頻率統計逐步合併高頻字元對，形成更穩定的詞彙表。

**核心精神**：
> 將語料中的字詞拆解為字元，然後根據出現頻率逐步合併最常見的字元對，形成更大的子詞單位。

---

## 二、BPE 的演算法步驟
假設語料如下：
```
low, lower, newest, widest
```

### Step 1️⃣：初始化
將所有字詞拆成字元並加上詞尾符號 `_`：
```
l o w _
l o w e r _
n e w e s t _
w i d e s t _
```

### Step 2️⃣：統計字元對（pair）頻率
| 字元對 | 出現次數 |
|--------|-----------|
| (l, o) | 2 |
| (o, w) | 2 |
| (e, s) | 2 |
| (s, t) | 2 |

### Step 3️⃣：合併最高頻的 pair
合併 (l, o) → 形成 `lo`：
```
lo w _
lo w e r _
n e w e s t _
w i d e s t _
```

### Step 4️⃣：重複合併直到詞彙表大小達到上限
最終詞彙可能包含：
```
{l, o, w, e, r, n, d, i, s, t, lo, low, ne, new, est}
```

---

## 三、BPE 的特點與優勢
| 特性 | 說明 |
|------|------|
| 靈活性高 | 能在字與詞之間找到平衡，減少未知詞 (OOV) |
| 可壓縮詞表 | 詞彙表可控制在 30K~50K 範圍 |
| 多語言通用 | 適用於中英混合語言環境 |
| 模型友好 | 被 GPT、BERT、LLaMA 等模型廣泛使用 |

---

## 四、BPE 在 NLP 模型中的應用
1. **OpenAI GPT 系列**：使用 BPE 子詞分割，詞彙表約 50,000 tokens。
   - 範例：`unbelievable → ['un', 'believ', 'able']`
2. **Hugging Face Tokenizer**：`ByteLevelBPETokenizer` 自動學習合併規則。
3. **SentencePiece (Google)**：延伸版 BPE，能直接處理 byte 層資料。

---

## 五、與其他分詞方法比較
| 方法 | 單位 | 優點 | 缺點 |
|------|------|------|------|
| Word-level | 以詞為單位 | 語義明確 | 詞表龐大，OOV 問題嚴重 |
| Character-level | 以字元為單位 | 無 OOV 問題 | 序列過長，語義弱 |
| BPE（Subword-level） | 子詞單位 | 平衡長度與語義 | 須離線訓練合併規則 |

---

## 六、Python 實作範例
```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 初始化 BPE 模型
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 訓練詞彙
trainer = trainers.BpeTrainer(vocab_size=2000, show_progress=True)
tokenizer.train(["data.txt"], trainer)

# 測試編碼
encoded = tokenizer.encode("unbelievable results in AI models")
print(encoded.tokens)
```

輸出範例：
```
['un', 'believ', 'able', 'results', 'in', 'AI', 'models']
```

---

## 七、重點總結
| 重點 | 說明 |
|------|------|
| 定義 | 根據字元頻率逐步合併的子詞編碼演算法 |
| 目的 | 降低 OOV、減少詞表、提升泛化能力 |
| 典型應用 | GPT、BERT、LLaMA、T5 等大型語言模型 |
| 核心價值 | 在字元與詞彙之間取得最佳平衡，保留語義又具運算效率 |

---

> 📘 **延伸閱讀**：
> - Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*.
> - Hugging Face Tokenizers 文檔：https://huggingface.co/docs/tokenizers
