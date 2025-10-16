# 🧠 Gensim.models 模組教學

## 一、模組概述

`gensim.models` 是 Gensim 的核心模組之一，提供多種 **統計式與語意式的自然語言建模工具**，可用於：

- 主題建模（Topic Modeling）
- 詞向量訓練（Word Embedding）
- 文件向量化（Document Vectorization）
- 語義索引（Semantic Indexing）
- 相似度與分類分析

其設計採 **模組化與串流式（streaming）** 架構，能在不載入整個語料庫的情況下高效運作。

---

## 二、主要模型分類總覽

| 類別 | 模型名稱 | 主要功能 | 對應類別名稱 |
|------|-----------|-----------|---------------|
| 🔹 統計式模型 | TF-IDF、LSI、LDA、HDP、RP、NMF | 主題建模與文件降維 | `TfidfModel`, `LsiModel`, `LdaModel`, `HdpModel`, `RpModel`, `Nmf` |
| 🔹 向量模型 | Word2Vec、FastText、Doc2Vec | 詞或文件嵌入 | `Word2Vec`, `FastText`, `Doc2Vec` |
| 🔹 相似度模型 | KeyedVectors、Phrases、Poincare | 向量相似性與詞組分析 | `KeyedVectors`, `Phrases`, `PoincareModel` |
| 🔹 深度學習式模型 | WordRank、Autoencoder-based | 語意表示學習 | `WordRank`, `AutoEncoder` |

---

## 三、各模型詳細說明

### 1️⃣ TF-IDF 模型（Term Frequency - Inverse Document Frequency）

**類別：** `gensim.models.TfidfModel`  
**用途：** 將文字轉換為權重化的數值向量。

```python
from gensim import models

tfidf = models.TfidfModel(corpus)
tfidf_corpus = tfidf[corpus]
```

**特點：**
- 根據詞頻與逆文件頻率計算權重。
- 可作為其他模型（如 LSI、LDA）的輸入。

---

### 2️⃣ LSI / LSA 模型（Latent Semantic Indexing / Analysis）

**類別：** `gensim.models.LsiModel`  
**用途：** 使用奇異值分解（SVD）將高維文字空間降維，以揭示潛在語意結構。

```python
lsi = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=2)
print(lsi.print_topics(2))
```

**特點：**
- 能揭示詞與主題的隱含語義。
- 適合資訊檢索與語意搜尋。

---

### 3️⃣ LDA 模型（Latent Dirichlet Allocation）

**類別：** `gensim.models.LdaModel`  
**用途：** 透過貝葉斯推論找出文件中的主題分佈。

```python
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=3, passes=10)
print(lda.print_topics(num_words=5))
```

**特點：**
- 主題建模中最常見方法。
- 可調參數：`num_topics`, `passes`, `alpha`, `eta`。
- 可延伸版本：`LdaMulticore`, `LdaSeqModel`。

---

### 4️⃣ HDP 模型（Hierarchical Dirichlet Process）

**類別：** `gensim.models.HdpModel`  
**用途：** 無需事先指定主題數量的非參數式主題建模方法。

```python
hdp = models.HdpModel(corpus, id2word=dictionary)
print(hdp.print_topics(3))
```

**特點：**
- 自動推斷主題數量。
- 適合探索性文本分析。

---

### 5️⃣ RP 模型（Random Projection Model）

**類別：** `gensim.models.RpModel`  
**用途：** 使用隨機投影降低向量維度以估計語義結構。

**特點：**
- 執行速度快、記憶體消耗低。
- 適合大型語料降維預處理。

---

### 6️⃣ NMF 模型（Non-negative Matrix Factorization）

**類別：** `gensim.models.nmf.Nmf`  
**用途：** 將矩陣分解為兩個非負矩陣，以獲得主題權重。

```python
nmf = models.Nmf(corpus=tfidf_corpus, id2word=dictionary, num_topics=3)
```

**特點：**
- 較 LDA 可解釋性高。
- 適合結構化文本資料。

---

### 7️⃣ Word2Vec 模型

**類別：** `gensim.models.Word2Vec`  
**用途：** 學習詞的連續向量表示。

```python
from gensim.models import Word2Vec

sentences = [["king", "queen", "man", "woman"]]
model = Word2Vec(sentences, vector_size=100, min_count=1)
print(model.wv.most_similar("king"))
```

**特點：**
- 支援 CBOW 與 Skip-gram。
- 可進行詞類比與相似度查詢。

---

### 8️⃣ FastText 模型

**類別：** `gensim.models.FastText`  
**用途：** 透過子詞分析提升對稀有詞的理解能力。

```python
from gensim.models import FastText

model = FastText(sentences, vector_size=100, window=3, min_count=1)
```

**特點：**
- 支援 OOV（Out-of-Vocabulary）詞。
- 對語言形態豐富的語言特別有效。

---

### 9️⃣ Doc2Vec 模型

**類別：** `gensim.models.Doc2Vec`  
**用途：** 將整份文件嵌入成向量，用於分類、聚類或相似度檢索。

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(words=['machine', 'learning'], tags=['doc1']),
             TaggedDocument(words=['deep', 'learning'], tags=['doc2'])]
model = Doc2Vec(documents, vector_size=50, window=2, min_count=1, workers=4)
print(model.dv['doc1'])
```

**特點：**
- 延伸 Word2Vec 思想至文件層級。
- 支援文字分類與語意搜尋。

---

## 四、進階與輔助模型

| 模型 | 功能 | 說明 |
|------|------|------|
| **Phrases** | 偵測常見詞組 | 自動將詞組如 “New York” 視為單一詞元 |
| **KeyedVectors** | 向量儲存與載入 | 提供 `.load()`、`.save_word2vec_format()` 方法 |
| **PoincareModel** | 階層式詞嵌入 | 適用於階層結構（如知識圖譜、WordNet） |
| **WordRank** | 順序式詞嵌入學習 | 使用排序損失函數進行詞表示學習 |

---

## 五、應用場景對照表

| 任務 | 推薦模型 | 備註 |
|------|-----------|------|
| 主題分析 | LDA、HDP、NMF | LDA 最常用，HDP 可自動決定主題數 |
| 語意搜尋 | LSI、TF-IDF、Doc2Vec | LSI 能揭示隱含語義 |
| 詞意比較 | Word2Vec、FastText | FastText 支援新詞 |
| 文件分類 | Doc2Vec、TF-IDF | 可搭配 scikit-learn 使用 |
| 知識圖譜表示 | PoincareModel | 適用於階層式結構 |

---

## 六、綜合範例：結合 TF-IDF 與 LDA 主題建模

```python
from gensim import corpora, models

documents = [
    ["human", "interface", "computer"],
    ["survey", "user", "computer", "system"],
    ["system", "human", "system", "eps"],
    ["user", "response", "time"],
    ["trees", "graph", "minors"],
    ["graph", "minors", "survey"]
]

# 建立字典與語料
id2word = corpora.Dictionary(documents)
corpus = [id2word.doc2bow(text) for text in documents]

# 建立 TF-IDF
from gensim.models import TfidfModel
tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# 應用 LDA 進行主題建模
lda = models.LdaModel(corpus_tfidf, num_topics=2, id2word=id2word, passes=10)
print(lda.print_topics(num_words=4))
```

---

## 七、總結

`gensim.models` 模組是文字語意分析的核心，結合統計方法（TF-IDF、LDA、NMF）與語意嵌入（Word2Vec、Doc2Vec、FastText），
能有效應用於主題分析、語意搜尋與文字分類等任務。

> ✅ 建議：在實務中可將多種模型串聯，例如 `TF-IDF → LSI/LDA` 或 `Word2Vec → 聚類`，以獲得更具語義結構的分析結果。