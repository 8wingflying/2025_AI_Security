# 🧠 NLTK `feature` 模組說明與教學範例

> **核心概念：特徵抽取（Feature Extraction）**  
> 目的：將文字轉換為可供機器學習模型使用的數值特徵，協助進行分類、分析與建模。

---

## 📍 一、模組定位

`nltk.feature` 模組屬於 NLTK 的 **機器學習核心子系統**，常與以下模組搭配使用：

| 模組 | 功能 | 說明 |
|------|------|------|
| `nltk.classify` | 分類器 | Naive Bayes、Decision Tree、MaxEnt 等 |
| `nltk.cluster` | 聚類 | 文字或文件分群 |
| `nltk.metrics` | 評估 | 準確率、F1、Precision 等 |
| `nltk.probability` | 統計支援 | 詞頻、條件機率分佈 |

> `nltk.feature` 是 **特徵工程的底層模組**，用於建立可供 `nltk.classify` 模型訓練的特徵資料。

---

## 🧩 二、特徵抽取的意義

自然語言文字無法直接被演算法理解，因此必須轉換成「特徵（Features）」：

| 原始文字 | 特徵字典表示 |
|------------|----------------|
| "I love Python" | `{ 'contains(love)': True, 'contains(Python)': True }` |
| "I hate bugs" | `{ 'contains(hate)': True, 'contains(bugs)': True }` |

---

## 🧠 三、常見功能與方法

| 功能 | 函數 | 用途 |
|------|------|------|
| 特徵抽取 | 自訂函數 `features(document)` | 將文字轉為特徵字典 |
| 特徵應用 | `nltk.classify.util.apply_features()` | 將特徵函數應用於語料 |
| 特徵格式 | Boolean / Count / TF | 控制特徵型態 |
| 模型整合 | `NaiveBayesClassifier.train()` | 以特徵資料訓練分類器 |

---

## 💡 四、範例 1：手動定義特徵函數

```python
import nltk

# 特徵抽取函數
def document_features(document):
    words = set(document)
    features = {}
    for word in ['love', 'hate', 'python', 'code']:
        features[f'contains({word})'] = (word in words)
    return features

# 範例資料
documents = [
    (['I', 'love', 'python'], 'pos'),
    (['I', 'hate', 'bugs'], 'neg'),
]

# 建立特徵集
featuresets = [(document_features(d), c) for (d, c) in documents]

# 訓練分類器
classifier = nltk.NaiveBayesClassifier.train(featuresets)

print(classifier.classify(document_features(['love', 'code'])))  # pos
print(classifier.classify(document_features(['hate'])))           # neg
```

---

## ⚙️ 五、範例 2：搭配 `apply_features()`

```python
from nltk.classify.util import apply_features

def features(words):
    return {word: True for word in words}

training_data = [
    (["great", "movie"], "pos"),
    (["boring", "plot"], "neg")
]

featuresets = apply_features(features, training_data)
classifier = nltk.NaiveBayesClassifier.train(featuresets)
print(classifier.classify({"great": True}))  # pos
```

---

## 🎬 六、範例 3：應用於情感分析（movie_reviews）

```python
import nltk
from nltk.corpus import movie_reviews
import random

# 準備資料
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# 特徵抽取函數
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def doc_features(doc):
    doc_words = set(doc)
    return {f'contains({w})': (w in doc_words) for w in word_features}

# 訓練與測試
train_set = [(doc_features(d), c) for (d, c) in documents[:1900]]
test_set  = [(doc_features(d), c) for (d, c) in documents[1900:]]

classifier = nltk.NaiveBayesClassifier.train(train_set)
print("Accuracy:", nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(10)
```

---

## 🧮 七、常見特徵型態

| 類型 | 說明 | 範例 |
|------|------|------|
| Boolean | 詞是否出現 | `{'contains(love)': True}` |
| Frequency | 詞出現次數 | `{'love': 2, 'hate': 1}` |
| N-gram | 詞對、詞組 | `{'bigram(love,python)': True}` |
| POS-based | 詞性標註特徵 | `{'verb_count': 3}` |
| Length | 文長或單字長度 | `{'len': 120}` |

---

## 🧱 八、模組整合關係

```
Raw Text
   ↓
Feature Extraction (nltk.feature)
   ↓
Training Set (features, labels)
   ↓
Classification (nltk.classify)
   ↓
Evaluation (nltk.metrics)
```

---

## 🔍 九、與現代 NLP 特徵抽取技術比較

| 模型 | 特徵來源 | 技術 | 範例 |
|------|------------|------|------|
| **NLTK Feature** | 手工特徵 | Boolean | `{ 'contains(love)': True }` |
| **TF-IDF / BOW** | 統計特徵 | 詞頻 + 權重 | `sklearn.feature_extraction.text` |
| **Word2Vec** | 分佈式語意 | 向量空間表示 | `gensim.models.Word2Vec` |
| **BERT / GPT** | 深度 contextual 特徵 | Transformer | `transformers.BertModel` |

---

## ⚖️ 十、優缺點分析

| 面向 | 優點 | 缺點 |
|------|------|------|
| 教學應用 | 簡單易懂，結構清楚 | 僅適用小型資料集 |
| 特徵設計 | 完全可自訂 | 無自動化特徵工程 |
| 整合性 | 與 NLTK 其他模組高度相容 | 不支援深度學習模型 |
| 效能 | 適合入門實驗 | 大資料集處理較慢 |

---

## 🧭 十一、延伸應用

| 應用任務 | 搭配模組 | 功能說明 |
|------------|------------|-------------|
| 情感分析 | `nltk.feature`, `nltk.classify` | 建立情感特徵集 |
| 垃圾郵件過濾 | `nltk.feature`, `nltk.metrics` | 根據文字特徵進行分類 |
| 主題辨識 | `nltk.cluster`, `nltk.feature` | 特徵向量化後分群 |
| 語言模型 | `nltk.probability`, `nltk.feature` | 計算詞機率分佈 |

---

## 📚 十二、參考資料

- 官方網站：[https://www.nltk.org/](https://www.nltk.org/)
- 教科書：*Natural Language Processing with Python* (Bird, Klein, Loper)
- GitHub 原始碼：[https://github.com/nltk/nltk](https://github.com/nltk/nltk)

---

> 📘 **摘要：**
> - `nltk.feature` 是 NLTK 中的特徵抽取基礎模組  
> - 特徵以字典形式表示，支援靜態與動態抽取  
> - 適合教學與研究用途，能與 `classify`、`metrics` 等模組協同運作  
> - 可作為理解現代 NLP 特徵工程（如 TF-IDF、Embedding）之前的重要基礎。

