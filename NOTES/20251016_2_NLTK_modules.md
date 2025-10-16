# 🧠 NLTK 模組功能對照與教學範例
Natural Language Toolkit (NLTK) 是 Python 最經典的自然語言處理套件之一，提供完整 NLP 流程支援：從語料庫存取、斷詞、詞性標註、語法分析到文本分類。

---

## 📍 一、NLTK 模組總覽表

| 類別 | 模組名稱 | 功能摘要 | 典型用途 |
|------|-----------|-----------|-----------|
| **語料與資源** | `nltk.corpus` | 語料庫、字典、WordNet | 載入語料、查詢詞彙關係 |
|  | `nltk.data` | 管理資源路徑 | 控制資料下載與存取 |
| **文字前處理** | `nltk.tokenize` | 斷詞、分句 | 分詞與句子切割 |
|  | `nltk.corpus.stopwords` | 停用詞 | 移除無意義字詞 |
|  | `nltk.stem` | 詞幹提取（Stemming） | 擷取詞根 |
|  | `nltk.stem.wordnet` | 詞形還原（Lemmatization） | 還原正規詞形 |
| **語言分析** | `nltk.tag` | 詞性標註（POS Tagging） | 為詞加上詞性標籤 |
|  | `nltk.chunk` | 命名實體辨識（NER） | 找出人名、地名、組織名 |
|  | `nltk.parse` | 語法剖析 | 建立語法樹 |
|  | `nltk.grammar` | 文法定義（CFG） | 自訂句法規則 |
| **語意與字彙** | `nltk.wordnet` | WordNet 詞彙語義網 | 查詢同義詞、反義詞 |
|  | `nltk.wsd` | 詞義消溯 | 判斷多義詞語意 |
| **分類與機器學習** | `nltk.classify` | 文本分類 | Naive Bayes、Decision Tree |
|  | `nltk.feature` | 特徵抽取 | 建立特徵向量 |
|  | `nltk.cluster` | 聚類分析 | 文件或詞聚類 |
| **統計與評估** | `nltk.probability` | 機率分佈 | 詞頻與樣本統計 |
|  | `nltk.metrics` | 評估指標 | Precision、Recall、F1 |
| **視覺化與互動** | `nltk.draw` | 視覺化句法樹 | 顯示語法結構 |
|  | `nltk.app` | 互動應用 | Concordance、聊天機器展示 |

---

## 📘 二、常用模組與程式範例

### 1. `nltk.corpus` 語料庫
```python
from nltk.corpus import brown
print(brown.categories())
print(brown.words(categories='news')[:10])
```

### 2. `nltk.tokenize` 斷詞
```python
from nltk.tokenize import word_tokenize, sent_tokenize
text = "NLTK is a great NLP toolkit. It helps with tokenization."
print(sent_tokenize(text))
print(word_tokenize(text))
```

### 3. `nltk.corpus.stopwords` 停用詞
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
words = word_tokenize("This is an example of removing stop words.")
filtered = [w for w in words if w.lower() not in stop_words]
print(filtered)
```

### 4. `nltk.stem` 詞幹提取
```python
from nltk.stem import PorterStemmer
ps = PorterStemmer()
print(ps.stem("running"))  # run
```

### 5. `nltk.stem.wordnet` 詞形還原
```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("better", pos="a"))  # good
```

### 6. `nltk.tag` 詞性標註
```python
from nltk import pos_tag, word_tokenize
sentence = "John loves natural language processing."
print(pos_tag(word_tokenize(sentence)))
```

### 7. `nltk.chunk` 命名實體辨識
```python
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
text = "Barack Obama was born in Hawaii."
tokens = word_tokenize(text)
tags = pos_tag(tokens)
tree = ne_chunk(tags)
for subtree in tree:
    if isinstance(subtree, Tree):
        print(subtree.label(), ' '.join(c[0] for c in subtree))
```

### 8. `nltk.parse` 語法剖析
```python
import nltk
grammar = nltk.CFG.fromstring("""
  S -> NP VP
  NP -> 'John' | 'Mary'
  VP -> 'runs' | 'eats'
""")
parser = nltk.ChartParser(grammar)
for tree in parser.parse(['John', 'runs']):
    print(tree)
```

### 9. `nltk.classify` 文本分類
```python
from nltk.classify import NaiveBayesClassifier
train = [
    ({'love': True, 'great': True}, 'pos'),
    ({'hate': True, 'bad': True}, 'neg')
]
classifier = NaiveBayesClassifier.train(train)
print(classifier.classify({'love': True}))  # pos
```

### 10. `nltk.wordnet` 詞彙語義網
```python
from nltk.corpus import wordnet as wn
syns = wn.synsets('dog')
print(syns[0].definition())
print(syns[0].lemmas()[0].name())
```

### 11. `nltk.draw` 視覺化句法樹
```python
from nltk.tree import Tree
t = Tree('S', [Tree('NP', ['John']), Tree('VP', ['runs'])])
t.draw()
```

---

## 🤖 三、NLTK Pipeline 統合流程

```
[Corpus] 
   ↓
[Tokenization] 
   ↓
[Stopword Removal] 
   ↓
[Stemming/Lemmatization] 
   ↓
[POS Tagging → NER → Parsing] 
   ↓
[Feature Extraction → Classification]
```

---

## 📊 四、NLTK 模組對照與應用

| 模組 | 功能焦點 | 教學應用 | 延伸用途 |
|------|------------|------------|------------|
| `tokenize` | 分詞與句子切割 | 文本前處理 | 支援中文分詞擴充 |
| `stopwords` | 停用詞清理 | 篩選關鍵詞 | 自定義停用詞表 |
| `stem` | 詞幹提取 | 詞形簡化 | 用於搜尋引擎詞比對 |
| `tag` | 詞性標註 | 文法教學 | 建立語料詞頻統計 |
| `chunk` | 實體辨識 | 語法分析課程 | NLP 專案實體擷取 |
| `parse` | 語法結構 | 語言學教材 | 自然語言生成前處理 |
| `classify` | 文本分類 | 情感分析範例 | Naive Bayes、SVM 教學 |
| `wordnet` | 語意查詢 | 詞彙學與語意分析 | 同義詞擴充、詞義消溯 |

---

## 🎓 五、與其他 NLP 框架比較

| 框架 | 特點 | 適合用途 |
|------|------|-----------|
| **NLTK** | 功能齊全、教育導向 | NLP 教學與研究 |
| **spaCy** | 高效、支援 GPU | 產業級 NLP 系統 |
| **TextBlob** | 簡單封裝 NLTK | 初學者快速應用 |
| **Transformers (Hugging Face)** | 深度學習預訓練模型 | 進階生成與理解任務 |

---

## 📚 六、參考資料
- 官方文件：https://www.nltk.org/
- 教材推薦：_Natural Language Processing with Python (Bird, Klein, Loper)_
- GitHub：https://github.com/nltk/nltk

---

> 📘 **摘要：**
> - NLTK 是 NLP 的入門基石  
> - 提供完整處理鏈（Token → POS → Parse → Classify）  
> - 適合教學、
