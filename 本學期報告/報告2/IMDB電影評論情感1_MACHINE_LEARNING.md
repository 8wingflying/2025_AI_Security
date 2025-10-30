# 使用 TF‑IDF 分析 IMDB 電影評論情感（完整教學與可執行程式碼）

> 目標：以 **TF‑IDF** 將文字轉為特徵，並使用 **Logistic Regression / Linear SVM** 建立二元情感分類器（正向/負向），評估準確率、F1、混淆矩陣，並示範超參數搜尋與錯誤分析。

---

## 目錄

* [環境需求與資料來源](#環境需求與資料來源)
* [方法總覽](#方法總覽)
* [實作步驟](#實作步驟)

  * [1. 載入資料（TFDS: `imdb_reviews`）](#1-載入資料tfds-imdb_reviews)
  * [2. 前處理（清理與分割）](#2-前處理清理與分割)
  * [3. 特徵工程（TF‑IDF）](#3-特徵工程tfidf)
  * [4. 建模（LR 與 Linear SVM）](#4-建模lr-與-linear-svm)
  * [5. 評估（分類報告與混淆矩陣）](#5-評估分類報告與混淆矩陣)
  * [6. 超參數搜尋（GridSearchCV）](#6-超參數搜尋gridsearchcv)
  * [7. 錯誤分析（Top 誤判案例）](#7-錯誤分析top-誤判案例)
  * [8. 模型保存與載入](#8-模型保存與載入)
* [延伸：中英文處理差異與常見技巧](#延伸中英文處理差異與常見技巧)
* [完整程式碼](#完整程式碼)

---

## 環境需求與資料來源

* Python ≥ 3.9
* 套件：`tensorflow-datasets`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`
* 資料集：**IMDB Reviews**（`tensorflow_datasets` 提供，50,000 篇影評，標註為正/負）

安裝：

```bash
pip install tensorflow-datasets pandas numpy scikit-learn matplotlib seaborn joblib
```

> 註：若在無法安裝 `tensorflow_datasets` 的環境，也可改用 Kaggle 的 IMDB CSV 檔；本文主流程以 TFDS 為例。

---

## 方法總覽

1. **讀取資料**：使用 `tfds.load('imdb_reviews')` 取得訓練/測試集文字與標籤。
2. **清理文字**：轉小寫、去 HTML tags、處理縮寫/標點（英文），停用詞根據驗證表現決定是否移除。
3. **向量化**：以 `TfidfVectorizer` 轉為稀疏矩陣，考慮 `ngram_range=(1,2)` 捕捉 bi-grams。
4. **建模**：

   * **Logistic Regression**：`liblinear` 或 `saga` 求解器。
   * **Linear SVM (LinearSVC)**：對高維稀疏特別有效。
5. **評估**：Accuracy、Precision、Recall、F1；繪製混淆矩陣。
6. **調參**：用 `GridSearchCV` 調 `C`、`max_df`、`min_df`、`ngram_range`、`class_weight` 等。
7. **錯誤分析**：列出置信度高的誤判文本以洞察失誤模式。
8. **落地**：保存向量器與模型（`joblib`）。

---

## 實作步驟

### 1. 載入資料（TFDS: `imdb_reviews`）

* 取得 `train`（25k）與 `test`（25k），標籤 0=負面, 1=正面。

### 2. 前處理（清理與分割）

* 英文處理建議：

  * 去 HTML 標籤與多餘空白
  * 規範化收縮詞（e.g., *don't → do not*）
  * 移除或保留停用詞需以驗證結果決策
* 使用訓練集切出驗證集（例如 10%）以促進調參。

### 3. 特徵工程（TF‑IDF）

* `max_features` 可先用 50k~100k 視資源調整
* `ngram_range=(1,2)` 常見、對情感詞與否定詞相鄰效果佳
* `min_df`/`max_df` 控制過稀/過常出現詞

### 4. 建模（LR 與 Linear SVM）

* **Logistic Regression**：可解釋的機率輸出，適合闡述重要特徵
* **LinearSVC**：訓練快速、常見於文本分類基線

### 5. 評估（分類報告與混淆矩陣）

* 使用 `classification_report`、`confusion_matrix`、`accuracy_score`
* 視覺化混淆矩陣以快速辨識偏誤

### 6. 超參數搜尋（GridSearchCV）

* 目標：找出最佳 `C`、`ngram_range`、`max_df` 等組合
* 管線：`Pipeline([('tfidf', TfidfVectorizer(...)), ('clf', LinearSVC(...))])`

### 7. 錯誤分析（Top 誤判案例）

* 篩選機率/決策函數最「自信」但誤判的樣本
* 觀察是否受反諷、比喻、冷門專有名詞影響

### 8. 模型保存與載入

* `joblib.dump(vectorizer, 'tfidf.joblib')`、`joblib.dump(model, 'model.joblib')`
* 推論時先 `joblib.load`，再 `vectorizer.transform` → `model.predict`

---

## 完整程式碼

> 直接複製此區塊可執行（建議在虛擬環境）。

```python
"""
使用 TF-IDF 進行 IMDB 影評情感分析
-------------------------------------------------
步驟：
1) 讀取資料 (tensorflow_datasets: imdb_reviews)
2) 前處理 (清理文字、訓練/驗證/測試切分)
3) TF-IDF 特徵化
4) 建模：Logistic Regression 與 LinearSVC
5) 評估：Accuracy/F1、分類報告、混淆矩陣
6) 超參數搜尋 (GridSearchCV)
7) 錯誤分析與模型保存

Author: You
"""

from __future__ import annotations
import re
import html
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ===============
# 工具：文本清理
# ===============
CONTRACTION_MAP = {
    "can't": "cannot", "won't": "will not", "n't": " not", "'re": " are",
    "'s": " is", "'d": " would", "'ll": " will", "'t": " not",
    "'ve": " have", "'m": " am"
}

def normalize_text(s: str) -> str:
    s = html.unescape(s)
    s = re.sub(r"<.*?>", " ", s)  # 去 HTML tag
    s = s.lower()
    # 展開常見縮寫
    for k, v in CONTRACTION_MAP.items():
        s = s.replace(k, v)
    # 移除非字母與數字（保留基本標點）
    s = re.sub(r"[^a-z0-9\s.,!?']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ===============
# 1) 載入資料
# ===============
print("[INFO] Loading IMDB dataset via TFDS ...")
ds_train, ds_test = tfds.load(
    'imdb_reviews', split=['train', 'test'], as_supervised=True
)

train_texts, train_labels = [], []
for text, label in tfds.as_numpy(ds_train):
    train_texts.append(text.decode('utf-8'))
    train_labels.append(int(label))

test_texts, test_labels = [], []
for text, label in tfds.as_numpy(ds_test):
    test_texts.append(text.decode('utf-8'))
    test_labels.append(int(label))

print(f"Train size: {len(train_texts)}, Test size: {len(test_texts)}")

# ===============
# 2) 前處理 & 切分
# ===============
print("[INFO] Cleaning texts ...")
train_texts_clean = [normalize_text(t) for t in train_texts]
test_texts_clean  = [normalize_text(t) for t in test_texts]

# 從訓練集中再切出驗證集 (10%)
X_train, X_valid, y_train, y_valid = train_test_split(
    train_texts_clean, train_labels, test_size=0.1, random_state=42, stratify=train_labels
)

# ===============
# 3) TF-IDF 特徵化
# ===============
tfidf = TfidfVectorizer(
    max_features=100_000,
    ngram_range=(1, 2),
    min_df=2,            # 低頻詞過濾
    max_df=0.9,          # 極常見詞過濾
    sublinear_tf=True,   # 使用 1 + log(tf)
)

X_train_vec = tfidf.fit_transform(X_train)
X_valid_vec = tfidf.transform(X_valid)
X_test_vec  = tfidf.transform(test_texts_clean)

# ===============
# 4) 建模：LR 與 LinearSVC
# ===============
print("[INFO] Training Logistic Regression ...")
clf_lr = LogisticRegression(max_iter=400, C=2.0, solver='liblinear')
clf_lr.fit(X_train_vec, y_train)

print("[INFO] Training LinearSVC ...")
clf_svc = LinearSVC(C=1.0)
clf_svc.fit(X_train_vec, y_train)

# ===============
# 5) 驗證與測試評估
# ===============
print("[INFO] Evaluating on validation set ...")
valid_preds_lr  = clf_lr.predict(X_valid_vec)
valid_preds_svc = clf_svc.predict(X_valid_vec)

acc_lr  = accuracy_score(y_valid, valid_preds_lr)
acc_svc = accuracy_score(y_valid, valid_preds_svc)
print(f"Valid Accuracy - LR: {acc_lr:.4f}, LinearSVC: {acc_svc:.4f}")

# 以驗證表現較佳者作為最終模型
best_model = clf_svc if acc_svc >= acc_lr else clf_lr
best_name  = 'LinearSVC' if best_model is clf_svc else 'LogisticRegression'
print(f"[INFO] Best model by valid accuracy: {best_name}")

print("[INFO] Evaluating on test set ...")
X_all_train = tfidf.transform(train_texts_clean)  # 可選：在選定模型後用全部訓練資料重訓
best_model.fit(X_all_train, train_labels)

test_preds = best_model.predict(X_test_vec)
print(classification_report(test_labels, test_preds, digits=4))

cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['neg', 'pos'], yticklabels=['neg', 'pos'])
plt.title(f'Confusion Matrix ({best_name})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# ===============
# 6) 超參數搜尋：以 LinearSVC 為例
# ===============
print("[INFO] Grid searching with Pipeline (may take time) ...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(sublinear_tf=True)),
    ('clf', LinearSVC())
])

param_grid = {
    'tfidf__max_features': [50_000, 100_000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__min_df': [2, 5],
    'tfidf__max_df': [0.9, 0.95],
    'clf__C': [0.5, 1.0, 2.0]
}

gs = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='f1')
gs.fit(train_texts_clean, train_labels)
print("Best params:", gs.best_params_)
print("Best CV F1:", gs.best_score_)

best_pipeline = gs.best_estimator_

# 以最佳管線在測試集評估
print("[INFO] Evaluating best pipeline on test set ...")
print(classification_report(test_labels, best_pipeline.predict(test_texts_clean), digits=4))

# ===============
# 7) 錯誤分析：列出 Top 誤判
# ===============
print("[INFO] Collecting hardest errors ...")
# 使用 decision_function 量測信心（LinearSVC 提供）；若為 LR，改用 predict_proba 的最大機率

def get_confidence(model, X):
    if hasattr(model, 'decision_function'):
        s = model.decision_function(X)
        # 將負數轉為接近 0 的信心，正數轉為接近 1
        return np.abs(s)
    elif hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)
        return np.max(p, axis=1)
    else:
        return np.zeros(X.shape[0])

conf_scores = get_confidence(best_model, X_test_vec)
errors_idx = np.where(test_preds != np.array(test_labels))[0]

# 取信心最高但仍誤判的前 10 筆
hard_errors = sorted(errors_idx, key=lambda i: conf_scores[i], reverse=True)[:10]

for rank, i in enumerate(hard_errors, 1):
    print("\n[Hard Error #%d]" % rank)
    print("True:", 'pos' if test_labels[i] == 1 else 'neg',
          "| Pred:", 'pos' if test_preds[i] == 1 else 'neg',
          "| Conf:", float(conf_scores[i]))
    print("Text:", test_texts[i][:500].replace('\n', ' '))

# ===============
# 8) 模型保存（向量器 + 分類器）
# ===============
print("[INFO] Saving artifacts ...")
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
joblib.dump(best_model, f'{best_name}_imdb_sentiment_model.joblib')
print("Saved: tfidf_vectorizer.joblib,", f"{best_name}_imdb_sentiment_model.joblib")

# 推論函式（部署時可複用）
class SentimentClassifier:
    """簡單封裝：載入 TF-IDF 與分類器後做推論。"""
    def __init__(self, vec_path: str, model_path: str):
        self.vectorizer = joblib.load(vec_path)
        self.model = joblib.load(model_path)
    def predict(self, texts: List[str]) -> List[int]:
        X = self.vectorizer.transform([normalize_text(t) for t in texts])
        return self.model.predict(X).tolist()

# 用法示例：
# clf = SentimentClassifier('tfidf_vectorizer.joblib', 'LinearSVC_imdb_sentiment_model.joblib')
# print(clf.predict(["This movie is absolutely fantastic!", "Worst film ever..."]))
```

---

## 延伸：中英文處理差異與常見技巧

* **中文文本**：

  * 需先斷詞（`jieba`、`ckip-transformers` 等），再做 TF‑IDF
  * 停用詞表需使用中文停用詞庫
  * 標點與全半形、同義詞規範化
* **常見提升技巧**：

  * 加入 **字串長度、感嘆號數量、全大寫比例** 等簡單統計特徵
  * 使用 **字符 n‑gram** 抗噪（對拼寫錯誤/俚語更穩健）
  * 負向片語偵測（如 *not good*、*no way*），可用 bi‑gram 捕捉

---

## 小結

本文以 **TF‑IDF + 線性分類器** 建立強力文本分類基線：快速、可解釋且資源需求低。若需更高表現，可嘗試 **預訓練語言模型（如 BERT/DistilBERT）** 做微調，或以 **TF‑IDF + 雙向 LSTM/Transformer** 的混合特徵做比較，以量化成本-效益差異。
