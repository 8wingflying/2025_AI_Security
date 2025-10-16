# IMDB電影評論情感
- 利用 IMDB 電影評論資料集（IMDb Movie Reviews Dataset），訓練底下模型 來預測影評是正面 (Positive) 還是負面 (Negative)
  - LSTM 
## LSTM
- https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- [An Empirical Exploration of Recurrent Network Architectures](https://proceedings.mlr.press/v37/jozefowicz15.html)
- [Exploring the Limits of Language Modeling](https://arxiv.org/abs/1602.02410)

## 🚀 延伸方向
| 方法 | 說明 |
|------|------|
| **BiLSTM** | 同時考慮前後文資訊 |
| **GloVe/Word2Vec** | 使用預訓練詞向量 |
| **Attention 機制** | 提升模型對關鍵詞的注意力 |
| **CNN-LSTM 結構** | 強化特徵抽取能力 |


# 🎬 使用 LSTM 分析 IMDB 電影評論情感

## 📘 一、實驗目標
利用 IMDB 電影評論資料集，訓練 LSTM 模型來預測影評是正面或負面。

---

## 📦 二、匯入套件與載入資料
```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# 參數設定
vocab_size = 10000
maxlen = 200
embedding_dim = 128

# 載入資料
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
```

---

## 🧹 三、資料前處理
```python
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
```

---

## 🧠 四、建立 LSTM 模型
```python
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=maxlen),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

---

## 🧩 五、訓練模型
```python
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)
```

---

## 📊 六、模型評估
```python
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"測試集準確率: {acc:.4f}")
```

---

## 📈 七、視覺化訓練結果
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Curve')
plt.legend()
plt.show()
```

---

## 🔍 八、單筆預測範例
```python
import numpy as np
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

# 顯示範例評論
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])
print(decoded_review)

# 預測
pred = model.predict(np.expand_dims(x_test[0], axis=0))[0][0]
print('預測結果：', '正面 👍' if pred > 0.5 else '負面 👎', f'(機率={pred:.2f})')
```

---

## 📊 九、結果摘要
| 項目 | 說明 |
|------|------|
| 資料集 | IMDB (50,000 筆影評) |
| 模型 | Embedding + LSTM + Dense(sigmoid) |
| 任務 | 二元分類（正面 / 負面） |
| 預期準確率 | 約 85%～90% |

---


