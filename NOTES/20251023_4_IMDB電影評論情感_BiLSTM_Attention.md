# 🎬 使用 LSTM 分析 IMDB 電影評論情感（含 BiLSTM + Attention 擴充版）

## 📘 一、實驗目標
利用 IMDB 電影評論資料集，訓練 LSTM 模型並進一步比較 **LSTM、BiLSTM、BiLSTM + Attention** 三種架構在情感分類任務的表現。

---

## 📦 二、匯入套件與載入資料
```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Layer
import numpy as np

# 基本參數
vocab_size = 10000
maxlen = 200
embedding_dim = 128

# 載入資料
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# 資料前處理
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
```

---

## 🧠 三、LSTM 模型 (Baseline)
```python
model_lstm = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=maxlen),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.summary()
```

---

## 🧩 四、BiLSTM 模型
```python
model_bilstm = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=maxlen),
    Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3)),
    Dense(1, activation='sigmoid')
])

model_bilstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_bilstm.summary()
```

---

## 🔦 五、自定義 Attention 機制
```python
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = inputs * a
        return K.sum(output, axis=1)
```

---

## 🧠 六、BiLSTM + Attention 模型
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

inputs = Input(shape=(maxlen,))
embedding = Embedding(vocab_size, embedding_dim, input_length=maxlen)(inputs)

bilstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(embedding)
attention = Attention()(bilstm)
output = Dense(1, activation='sigmoid')(attention)

model_attention = Model(inputs, output)
model_attention.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_attention.summary()
```

---

## 🚀 七、訓練與評估
```python
history_lstm = model_lstm.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2, verbose=1)
history_bilstm = model_bilstm.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2, verbose=1)
history_attention = model_attention.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2, verbose=1)

# 評估
print("LSTM 準確率:", model_lstm.evaluate(x_test, y_test, verbose=0)[1])
print("BiLSTM 準確率:", model_bilstm.evaluate(x_test, y_test, verbose=0)[1])
print("BiLSTM + Attention 準確率:", model_attention.evaluate(x_test, y_test, verbose=0)[1])
```

---

## 📊 八、三種模型比較
| 模型 | 結構特點 | 語意理解 | 準確率(approx) | 特點 |
|------|-----------|------------|----------------|------|
| LSTM | 單向記憶序列 | 中等 | 85% | 結構簡單，訓練快速 |
| BiLSTM | 雙向序列 | 良好 | 90% | 考慮上下文語意 |
| **BiLSTM + Attention** | 雙向 + 注意力 | **極佳** | **92–94%** | 聚焦關鍵詞，提高可解釋性 |

---

## 📈 九、訓練視覺化
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
plt.plot(history_lstm.history['accuracy'], label='LSTM')
plt.plot(history_bilstm.history['accuracy'], label='BiLSTM')
plt.plot(history_attention.history['accuracy'], label='BiLSTM+Attention')
plt.title('模型準確率比較')
plt.legend()
plt.show()
```

---

## 🔍 十、結論與延伸建議
| 延伸方向 | 說明 |
|-----------|------|
| **Self-Attention / Transformer** | 提升至 Transformer 架構，取代 RNN |
| **使用 GloVe/BERT 向量** | 提升語意理解與泛化能力 |
| **可視化注意力權重** | 分析模型對文字關鍵區段的聚焦效果 |
| **多層 Attention** | 疊加注意力層以學習更深層語意模式 |
