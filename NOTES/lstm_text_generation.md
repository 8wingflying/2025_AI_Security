# 🧠 使用 LSTM 進行文本生成（Text Generation）

## 一、概念說明

**文本生成** 是自然語言處理（NLP）中的一項經典任務，目的是：
> 根據過去的文字序列，自動預測並生成下一個字或詞，進而組成新句子。

### 為何使用 LSTM？
LSTM 是一種特殊的 **RNN（Recurrent Neural Network）**，可有效解決：
- **長期依賴問題（Long-term dependencies）**
- **梯度消失/爆炸問題**

因此非常適合處理文字序列等長序列資料。

---

## 二、LSTM 文本生成的流程

| 階段 | 說明 |
|------|------|
| 1️⃣ 資料收集 | 準備一段文本資料（例如：小說、文章、詩詞） |
| 2️⃣ 前處理 | 將文字轉為數字序列（tokenize + padding） |
| 3️⃣ 建立模型 | 使用 LSTM 層學習上下文關係 |
| 4️⃣ 訓練模型 | 以序列的前 n-1 個字預測第 n 個字 |
| 5️⃣ 生成文本 | 根據起始文字遞迴生成新字 |

---

## 三、範例：使用 Keras 建立 LSTM 文本生成模型

### 📦 套件安裝
```bash
pip install tensorflow numpy pandas
```

---

### 📚 範例資料
```python
text = """人工智慧正在改變世界。
生成式人工智慧讓電腦學會創造，
從文字、圖像到音樂，皆可由模型自動生成。"""
```

---

### 🧹 資料前處理
```python
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# 建立 Tokenizer
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# 產生訓練序列
sequences = []
for i in range(1, len(text)):
    seq = text[:i+1]
    token_list = tokenizer.texts_to_sequences([seq])[0]
    sequences.append(token_list)

# 對齊長度
max_seq_len = max([len(x) for x in sequences])
sequences = np.array(pad_sequences(sequences, maxlen=max_seq_len, padding='pre'))

# X: 前 n-1 個字, y: 第 n 個字
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=total_words)
```

---

### 🧠 建立 LSTM 模型
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(total_words, 64, input_length=max_seq_len-1),
    LSTM(128),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

---

### 🏋️‍♀️ 模型訓練
```python
history = model.fit(X, y, epochs=200, verbose=1)
```

---

### ✨ 文本生成函式
```python
def generate_text(seed_text, next_chars=50):
    for _ in range(next_chars):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0))
        output_char = ""
        for char, index in tokenizer.word_index.items():
            if index == predicted:
                output_char = char
                break
        seed_text += output_char
    return seed_text

print(generate_text("人工智慧", 100))
```

---

## 四、進階技巧

| 技術 | 說明 |
|------|------|
| **多層 LSTM** | 增加模型深度提升語意理解能力 |
| **Bidirectional LSTM** | 捕捉前後文語境 |
| **Dropout** | 防止過擬合 |
| **Temperature Sampling** | 在生成時控制創造性（越高越隨機） |
| **Word-level Tokenization** | 以詞為單位生成更自然語句 |

---

## 五、示範輸出

輸入：
```
人工智慧
```

輸出：
```
人工智慧正在改變世界，讓人類與機器之間的界線越來越模糊。
```

---

## 六、延伸應用

- 📜 詩詞生成：學習古詩句模式，自動創作詩歌。
- 💬 對話生成：作為聊天機器人核心模型。
- 📰 新聞續寫：根據開頭生成完整文章。
- 🎨 多模態生成：與圖像、音樂模型結合成為 Multimodal LLM。

---

✅ 本教學展示如何以 LSTM 進行文本生成，從資料預處理、模型訓練到生成新文本的完整流程，並可延伸應用於多種創作型任務。

