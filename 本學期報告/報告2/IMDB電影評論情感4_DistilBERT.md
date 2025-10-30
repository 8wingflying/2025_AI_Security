#

```python
# 🎬 TensorFlow × DistilBERT IMDB 情感分析教學
> 使用 TensorFlow + Hugging Face Transformers 微調 DistilBERT 模型於 IMDB 電影評論情感分析  
> 作者：T Ben × ChatGPT GPT-5 ｜ 日期：2025-10-30  

---

## 🧠 一、理論簡介

### 🔍 什麼是 DistilBERT？
**DistilBERT** 是由 **BERT** 精簡而來的輕量 Transformer 模型。  
透過 **知識蒸餾（Knowledge Distillation）** 技術，  
小模型可學習大模型的語意表徵，保留約 97% 的性能但速度快約 60%。

### 🌟 優點
- ⚡ 訓練與推論速度更快  
- 🧩 模型體積更小，適合部署  
- 🎯 效能接近原始 BERT  

### 💡 應用場景
- 影評或商品評論的情感分析  
- 文字分類與意圖識別  
- FAQ／客服聊天機器人  

---

## ⚙️ 二、環境安裝

```bash
pip install tensorflow transformers datasets
```
# 📦 三、資料載入與前處理

```PYTHON
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from datasets import load_dataset

# 1️⃣ 載入 IMDB 資料集
imdb = load_dataset("imdb")

# 2️⃣ 取出訓練與測試資料
train_texts = imdb["train"]["text"]
train_labels = imdb["train"]["label"]
test_texts  = imdb["test"]["text"]
test_labels = imdb["test"]["label"]

# 3️⃣ 載入 DistilBERT Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 4️⃣ Tokenize 資料
def preprocess(texts, labels, max_length=128):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}, tf.convert_to_tensor(labels)

train_inputs, train_labels_tf = preprocess(train_texts, train_labels)
test_inputs,  test_labels_tf  = preprocess(test_texts,  test_labels)

# 🧠 四、建立與微調 DistilBERT 模型

# 1️⃣ 載入預訓練 DistilBERT 模型
model = TFDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# 2️⃣ 編譯模型
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
model.compile(
    optimizer=optimizer,
    loss=model.compute_loss,  # Transformers 模型自帶 loss
    metrics=["accuracy"]
)

# 3️⃣ 訓練模型
history = model.fit(
    x=train_inputs,
    y=train_labels_tf,
    validation_split=0.1,
    epochs=3,
    batch_size=16
)

# 4️⃣ 評估模型
loss, acc = model.evaluate(test_inputs, test_labels_tf)
print(f"測試準確率：{acc:.4f}")

#🎯 五、預測新評論

def predict_review(text):
    enc = tokenizer(text, padding=True, truncation=True, return_tensors="tf", max_length=128)
    logits = model(enc["input_ids"], enc["attention_mask"])[0]
    pred = tf.math.argmax(logits, axis=1).numpy()[0]
    return "正面評論 👍" if pred == 1 else "負面評論 👎"

print(predict_review("This movie was fantastic! I really enjoyed it."))
print(predict_review("I hated this film. It was boring and too long."))
```
## 📊 六、模型比較表

```
模型	架構	優點	缺點	準確率 (約)
GRU	RNN	輕量快速	難捕捉長語境	0.85
BiLSTM	雙向 LSTM	語意理解強	訓練慢	0.88
Transformer	Self-Attention	並行高效	資料需求大	0.91
DistilBERT	精簡 BERT	快速精準兼顧	需 GPU 訓練	0.93–0.95
```
## 🧭 七、延伸練習與優化建議
```
超參數微調

嘗試不同的 max_length、batch_size、learning_rate。

凍結部分層

只訓練最後幾層以加快 fine-tuning。

多語言支援

使用 distilbert-base-multilingual-cased 以支援中英混合評論。

注意力可視化

顯示 Attention Map，觀察模型關注哪些詞。

自訂資料集

將中文影評、商品評論導入再訓練，建立在地化情感模型。
```

# 🏁 八、結論
```
DistilBERT 是兼具速度與準確率的高效 NLP 模型。
在 IMDB 任務中，它能以約一半參數達成接近 BERT 的效果（約 93%–95% 準確率）。
非常適合應用於：

客服意圖識別

即時評論分析

行動裝置端 NLP 任務
```
