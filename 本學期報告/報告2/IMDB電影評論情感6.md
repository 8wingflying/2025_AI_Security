# 使用 Transformer 分析 IMDB 電影評論情感（含早停機制）

> 目標：以 **Hugging Face Transformers**（如 DistilBERT/BERT）微調二元情感分類器，並在 **IMDB** 影評資料集上評估 Accuracy/F1、混淆矩陣與錯誤分析，**加入早停（Early Stopping）機制** 提升訓練穩定性與防止過擬合。

---

## 目錄

* [環境需求與安裝](#環境需求與安裝)
* [資料與方法總覽](#資料與方法總覽)
* [實作步驟（含 Early Stopping）](#實作步驟含-early-stopping)

  * [1. 讀取資料（`datasets` 的 `imdb`）](#1-讀取資料datasets-的-imdb)
  * [2. 標記化（Tokenizer）與資料集轉換](#2-標記化tokenizer與資料集轉換)
  * [3. 建模（`AutoModelForSequenceClassification`）](#3-建模automodelforsequenceclassification)
  * [4. 訓練設定（含 EarlyStoppingCallback）](#4-訓練設定含-earlystoppingcallback)
  * [5. 訓練與驗證、混淆矩陣](#5-訓練與驗證混淆矩陣)
  * [6. 錯誤分析（Top 誤判）](#6-錯誤分析top-誤判)
  * [7. 模型保存、載入與推論](#7-模型保存載入與推論)
* [延伸：注意力視覺化與 Grad-CAM](#延伸注意力視覺化與-grad-cam)
* [完整程式碼](#完整程式碼)

---

## 4. 訓練設定（含 EarlyStoppingCallback）

Transformers 提供 **`EarlyStoppingCallback`**，可在驗證集指標（如 F1）於多次評估後無明顯提升時，自動停止訓練。

主要參數：

* `early_stopping_patience`: 容忍 epoch 數（如 2 → 若連續 2 次無提升則停止）
* `early_stopping_threshold`: 提升閾值（小於此值視為無進步）
* 需設定 `load_best_model_at_end=True` 並指定 `metric_for_best_model`

---

## 完整程式碼（含早停）

```python
from __future__ import annotations
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import evaluate
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
EPOCHS = 6
SEED = 42
BATCH_TRAIN = 16
BATCH_EVAL = 32

# 固定隨機種子
np.random.seed(SEED)
torch.manual_seed(SEED)

# === 1. 載入 IMDB ===
print("[INFO] Loading IMDB dataset ...")
raw_ds = load_dataset("imdb")
train_valid = raw_ds["train"].train_test_split(test_size=0.1, seed=SEED)
train_ds, valid_ds, test_ds = train_valid["train"], train_valid["test"], raw_ds["test"]

# === 2. Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

train_enc = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
valid_enc = valid_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
test_enc  = test_ds.map(tokenize_fn,  batched=True, remove_columns=["text"])

train_enc.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
valid_enc.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_enc.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# === 3. 模型 ===
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, id2label={0: 'NEG', 1: 'POS'}, label2id={'NEG':0,'POS':1}
)

# === 4. 評估與 Early Stopping ===
acc_metric = evaluate.load("accuracy")
f1_metric  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1  = f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"]
    return {"accuracy": acc, "f1": f1}

args = TrainingArguments(
    output_dir="./imdb_bert_earlystop",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_TRAIN,
    per_device_eval_batch_size=BATCH_EVAL,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_enc,
    eval_dataset=valid_enc,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0001)]
)

# === 5. 訓練與驗證 ===
trainer.train()

# === 6. 測試集評估 ===
raw_preds = trainer.predict(test_enc)
labels = raw_preds.label_ids
preds = np.argmax(raw_preds.predictions, axis=-1)
print(classification_report(labels, preds, target_names=['NEG','POS']))

cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NEG','POS'], yticklabels=['NEG','POS'])
plt.title('Confusion Matrix with Early Stopping')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# === 7. 保存模型 ===
trainer.save_model("./imdb_bert_earlystop_model")
tokenizer.save_pretrained("./imdb_bert_earlystop_model")
```

---

### 📘 Early Stopping 工作原理

* 每個 epoch 結束後執行評估。
* 若 F1 分數連續兩次未明顯提升（差距 < 0.0001），則提前結束訓練。
* `load_best_model_at_end=True` 會自動載入最佳模型權重（依據 metric_for_best_model）。

---

## 小結

早停（Early Stopping）能有效防止過擬合，節省訓練時間，同時保持最佳泛化性能。結合 **Transformer 模型微調**，可在 IMDB 情感分析中達到高準確率與穩定訓練表現。
