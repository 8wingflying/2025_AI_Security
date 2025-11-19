# 使用 Transformer 分析 IMDB 電影評論情感（完整教學與可執行程式碼）

> 目標：以 **Hugging Face Transformers**（如 DistilBERT/BERT）微調二元情感分類器，並在 **IMDB** 影評資料集上評估 Accuracy/F1，繪製混淆矩陣、觀察注意力，進行錯誤分析與模型保存/推論。

---

## 目錄

* [環境需求與安裝](#環境需求與安裝)
* [資料與方法總覽](#資料與方法總覽)
* [實作步驟（PyTorch + Transformers Trainer）](#實作步驟pytorch--transformers-trainer)

  * [1. 讀取資料（`datasets` 的 `imdb`）](#1-讀取資料datasets-的-imdb)
  * [2. 標記化（Tokenizer）與資料集轉換](#2-標記化tokenizer與資料集轉換)
  * [3. 建模（`AutoModelForSequenceClassification`）](#3-建模automodelforsequenceclassification)
  * [4. 訓練設定（`TrainingArguments`）與評估指標](#4-訓練設定trainingarguments與評估指標)
  * [5. 訓練與驗證、混淆矩陣](#5-訓練與驗證混淆矩陣)
  * [6. 錯誤分析（Top 誤判）](#6-錯誤分析top-誤判)
  * [7. 模型保存、載入與推論](#7-模型保存載入與推論)
* [效能與實務技巧](#效能與實務技巧)
* [延伸：注意力視覺化與 Grad-CAM 風格解釋](#延伸注意力視覺化與-grad-cam-風格解釋)
* [完整程式碼](#完整程式碼)

---

## 環境需求與安裝

* Python ≥ 3.9
* 套件：`transformers`, `datasets`, `evaluate`, `accelerate`, `scikit-learn`, `torch`, `matplotlib`, `seaborn`

安裝：

```bash
pip install -U transformers datasets evaluate accelerate scikit-learn matplotlib seaborn torch
# 若在 Windows 且無 GPU，可先安裝 CPU 版 torch；若有 NVIDIA GPU，請依 PyTorch 官網指令安裝相容 CUDA 版。
```

---

## 資料與方法總覽

* **資料集**：`datasets` 提供的 **IMDB**（train/test 各 25k 篇，標籤：0=負面、1=正面）。
* **模型**：以 **DistilBERT** 為基線（速度快、參數少），可輕鬆切換為 `bert-base-uncased`、`roberta-base`。
* **流程**：文字 → Tokenizer → Transformer Encoder → `CLS` 分類頭 → Cross-Entropy 損失 → 指標（Accuracy、F1）。
* **優勢**：能捕捉長距依賴與上下文語意，對否定詞與諷刺相對更穩健（但仍需錯誤分析）。

---

## 實作步驟（PyTorch + Transformers Trainer）

### 1. 讀取資料（`datasets` 的 `imdb`）

* 使用 `load_dataset('imdb')`，並以 `train` 切分出 `validation`（例如 10%）。

### 2. 標記化（Tokenizer）與資料集轉換

* `AutoTokenizer.from_pretrained('distilbert-base-uncased')`
* `truncation=True`、`padding='max_length'`（訓練用）、`max_length=256`（可依 GPU 記憶體調整）

### 3. 建模（`AutoModelForSequenceClassification`）

* `num_labels=2`，注意 ID-to-Label/Label-to-ID 對應。

### 4. 訓練設定（`TrainingArguments`）與評估指標

* 常用設定：`per_device_train_batch_size=16`、`per_device_eval_batch_size=32`、`num_train_epochs=3~5`、`warmup_ratio=0.1`、`weight_decay=0.01`、`fp16=True`（有支援時）。
* 指標：`accuracy`, `f1`（binary）。

### 5. 訓練與驗證、混淆矩陣

* 用 `Trainer.train()` 訓練，`Trainer.evaluate()` 驗證。
* 以 `Trainer.predict()` 在測試集產生預測，繪製混淆矩陣以觀察類別偏誤。

### 6. 錯誤分析（Top 誤判）

* 找出置信度高但誤判的樣本，檢視語意現象（反諷、比喻、多義詞、專有名詞、劇透詞彙等）。

### 7. 模型保存、載入與推論

* `trainer.save_model()` 與 `tokenizer.save_pretrained()`。
* 推論：`pipeline('text-classification', model=..., tokenizer=...)` 或手動 `tokenizer → model → softmax`。

---

## 效能與實務技巧

* **選模**：`distilbert-base-uncased`（快）→ `bert-base-uncased`（表現略好）→ `roberta-base`（常見強力基線）。
* **序列長度**：`max_length=256/320/384` 視 GPU 記憶體與長文比例調整。
* **資料清理**：移除 HTML 或雜訊；保留關鍵標點可能有助情感判斷。
* **類別不平衡**：IMDB 原則上平衡；若遇到偏斜資料，可考慮 class weights 或 focal loss。
* **正則化**：`weight_decay`、dropout、早停（`load_best_model_at_end=True` + `metric_for_best_model='f1'`）。
* **加速**：混合精度（`fp16/bf16`）、gradient accumulation、梯度裁剪、`accelerate` 自動裝置管理。

---

## 延伸：注意力視覺化與 Grad-CAM 風格解釋

* 取出 `attentions=True` 的注意力權重，視覺化 token 與 `CLS` 的關聯熱圖。
* 以 **Integrated Gradients** 或 **Layer Integrated Gradients**（captum）做敏感度分析，強化可解釋性。

---

## 完整程式碼

> 可直接複製執行（建議在 GPU/Colab）。若無 GPU，也可縮小 `batch_size` 與 `max_length` 降低資源需求。

```python
"""
IMDB 影評情感分類（Transformer 微調：DistilBERT）
-------------------------------------------------
步驟：
1) 載入 IMDB (datasets)
2) 標記化 (AutoTokenizer)
3) 建模 (AutoModelForSequenceClassification)
4) 訓練 (Trainer) 與評估（Accuracy/F1）
5) 測試集預測、混淆矩陣
6) 錯誤分析（Top 誤判）
7) 模型保存與推論
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_NAME = "distilbert-base-uncased"  # 可改："bert-base-uncased", "roberta-base"
MAX_LEN = 256
BATCH_TRAIN = 16
BATCH_EVAL = 32
EPOCHS = 3
SEED = 42

# 設定隨機種子
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("[INFO] Loading IMDB dataset ...")
raw_ds = load_dataset("imdb")  # splits: train(25k) / test(25k)

# 由 train 切分出 validation（10%）
raw_train_valid = raw_ds["train"].train_test_split(test_size=0.1, seed=SEED)
train_ds = raw_train_valid["train"]
valid_ds = raw_train_valid["test"]
test_ds  = raw_ds["test"]

print(train_ds)
print(valid_ds)
print(test_ds)

print("[INFO] Loading tokenizer and model ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenize 函式
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

# 對 train/valid/test 進行標記化
train_enc = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
valid_enc = valid_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
test_enc  = test_ds.map(tokenize_fn,  batched=True, remove_columns=["text"])

# 設定格式為 torch.Tensor
train_enc.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
valid_enc.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_enc.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

num_labels = 2
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
)

# 評估指標（accuracy + f1）
acc_metric = evaluate.load("accuracy")
f1_metric  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1  = f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"]
    return {"accuracy": acc, "f1": f1}

args = TrainingArguments(
    output_dir="./outputs_imdb_distilbert",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_TRAIN,
    per_device_eval_batch_size=BATCH_EVAL,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=100,
    seed=SEED,
    fp16=torch.cuda.is_available(),
    report_to=["none"],  # 若要使用 wandb/tensorboard 可修改
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_enc,
    eval_dataset=valid_enc,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("[INFO] Start training ...")
trainer.train()

print("[INFO] Evaluate on validation ...")
val_metrics = trainer.evaluate()
print(val_metrics)

print("[INFO] Predict on test set ...")
raw_preds = trainer.predict(test_enc)
logits = raw_preds.predictions
labels = raw_preds.label_ids
preds = np.argmax(logits, axis=-1)

print("[Classification Report]\n", classification_report(labels, preds, target_names=["NEGATIVE", "POSITIVE"]))

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=["NEGATIVE", "POSITIVE"], yticklabels=["NEGATIVE", "POSITIVE"])
plt.title("Confusion Matrix (DistilBERT on IMDB)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# 取得每筆的置信度（softmax）以做錯誤分析
probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
conf = probs.max(axis=1)

# 錯誤索引（預測 != 真值）
errors = np.where(preds != labels)[0]
# 取出最「自信」但仍誤判的前 10 筆
hard_errors = errors[np.argsort(conf[errors])[::-1][:10]]

print("\n[Hard Errors - Top 10]")
for rank, idx in enumerate(hard_errors, 1):
    print(f"#{rank:02d} | True: {id2label[int(labels[idx])]} | Pred: {id2label[int(preds[idx])]} | Conf: {conf[idx]:.4f}")
    # 若要顯示原文需保留 test_ds['text']，此處示例：
    # 為避免記憶體壓力，可先將 test 文本另存：

# ==== 保存模型與 Tokenizer ====
print("[INFO] Saving model & tokenizer ...")
trainer.save_model("./imdb_distilbert_model")
tokenizer.save_pretrained("./imdb_distilbert_model")

# ==== 推論函式 ====
from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="./imdb_distilbert_model",
    tokenizer="./imdb_distilbert_model",
    device=0 if torch.cuda.is_available() else -1,
)

examples = [
    "This movie is absolutely fantastic! A must watch.",
    "Terrible plot and wooden acting. I want my time back.",
]
print("\n[Inference Examples]")
for t in examples:
    print(t, "=>", clf(t))
```

---

### （可選）嘗試更強基線

* 將 `MODEL_NAME` 改為：

  * `bert-base-uncased`：常見基線，表現與可解釋性良好。
  * `roberta-base`：往往有更佳 F1，但資源需求略高。
* 增加 `num_train_epochs` 至 4~5，或微調 `learning_rate ∈ [1e-5, 3e-5]`、`max_length ∈ [256, 384]`。

### （可選）早停與類別權重

* 早停：將 `evaluation_strategy="steps"` + 較小 `save_steps`，並監控驗證 F1。
* 類別權重：若資料不平衡，可在 `loss` 前加權或覆寫 `Trainer` 的 `compute_loss`。

---

## 小結

本文示範以 **Transformer（DistilBERT/BERT）** 微調在 IMDB 上的二元情感分類流程，提供從資料讀取、標記化、訓練設定、評估、混淆矩陣、錯誤分析到推論與保存的完整範例。與傳統 **TF‑IDF + 線性分類器** 相比，Transformer 對語意上下文更敏感，通常能取得更佳的泛化性能；實務上建議同時保留傳統基線以便成本‑效益比較與回歸驗證。
