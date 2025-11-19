# IMDB 影評情緒分類：Fine-Tuned LLaMA vs BERT vs DistilBERT 比較報告

本報告整理三種模型在 IMDB 影評情緒分類任務上的微調方式與比較：

- **LLaMA 3 8B（LoRA 微調，Causal LM）**
- **BERT-base（Sequence Classification 微調）**
- **DistilBERT（Sequence Classification 微調）**

並提供：

1. 模型比較總覽表  
2. 共同實驗設定說明  
3. 三個模型的典型微調程式架構（簡化版）  
4. 推論速度 Benchmark 範例程式架構（簡化版）  

> ⚠️ 注意：實際可訓練與否取決於你的 GPU/VRAM 環境與是否有 LLaMA 權重存取權限。

---

## 1. 模型比較總覽表

| 模型 | 類型 | 參數量 | 微調方式 | IMDB Accuracy（典型） | 訓練時間（A100） | 推論速度 | 適用場景 |
|------|------|--------|-----------|-------------------------|------------------|-----------|-----------|
| **LLaMA 3 8B（LoRA）** | Causal LM | ~8B | LoRA / QLoRA | 96%–97% | 1–2 小時 | 較慢 | 高語意理解、多任務（分類＋生成） |
| **BERT-base** | Encoder-only | 110M | Full Fine-tune | 94%–95% | 10–20 分鐘 | 快 | 標準 NLP 任務（分類、NER、QA） |
| **DistilBERT** | Encoder-only | 66M | Full Fine-tune | 92%–94% | 5–10 分鐘 | 最快 | 輕量部署、低延遲需求 |

---

## 2. 共同實驗設定（建議）

- **資料集**：`datasets.load_dataset("imdb")`
- **任務**：二元情緒分類（positive / negative）
- **訓練 / 測試切分**：官方 IMDB 預設切分
- **最大序列長度**：`max_length = 256 或 512`
- **評估指標**：Accuracy, F1 (macro)

---

## 3. BERT / DistilBERT 微調範例架構（Sequence Classification）

### 3.1 共同前置程式：下載 IMDB

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
print(dataset)
```

### 3.2 BERT-base 微調範例（簡化版）

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
import evaluate

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("imdb")

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)

tokenized = dataset.map(preprocess, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

args = TrainingArguments(
    output_dir="./bert_imdb",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./bert_imdb")
```

### 3.3 DistilBERT 微調範例（簡化版）

只需把 `model_name` 換成 `distilbert-base-uncased`，其餘流程相同：

```python
model_name = "distilbert-base-uncased"
```

---

## 4. LLaMA 3（LoRA）微調架構（Causal LM）

LLaMA 常以 **Causal LM** 的方式訓練，因此微調時，我們通常把完整的輸入（包含「問題 + 答案」）餵給模型，並讓模型學會在 prompt 後產生正確的標籤文字（`positive` 或 `negative`）。

### 4.1 安裝建議套件

```bash
pip install transformers datasets accelerate peft bitsandbytes sentencepiece
```

### 4.2 LLaMA LoRA 微調架構（簡化示範）

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

base_model = "meta-llama/Meta-Llama-3-8B-Instruct"  # 需具備存取權

dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

def build_prompt(text, label=None):
    # label: 0=negative, 1=positive
    base = (
        "You are a sentiment analysis assistant.\n"
        "Classify the following IMDB movie review as positive or negative.\n\n"
        f"Review:\n{text}\n\nAnswer (positive or negative):"
    )
    if label is None:
        return base
    else:
        label_text = "positive" if label == 1 else "negative"
        return base + " " + label_text

def preprocess_function(examples):
    prompts = [build_prompt(t, l) for t, l in zip(examples["text"], examples["label"])]
    model_inputs = tokenizer(
        prompts,
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

args = TrainingArguments(
    output_dir="./llama3_imdb_lora",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=50,
    save_steps=500,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

trainer.train()
model.save_pretrained("./llama3_imdb_lora")
tokenizer.save_pretrained("./llama3_imdb_lora")
```

---

## 5. 推論與推論速度 Benchmark

### 5.1 BERT / DistilBERT 推論

```python
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

model_name = "./bert_imdb"  # 或 "./distilbert_imdb"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
dataset = load_dataset("imdb")
texts = dataset["test"]["text"][:500]  # 取 500 筆測試

def predict_batch(texts, batch_size=32):
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(list(preds))
    return np.array(all_preds)

start = time.time()
preds = predict_batch(texts, batch_size=32)
end = time.time()

print(f"推論 500 筆耗時：{end - start:.2f} 秒")
print(f"每秒處理樣本數：約 {len(texts) / (end - start):.2f} 筆/s")
```

### 5.2 LLaMA LoRA 推論（生成式分類）

```python
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "./llama3_imdb_lora"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

dataset = load_dataset("imdb")
texts = dataset["test"]["text"][:100]  # LLaMA 較慢，示範 100 筆

def build_prompt_only(text):
    return (
        "You are a sentiment analysis assistant.\n"
        "Classify the following IMDB movie review as positive or negative.\n\n"
        f"Review:\n{text}\n\nAnswer (positive or negative):"
    )

def predict_llama(texts):
    preds = []
    for t in texts:
        prompt = build_prompt_only(t)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = decoded.split("Answer (positive or negative):")[-1].strip().lower()
        if "positive" in answer and "negative" not in answer:
            preds.append(1)
        elif "negative" in answer and "positive" not in answer:
            preds.append(0)
        else:
            preds.append(1)
    return preds

start = time.time()
preds = predict_llama(texts)
end = time.time()

print(f"LLaMA 推論 {len(texts)} 筆耗時：{end - start:.2f} 秒")
print(f"每秒處理樣本數：約 {len(texts) / (end - start):.2f} 筆/s")
```

---

## 6. 報告重點總結

1. **LLaMA 3 8B（LoRA 微調）**：  
   - 準確率最高（約 96%–97%），同時兼具生成能力與推理能力。  
   - 適合同時需要「分類 + 解釋 + 生成」的複合任務。  
   - 訓練與推論成本較高，適合有 GPU 環境的伺服器端部署。  

2. **BERT-base 微調**：  
   - 訓練快、推論快、準確率高（約 94%–95%）。  
   - 是 IMDB 及多數 NLP 任務的穩健基線模型。  

3. **DistilBERT 微調**：  
   - 略微犧牲準確率，換取更快推論速度與更小模型大小。  
   - 適合 Edge / 手機裝置、即時情緒監控、低延遲場景。  

你可以依照：**效能需求（Accuracy） vs 成本／延遲（Latency） vs 設備環境（GPU / Edge）**，選擇最適合的模型與部署策略。
