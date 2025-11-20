## Google Gemma 3 模型（可能您是在用簡寫指代 的 4-bit 量化版本，或者是 “Gem for Gemma” 的某種昵稱）
- Gemma 3 是 Google 在 2025 年發佈的多模態開放模型，非常適合用於文本分析任務。
- 本指南將展示如何使用 Gemma 3-4B-IT（指令微調版）對 IMDB 電影評論數據集 進行情感分析（Sentiment Analysis）。
- 為了在單張 GPU（如 Colab T4 或本地顯卡）上高效運行，我們將使用 4-bit 量化載入模型。
- 版本1:精簡版
- 版本2:微調版
  - 如果您想讓 Gemma 3 在 IMDB 上達到極致的準確率（例如用於學術競賽），您可以使用 PEFT 和 LoRA 技術進行微調：
  - 1.	準備數據: 將 IMDB 處理成 {"text": "...", "label": "..."} 格式。
  - 2.	使用 LoRA: 凍結模型大部分參數，只訓練 Low-Rank Adapters。
  - 3.	訓練庫: 使用 Hugging Face 的 trl 庫中的 SFTTrainer。

## 1. 環境準備
- 首先，您需要安裝必要的 Python 庫。 Gemma 3 需要較新的 版本。transformers
```bash
pip install -U transformers torch accelerate bitsandbytes datasets
```
## 2. 完整代碼範例：使用 Gemma 3 分析 IMDB
- 範例程式會執行以下操作：
  - 載入IMDB數據集。
  - 載入 4-bit 量化的 Gemma 3 模型（節省顯存）。
  - 構建 Prompt 讓模型判斷評論是 “Positive” （正面） 還是 “Negative” （負面）。
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm

# 1. 設置模型 ID (使用 Google 的官方 Gemma 3 指令微調版)
# 如果您的顯存較少，建議使用 4B 或 1B 版本
model_id = "google/gemma-3-4b-it"

# 2. 配置 4-bit 量化 (這對應了您可能指的 '4gemma' / 4-bit)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("正在載入模型...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 3. 載入 IMDB 資料集 (僅取前 5 條作為演示)
print("正在載入 IMDB 資料...")
dataset = load_dataset("imdb", split="test").shuffle(seed=42).select(range(5))

# 4. 定義分析函數
def analyze_sentiment(review_text):
    # 構建 Prompt (提示詞)
    # Gemma 3 對指令遵循能力很強，我們直接要求它輸出 json 或特定格式
    prompt = f"""
    Analyze the sentiment of the following movie review. 
    Classify it as either 'Positive' or 'Negative'.
    Provide the output in the format: Sentiment: [Label]

    Review: "{review_text[:1000]}..." 
    
    Answer:
    """
    
    # 使用聊天範本 (推薦用於 instruct 模型)
    messages = [
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
    
    outputs = model.generate(
        input_ids, 
        max_new_tokens=10, # 我們只需要簡短的回答
        do_sample=False,   # 使用貪婪解碼以獲得確定性結果
        temperature=0.0
    )
    
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# 5. 運行分析
print("-" * 50)
print(f"使用模型: {model_id} 進行分析")
print("-" * 50)

for i, item in enumerate(dataset):
    review = item['text']
    true_label = "Positive" if item['label'] == 1 else "Negative"
    
    # 讓 Gemma 3 進行預測
    prediction = analyze_sentiment(review)
    
    print(f"Review #{i+1}:")
    print(f"原文片段: {review[:100]}...")
    print(f"真實標籤: {true_label}")
    print(f"Gemma預測: {prediction}")
    print("-" * 30)
```

# 版本2:微調版
- 使用 PEFT (LoRA) 和 TRL (SFTTrainer) 對 Gemma 3 進行微調的程式碼。
- 這個方法只需要訓練模型的一小部分參數，極大地減少了顯存需求和訓練時間。
- 🚀 1. 環境與函式庫準備
- 確保已安裝：transformers, torch, accelerate, bitsandbytes, datasets
- pip install peft trl
- 💻 2. Gemma 3 Lora 微調程式碼
```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer # Supervised Fine-Tuning Trainer

# --- 參數設定 ---
# 選擇 Gemma 3 指令微調版本
model_id = "google/gemma-3-4b-it" 
dataset_id = "imdb" 
output_dir = "gemma3_imdb_sentiment_lora" # 訓練結果儲存路徑

# 1. 定義 Prompt 模板 (必須將資料轉為模型能理解的格式)
def format_imdb_data(example):
    # 'text' 是電影評論，'label' 是 0 (負面) 或 1 (正面)
    sentiment = "正面 (Positive)" if example["label"] == 1 else "負面 (Negative)"
    
    # 這是 SFT 的標準格式：用戶輸入 -> 模型輸出
    text = f"分析以下電影評論的情感傾向。請只回答 '正面 (Positive)' 或 '負面 (Negative)'。\n\n"
    text += f"評論: {example['text']}\n\n"
    text += f"情感傾向: {sentiment}"
    return {"text": text}

# 2. 載入資料集
print("正在載入 IMDB 訓練資料集...")
# 我們使用 train split 進行訓練，並限制大小以進行快速演示
dataset = load_dataset(dataset_id, split="train[:5000]").map(format_imdb_data) 


# 3. 配置 4-bit 量化 (Q-LoRA 基礎)
print("正在配置 4-bit 量化...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Normal Float 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16 # 使用 bf16 進行計算，提高性能
)

# 4. 載入模型和 Tokenizer
print(f"正在載入模型：{model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 確保模型的 pad_token 設置正確
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# 5. 配置 Q-LoRA
# 在 k-bit 訓練中，必須先準備模型
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA 配置，這是 PEFT 的核心
lora_config = LoraConfig(
    r=16, # LoRA 的秩 (rank)，值越大性能可能越好，但計算量更大
    lora_alpha=16, # LoRA 縮放參數
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM", # 任務類型：因果語言模型
    # 針對 Gemma 3 的注意力層進行微調
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
)

# 6. 定義訓練參數
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,                     # 訓練的 Epoches 數量
    per_device_train_batch_size=4,          # 每個裝置的批次大小
    gradient_accumulation_steps=4,          # 梯度累積步數 (等效批次大小為 16)
    optim="paged_adamw_8bit",               # 優化器 (paged_adamw_8bit 節省顯存)
    logging_steps=25,
    learning_rate=2e-4,                     # 學習率
    fp16=False,
    bf16=True,                              # 使用 bfloat16
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="none",                       # 可選：設定為 "wandb" 等報告工具
)

# 7. 初始化 SFTTrainer 並開始訓練
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=512, # 限制最大序列長度，防止 OOM
    tokenizer=tokenizer,
    args=training_args,
)

print("\n🚀 開始 Gemma 3 Q-LoRA 微調...")
trainer.train()

# 8. 儲存微調後的 LoRA Adapters
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\n✅ 微調完成！LoRA 權重已儲存至：{output_dir}")
```

#### 使用微調後的 Gemma 3 模型（含 LoRA 權重）進行推論和分析新 IMDB 評論的程式碼。
- 推論的關鍵在於使用 PEFT 的 PeftModel 函式庫來載入訓練好的 LoRA 權重，並將其連接到原始的 Gemma 3 模型上。
- 🚀 推論程式碼示例
- 此程式碼將使用您在上一階段儲存的 LoRA 權重目錄 (gemma3_imdb_sentiment_lora)
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- 參數設定 (與訓練時保持一致) ---
BASE_MODEL_ID = "google/gemma-3-4b-it" 
LORA_WEIGHTS_PATH = "gemma3_imdb_sentiment_lora" # 訓練階段儲存 LoRA 權重的路徑

# 1. 載入模型和 LoRA 權重
def load_fine_tuned_model():
    print("正在載入原始 Gemma 3 模型 (4-bit 量化)...")
    
    # 設置 4-bit 量化配置 (與訓練時相同)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 載入 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    
    # 載入原始 Gemma 3 模型
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # 載入並附加 LoRA 權重
    print(f"正在載入微調權重 (LoRA Adapters) 從: {LORA_WEIGHTS_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH)
    
    # 可選：將 LoRA 權重與基底模型合併，以便於部署（會增加顯存/硬碟使用）
    # print("正在合併 LoRA 權重到基底模型...")
    # model = model.merge_and_unload()
    
    print("✅ 模型載入完成。")
    return model, tokenizer

# 2. 定義推論分析函數
def analyze_review_with_gemma3(model, tokenizer, review_text):
    # **重要：使用與訓練時完全相同的 Prompt 格式**
    prompt = f"分析以下電影評論的情感傾向。請只回答 '正面 (Positive)' 或 '負面 (Negative)'。\n\n"
    prompt += f"評論: {review_text}\n\n"
    prompt += f"情感傾向:" # 模型應該從這裡開始生成
    
    # 使用聊天模板處理輸入
    messages = [
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
    
    # 執行生成
    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            max_new_tokens=20,          # 稍微多給一些 Token 以確保模型完成回答
            do_sample=False,            # 確保結果確定性
            temperature=0.0,            # 溫度設為 0 以獲得最可靠的分類
        )
    
    # 解碼模型的回應
    # 我們只取生成的部分
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip().split('\n')[0] # 只取第一行，確保輸出乾淨

# 3. 執行推論
model, tokenizer = load_fine_tuned_model()

# 測試用新的 IMDB 評論
new_reviews = [
    "This film is an absolute masterpiece! The acting, the script, and the cinematography were all flawless. A must-see.",
    "I was incredibly disappointed. The plot was confusing, the pacing was agonizingly slow, and the main actor seemed completely bored.",
    "It's a decent movie for a rainy afternoon. Nothing groundbreaking, but it kept me mildly entertained. I wouldn't rush to see it again.",
]

print("-" * 50)
print("🎬 開始使用微調後的 Gemma 3 分析新評論:")
print("-" * 50)

for i, review in enumerate(new_reviews):
    print(f"--- 評論 #{i+1} ---")
    print(f"原文: {review[:80]}...")
    
    # 進行情感分析
    prediction = analyze_review_with_gemma3(model, tokenizer, review)
    
    print(f"**Gemma 3 情感分析結果:** {prediction}")
    print("-" * 20)
```
- 關鍵步驟解析
  - 載入基底模型與 LoRA (PeftModel.from_pretrained)
  - 首先，我們使用與訓練時相同的 BitsAndBytesConfig 載入原始的 Gemma 3 基底模型。
  - 接著，使用 PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH) 將您微調的 LoRA 權重檔案（位於 gemma3_imdb_sentiment_lora 目錄內）掛載到基底模型上。這樣，模型在推論時就會同時利用原始知識和微調學到的新知識。
  - 推論 Prompt 格式
  - 確保在 analyze_review_with_gemma3 函數中使用的 Prompt (提示詞) 格式與您在訓練階段 (format_imdb_data 函數) 中使用的格式完全一樣。格式不一致是推論失敗或準確率低下的最常見原因。
  - 將 temperature 設置為 0.0 確保模型輸出最可靠的分類結果，而不是發揮創造力。
  - 透過這個方法，您的 Gemma 3 模型現在已經專門針對 IMDB 評論情感分析任務進行了優化，應該能夠提供比原始模型更準確的結果。
