# 🤗 Hugging Face 各項功能與常用函數總覽表  
> **中英對照 + 模組分類 + 範例說明**

---

## 一、主要模組（Core Modules Overview）

| 模組名稱 | 中文說明 | 功能重點 |
|------------|------------|------------|
| `transformers` | 模型與推論 | 預訓練模型載入、微調、推論 |
| `datasets` | 資料模組 | 提供 10,000+ NLP / CV / Audio 資料集 |
| `tokenizers` | 分詞模組 | 快速斷詞與子詞編碼（BPE、WordPiece） |
| `evaluate` | 評估模組 | 各類指標（BLEU、ROUGE、F1、Accuracy） |
| `accelerate` | 加速模組 | 分散式訓練與多 GPU 支援 |
| `peft` | 參數高效微調 | LoRA、Prefix Tuning、Adapters |
| `diffusers` | 生成模型（影像） | Stable Diffusion、ControlNet 等 |
| `gradio` | Web 介面 | 快速建立模型 Demo UI |
| `huggingface_hub` | 模型中心 API | 上傳 / 下載模型、資料集、空間 |
| `trl` | 強化學習微調 | 支援 RLHF / PPO / DPO 等 |
| `optimum` | 部署與最佳化 | ONNX / TensorRT / Intel / Apple 優化部署 |

---

## 二、Transformers 模組（`transformers`）

| 函數 / 類別 | 功能說明 | 範例 | 備註 |
|---------------|------------|-------|------|
| `pipeline()` | 快速推論管線 | `pipeline("sentiment-analysis")` | 自動載入模型 |
| `AutoTokenizer.from_pretrained()` | 載入對應 tokenizer | `AutoTokenizer.from_pretrained("bert-base-uncased")` | 自動匹配模型 |
| `AutoModel.from_pretrained()` | 載入模型 | `AutoModel.from_pretrained("bert-base-uncased")` | — |
| `AutoModelForSequenceClassification` | 序列分類模型 | — | 用於情感分析、主題分類 |
| `AutoModelForCausalLM` | 語言生成模型 | GPT 系列 | — |
| `AutoModelForTokenClassification` | 命名實體辨識模型 | — | NER 任務 |
| `AutoModelForQuestionAnswering` | 問答模型 | — | SQuAD 任務 |
| `AutoModelForImageClassification` | 圖像分類模型 | — | ViT / DeiT 模型 |
| `.generate()` | 文字生成 | `model.generate(inputs, max_new_tokens=50)` | 支援 beam search |
| `.from_pretrained()` | 載入 Hugging Face Hub 模型 | 支援本地或遠端 | — |
| `.save_pretrained()` | 儲存模型與 tokenizer | `model.save_pretrained("./model")` | — |
| `.to("cuda")` | GPU 加速 | — | CUDA / MPS / CPU |

---

## 三、Tokenizers 模組（`tokenizers`）

| 函數 / 類別 | 功能說明 | 範例 | 備註 |
|---------------|------------|-------|------|
| `Tokenizer()` | 建立自訂分詞器 | `Tokenizer(models.BPE())` | — |
| `BPE`, `WordPiece`, `Unigram` | 分詞演算法 | — | 可訓練自訂語料 |
| `.train_from_iterator()` | 自建字典 | `tokenizer.train_from_iterator(texts)` | — |
| `.encode()` / `.decode()` | 字串轉 ID / 反解 | — | — |
| `.save()` / `.from_file()` | 儲存與載入分詞器 | — | — |
| `.get_vocab()` | 取得詞彙表 | — | — |

---

## 四、Datasets 模組（`datasets`）

| 函數 / 類別 | 功能說明 | 範例 | 備註 |
|---------------|------------|-------|------|
| `load_dataset()` | 載入資料集 | `load_dataset("imdb")` | 自動下載 |
| `load_from_disk()` | 載入本地資料集 | — | — |
| `.map()` | 映射函數轉換資料 | `dataset.map(tokenize_function)` | 用於分詞 |
| `.filter()` | 篩選樣本 | `dataset.filter(lambda x: len(x["text"])>50)` | — |
| `.shuffle()` | 打亂資料 | — | — |
| `.train_test_split()` | 資料分割 | — | — |
| `.to_pandas()` | 轉成 DataFrame | — | — |
| `.save_to_disk()` | 儲存處理後資料 | — | — |

---

## 五、Evaluate 模組（`evaluate`）

| 函數名稱 | 功能說明 | 範例 | 備註 |
|------------|------------|-------|------|
| `load("accuracy")` | 準確率指標 | `metric = evaluate.load("accuracy")` | — |
| `load("f1")` | F1-score | — | — |
| `load("precision")`, `load("recall")` | 精確率 / 召回率 | — | — |
| `load("bleu")` | BLEU 分數（翻譯） | — | — |
| `load("rouge")` | ROUGE 分數（摘要） | — | — |
| `.compute(predictions, references)` | 計算指標 | — | 輸入預測與真實值 |

---

## 六、PEFT 模組（Parameter-Efficient Fine-Tuning）

| 方法 / 模型 | 功能說明 | 常用參數 | 備註 |
|---------------|------------|-------------|------|
| `LoraConfig()` | LoRA 微調設定 | `r`, `alpha`, `target_modules` | — |
| `get_peft_model()` | 將模型轉為 PEFT 模型 | `get_peft_model(base_model, peft_config)` | — |
| `.save_pretrained()` | 儲存 LoRA 權重 | — | 可合併至主模型 |
| `.merge_and_unload()` | 合併微調權重 | — | — |

---

## 七、Diffusers 模組（影像生成）

| 類別 / 函數 | 功能說明 | 範例 | 備註 |
|---------------|------------|-------|------|
| `StableDiffusionPipeline()` | 載入 Stable Diffusion 模型 | `pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")` | — |
| `.to("cuda")` | GPU 加速 | — | — |
| `.enable_attention_slicing()` | 降低 VRAM 使用 | — | — |
| `.__call__()` | 文生圖 | `pipe("A cat wearing sunglasses")` | — |
| `.save_pretrained()` | 儲存模型 | — | — |

---

## 八、Accelerate 模組（多 GPU / 分散式訓練）

| 函數 / 類別 | 功能說明 | 範例 |
|---------------|------------|-------|
| `Accelerator()` | 建立加速器 | `accelerator = Accelerator()` |
| `.prepare()` | 包裝模型 / 資料 | `model, optimizer, dataloader = accelerator.prepare(...)` |
| `.unwrap_model()` | 拆解包裝模型 | — |
| `.print()` | 分散式打印 | — |

---

## 九、Hugging Face Hub（雲端模型中心）

| 函數 / 類別 | 功能說明 | 範例 |
|---------------|------------|-------|
| `huggingface_hub.login()` | 登入帳號 | `huggingface_hub.login(token="hf_...")` |
| `hf_hub_download()` | 下載模型或資料 | `hf_hub_download(repo_id="bert-base-uncased")` |
| `HfApi()` | 操作 Hub API | `api = HfApi()` |
| `api.upload_file()` | 上傳檔案 | — |
| `api.list_models()` / `api.list_datasets()` | 查詢模型 / 資料 | — |
| `model_card` | 模型說明文件 | Markdown 格式 | — |

---

## 十、Gradio 模組（Web Demo 介面）

| 函數 / 類別 | 功能說明 | 範例 |
|---------------|------------|-------|
| `gr.Interface()` | 建立互動介面 | `gr.Interface(fn=predict, inputs="text", outputs="label")` |
| `.launch()` | 啟動伺服器 | — |
| `Blocks()` | 複合式介面 | `with gr.Blocks() as demo:` | 支援多模組 UI |
| `Textbox`, `Image`, `Audio`, `Dropdown` | 互動元件 | — |
| `.queue()` | 非同步執行 | — |

---

## 十一、Transformers - 推論與微調範例（NLP）

| 任務 | 使用模型 | 範例程式 |
|-------|------------|-------------|
| **情感分析** | `distilbert-base-uncased-finetuned-sst-2-english` | `pipeline("sentiment-analysis")("I love AI!")` |
| **翻譯** | `Helsinki-NLP/opus-mt-en-fr` | `pipeline("translation_en_to_fr")("Hello world")` |
| **摘要** | `facebook/bart-large-cnn` | `pipeline("summarization")(text)` |
| **問答系統** | `deepset/roberta-base-squad2` | `pipeline("question-answering")(...)` |
| **文本生成** | `gpt2`, `tiiuae/falcon-7b-instruct` | `pipeline("text-generation")("Once upon a time...")` |
| **零樣本分類** | `facebook/bart-large-mnli` | `pipeline("zero-shot-classification")("This is amazing", ["positive", "negative"])` |

---

## 十二、範例：簡易生成模型 API

```python
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
generator = pipeline("text-generation", model="gpt2")

@app.get("/generate")
def generate(prompt: str):
    result = generator(prompt, max_new_tokens=50)
    return {"result": result[0]["generated_text"]}
