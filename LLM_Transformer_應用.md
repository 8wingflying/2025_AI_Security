### 範例
- [Hands-On Generative AI with Transformers and Diffusion Models](https://learning.oreilly.com/library/view/hands-on-generative-ai/9781098149239/)
- [GITHUB](https://github.com/genaibook/genaibook/tree/main)
- 安裝套件 == > !pip install genaibook

# 載入模組
```python
import diffusers
import huggingface_hub
import transformers

diffusers.logging.set_verbosity_error()
huggingface_hub.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()
```
## 檢視設定
```python
from genaibook.core import get_device

device = get_device()
print(f"Using device: {device}")
```
## 產生圖像
- 使用[huggingface/diffusers](https://github.com/huggingface/diffusers)模組
- Diffusers is the go-to library for state-of-the-art pretrained diffusion models for generating images, audio, and even 3D structures of molecules. 
```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
).to(device)

prompt = "a photograph of an astronaut riding a horse"
pipe(prompt).images[0]
```
## 產生文本分類(text Classification)
```python
import torch
torch.manual_seed(0)

from transformers import pipeline

classifier = pipeline("text-classification", device=device)
classifier("This movie is disgustingly good !")
```

## 產生文本生成(text Generation)
```python
from transformers import set_seed

# Setting the seed ensures we get the same results every time we run this code
set_seed(10)

generator = pipeline("text-generation")
prompt = "It was a dark and stormy"
generator(prompt)[0]["generated_text"]
```

## 產生聲音
```python
pipe = pipeline("text-to-audio", model="facebook/musicgen-small", device=device)
data = pipe("electric rock solo, very intense")
print(data)

import IPython.display as ipd

display(ipd.Audio(data["audio"][0], rate=data["sampling_rate"]))
```


### [Hands-On Generative AI with Transformers and Diffusion Models](https://learning.oreilly.com/library/view/hands-on-generative-ai/9781098149239/)
- [GITHUB](https://github.com/genaibook/genaibook/tree/main)
- 安裝套件 == > !pip install genaibook

### I. Leveraging Open Models

1. An Introduction to Generative Media

2. Transformers

3. Compressing and Representing Information

4. Diffusion Models

5. Stable Diffusion and Conditional Generation

### II. Transfer Learning for Generative Models

6. Fine-Tuning Language Models

7. Fine-Tuning Stable Diffusion

### III. Going Further

8. Creative Applications of Text-to-Image Models

9. Generating Audio

10. Rapidly Advancing Areas in Generative AI

A. Open Source Tools

B. LLM Memory Requirements

C. End-to-End Retrieval-Augmented Generation
