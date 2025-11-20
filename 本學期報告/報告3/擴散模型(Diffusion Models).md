## 擴散模型(Diffusion Models)

## 參考資料
- [Generative Deep Learning, 2nd Edition](https://learning.oreilly.com/library/view/generative-deep-learning/9781098134174/)
- https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition
- chapter 8　擴散模型(Diffusion Models)

## 論文
- 66個擴散模型Diffusion Models經典論文
- 去噪擴散概率模型 |DDPM(2020）
  - [Denoising Diffusion Probabilistic Models ](https://arxiv.org/abs/2006.11239)
- 去噪擴散隱式模型 |DDIM(20200）
  - [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) 

## REVIEW
- 2022 [Diffusion Models: A Comprehensive Survey of Methods and Applications](https://arxiv.org/abs/2209.00796)
- 2024 [Diffusion Models and Representation Learning: A Survey(https://arxiv.org/abs/2407.00783)
- 2024 [Generative diffusion models: A survey of current theoretical developments](https://www.sciencedirect.com/science/article/abs/pii/S0925231224011445)

## 導讀
- https://ycc.idv.tw/diffusion-model.html
- https://ithelp.ithome.com.tw/articles/10328909
- https://ithelp.ithome.com.tw/articles/10330174
- https://ithelp.ithome.com.tw/articles/10329130
- https://ithelp.ithome.com.tw/articles/10331028
## 書籍
- [Generative AI - Diffusion Model 擴散模型現場實作精解|楊靈、張至隆、張文濤、崔斌](https://www.tenlong.com.tw/products/9786267383414?list_name=trs-t)
- [擴散模型從原理到實戰|李忻瑋 蘇步升 徐浩然 餘海銘](https://www.tenlong.com.tw/products/9787115618870?list_name=srh)
- [Hands-On Generative AI with Transformers and Diffusion Models](https://learning.oreilly.com/library/view/hands-on-generative-ai/9781098149239/)
- [Using Stable Diffusion with Python](https://learning.oreilly.com/library/view/using-stable-diffusion/9781835086377/)
## Diffusion Model 的數學原理

## 範例程式
  - [Using Stable Diffusion with Python](https://learning.oreilly.com/library/view/using-stable-diffusion/9781835086377/)
- Ch 1.Generate image from text
```python
# install diffusers
%pip install -U diffusers


import torch
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "stablediffusionapi/deliberate-v2"
    , torch_dtype = torch.float16
).to("cuda:0")

prompt = "a photograph of an astronaut riding a horse"
image = pipe(
    prompt          = prompt
    , generator     = torch.Generator("cuda:0").manual_seed(6)
    , width         = 768
    , height        = 512
).images[0]

display(image)

image
```
- Detect the backgroud
```python
from transformers import CLIPSegProcessor,CLIPSegForImageSegmentation

processor = CLIPSegProcessor.from_pretrained(
    "CIDAS/clipseg-rd64-refined"
)
model = CLIPSegForImageSegmentation.from_pretrained(
    "CIDAS/clipseg-rd64-refined"
)

# generate mask data
import matplotlib.pyplot as plt
prompts = ['the background']
inputs = processor(
    text             = prompts
    , images         = [image] * len(prompts)
    , padding        = True
    , return_tensors = "pt"
)

with torch.no_grad():
    outputs = model(**inputs)

preds = outputs.logits

mask_data = torch.sigmoid(preds)[0]
print(mask_data.shape)
plt.imshow(mask_data)

# genearte mask binary image
import cv2
from PIL import Image
mask_file_name = f"bg_mask.png"
plt.imsave(mask_file_name,mask_data) 
mask_data_cv = cv2.imread(mask_file_name) # -> (352, 352, 3)

def get_mask_img(mask_data):
    gray_image = cv2.cvtColor(mask_data,cv2.COLOR_BGR2GRAY)
    thresh, bw_image = cv2.threshold(gray_image,100,255,cv2.THRESH_BINARY)
    cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(bw_image)

bw_image = get_mask_img(mask_data=mask_data_cv)
#cv2.imwrite(bw_image)
bw_image = bw_image.resize((768,512))
bw_image

# Start a inpaint pipeline
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler
inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4"
    , torch_dtype = torch.float16
    , safety_checker = None
).to("cuda:0")
#inpaint_pipe.scheduler = EulerDiscreteScheduler.from_config(inpaint_pipe.scheduler.config)

# change the background
sd_prompt = "blue sky and mountains"
out_image = inpaint_pipe(
    prompt          = sd_prompt
    , image         = image
    , mask_image    = bw_image
    , strength      = 0.9
    , generator     = torch.Generator("cuda:0").manual_seed(7)
    # , guidance_scale = 7.5
    # , num_inference_steps = 50
    # , width = 768
    # , height = 512
).images[0]
out_image

```
## 延伸閱讀
- [擴散模型從原理到實戰|李忻瑋 蘇步升 徐浩然 餘海銘](https://www.tenlong.com.tw/products/9787115618870?list_name=srh)
- [擴散模型：生成式 AI 模型的理論、應用與代碼實踐|楊靈 ](https://www.tenlong.com.tw/products/9787121459856?list_name=srh)
  - [Generative AI - Diffusion Model 擴散模型現場實作精解|楊靈、張至隆、張文濤、崔斌 編著](https://www.tenlong.com.tw/products/9786267383414?list_name=trs-t)
  - [Hands-On Generative AI with Transformers and Diffusion Models](https://learning.oreilly.com/library/view/hands-on-generative-ai/9781098149239/)
    - https://github.com/genaibook/genaibook
  - [Using Stable Diffusion with Python](https://learning.oreilly.com/library/view/using-stable-diffusion/9781835086377/)
    - https://github.com/PacktPublishing/Using-Stable-Diffusion-with-Python 
  - [Diffusions in Architecture: Artificial Intelligence and Image Generators](https://learning.oreilly.com/library/view/diffusions-in-architecture/9781394191772/)
  - [Applied Generative AI for Beginners: Practical Knowledge on Diffusion Models, ChatGPT, and Other LLMs](https://learning.oreilly.com/library/view/applied-generative-ai/9781484299944/)
