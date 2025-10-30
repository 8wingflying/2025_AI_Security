# 電腦視覺與CNN實測報告
- 電腦視覺主題
- 電腦基本處理 ==> OPENCV
  - 卷積與濾波
  - 邊緣偵測
- 電腦視覺核心技術1:CNN與影像識別
  - CNN 基本元件: 函式功能說明
    - https://keras.io/api/layers/convolution_layers/ 
  - CNN 模型
- 電腦視覺核心技術2:Transfer Learning
- 電腦視覺核心技術3:Vision Transformer (ViT)
- ....

## 電腦視覺主題
- 任務(Tasks)
  - 影像辨識|圖像分類(Image Classification)
  - 物件偵測(Object detection)
  - `影像`分割(`Image` segmentation)
    - `語意`分割(`Semantic` segmentation)
    - `實例`分割(`Instance` segmentation)
    - `全景`分割(`Panoptic` segmentation)
  - `影像`生成與合成(`Image` Generation & Synthesis）
  - 更進階 ==>`影片`生成與合成(`Video` Generation & Synthesis）

## 電腦基本處理==> OPENCV
  - 卷積與濾波
  - 邊緣偵測
## CNN 基本元件: 函式功能說明
- https://keras.io/api/layers/convolution_layers/ 

##
- CNN 卷積神經網路 Convolutional neural network(2012–2020）
    - 基本元件
      - Convolution運算: padding(填補) | stride(步幅)
      - Pooling運算
      - Dropout
      - 展平層(Flatten Layer)
      - 全連接層(Fully Connected Layer, Dense Layer)
    - ImageNet(2009)與 ILSVRC競賽2010-2017(ImageNet Large-Scale Visual Recognition Challenge) 
### CNN 模型
- LeNet(1989) ==> LeNet-5 (1998)
- AlexNet(2012)
- ZFNet(2013)
- GoogLeNet（Inception 有很多版本）2014
- VGG-16(層) VGG-19(層) 2014
- ResNet（Residual Network）2015
  - 引入「殘差連接」（或跳躍連接）
  - 解決了訓練極深層網路時的梯度消失和模型退化問題
  - 首度構建破百層的神經網路
- DenseNet(Dense Convolutional Network) 2016
- MobileNet(Google, 2017  有很多版本)/EfficientNet(Google, 2019) ==> 聚焦(應用)到edge computing| Mobile Computing
### 進階發展
- Vision Transformer (ViT) 2020(中階主題)
  - 把Transformer 架構應用到 `電腦視覺`
- 邁向 `多模態`大語言模型
    - CLIP（OpenAI,,2021）：CLIP（Contrastive Language-Image Pre-Training）(中階主題)
    - 生成式AI ==> diffusion model(2021/2022)
    - .....
   

   
