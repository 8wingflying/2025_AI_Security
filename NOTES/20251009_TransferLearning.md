# 重點
- 檢視重要模型
  - Google InceptionV3
  - VGG16  VGG19
  - Resnet
  - Densenet 
- 如何使用這些模型(Transfer Learning)

## 檢視重要模型1;Google INception
```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
model = tf.keras.applications.InceptionV3(include_top=True, weights='imagenet')
model.summary()
```

## 如何使用這些模型(Transfer Learning) 1

```python
# ---- 建立並訓練密集層分類器 ---- #

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(include_top=False,
              weights='imagenet',
              input_shape=(150,150,3))

model = Sequential()
model.add(vgg16)    # 將 vgg16 做為一層
model.add(Flatten())
model.add(Dense(512, activation='relu', input_dim=4 * 4 * 512))
model.add(Dropout(0.5))  # 丟棄法
model.add(Dense(1, activation='sigmoid'))

vgg16.trainable = False     # 凍結權重
model.summary()

model.compile(optimizer=RMSprop(lr=2e-5),   # 學習速率降低一點
              loss='binary_crossentropy',
              metrics=['acc'])

## 使用資料擴增法生成訓練資料
## 用ImageDataGenerator 來讀取資料，並使用資料擴增法進行100個週期的訓練

from tensorflow.keras.preprocessing.image import ImageDataGenerator

gobj = ImageDataGenerator(rescale=1./255, validation_split=0.75,
    rotation_range=40,      #←隨機旋轉 -40~40 度
    width_shift_range=0.2,  #←隨機向左或右平移 20% 寬度以內的像素
    height_shift_range=0.2, #←隨機向上或下平移 20% 高度以內的像素
    shear_range=10,         #←隨機順時針傾斜影像 0~10 度
    zoom_range=0.2,         #←隨機水平或垂直縮放影像 20% (80%~120%)
    horizontal_flip=True)   #←隨機水平翻轉影像

trn_gen = gobj.flow_from_directory( #←建立生成訓練資料的走訪器
    'cat_dog/train',         #←指定目標資料夾
    target_size=(150, 150),  #←調整所有影像大小成 150x150
    batch_size=50,        #←每批次要生成多少筆資料
    class_mode='binary',     #←指定分類方式, 這裡是設為二元分類
    subset='training')       #←只生成前 75% 的訓練資料

gobj = ImageDataGenerator(rescale=1./255)

val_gen = gobj.flow_from_directory( #←建立生成驗證資料的走訪器
    'cat_dog/test',          #←指定要讀取測試資料夾
    target_size=(150, 150),
    batch_size=50,
    class_mode='binary')

history = model.fit(trn_gen,        #←指定訓練用的走訪器
                    epochs=30, verbose=2,
                    validation_data=val_gen)    #←指定驗證用的走訪器
```
```python
import matplotlib.pyplot as plt

# 繪製線圖 (可將訓練時所傳回的損失值或準確率等歷史記錄繪製成線圖)
# history: 內含一或多筆要繪資料的字典, 例如：{'loss': [4,2,1,…], 'acc': [2,3,5,…]}
# keys: 以 tuple 或串列指定 history 中要繪製的 key 值, 例如：('loss', 'acc')
# title: 以字串指定圖表的標題文字
# xyLabel: 以 tuple 或串列指定 x, y 軸的說明文字, 例如：('epoch', 'Accuracy')
# ylim: 以 tuple 或串列指定 y 軸的最小值及最大值, 例如 (1, 3), 超出範圍的值會被忽略
# size: 以 tuple 指定圖的尺寸, 預設為 (6, 4) (即寬 6 高 4 英吋)
def plot(history_dict, keys, title=None, xyLabel=[], ylim=(), size=()):
    lineType = ('-', '--', '.', ':')    # 線條的樣式, 畫多條線時會依序採用
    if len(ylim)==2: plt.ylim(*ylim)    # 設定 y 軸最小值及最大值
    if len(size)==2: plt.gcf().set_size_inches(*size)  # size預設為 (6,4)
    epochs = range(1, len(history_dict[keys[0]])+1)  # 計算有幾週期的資料
    for i in range(len(keys)):   # 走訪每一個 key (例如 'loss' 或 'acc' 等)
        plt.plot(epochs, history_dict[keys[i]], lineType[i])  # 畫出線條
    if title:   # 是否顯示標題欄
        plt.title(title)
    if len(xyLabel)==2:  # 是否顯示 x, y 軸的說明文字
        plt.xlabel(xyLabel[0])
        plt.ylabel(xyLabel[1])
    plt.legend(keys, loc='best') # 顯示圖例 (會以 key 為每條線的說明)
    plt.show()  # 顯示出畫好的圖
```
```python
plot( history.history,   # 繪製準確率與驗證準確度的歷史線圖
        ('acc', 'val_acc'),
        'Training & Vaildation Acc',
        ('Epoch','Acc'), 
        )     

plot( history.history,   #  繪製損失及驗證損失的歷史線圖
        ('loss', 'val_loss'),
        'Training & Vaildation Loss',
        ('Epoch','Loss'), 
        )
```
```python
## 使用Google Drive存要分析的檔案
from google.colab import drive
drive.mount('/gdrive')
%cd /gdrive

import os
os.chdir('/content/drive/My Drive/TF2020')
#os.chdir('..')
```
## 
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(include_top=False,    
              weights='imagenet',
              input_shape=(150,150,3))

unfreeze = ['block5_conv1', 'block5_conv2', 'block5_conv3'] # 最後 3 層的名稱

for layer in vgg16.layers:
    if layer.name in unfreeze:
        layer.trainable = True  # 最後 3 層解凍
    else:
        layer.trainable = False # 其他凍結權重

vgg16.summary()

model = Sequential()
model.add(vgg16)    
model.add(Flatten())
model.add(Dense(512, activation='relu', input_dim=4 * 4 * 512))
model.add(Dropout(0.5))  # 丟棄法
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(lr=1e-5),   # 學習速率從 2e-5 -> 1e-5
              loss='binary_crossentropy',
              metrics=['acc'])

def to_EMA(points, a=0.3):  #←將歷史資料中的數值轉為 EMA 值
  ret = []          # 儲存轉換結果的串列
  EMA = points[0]   # 第 0 個 EMA 值
  for pt in points:
    EMA = pt*a + EMA*(1-a)  # 本期EMA = 本期值*0.3 + 前期EMA * 0.7
    ret.append(EMA)         # 將本期EMA加入串列中
  return ret

hv = to_EMA(history.history['val_acc'])  # 將 val_acc 歷史資料的值轉成 EMA 值

history.history['ema_acc'] = hv


plot(history.history, ('acc','val_acc', 'ema_acc'),    # 繪製準確度歷史線圖
        'Training & Validation accuracy', ('Epochs', 'Accuracy'))


```
