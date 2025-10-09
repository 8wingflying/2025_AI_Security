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

## 如何使用這些模型(Transfer Learning)
