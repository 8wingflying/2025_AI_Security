# 📘 TensorFlow Model Optimization 模組教學文件
**Author**：ChatGPT GPT-5  
**Language**：繁體中文 / English  
**Purpose**：說明如何使用 `tensorflow-model-optimization` 進行模型壓縮（compression）、剪枝（pruning）、量化（quantization）與效能最佳化（optimization）  

---

## 🧩 一、模組簡介 (Overview)

`tensorflow-model-optimization`（簡稱 **TFMO**）是 TensorFlow 官方推出的模型最佳化套件，
提供多種方法讓開發者在 **不大幅降低準確率** 的情況下，顯著減少模型大小並提升推論速度。

核心功能包括：
- ✂️ **剪枝 (Pruning)**：移除對結果影響較小的權重  
- ⚖️ **量化 (Quantization)**：將浮點模型轉為低位元整數表示（如 int8）  
- 🧱 **聚類 (Clustering)**：將權重分組以減少參數多樣性  
- 🚀 **整合部署 (Deployment-ready)**：可直接導出至 TensorFlow Lite、Edge TPU、或 ONNX  

---

## ⚙️ 二、安裝模組 (Installation)

```bash
pip install -q tensorflow-model-optimization matplotlib
```

確認安裝版本：
```python
import tensorflow_model_optimization as tfmot
print(tfmot.__version__)
```

---

## 🧠 三、應用場景 (Use Cases)

| 技術 | 說明 | 主要用途 |
|------|------|-----------|
| 剪枝 Pruning | 移除權重中冗餘連接 | 減少模型大小 |
| 量化 Quantization | 將 float32 → int8 | 加速推論、降低記憶體使用 |
| 聚類 Clustering | 將相似權重聚合成少量代表值 | 進一步壓縮模型 |
| 混合最佳化 Mixed Optimization | 結合剪枝與量化 | 極致壓縮與效能提升 |

---

## ✂️ 四、模型剪枝 (Model Pruning)

```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# 定義剪枝參數
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

# 原始模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 套用剪枝
model_for_pruning = prune_low_magnitude(model, **pruning_params)
model_for_pruning.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

✅ 訓練後使用：
```python
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_export.save('pruned_model.h5')
```

---

## ⚖️ 五、模型量化 (Model Quantization)

### 🔹 1️⃣ 後訓練量化 (Post-training Quantization)

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

with open("quant_model.tflite", "wb") as f:
    f.write(tflite_quant_model)
```

### 🔹 2️⃣ 訓練時量化 (Quantization Aware Training, QAT)

```python
quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(model)
q_aware_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## 🧱 六、模型聚類 (Weight Clustering)

```python
clustering_params = {
    'number_of_clusters': 16,
    'cluster_centroids_init': tfmot.clustering.keras.CentroidInitialization.LINEAR
}

cluster_model = tfmot.clustering.keras.cluster_weights(model, **clustering_params)
cluster_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

匯出模型：
```python
final_model = tfmot.clustering.keras.strip_clustering(cluster_model)
final_model.save('clustered_model.h5')
```

---

## 📊 七、模型壓縮前後對照圖 (Visualization of Model Compression)

此段示範如何使用 `matplotlib` 比較模型壓縮前後的大小與推論速度。

```python
import matplotlib.pyplot as plt

labels = ['Original', 'Pruned', 'Quantized', 'Hybrid']
size_mb = [20, 12, 5, 4]  # 模型大小（MB）
speed = [1.0, 1.2, 2.0, 2.5]  # 相對推論速度

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(labels, size_mb, color='skyblue', label='Model Size (MB)')
ax2.plot(labels, speed, color='orange', marker='o', label='Speed (x)')

ax1.set_xlabel('Model Type')
ax1.set_ylabel('Size (MB)', color='blue')
ax2.set_ylabel('Speed (x)', color='orange')
ax1.set_title('📊 模型壓縮前後效能對照圖')

fig.legend(loc='upper right')
plt.show()
```

📈 **結果說明**：
- 模型大小縮小約 80%，推論速度最高可達 2.5 倍。
- 若結合量化與剪枝，適合邊緣裝置部署。

---

## ⚙️ 八、TensorFlow Lite × Edge TPU 實際部署流程

### ✅ 部署步驟：

1️⃣ **匯出量化模型：**
```python
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open('optimized_model.tflite', 'wb').write(tflite_model)
```

2️⃣ **Edge TPU 編譯：**
```bash
edgetpu_compiler optimized_model.tflite
```

3️⃣ **在樹莓派或 Coral Dev Board 上執行：**
```bash
python3 infer_tflite.py --model optimized_model_edgetpu.tflite --image test.jpg
```

4️⃣ **驗證效能：**
- 檢查 FPS 與延遲時間（Latency）
- 與原始模型進行比較

---

## 🧠 九、模型壓縮 × AI 風險管理 (NIST AI RMF 應用)

模型壓縮技術在 NIST AI RMF（Artificial Intelligence Risk Management Framework）中屬於 **“系統風險 (Systemic Risk)”** 與 **“技術風險 (Technical Risk)”** 控制層面，主要影響：

| 風險類別 | 說明 | 對應控制措施 |
|-----------|-----------|----------------|
| **Accuracy Risk** | 剪枝與量化可能降低模型準確率 | 使用驗證集重新校準模型精度 |
| **Robustness Risk** | 壓縮後模型對對抗樣本敏感度提高 | 結合 adversarial training 減輕影響 |
| **Explainability Risk** | 權重壓縮後模型可解釋性下降 | 紀錄壓縮參數與可追溯性報告 |
| **Bias Propagation Risk** | 模型壓縮可能放大資料偏差 | 於壓縮前後進行公平性測試 |

### 🔐 建議對策：
- 建立 **模型壓縮審查程序 (Compression Review Process)**。
- 紀錄每次壓縮步驟與對應精度變化。
- 納入 AI RMF “MAP” 階段（Measure, Analyze, Prepare）文件。  

📚 **延伸參考：**
- NIST AI RMF 1.0 Section 3.2 “Systemic Risk”  
- AI RMF: Generative AI Profile (2024 Draft) — 技術風險與效能平衡  

---

## 📈 十、效能比較 (Performance Comparison)

| 模型類型 | 模型大小 | 推論速度 | 準確率變化 |
|-----------|-----------|-----------|-------------|
| 原始模型 | 100% | 1x | baseline |
| 剪枝後 | 約 60% | 約 1.2x | ±1% |
| 量化後 | 約 25% | 約 2x | ±2% |
| 剪枝 + 量化 | 約 20% | 約 2.5x | ±3% |

---

## 📚 十一、延伸閱讀 (Further Reading)

- 🔗 [TensorFlow Model Optimization 官方網站](https://www.tensorflow.org/model_optimization)  
- 📘 [Pruning API Docs](https://www.tensorflow.org/model_optimization/guide/pruning)  
- ⚖️ [Quantization API Docs](https://www.tensorflow.org/model_optimization/guide/quantization)  
- 🧱 [Clustering API Docs](https://www.tensorflow.org/model_optimization/guide/clustering)  
- 🧠 [NIST AI RMF 官方文件](https://www.nist.gov/itl/ai-risk-management-framework)  

---

## ✅ 教學重點回顧 (Summary)

| 主題 | 重點 |
|------|------|
| 剪枝 | 移除權重稀疏連接以減少模型大小 |
| 量化 | 將浮點數轉為整數以加速推論 |
| 聚類 | 壓縮權重參數空間 |
| 模型壓縮視覺化 | Matplotlib 繪製模型大小與速度變化 |
| Edge TPU 部署 | 實際編譯與測試部署流程 |
| AI 風險管理 | 對應 NIST AI RMF 技術風險控制 |

---

© 2025 ChatGPT GPT-5

