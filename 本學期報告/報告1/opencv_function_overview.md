# 🧠 OpenCV 功能總覽教學文件
**版本：2025年版｜語言：Python｜編碼：UTF-8**

---

## 📘 一、OpenCV 簡介
OpenCV（Open Source Computer Vision Library）是一個跨平台的電腦視覺與機器學習開源函式庫，  
支援 C++、Python、Java 等語言，能處理影像、影片、機器學習、物件偵測與 3D 建模。

---

## 🧩 二、影像處理（Image Processing）

### 🔹 基本操作
```python
import cv2

img = cv2.imread('example.jpg')
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 🔹 顏色轉換
- `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`
- `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)`

### 🔹 幾何變換
| 功能 | 函式 |
|------|------|
| 縮放 | `cv2.resize()` |
| 旋轉 | `cv2.rotate()` |
| 平移/仿射 | `cv2.warpAffine()` |
| 透視變換 | `cv2.warpPerspective()` |

### 🔹 濾波與邊緣
| 類型 | 範例 |
|------|------|
| 模糊化 | `cv2.GaussianBlur(img, (5,5), 0)` |
| 邊緣偵測 | `cv2.Canny(img, 100, 200)` |

### 🔹 直方圖處理
- 計算：`cv2.calcHist([img],[0],None,[256],[0,256])`
- 均衡化：`cv2.equalizeHist(gray_img)`

---

## 🧬 三、特徵偵測（Feature Detection & Description）

### 🔹 常見演算法
| 功能 | 函式 |
|------|------|
| 角點偵測 | Harris、Shi-Tomasi |
| 特徵點偵測 | SIFT、SURF、ORB |
| 特徵匹配 | `cv2.BFMatcher`、`cv2.FlannBasedMatcher` |

### 🔹 範例：ORB 特徵匹配
```python
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(des1, des2)
result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
```

---

## 👱‍♂️ 四、人臉與姿態辨識（Face & Pose Recognition）

### 🔹 臉部偵測
```python
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
```

### 🔹 臉部辨識
- LBPH（Local Binary Pattern Histogram）
- Eigenfaces、Fisherfaces
- 模型建立：
  ```python
  recognizer = cv2.face.LBPHFaceRecognizer_create()
  recognizer.train(images, labels)
  recognizer.save('model.yml')
  ```

---

## 🎥 五、影片處理與即時影像（Video Processing）

### 🔹 影片讀取與顯示
```python
cap = cv2.VideoCapture('video.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

### 🔹 背景分離
- `cv2.createBackgroundSubtractorMOG2()`
- `cv2.createBackgroundSubtractorKNN()`

### 🔹 光流分析
- `cv2.calcOpticalFlowFarneback()`
- `cv2.calcOpticalFlowPyrLK()`

---

## 🧮 六、機器學習模組（cv2.ml）

| 類型 | 範例函式 |
|------|-----------|
| KNN | `cv2.ml.KNearest_create()` |
| SVM | `cv2.ml.SVM_create()` |
| 決策樹 | `cv2.ml.DTrees_create()` |
| 隨機森林 | `cv2.ml.RTrees_create()` |
| Boosting | `cv2.ml.Boost_create()` |
| PCA | `cv2.PCACompute()` |

---

## 🔍 七、相機校正與立體視覺（Camera Calibration / 3D）

| 功能 | 函式 |
|------|------|
| 校正 | `cv2.calibrateCamera()` |
| 去異變 | `cv2.undistort()` |
| 雙目視覺 | `cv2.StereoBM_create()` |
| 深度圖生成 | `cv2.StereoSGBM_create()` |
| ArUco 標記 | `cv2.aruco.detectMarkers()`、`cv2.aruco.estimatePoseSingleMarkers()` |

---

## 🧪 八、進階模組與擴充套件

| 模組 | 功能 |
|------|------|
| `cv2.xfeatures2d` | 高階特徵（SIFT, SURF, FREAK） |
| `cv2.aruco` | AR 標記與姿態追蹤 |
| `cv2.bgsegm` | 背景建模加強版 |
| `cv2.text` | OCR 文字偵測與辨識 |
| `cv2.ximgproc` | 超像素與邊緣保留濾波 |
| `cv2.face` | 臉部辨識 API |

---

## 📊 九、常見應用場景
| 應用 | 涉及技術 |
|------|-----------|
| 車牌辨識 | 邊緣偵測 + OCR |
| 手勢追蹤 | 膚色範圍 + 輪廓分析 |
| 虛擬實境 (AR) | ArUco 標記 + 姿態估計 |
| 安全監控 | 背景分離 + 物件追蹤 |
| 影像拼接 | 特徵匹配 + Homography |
| 深度學習辨識 | DNN 模組（YOLO、SSD、Mask R-CNN） |

---

## 🧩 十、延伸學習建議
1. **學習 OpenCV DNN 模組**：匯入 YOLO / ONNX 模型  
2. **結合 MediaPipe / Dlib**：進行人臉關鍵點偵測  
3. **與 TensorFlow / PyTorch 整合**：進行 AI 強化辨識  
4. **使用 CUDA 加速**：`cv2.cuda` 模組支援 GPU 加速  

---

## 📚 參考資源
- [OpenCV 官方文件](https://docs.opencv.org/)
- [OpenCV GitHub 原始碼](https://github.com/opencv/opencv)
- [LearnOpenCV 教學網](https://learnopencv.com/)
- [PyImageSearch 部落格](https://www.pyimagesearch.com/)

---

**製作：ChatGPT（GPT-5）｜更新日期：2025-10-29**

