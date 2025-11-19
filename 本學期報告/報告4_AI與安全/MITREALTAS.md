## MITRE ALTAS

## MITRE ALTAS
## 1. 對照總覽表（Summary Mapping）

| OWASP 編號 | 攻擊名稱（EN） | 攻擊名稱（繁中） | 典型 MITRE ATLAS 技術\* | NIST AI RMF 主要 Function | ISO/IEC 42001 相關主題 |
|------------|----------------|-------------------|--------------------------|---------------------------|------------------------|
| **ML01:2023** | Input Manipulation Attack | 輸入操控攻擊 | Adversarial Input / Evasion (對抗樣本規避偵測) | **MAP, MEASURE, MANAGE**：情境盤點、健全性測試、風險處置 | 風險管理、模型健全性、監控與修正（Clauses 6, 8, 9, 10） |
| **ML02:2023** | Data Poisoning Attack | 資料投毒攻擊 | Training Data Poisoning / Prediction Poisoning | **MAP, MEASURE, MANAGE**：資料治理、資料品質與漂移監控、供應鏈控管 | 資料治理、AI 風險管理、供應鏈管理（Clauses 6, 8；附錄控制：資料與供應商管理） |
| **ML03:2023** | Model Inversion Attack | 模型反演攻擊 | Model Inversion & Training Data Extraction | **GOVERN, MAP, MEASURE**：隱私與機敏度盤點、隱私強化設計與測試 | 隱私與資訊安全、影響評估、法規遵循（Clauses 4, 5, 6, 8） |
| **ML04:2023** | Membership Inference Attack | 成員推論攻擊 | Training Data Extraction / Membership Inference 類型技術 | **GOVERN, MAP, MEASURE**：資料最小化、隱私風險量化與測試 | 資料主體保護、隱私影響評估、模型輸出管控（Clauses 6, 8, 9） |
| **ML05:2023** | Model Theft | 模型竊取攻擊 | Model Extraction / Model Stealing | **GOVERN, MANAGE**：智慧財產與存取控制、API 保護與紀錄 | 智慧財產保護、存取控制、外部接口管理（Clauses 5, 6, 8） |
| **ML06:2023** | AI Supply Chain Attacks | AI 供應鏈攻擊 | Compromised Components / Data / Pretrained Models | **GOVERN, MAP, MANAGE**：供應鏈治理、第三方風險管理 | AI 供應鏈治理、外部供應商控管、採購與委外（Clauses 4, 5, 6, 8） |
| **ML07:2023** | Transfer Learning Attack | 遷移學習攻擊 | Poisoned / Backdoored Pretrained Models | **MAP, MEASURE, MANAGE**：前訓練模型信任度評估與測試 | 模型生命週期控管、重用與遷移評估（Clauses 6, 8, 9） |
| **ML08:2023** | Model Skewing | 模型偏移攻擊 | Online Prediction Poisoning / Feedback Loop Manipulation | **MAP, MEASURE, MANAGE**：線上行為監控、偏差與漂移偵測 | 模型監控、效能與偏差管理、持續改進（Clauses 8, 9, 10） |
| **ML09:2023** | Output Integrity Attack | 輸出完整性攻擊 | Output Tampering / Post-Processing Manipulation | **MEASURE, MANAGE**：端點安全、結果驗證與多重校驗 | 介面與輸出治理、驗證與監測控制（Clauses 8, 9） |
| **ML10:2023** | Model Poisoning | 模型中毒攻擊 | Training Pipeline / Parameter Poisoning | **MAP, MEASURE, MANAGE**：訓練流程安全、防篡改、完整性監控 | 模型與管線安全、變更管理、紀錄與追蹤（Clauses 6, 8, 9, 10） |

\* 註：MITRE ATLAS 官方網站將攻擊技術以 **技術（Technique）** 與 **戰術（Tactic）** 歸類，例如 *AI/ML Training Data and Prediction Poisoning*、*Model Extraction*、*Model Poisoning* 等，本表為概念性對應，方便做風險盤點與課程教學使用。

---

## 2. 各類攻擊 × 防禦措施 × 框架對照

下面針對每個 OWASP ML Top 10 風險，整理「威脅說明 → 典型場景 → 防禦措施 → NIST AI RMF / ISO 42001 對照」。

### 2.1 ML01:2023 Input Manipulation Attack（輸入操控 / 對抗樣本）

**威脅說明：**  
攻擊者透過微小但刻意設計的輸入修改（對抗樣本、prompt 攻擊、特徵污染）誘導模型產生錯誤或偏斜輸出。

**典型場景：**

- 影像分類被對抗樣本誤導（路牌辨識、醫療影像等）
- LLM prompt injection 使系統忽略原本防護策略
- 權限判斷 / 偵測模型遭刻意繞過

**防禦措施（技術）：**

- 對抗訓練與資料增強（adversarial training）
- 輸入驗證與正規化（input validation / normalization）
- 信任邊界設計：對來自不可信來源的輸入套用額外檢查 / sandbox
- Ensembling / 多模型一致性檢查
- 針對 LLM：prompt filtering / instruction grounding / content policy 檢查

**治理與流程（NIST AI RMF）：**

- **MAP**：釐清高風險使用情境、辨識輸入管道與風險來源  
- **MEASURE**：導入對抗穩健性測試、紅隊測試、攻擊模擬  
- **MANAGE**：針對高風險場景設定風險容忍度、修補流程與 incident playbook  

**ISO/IEC 42001 對應：**

- Clause 6：AI 風險與機會之識別與處理  
- Clause 8：運作管制（包含資料、模型與介面設計）  
- Clause 9：效能評估、測試與監控（包含對抗穩健性測試）

---

### 2.2 ML02:2023 Data Poisoning Attack（資料投毒）

**威脅說明：**  
在訓練 / 驗證 / 測試資料集中植入惡意樣本，使模型在部署後被觸發特定錯誤行為。

**典型場景：**

- 開放資料集被人為「污染」
- 使用者回饋機制被濫用（惡意標註、假評分）
- MLOps 管線中，測試資料被替換

**防禦措施：**

- 資料來源驗證與供應鏈審查（data provenance, lineage）
- 資料品質與離群偵測（anomaly detection on datasets）
- 訓練前後一致性檢查（train-test consistency）
- 使用差分隱私或對抗檢查模型 (e.g., student-teacher 檢查)

**NIST AI RMF：**

- **MAP**：盤點資料來源、標註流程與供應商  
- **MEASURE**：資料品質、異常值檢測、dataset shift 監控  
- **MANAGE**：供應鏈與資料治理  policy、第三方審查與稽核  

**ISO/IEC 42001：**

- 風險管理（Clause 6）  
- 供應鏈與外部供應者控管（Clause 8 + 附錄控制）  
- 持續改進與矯正措施（Clause 10）

---

### 2.3 ML03:2023 Model Inversion Attack（模型反演）

**威脅說明：**  
攻擊者透過模型輸出反推出訓練資料的敏感資訊（例如某位病患的影像或特徵）。

**防禦措施：**

- 採用差分隱私訓練與輸出限制（DP-SGD、noising）
- 限制模型對外提供的信心分數 / logits
- 模型壓縮 / 蒸餾（降低過度記憶訓練資料的風險）
- 針對隱私敏感場景進行 inversion 測試與紅隊演練

**NIST AI RMF：**

- **GOVERN**：資料與隱私治理策略（合法性、合規性）  
- **MAP**：建立個資 / 機敏資料目錄，辨識高敏感模型  
- **MEASURE**：隱私攻擊測試（inversion / membership inference）  

**ISO/IEC 42001：**

- 組織環境與利害關係人需求（Clauses 4–5）  
- 風險與影響評估（Clause 6）  
- 資訊安全與隱私控制（Clause 8）

---

### 2.4 ML04:2023 Membership Inference Attack（成員推論）

**威脅說明：**  
判斷特定樣本是否被用於模型訓練，導致隱私洩漏（例如：病患是否在某醫療數據庫中）。

**防禦措施：**

- 資料最小化與匿名化，減少可識別特徵  
- 差分隱私訓練與 regularization  
- 限制模型輸出細節與 API 回應（避免返回過多信心分數）  
- 針對 membership inference 的攻擊測試與指標量測

**框架對應：**

- **NIST AI RMF**：GOVERN（隱私政策）、MAP（識別受影響群體）、MEASURE（隱私健全性測試）  
- **ISO/IEC 42001**：隱私與法規遵循、資料主體權利保護、影響評估

---

### 2.5 ML05:2023 Model Theft（模型竊取）

**威脅說明：**  
透過反覆查詢 API 或入侵基礎設施複製模型行為或權重，造成智慧財產與安全風險。

**防禦措施：**

- API 調用速率限制與異常行為偵測  
- 強化身分驗證、授權與金鑰管理  
- 模型水印（watermarking）與指紋識別  
- 加密與安全部署（如硬體信任根、TEE）

**框架對應：**

- **NIST AI RMF**：GOVERN（IP 保護政策）、MANAGE（API 安全與事件回應）  
- **ISO/IEC 42001**：智慧財產、存取控制、外部接口與雲端服務管理（Clauses 5, 6, 8）

---

### 2.6 ML06:2023 AI Supply Chain Attacks（供應鏈攻擊）

**威脅說明：**  
針對資料、模型、程式庫、MLOps 工具等整個 AI 供應鏈的攻擊（惡意套件、被植後門的預訓練模型等）。

**防禦措施：**

- 軟體供應鏈安全（SBOM、簽章、完整性驗證）  
- 第三方與開源元件審查與白名單管理  
- 預訓練模型的安全與偏差評估  
- 供應商安全要求（合約條款、稽核）

**框架對應：**

- **NIST AI RMF**：GOVERN（供應鏈治理）、MAP（盤點第三方依賴）、MANAGE（採購與委外安全要求）  
- **ISO/IEC 42001**：供應鏈與外部供應者管理、生命周期與變更管理

---

### 2.7 ML07:2023 Transfer Learning Attack（遷移學習攻擊）

**威脅說明：**  
攻擊者將惡意後門嵌入預訓練模型，當被下游任務 fine-tune 使用時觸發攻擊。

**防禦措施：**

- 僅使用可信來源的預訓練模型（官方庫、簽章驗證）  
- 對預訓練模型做安全掃描與後門檢測  
- fine-tune 前後比較行為差異（尤其在觸發樣本下）  
- 對高風險任務採「從頭訓練」或多來源模型比對

**框架對應：**

- **NIST AI RMF**：MAP（模型來源盤點）、MEASURE（穩健性與後門測試）、MANAGE（reuse policy）  
- **ISO/IEC 42001**：模型生命周期、重用與遷移評估、變更管理

---

### 2.8 ML08:2023 Model Skewing（模型偏移）

**威脅說明：**  
利用回饋迴圈或線上互動，逐漸將模型行為偏向攻擊者想要的方向（如推薦系統被刷評、詐騙樣本漸漸被當成正常）。

**防禦措施：**

- 線上資料與回饋的品質監控（anti-spam / anti-bot）  
- 模型輸出分佈、指標與 fairness 監控  
- 定期重新訓練與偏差校正（含人類審查）  
- 強制樣本抽樣機制避免被單一族群主導

**框架對應：**

- **NIST AI RMF**：MEASURE（效能與偏差指標）、MANAGE（drift 回應計畫）  
- **ISO/IEC 42001**：績效監控、偏差管理與持續改善（Clauses 8–10）

---

### 2.9 ML09:2023 Output Integrity Attack（輸出完整性攻擊）

**威脅說明：**  
攻擊者不必修改模型本身，而是操控 **輸出路徑**（UI、API Gateway、前置/後處理程式）來篡改或偽造結果。

**防禦措施：**

- 在關鍵流程中加入結果驗證與多重校驗（multi-party / multi-model check）  
- 保護推論管線與服務端程式（程式碼審查、檔案完整性檢查）  
- 加強端點與 API 安全（WAF、認證、加密）  

**框架對應：**

- **NIST AI RMF**：MEASURE（輸出監控）、MANAGE（incident response, change management）  
- **ISO/IEC 42001**：運作管制與輸出管理、記錄與稽核追蹤（Clauses 8, 9）

---

### 2.10 ML10:2023 Model Poisoning（模型中毒）

**威脅說明：**  
直接在訓練過程或模型參數層級進行篡改，例如在 MLOps 管線、訓練節點或儲存體中植入惡意更新。

**防禦措施：**

- 訓練環境隔離與硬體安全（安全集群、TEE）  
- 模型版本管理與完整性檢查（hash、簽章、artifact repository）  
- MLOps pipeline 安全（CI/CD 權限最小化、審核）  
- 部署前後模型行為與權重差異檢查

**框架對應：**

- **NIST AI RMF**：MAP（模型與管線盤點）、MEASURE（完整性與穩健性測試）、MANAGE（pipeline security、incident plan）  
- **ISO/IEC 42001**：AI 管理系統之運作與變更管理、事件處理與持續改善（Clauses 8–10）

---

## 3. 如何使用本對照表

- **威脅建模**：將 OWASP ML Top 10 當作 AI 威脅清單，對照 MITRE ATLAS 技術，建立攻擊情境。  
- **治理與稽核**：利用 NIST AI RMF 與 ISO/IEC 42001 對照，規劃政策、程序與控管項目。  
- **教學與訓練**：本表可直接作為課堂講義或企業教育素材，搭配實作／紅隊演練。  

---
