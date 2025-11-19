# MITRE ATLAS 戰術（Tactics）＋技術（Techniques）

## 目錄
1. Reconnaissance（偵察）
2. Resource Development（資源建立）
3. Initial Access（初始存取）
4. Data Collection（資料收集）
5. AI Model Access（模型存取）
6. Discovery（探索）
7. Attack Staging（攻擊準備）
8. Execution（執行）
9. Persistence（持續存留）
10. Privilege Escalation（權限提升）
11. Defense Evasion（防禦規避）
12. Credential Access（憑證竊取）
13. Exfiltration（外洩）
14. Impact（影響）
15. Supply Chain Compromise（供應鏈妥協）

---

## 1. Reconnaissance（偵察）
### 技術
- Public Training Data Enumeration
- Model API Fingerprinting
- Model Card Analysis

### 範例
攻擊者從 GitHub 推估 API 行為差異準備模型抽取。

---

## 2. Resource Development（資源建立）
### 技術
- Acquire Poisoning Dataset
- Malicious Model Pre-training
- Compromise ML Libraries

### 範例
於 HuggingFace 上上傳帶後門模型供受害者誤用。

---

## 3. Initial Access（初始存取）
### 技術
- Exploit AI API Endpoint
- Abuse Public Web Form
- Shadow Deployment Access

### 範例
攻擊者利用未驗證的 /predict API 存取模型。

---

## 4. Data Collection（資料收集）
### 技術
- Dataset Scraping
- Label Observation

### 範例
透過大量 Query 建立分類模型的資料分佈。

---

## 5. AI Model Access（模型存取）
### 技術
- Model Weight Access
- Model API Access
- Shadow Copy Extraction

### 範例
駭客從 S3 bucket 下載未加密模型權重。

---

## 6. Discovery（探索）
### 技術
- Pipeline Enumeration
- Feature Map Discovery
- Model Metadata Exploration

### 範例
發現企業使用固定 random seed 加速模型抽取。

---

## 7. Attack Staging（攻擊準備）
### 技術
- Adversarial Sample Crafting
- Poison Data Crafting
- Backdoor Trigger Design

### 範例
產生大量 FGSM 圖像測試防禦能力。

---

## 8. Execution（執行）
### 技術
- Evasion Attack
- Poisoning Attack
- Model Inversion
- Model Extraction
- Prompt Injection

### 範例
讓自駕車把「停」標誌錯分類為「限速」。

---

## 9. Persistence（持續存留）
### 技術
- Training Pipeline Backdoor
- Model Replacement
- API Key Persistence

### 範例
攻擊者替換推論模型版本為帶後門版本。

---

## 10. Privilege Escalation（權限提升）
### 技術
- Upgrade to Training Access
- Model Registry Write Access
- Authorization Bypass

### 範例
取得 MLflow admin 權限覆寫模型。

---

## 11. Defense Evasion（規避防禦）
### 技術
- Adversarial Perturbation
- Log Tampering
- Model Watermark Removal

### 範例
用微小擾動規避惡意程式檢測。

---

## 12. Credential Access（憑證竊取）
### 技術
- API Token Theft
- MLflow/Kubeflow Credential Theft
- Cloud IAM Key Theft

### 範例
從 CI/CD Log 中竊取 HuggingFace Token。

---

## 13. Exfiltration（外洩）
### 技術
- Model Exfiltration
- Training Data Exfiltration
- Feature Vector Leakage

### 範例
壓縮 LLM 權重並傳送到外部伺服器。

---

## 14. Impact（影響）
### 技術
- Model Degradation
- Model Misbehavior
- Data Integrity Impact
- Safety Failure Induction

### 範例
病理診斷模型被污染後誤判癌症。

---

## 15. Supply Chain Compromise（供應鏈妥協）
### 技術
- Malicious Pre-trained Model Supply
- Dependency Poisoning
- Model Hub Poisoning

### 範例
在 tokenizer 中植入後門導致企業誤用。
