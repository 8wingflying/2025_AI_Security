# AI 與安全(三大主題)
- AI-powered attack
- AI-powered defense
- Attacking AI(Machine Learning)
  - [AI Security 101](https://atlas.mitre.org/resources/ai-security-101) [說明](ai-security-101.md)
  - [MITRE ALTAS](https://atlas.mitre.org/) [說明](MITREALTAS.md)
  - [Top 10 Machine Learning Security Risks](https://owasp.org/www-project-machine-learning-security-top-10/)  [TOP20_ML說明](TOP20_ML.md)
## 自動化滲透測試
## AI Detection
- [An Evaluation Framework for Network IDS/IPS Datasets: Leveraging MITRE ATT&CK and Industry Relevance Metrics](https://arxiv.org/abs/1903.08689)
  - COMPREHENSIVE IDS DATASET COMPARISON (2009–2024)
  - five key evaluation metrics: ARS, TRS, TeRS, ECS and DQS
  - IDS/IPS Dataset Evaluation Framework

#### Model Performance Table

| 模型名稱                                       |   Exact Match % |   Partial Match % |   Overall Accuracy % |
|:-------------------------------------------------|----------------:|------------------:|---------------------:|
| acedev003/gte-small-mitre                        |           21.43 |             57.14 |                78.57 |
| sentence-transformers/all-MiniLM-L12-v2          |           28.57 |             42.86 |                71.43 |
| basel/ATTACK-BERT                                |           42.86 |             21.43 |                64.29 |
| sentence-transformers/all-MiniLM-L6-v2           |           21.43 |             42.86 |                64.29 |
| all-mpnet-base-v2                                |           35.71 |             21.43 |                57.14 |
| multi-qa-mpnet-base-dot-v1                       |            7.14 |             50    |                57.14 |
| sentence-transformers/multi-qa-mpnet-base-dot-v1 |            0    |             50    |                50    |
| BAAI/bge-m3                                      |           42.86 |              7.14 |                50    |
| Sentence-t5-base                                 |            0    |             28.57 |                28.57 |
| paraphrase-albert-small-v2                       |            0    |             28.57 |                28.57 |
| google/flan-t5-large                             |            0    |             21.43 |                21.43 |
| markusbayer/CySecBERT                            |            0    |             14.29 |                14.29 |
| microsoft/deberta-v3-base                        |            0    |              0    |                 0    |
| ehsanaghaei/SecureBERT                           |            0    |              0    |                 0    |
| jackaduma/SecBERT                                |            0    |              0    |                 0    |
| sarahvei/MITRE-v15-tactic-bert-case-based        |            0    |              0    |                 0    |
| bencycl129/mitre-bert-base-cased                 |            0    |              0    |                 0    |
