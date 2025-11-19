## 報告1A:IMDB自然語言實戰
- [使用 TF‑IDF+ 機器學習 分析](IMDB電影評論情感1_MACHINE_LEARNING.md)
  - 以 TF‑IDF 將文字轉為特徵，並使用 Logistic Regression / Linear SVM 建立二元情感分類器（正向/負向），評估準確率、F1、混淆矩陣，並示範超參數搜尋與錯誤分析。 
- [使用 RNN 分析](IMDB電影評論情感2_RNN.md)
  - 單層SimpleRNN, LSTM, GRU
  - 多層 LSTM 模型
  - 多層 GRU 模型
  - 雙向LSTM
  - 雙向GRU(Bidirectional GRU) 
- [使用 Transformer 分析加入早停（Early Stopping）機制](IMDB電影評論情感6.md)
  - 以 Hugging Face Transformers（如 DistilBERT/BERT）微調二元情感分類器，並在 IMDB 影評資料集上評估 Accuracy/F1、混淆矩陣與錯誤分析，加入早停（Early Stopping）機制 提升訓練穩定性與防止過擬合。
- [使用DistilBERT分析](IMDB電影評論情感4_DistilBERT.md)
 
## 報告1B:自然語言實戰
- 自然語言理解 主題
  - 情感分析(Sentiment Analysis) ==> Binary classification ==> Multiclass classification 
  - Text classification (TC) 主題分類 ==> Multiclass vs Multi-label
    - 股票分析  名嘴  投顧 
  - Text Generation
  - Machine Translation(機器翻譯)
- NLP-1 自然語言前處理 Tokenization ==>
- NLP-2 word == > Vector
- NLP-3 傳統序列模型：RNN、LSTM、GRU
- NLP-4 Transformer 架構
- NLP-5 LLM 預訓練語言模型== >
- NLP-6 LLM Fine-Tunning



## 報告2:Time series analysis using Custom-build LLM
- ETF analysis using Deep learning
- 參考資料
  - 時空序列分析-關鍵籌碼分析 系列 https://ithelp.ithome.com.tw/users/20168322/ironman/7065 
- 使用套件
  - https://www.finlab.tw/python-taiwan-stock-market-selection/
  - 移動平均線 (MA)：計算特定期間內的平均價格，有助於識別趨勢。
  - 相對強弱指標 (RSI)：衡量某個股票在特定期間內的過度買入或賣出狀態 
- 資料擷取 ==>
- 資料儲存
  - sqlite
  - Mysql
  - NOSQL
- 資料分析
  - 統計分析 ==> ARMA
  - 機器學習 ==>
  - 深度學習 ==> LSTM | Transformer | LLM | MLLM
- Deployment
  - line
  - web(Chat)
  - API(FastAPI)
  - CLOUD
    - Amazon
    - Azure
    - Google  
