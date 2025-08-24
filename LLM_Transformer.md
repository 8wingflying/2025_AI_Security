# Transformer
- 一種基於`自注意力機制（Self-Attention Mechanism`的神經網路架構，由 Google 在 2017 年提出。
- 它的目標是有效地處理序列資料（如文字、時間序列數據），特別擅長應對長距離相依性問題。
- 這讓它在翻譯、對話生成、文本分類等任務中表現出色。
- 解決傳統的遞迴式神經網路（RNN）的問題
  - 在處理長序列資料時，RNN依賴於逐步計算序列中每個元素之間的相依關係，這樣的設計不僅會導致訊息遺失，也難以捕捉長期依賴性。
  - 而且RNN必須按序列的順序進行運算，無法進行平行計算，效率比較低。
# 經典論文與實作
- 2017年Transformer經典論文 [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- huggingface實作**"Transformers: State-of-the-Art Natural Language Processing"**. *Thomas Wolf et al.* EMNLP 2020. [[Paper](https://arxiv.org/abs/1910.03771)] [[Source](https://huggingface.co/)]
- [Introduction to Transformers: an NLP Perspective(ARXIV：2311.17633)](https://arxiv.org/abs/2311.17633)

## Transformer 架構
- https://ithelp.ithome.com.tw/articles/10363257
- Transformer的架構，主要由兩個部分組成：編碼器（Encoder） 和 解碼器（Decoder）
- 編碼器:
  - Input Embedding：輸入的序列（例如詞彙）會先轉換成嵌入向量（embedding），這是每個詞的向量表示。
  - Positional Encoding：由於 Transformer 本身不具備序列的時間步概念，因此需要引入位置編碼（Positional Encoding）來讓模型了解輸入序列的順序。
  - Multi-Head Attention：這是 Transformer 的核心機制，利用**多頭自注意力（Multi-Head Self-Attention）**來讓每個輸入詞彙與其他所有詞進行關聯，捕捉整個序列的依賴關係。
    - Attention
    - Self-attention
    - Multi-head Self-attention 
  - Add & Norm：這是一種殘差連接，將多頭注意力的輸出和輸入相加，並進行正規化。
  - Feed Forward：每個編碼器層都有一個全連接層來進一步處理數據。
- 解碼器:
  - 解碼器結構與編碼器類似
  - 多了一個額外的部分：Masked Multi-Head Attention。
  - 這是為了確保在訓練生成時，解碼器只能看到之前的輸出，而不能看到未來的輸出，避免數據洩漏。
- 解碼器的輸出
  - 通過一個線性層，再進行 Softmax，生成對應的機率分布，代表模型對下一個詞的預測。
