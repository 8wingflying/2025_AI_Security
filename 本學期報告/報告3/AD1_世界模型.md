# 世界模型(World Models)
- [Awesome-World-Model](https://github.com/tsinghua-fib-lab/World-Model) 


## 論文
- 2018 `開山巨作`2018[World Models|David Ha, Jürgen Schmidhuber](https://arxiv.org/abs/1803.10122)
  - https://worldmodels.github.io/
- [[1809.01999] Recurrent World Models Facilitate Policy Evolution](https://arxiv.org/abs/1809.01999)
- 2024[(2402)World Model on Million-Length Video And Language With Blockwise RingAttention](https://arxiv.org/abs/2402.08268)
  - https://github.com/LargeWorldModel/lwm 
  - 訓練了迄今為止上下文尺寸最大的神經網路之一，在複雜的檢索任務和長視頻理解方面取得了最佳成績。
  - 提出了克服視覺-語言訓練挑戰的解決方案，包括使用masked sequence packing混合不同序列長度，loss weighting平衡語言和視覺，以及利用模型生成的QA資料集進行長序列對話。
  - 開源了一個高度優化的實現，包括RingAttention、masked sequence packing等關鍵特性，用於訓練百萬級長度的多模態序列。
  - 開源了一系列能夠處理超過1M token長文本（LWM-Text、LWM-Text-Chat）和視頻（LWM、LWM-Chat）的7B參數模型。
  - 【導讀】https://zhuanlan.zhihu.com/p/685851325

## REVIEW
- 2024.11 [Understanding World or Predicting Future? A Comprehensive Survey of World Models](https://arxiv.org/abs/2411.14499v1)
  - https://arxiv.org/abs/2411.14499 
  - https://github.com/LargeWorldModel/lwm

## 參考資料
- https://worldmodels.github.io/
- https://zhuanlan.zhihu.com/p/25896058607
- https://blog.csdn.net/weixin_74887700/article/details/145644596

## 世界模型(World Models)
- 2023年6月18日，特斯拉自動駕駛負責人阿肖克·埃盧斯瓦米（Ashok Elluswamy）在CVPR2023上作了一個名為“自動駕駛的基礎模型”的主題演講
  - 解釋了特斯拉正在打造的“通用世界模型”如何能夠通過過往的視頻片段和行動提示，生成“可能的未來”的全新視頻。
- 2025年開年Deepseek大火的情況下，世界模型的研究對於數位孿生城市發展具有重要的現實意義
  - [李飛飛創業公司World Labs:引領AI新方向的“大世界模型”](https://blog.csdn.net/weixin_41496173/article/details/142441574)
  - 大世界模型(Large WorldModel, LWM)是World Labs的另一個核心

#### [Understanding World or Predicting Future? A Comprehensive Survey of World Models](https://arxiv.org/abs/2411.14499)
- 由於 GPT-4 等多模態大型語言模型和 Sora 等視頻生成模型的進步，世界模型的概念引起了廣泛關注，這些模型是追求通用人工智慧的核心。
- 本文對世界模型的文獻進行了全面回顧。
- 通常，世界模型被視為理解世界現狀或預測其未來動態的工具。
- 本文對世界模型進行了系統分類，強調兩個主要功能：
  - （1） 構建內部表示以瞭解世界的機制，以及
  - （2） 預測未來狀態以類比和指導決策。
- 首先，我們研究這兩個類別的當前進展。
- 然後，我們探討了世界模型在關鍵領域的應用，包括自動駕駛、機器人和社會類比，
- 並重點介紹每個領域如何利用這些方面。
- 最後，我們概述了關鍵挑戰，並對未來潛在的研究方向提供了見解。


#### 範例程式
- chapter 12　世界模型
- 強化學習
- 世界模型概述
- 收集隨機推演資料
- 訓練 VAE
- 收集資料以訓練 MDN-RNN
- 訓練 MDN-RNN
- 訓練控制器
　夢境訓練
