## LlamaIndex
- LlamaIndex, Data Framework for LLM Applications
- https://llamahub.ai/
- 資料來源:https://claire-chang.com/2024/05/17/llamaindex%E4%BB%8B%E7%B4%B9/
- LlamaIndex 專注於將非結構化數據（如文本文件、PDF、網頁內容等）轉換為可以用於查詢和分析的結構化索引。它為用戶提供了一種簡單的方法來構建和查詢這些索引，從而更好地利用大型語言模型進行數據處理和檢索。
- LlamaIndex 提供了以下功能來幫忙構建上下文增強 LLM 應用程式：
  - 數據索引和向量化：LlamaIndex 可以將非結構化數據轉換為向量表示，這對於 LLM 來說更易於處理。可以輕鬆地將自己的數據添加到 LlamaIndex 中，並讓 LLM 從中學習。
  - 數據檢索：LlamaIndex 可以根據用戶查詢快速準確地檢索相關數據。這意味著 LLM 應用程式將能夠快速找到其所需的信息，以提供準確的響應。
  - LLM 集成：LlamaIndex 可以與各種 LLM 集成，例如 GPT-3 和 Jurassic-1 Jumbo。這意味著可以選擇最適合的應用程式的 LLM。
     
## 範例學習  !pip install llama-index
- !mkdir data
- 上傳一個新加坡 ai 治理的 pdf檔案
- !pip install openai
- 範例程式
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import openai
from google.colab import userdata
openai.api_key = userdata.get('GenAI20240912')
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is ai governance")
print(response)
```
## 延伸閱讀
- [RAG-Driven Generative AI](https://www.packtpub.com/en-us/product/rag-driven-generative-ai-9781836200901)
- [Building Data-Driven Applications with LlamaIndex](https://www.packtpub.com/en-us/product/building-data-driven-applications-with-llamaindex-9781805124405)
