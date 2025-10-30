## 報告1:自然語言實戰
- 自然語言理解 主題
  - 情感分析(Sentiment Analysis) ==> Binary classification ==> Multiclass classification 
  - Text classification (TC) 主題分類 ==> Multiclass vs Multi-label
  - Text Generation
  - Machine Translation(機器翻譯)
- NLP-1自然語言前處理 Tokenization ==>
- NLP-2 word == > Vector
- NLP-3 傳統序列模型：RNN、LSTM、GRU
- NLP-4 Transformer 架構
- NLP-5 LLM 預訓練語言模型== >
- NLP-6 LLM Fine-Tunning



## 報告2:Time series analysis using Custom-build LLM
- ETF analysis using Deep learning
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
```
# 台灣ETF資料分析：Yahoo Finance + TWSE + Plotly Dash
# 作者：T Ben

import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from dash import Dash, dcc, html

# ====== 1️⃣ 基本設定 ======
etfs = {
    "0050": "元大台灣50",
    "0056": "元大高股息",
    "00878": "國泰永續高股息",
    "00692": "富邦公司治理"
}

# ====== 2️⃣ Yahoo Finance 歷史價格 ======
data = {}
for code in etfs:
    df = yf.download(f"{code}.TW", start="2020-01-01")
    df["Code"] = code
    df["Name"] = etfs[code]
    data[code] = df

prices = pd.concat(data.values())
pivot_close = prices.pivot_table(values="Close", index="Date", columns="Code")
returns = pivot_close.pct_change().dropna()
cumulative = (1 + returns).cumprod()

# ====== 3️⃣ TWSE 每日基金資料 ======
def get_twse_etf_info(code: str) -> pd.DataFrame:
    """
    從證交所網站抓取ETF每日資料
    """
    url = f"https://www.twse.com.tw/exchangeReport/FMTQIK?response=html&selectType=30&_=&stockNo={code}"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "lxml")

    table = soup.find("table")
    if table is None:
        print(f"⚠️ 找不到ETF代號 {code} 的資料")
        return pd.DataFrame()

    df = pd.read_html(str(table))[0]
    df.columns = ["日期", "開盤價", "最高價", "最低價", "收盤價", "漲跌價差", "成交張數"]
    df["日期"] = pd.to_datetime(df["日期"], format="%Y/%m/%d")
    df["代號"] = code
    df["名稱"] = etfs.get(code, "未知ETF")
    return df

twse_data = pd.concat([get_twse_etf_info(code) for code in etfs])

# ====== 4️⃣ 總覽摘要 ======
summary = pd.DataFrame({
    "平均日報酬": returns.mean(),
    "年化報酬率": (1 + returns.mean())**252 - 1,
    "波動度": returns.std() * (252**0.5)
})
summary["Sharpe比率"] = summary["年化報酬率"] / summary["波動度"]
summary = summary.round(4)

print("📊 ETF績效摘要：")
print(summary)

# ====== 5️⃣ Plotly 互動儀表板 ======

app = Dash(__name__)
app.title = "台灣ETF互動分析儀表板"

app.layout = html.Div([
    html.H1("📈 台灣ETF分析儀表板", style={"textAlign": "center"}),
    
    dcc.Dropdown(
        id="etf-dropdown",
        options=[{"label": f"{code} {name}", "value": code} for code, name in etfs.items()],
        value="0050",
        clearable=False,
        style={"width": "50%", "margin": "auto"}
    ),
    
    dcc.Graph(id="price-chart"),
    dcc.Graph(id="return-chart"),
])

@app.callback(
    [dcc.Output("price-chart", "figure"),
     dcc.Output("return-chart", "figure")],
    [dcc.Input("etf-dropdown", "value")]
)
def update_graph(selected_code):
    # Yahoo Finance 累積報酬
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=cumulative.index, y=cumulative[selected_code],
        mode="lines", name=f"{selected_code} 累積報酬"
    ))
    fig1.update_layout(title=f"{selected_code} {etfs[selected_code]}｜累積報酬率走勢",
                       xaxis_title="日期", yaxis_title="累積報酬率")

    # TWSE 收盤價
    df = twse_data[twse_data["代號"] == selected_code]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["日期"], y=df["收盤價"],
        mode="lines+markers", name=f"{selected_code} 收盤價"
    ))
    fig2.update_layout(title=f"{selected_code} {etfs[selected_code]}｜TWSE 收盤價",
                       xaxis_title="日期", yaxis_title="收盤價 (NTD)")

    return fig1, fig2

if __name__ == "__main__":
    print("🚀 啟動 Dash 互動儀表板：http://127.0.0.1:8050/")
    app.run_server(debug=True)

```
