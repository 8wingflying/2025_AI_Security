# 📘 FinMind 台股事件驅動分析教學  
**完整教學檔（含程式碼與圖示範例）**  
版本：v1.0 ｜ 作者：ChatGPT × T.Ben ｜ 編碼：UTF-8  

---

## 🧬 一、教學目標

本教學示範如何使用 **FinMind API** 整合多來源資料（股價、三大法人、新聞、法說會等），  
進行 **事件驅動式（Event-Driven）股價分析與預測**。

> 📈 適用場景  
> - 台股事件影響分析（如法說會、財報公告、新聞）  
> - 機器學習預測事件後報酬  
> - 投資研究、風險管理、量化策略開發  
> - 教學課程、AI 金融資料分析實作

---

## 🔑 二、環境準備與 FinMind 介紹

### 🔹 FinMind 簡介

**FinMind** 是台灣開源金融資料平台，提供 RESTful API 與 Python SDK，  
涵蓋範圍包括：
- 台股每日報價（上市／上櫃）
- 三大法人買賣超
- 財報資料
- 公司新聞與重大事件
- 匯率、利率等廣域經濟數據

官方文件：[https://finmind.github.io/](https://finmind.github.io/)

---

### 🔹 安裝與授權設定

```bash
pip install FinMind
```

登入帳號並在 FinMind 平台取得 **API Token**：

```python
from FinMind.data import DataLoader

token = "你的_API_TOKEN"
dl = DataLoader()
dl.login_by_token(api_token=token)
```

---

## 📊 三、資料擷取範例：以台積電（2330）為例

### 1️⃣ 股價資料
```python
data_price = dl.taiwan_stock_daily(
    stock_id="2330",
    start_date="2022-01-01",
    end_date="2024-12-31"
)
```

### 2️⃣ 三大法人買賣超
```python
data_investor = dl.taiwan_stock_institutional_investors_buy_sell(
    stock_id="2330",
    start_date="2022-01-01",
    end_date="2024-12-31"
)
```

### 3️⃣ 公司新聞
```python
data_news = dl.taiwan_stock_news(
    stock_id="2330",
    start_date="2022-01-01",
    end_date="2024-12-31"
)
```

### 4️⃣ 法說會與事件公告
```python
data_event = dl.taiwan_stock_event(
    stock_id="2330",
    start_date="2022-01-01",
    end_date="2024-12-31"
)
```

---

## 🦯 四、整合資料與建立事件分析框架

```python
import pandas as pd
import matplotlib.pyplot as plt

# 股價
df_price = pd.DataFrame(data_price)
df_price['date'] = pd.to_datetime(df_price['date'])
df_price = df_price[['date', 'close']].rename(columns={'close': 'Close'})

# 三大法人買賣超（外資）
df_investor = pd.DataFrame(data_investor)
df_investor['date'] = pd.to_datetime(df_investor['date'])
df_investor = df_investor[df_investor['name'] == 'Foreign_Investor']
df_investor = df_investor[['date', 'buy', 'sell']]
df_investor['net_buy'] = df_investor['buy'] - df_investor['sell']

# 合併
df = pd.merge(df_price, df_investor, on='date', how='left').fillna(0)

# 日報酬率
df['Return'] = df['Close'].pct_change()
```

### 加入事件旗標（法說會或重大公告）

```python
df_event = pd.DataFrame(data_event)
df_event['date'] = pd.to_datetime(df_event['date'])

df['event_flag'] = df['date'].isin(df_event['date']).astype(int)
```

---

## 📈 五、視覺化範例：事件與股價的關聯

```python
plt.figure(figsize=(12,6))
plt.plot(df['date'], df['Close'], label='TSMC Close Price')
plt.scatter(df[df['event_flag']==1]['date'],
            df[df['event_flag']==1]['Close'],
            color='red', label='Events (法說會／公告)')
plt.title('台積電股價與事件日期標記')
plt.xlabel('日期')
plt.ylabel('收盤價 (NTD)')
plt.legend()
plt.show()
```

> 🔍 紅點標示代表「法說會或重大事件」日期，可視覺化觀察股價反應。

---

## 🦮 六、事件前後報酬比較分析

```python
event_dates = df[df['event_flag']==1]['date']
event_window = 5  # 事件前後各5天

returns = []
for d in event_dates:
    before = df.loc[(df['date']>=d-pd.Timedelta(days=event_window)) &
                    (df['date']<d)]['Return'].mean()
    after = df.loc[(df['date']>d) &
                   (df['date']<=d+pd.Timedelta(days=event_window))]['Return'].mean()
    returns.append({'event_date':d, 'before':before, 'after':after})

df_event_return = pd.DataFrame(returns)
df_event_return['diff'] = df_event_return['after'] - df_event_return['before']
print(df_event_return.head())
```

### 📊 視覺化事件前後報酬變化

```python
plt.bar(df_event_return['event_date'], df_event_return['diff'])
plt.title('事件前後報酬變化 (平均值差異)')
plt.ylabel('報酬差異')
plt.xticks(rotation=45)
plt.show()
```

> 若 diff 為正 → 表示事件後平均報酬上升。  
> 若 diff 為負 → 表示事件後下跌。

---

## 🧠 七、進階應用方向

| 模組 | 說明 |
|------|------|
| 🧮 **多事件回歸分析** | 結合法人買賣、新聞情緒、事件旗標 → 預測短期報酬率 |
| 📊 **情緒分析 (NLP)** | 對 `data_news` 的 `title` 做情緒分數（TextBlob、BERT、CKIP） |
| 🤖 **機器學習建模** | 使用 XGBoost / LSTM 預測報酬率或波動方向 |
| 🔍 **風險事件監測** | 自動監測重大公告／新聞變動並即時繪製影響圖表 |
| 🕹 **Streamlit 儀表板** | 將分析整合成互動式可視化平台（顯示股價※事件※法人※新聞情緒） |

---

## 🧬 八、Streamlit 可視化儀表板範例（可選）

```python
import streamlit as st
import altair as alt

st.title("台積電事件驅動分析儀表板")

chart = alt.Chart(df).mark_line().encode(
    x='date:T',
    y='Close:Q',
    tooltip=['date', 'Close']
).properties(title='TSMC 股價走勢')

event_points = alt.Chart(df[df['event_flag']==1]).mark_point(color='red', size=60).encode(
    x='date:T',
    y='Close:Q',
    tooltip=['date']
)

st.altair_chart(chart + event_points, use_container_width=True)
```

---

## 🧾 九、參考資料
- FinMind 官方文件：[https://finmind.github.io/](https://finmind.github.io/)
- Kaggle 台股資料集：
  - [Taiwan Capitalization Weighted Stock Index](https://www.kaggle.com/datasets/chunghaoleeyginger/taiwan-capitalization-weighted-stock-index)
 

