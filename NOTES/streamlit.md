# 📚 **Streamlit**

> **Streamlit** 是資料科學家與 AI 工程師最友善的應用框架
> Python 最受歡迎的 資料應用與機器學習可視化前端框架，讓開發者能以最少代碼快速建立互動式網頁應用。
> 特色包括：
> - 📈 **快速開發**：用 Python 即可建立前端介面  
> - 🔁 **即時互動**：內建狀態回呼與動態更新  
> - 🧩 **高度整合**：支援 Pandas、Matplotlib、Plotly、OpenAI API  
> - ☁️ **一鍵部署**：可直接上傳至 Streamlit Cloud 或 Docker  

---

# 🌐 Streamlit 各項功能與常用函數總覽表  


---

## 一、Streamlit 概要（Overview）

| 模組 / 類別 | 中文說明 | 功能重點 |
|--------------|------------|------------|
| `st.write` | 智慧輸出 | 自動判斷輸出型態（文字、表格、圖表、DataFrame） |
| `st.title`、`st.header`、`st.subheader` | 文字標題 | 設定不同層級的標題 |
| `st.markdown` | Markdown 支援 | 可嵌入格式化文字與 HTML 標籤 |
| `st.text`、`st.code`、`st.latex` | 顯示文字、程式碼、數學公式 | 靜態內容呈現 |
| `st.sidebar` | 側邊欄容器 | 建立互動控制元件區 |

---

## 二、文字與結構顯示（Text & Layout）

| 函數名稱 | 功能說明 | 主要參數 | 備註 |
|------------|------------|-------------|------|
| `st.title()` | 主標題 | `st.title("AI Dashboard")` | 粗體大字 |
| `st.header()` | 小標題 | `st.header("Data Overview")` | 中字體 |
| `st.subheader()` | 次標題 | `st.subheader("Model Performance")` | 較小標題 |
| `st.text()` | 純文字 | `st.text("Plain text")` | 等寬字型 |
| `st.markdown()` | Markdown 格式 | `st.markdown("**Bold Text**")` | 支援 HTML 標籤 |
| `st.code()` | 顯示程式碼區塊 | `st.code("print('Hello')", language="python")` | 語法高亮 |
| `st.latex()` | 顯示數學公式 | `st.latex(r"E=mc^2")` | LaTeX 語法支援 |
| `st.caption()` | 小註解文字 | `st.caption("Powered by Streamlit")` | 置底小字說明 |
| `st.divider()` | 水平線 | — | 分隔內容區塊 |

---

## 三、資料顯示（Data Display）

| 函數名稱 | 功能說明 | 主要參數 | 備註 |
|------------|------------|-------------|------|
| `st.write()` | 智慧輸出函數 | 自動識別型態 | 可輸出文字、DataFrame、Matplotlib 等 |
| `st.dataframe()` | 互動式表格 | `st.dataframe(df, use_container_width=True)` | 可排序 / 捲動 |
| `st.table()` | 靜態表格 | `st.table(df.head())` | 無互動功能 |
| `st.metric()` | 顯示 KPI 指標 | `st.metric(label="Accuracy", value="95%", delta="+2%")` | 適用儀表板 |
| `st.json()` | 顯示 JSON 格式 | `st.json(data_dict)` | 自動縮排與摺疊 |
| `st.image()` | 顯示圖片 | `st.image("img.png", caption="Sample Image", width=300)` | 支援 PIL、URL |
| `st.video()` | 顯示影片 | `st.video("video.mp4")` | 可播放控制 |
| `st.audio()` | 播放音訊 | `st.audio("sound.mp3")` | — |

---

## 四、繪圖與視覺化（Charts & Visualization）

| 函數名稱 | 功能說明 | 支援類型 / 套件 | 範例 |
|------------|------------|----------------|--------|
| `st.line_chart()` | 折線圖 | Pandas DataFrame / Numpy | `st.line_chart(df)` |
| `st.bar_chart()` | 長條圖 | Pandas DataFrame / Numpy | `st.bar_chart(df)` |
| `st.area_chart()` | 區域圖 | Pandas DataFrame / Numpy | `st.area_chart(df)` |
| `st.pyplot()` | Matplotlib 圖 | `plt` 物件 | `st.pyplot(fig)` |
| `st.altair_chart()` | Altair 圖 | Altair Chart 物件 | `st.altair_chart(chart)` |
| `st.plotly_chart()` | Plotly 圖 | Plotly Figure | `st.plotly_chart(fig)` |
| `st.bokeh_chart()` | Bokeh 圖 | Bokeh Figure | `st.bokeh_chart(fig)` |
| `st.map()` | 地圖散點圖 | 經緯度 DataFrame | `st.map(df[['lat', 'lon']])` |
| `st.pydeck_chart()` | 3D 地圖 | PyDeck 視覺化 | `st.pydeck_chart(deck_obj)` |

---

## 五、使用者互動元件（Widgets）

| 函數名稱 | 控制項類型 | 功能說明 | 回傳型態 |
|------------|------------|------------|-------------|
| `st.button()` | 按鈕 | 觸發動作 | `True/False` |
| `st.download_button()` | 下載按鈕 | 提供檔案下載 | — |
| `st.checkbox()` | 勾選框 | 二元選擇 | `True/False` |
| `st.radio()` | 單選按鈕 | 多選一 | 字串 |
| `st.selectbox()` | 下拉式選單 | 單一選擇 | 字串 |
| `st.multiselect()` | 多選下拉 | 多選選項 | list |
| `st.slider()` | 數值滑桿 | 設定數值範圍 | 數值 / tuple |
| `st.text_input()` | 文字輸入框 | 輸入字串 | str |
| `st.number_input()` | 數字輸入框 | 輸入數字 | int/float |
| `st.text_area()` | 多行文字框 | 長文字輸入 | str |
| `st.file_uploader()` | 上傳檔案 | 接收上傳的檔案 | BytesIO |
| `st.date_input()` | 日期選擇 | 選擇日期 | `datetime.date` |
| `st.time_input()` | 時間選擇 | 選擇時間 | `datetime.time` |
| `st.color_picker()` | 顏色選擇器 | 選取顏色 | HEX 色碼 |

---

## 六、介面佈局（Layout & Containers）

| 函數名稱 | 功能說明 | 使用方式 | 備註 |
|------------|------------|-------------|------|
| `st.sidebar` | 側邊欄容器 | `st.sidebar.selectbox()` | 常用於控制面板 |
| `st.columns()` | 建立多欄版面 | `col1, col2 = st.columns(2)` | 分欄顯示內容 |
| `st.tabs()` | 建立分頁介面 | `tab1, tab2 = st.tabs(["EDA", "Model"])` | 頁籤式切換 |
| `st.expander()` | 可收合區塊 | `with st.expander("Details"):` | — |
| `st.container()` | 自訂內容區塊 | `with st.container():` | 控制區域組合 |
| `st.empty()` | 動態占位元件 | 用於更新內容 | `placeholder.text("...")` |
| `st.spinner()` | 顯示載入動畫 | `with st.spinner("Processing..."):` | 執行中提示 |
| `st.progress()` | 進度條 | `st.progress(value)` | 支援動態更新 |

---

## 七、狀態與通知（Status & Messages）

| 函數名稱 | 功能說明 | 範例 |
|------------|------------|------|
| `st.success()` | 成功訊息 | `st.success("Training completed!")` |
| `st.error()` | 錯誤訊息 | `st.error("File not found!")` |
| `st.warning()` | 警告訊息 | `st.warning("Low accuracy!")` |
| `st.info()` | 提示訊息 | `st.info("Upload a CSV file to begin")` |
| `st.exception()` | 顯示例外錯誤 | `st.exception(e)` |
| `st.toast()` *(v1.20+)* | 浮動提示訊息 | `st.toast("Saved successfully!")` |

---

## 八、Session 狀態（State & Caching）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 快取結果 | 函數執行快取 | `@st.cache_data` | 用於資料載入 |
| 快取物件 | 模型 / 檔案快取 | `@st.cache_resource` | 用於 ML 模型 |
| Session 狀態 | 儲存互動變數 | `st.session_state["key"]` | 跨頁共享變數 |
| 重設狀態 | 清除快取 | `st.cache_data.clear()` | — |

---

## 九、互動回呼與動態更新（Events & Interaction）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 即時更新 | 動態改變內容 | `placeholder = st.empty()` → `placeholder.write("Updated!")` | 實時互動 |
| 重新執行 | 觸發重繪 | `st.experimental_rerun()` | 重新執行腳本 |
| 狀態控制 | 回呼函數 | `on_change`、`args`, `kwargs` | 控制互動行為 |

---

## 十、應用設定與部署（App Config & Deployment）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 應用設定 | 修改頁面標題與圖示 | `st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")` | 建議放在程式最前方 |
| 網頁版面 | 控制寬度 | `layout='wide' / 'centered'` | — |
| 移除預設樣式 | 自訂 CSS | `st.markdown("<style>...</style>", unsafe_allow_html=True)` | — |
| 雲端部署 | 上傳到 Streamlit Cloud | [https://streamlit.io/cloud](https://streamlit.io/cloud) | 免費服務 |
| 命令執行 | 啟動應用 | `streamlit run app.py` | 終端機命令 |

---



✅ **典型範例：AI Dashboard**

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Dashboard", layout="wide")
st.title("📊 Machine Learning Dashboard")

uploaded = st.file_uploader("Upload CSV file")
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())

    feature = st.selectbox("Select feature", df.columns)
    fig, ax = plt.subplots()
    df[feature].hist(ax=ax, bins=20)
    st.pyplot(fig)

st.sidebar.header("Model Settings")
lr = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.01)
st.sidebar.write(f"Current LR: {lr}")
