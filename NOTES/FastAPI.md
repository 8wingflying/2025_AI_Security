# 📚 **FastAPI** 
> **FastAPI** 是現代 Python Web 應用的首選框架，  
> 特點包括：
> - ⚡ **超高效能**（比 Flask 快 3–5 倍）  
> - 🧩 **型別安全**（Pydantic 驗證 + 型別提示）  
> - 🧠 **自動產生 Swagger / ReDoc 文件**  
> - 🧵 **支援 async / await 非同步設計**  
> - 🧰 **可模組化、易測試、易維護**

---

# ⚡ FastAPI 各項功能與常用函數總覽表  

---

## 一、FastAPI 概要（Overview）

| 模組 / 類別 | 中文說明 | 功能重點 |
|---------------|------------|------------|
| `FastAPI()` | 主應用實例 | 建立 API 主體物件 |
| `@app.get()` / `@app.post()` | 路由修飾器 | 定義 HTTP 方法端點 |
| `Request`, `Response` | 請求 / 回應物件 | 控制 HTTP 資料交換 |
| `status` | 狀態碼模組 | 提供標準 HTTP 狀態常數 |
| `Depends` | 依賴注入系統 | 控制安全性、驗證、邏輯重用 |
| `BackgroundTasks` | 背景任務模組 | 非同步處理長時間任務 |
| `Pydantic BaseModel` | 資料模型驗證 | 自動生成文件與參數檢查 |
| `Path`, `Query`, `Body`, `Header`, `Cookie`, `Form`, `File` | 請求參數類型 | 提供靈活的輸入方式 |
| `APIRouter` | 子路由系統 | 模組化應用結構 |
| `StaticFiles` | 靜態資源服務 | 提供圖片 / JS / CSS 檔案 |

---

## 二、應用建立與路由定義（App Initialization & Routing）

| 函數 / 修飾器 | HTTP 方法 | 功能說明 | 主要參數 | 範例 |
|----------------|------------|------------|-------------|------|
| `@app.get()` | GET | 讀取資源 | `path`, `response_model`, `status_code` | `@app.get("/items")` |
| `@app.post()` | POST | 建立資源 | `response_model`, `summary` | `@app.post("/users")` |
| `@app.put()` | PUT | 更新整個資源 | `path`, `status_code` | `@app.put("/items/{id}")` |
| `@app.patch()` | PATCH | 局部更新 | — | — |
| `@app.delete()` | DELETE | 刪除資源 | — | — |
| `@app.options()` | OPTIONS | 查詢支援的 HTTP 方法 | — | — |
| `@app.head()` | HEAD | 只取得標頭資訊 | — | — |

---

## 三、請求參數與驗證（Request Parameters & Validation）

| 類型 | 使用函數 | 功能說明 | 常用參數 / 屬性 | 範例 |
|-------|------------|------------|------------------|------|
| 路徑參數 | `Path()` | 從 URL 路徑取得資料 | `title`, `ge`, `le`, `regex` | `id: int = Path(..., ge=0)` |
| 查詢參數 | `Query()` | 從 URL query string 取值 | `min_length`, `max_length`, `default` | `q: str = Query(None, max_length=50)` |
| 請求主體 | `Body()` | 從 JSON 取得 body | `embed=True` | `item: Item = Body(...)` |
| 標頭參數 | `Header()` | 取得 request header | `convert_underscores` | `user_agent: str = Header(None)` |
| Cookie | `Cookie()` | 取得 cookie | — | `cookie_id: str = Cookie(None)` |
| 表單 | `Form()` | 接收表單資料 | `Form(...)` | 用於 HTML 表單 |
| 檔案上傳 | `File()` | 接收上傳檔案 | `File(...)` | 支援多檔案上傳 |

---

## 四、資料模型與驗證（Data Models with Pydantic）

| 類別 / 函數 | 功能說明 | 主要屬性 / 功能 | 範例 |
|---------------|------------|------------------|------|
| `BaseModel` | 定義資料結構 | 型別檢查、自動轉換 | `class User(BaseModel): name:str age:int` |
| `Field()` | 欄位設定 | `default`, `title`, `max_length`, `regex` | `name: str = Field(..., max_length=50)` |
| `.dict()` / `.json()` | 模型序列化 | 轉換為 dict / JSON | — |
| `Config` 類別 | 模型設定 | `orm_mode=True` 允許 ORM 資料 | — |

---

## 五、回應（Responses）

| 類別 / 函數 | 功能說明 | 主要參數 / 屬性 | 備註 |
|---------------|------------|------------------|------|
| `Response` | 回應基底類別 | `content`, `status_code`, `media_type` | — |
| `JSONResponse` | 回傳 JSON | 自動設定 `application/json` | — |
| `HTMLResponse` | 回傳 HTML | `media_type='text/html'` | — |
| `PlainTextResponse` | 回傳純文字 | — | — |
| `FileResponse` | 回傳檔案 | `path`, `filename`, `media_type` | — |
| `RedirectResponse` | 重導至其他路徑 | `url`, `status_code=302` | — |
| `StreamingResponse` | 串流回應 | 用於大檔或即時資料 | — |

---

## 六、依賴注入（Dependency Injection）

| 函數 / 類別 | 功能說明 | 使用方式 | 範例 |
|----------------|------------|-------------|------|
| `Depends()` | 宣告依賴項 | 可重用安全性 / 資料庫 / 驗證邏輯 | `user = Depends(get_current_user)` |
| `Security()` | 搭配 OAuth2 | 控制 API 權限 | `Security(get_current_user, scopes=["admin"])` |
| `BackgroundTasks` | 背景任務 | 非同步執行任務 | `tasks.add_task(send_email)` |
| `Annotated` *(Python 3.9+)* | 簡化依賴型別提示 | `Annotated[str, Depends(func)]` | — |

---

## 七、錯誤處理與例外（Errors & Exceptions）

| 類別 / 函數 | 功能說明 | 主要參數 | 範例 |
|---------------|------------|-------------|------|
| `HTTPException` | 拋出 HTTP 錯誤 | `status_code`, `detail`, `headers` | `raise HTTPException(404, "Item not found")` |
| `RequestValidationError` | 驗證錯誤事件 | — | 由 FastAPI 自動處理 |
| `app.exception_handler()` | 自訂例外處理器 | 處理自定義錯誤格式 | — |
| `ValidationError` | 由 Pydantic 產生 | 驗證失敗時觸發 | — |

---

## 八、中介軟體（Middleware）

| 函數 / 裝飾器 | 功能說明 | 範例 |
|----------------|------------|------|
| `@app.middleware("http")` | 攔截每次 HTTP 請求 | 用於日誌、CORS、驗證等 |
| `add_middleware()` | 新增現成中介層 | `app.add_middleware(CORSMiddleware, allow_origins=["*"])` |
| `CORSMiddleware` | 解決跨域問題 | — |
| `GZipMiddleware` | 壓縮回應 | 提高傳輸效率 |

---

## 九、應用模組化（Routers & Modularization）

| 類別 / 函數 | 功能說明 | 範例 |
|---------------|------------|------|
| `APIRouter()` | 建立子路由 | `router = APIRouter(prefix="/users")` |
| `include_router()` | 將子路由導入主應用 | `app.include_router(router)` |
| `tags` | 文件分組標籤 | `tags=["users"]` |
| `dependencies` | 模組依賴項 | — |

---

## 十、文件自動化（API Docs & Schema）

| 功能類別 | 功能說明 | 路徑 | 備註 |
|------------|------------|------|------|
| Swagger UI | 互動式文件 | `/docs` | 預設開啟 |
| ReDoc | 書本式文件 | `/redoc` | — |
| OpenAPI JSON | 結構化定義 | `/openapi.json` | — |
| `title`, `description`, `version` | 文件標頭設定 | 在 `FastAPI()` 初始化中設定 | — |

---

## 十一、資料庫整合（Database Integration）

| 類型 | 工具 / 模組 | 功能說明 | 範例 |
|--------|--------------|------------|------|
| ORM | `SQLAlchemy` | 支援同步 / 非同步 DB | `SessionLocal()` |
| ODM | `Tortoise ORM`, `Beanie` | 支援 NoSQL (MongoDB) | — |
| 驗證模型 | `Pydantic` | 自動驗證 ORM 物件 | `Config.orm_mode = True` |
| 非同步引擎 | `asyncpg`, `databases` | 搭配 async / await | — |

---

## 十二、安全與認證（Security & Auth）

| 模組 / 類別 | 功能說明 | 範例 |
|---------------|------------|------|
| `OAuth2PasswordBearer()` | Bearer Token 驗證 | `oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")` |
| `OAuth2PasswordRequestForm` | 表單登入格式 | — |
| `APIKeyHeader`, `APIKeyQuery` | API Key 驗證 | — |
| `JWT` | JSON Web Token | 可結合 `pyjwt` 套件 |
| `SecurityScopes` | 權限控制 | — |

---

## 十三、靜態檔案與範本（Static Files & Templates）

| 功能類別 | 功能說明 | 範例 |
|------------|------------|------|
| 靜態檔案 | 提供前端資源 | `app.mount("/static", StaticFiles(directory="static"))` |
| 模板引擎 | HTML 模板渲染 | `Jinja2Templates(directory="templates")` |
| 模板回應 | 顯示 HTML | `templates.TemplateResponse("index.html", {"request": request})` |

---

## 十四、測試與開發（Testing & Dev Tools）

| 功能類別 | 功能說明 | 主要模組 / 函數 | 範例 |
|------------|------------|------------------|------|
| 測試工具 | 單元測試支援 | `TestClient(app)` | `from fastapi.testclient import TestClient` |
| 自動重載 | 開發模式 | `uvicorn main:app --reload` | 監控程式變更 |
| 伺服器啟動 | ASGI 伺服器 | `uvicorn.run(app, host="0.0.0.0", port=8000)` | — |

---



✅ **典型範例：FastAPI CRUD 範例**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="🚀 FastAPI Example")

class Item(BaseModel):
    id: int
    name: str
    price: float

items = {}

@app.post("/items")
def create_item(item: Item):
    items[item.id] = item
    return item

@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return items[item_id]

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    items[item_id] = item
    return item

@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    if item_id in items:
        del items[item_id]
    return {"deleted": item_id}
