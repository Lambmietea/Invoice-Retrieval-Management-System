# Mini-Demo — 发票智能助手（COMP6708 Assignment 3）

基于 **Streamlit** 的发票智能助手，集成三个核心模块：

| 模块 | 功能 | 技术栈 |
|------|------|--------|
| **Module A** | PDF 发票解析 | Qwen-VL-Max 多模态大模型（DashScope） |
| **Module B** | 混合 RAG 检索 | SQLite + ChromaDB 向量数据库 |
| **Module C** | LLM 智能分析 | Gemini Agent + 5 技能 |

用户在前端网页**上传 PDF 发票** → AI 自动提取结构化数据 → 索引到向量数据库 → 支持**自然语言搜索**和**智能分析**。

---

## 快速启动

```bash
cd mini-demo

# 1. 创建虚拟环境并安装依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置 API Key（可选 — 代码内有默认 key 供本地 demo）
export GEMINI_API_KEY="your-gemini-key"        # Module C: 智能分析
export DASHSCOPE_API_KEY="your-dashscope-key"   # Module A: PDF 解析

# 3. 启动 Streamlit
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可使用。

---

## 环境变量

所有 API 密钥统一在 `config.py` 中管理，支持环境变量覆盖：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GEMINI_API_KEY` | 代码内有默认值 | Google Gemini API 密钥（Module C 智能分析） |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini 模型名称 |
| `DASHSCOPE_API_KEY` | 代码内有默认值 | 阿里云 DashScope API 密钥（Module A PDF 解析） |
| `DASHSCOPE_MODEL` | `qwen-vl-max-2025-04-02` | Qwen-VL 视觉模型名称 |

启动后，侧边栏会显示当前 API 配置状态（密钥脱敏显示）。

---

## 功能概览

### Tab 1：Upload & Parse（上传与解析）
上传 PDF 发票 → **Qwen-VL 多模态大模型**自动识别发票图像并提取结构化 JSON → 索引入 Module B（SQLite + ChromaDB），后续可通过 Search 检索。

### Tab 2：Search（智能搜索）⭐
**核心功能** — 自然语言输入，Module B 混合 RAG 引擎自动完成检索。

支持三种检索模式（自动选择）：

| 模式 | 说明 | 示例查询 |
|------|------|----------|
| 结构化过滤 | 按日期/月份/金额/供应商/类别精确筛选 | `2024年7月的发票` |
| 语义搜索 | 基于向量相似度的模糊检索 | `和差旅相关的票据` |
| 混合检索 | SQL 缩小范围 → 向量重排序 | `2024年10月超过100元的发票` |

**示例查询：**
```
2024年7月的发票          → 月份过滤
金额超过500的发票        → 金额阈值过滤
和差旅相关的票据          → 纯语义检索
京东的发票               → 供应商关键词
2024年10月超过100元的发票 → 混合：月份 + 金额 + 语义
食品饮料类的消费          → 语义检索
March dining invoices    → 英文也支持
```

### Tab 3：Tasks（智能分析任务）
通过 Module C（Gemini LLM Agent）执行五种分析技能：

| 技能 | 示例指令 |
|------|----------|
| 汇总统计 | `按销售方统计各商家的消费总金额` |
| 重复检测 | `帮我看看有没有重复报销的发票` |
| 异常检测 | `查一下有没有金额异常或信息不全的发票` |
| 供应商画像 | `钱都被哪几个公司赚走了？` |
| 报销台账 | `帮我做成报销台账表，按日期排好` |

### Tab 4：Dashboard（数据看板）
自动汇总发票数据：总发票数、总金额、按类别消费柱状图、月度趋势折线图、Top 供应商排行。

---

## 架构说明

```
┌─────────────────────────────────────────────────┐
│  Streamlit UI (app.py)                          │
│  ├── Upload & Parse (ui/upload_page.py)         │
│  ├── Search        (ui/search_page.py)          │
│  ├── Tasks         (ui/tasks_page.py)           │
│  └── Dashboard     (ui/dashboard_page.py)       │
└────────────────┬────────────────────────────────┘
                 │ 所有调用通过 api_adapter
                 ▼
┌─────────────────────────────────────────────────┐
│  services/api_adapter.py                        │
│  ├── parse_invoice() ──→ Module A (PDF 解析)    │
│  ├── query_invoices() ──→ Module B (混合 RAG)   │
│  ├── index_invoice()  ──→ Module B (入库)       │
│  ├── run_task()       ──→ Module C (LLM Agent)  │
│  └── get_all_invoices()                         │
└───┬──────────────────┬──────────────────────────┘
    ▼                  ▼                  ▼
┌──────────┐  ┌──────────────┐  ┌──────────────────────┐
│ Module A │  │  Module B    │  │  Module C            │
│ PDF→JSON │  │  混合RAG检索  │  │  LLM智能分析         │
│ Qwen-VL  │  │  SQLite+     │  │  Gemini Agent +      │
│ DashScope│  │  ChromaDB    │  │  5 Skills            │
└──────────┘  └──────────────┘  └──────────────────────┘
```

**数据流：**
- **Upload tab**: PDF 文件 → `api_adapter.parse_invoice()` → Module A `parse_invoice_pdf()` → Qwen-VL 解析 → 中文 JSON → 索引到 Module B
- **Search tab**: 用户输入 → `api_adapter.query_invoices()` → Module B `InvoiceService.query_invoices()` → 混合检索 → 返回结果
- **Tasks tab**: 用户指令 → `api_adapter.run_task()` → Module C `agent_router()` → 技能执行 → 返回结果
- **首次启动**: adapter 自动将 `data/*.json`（30 张发票）索引到 Module B

---

## Module A 集成细节

### PDF 解析流程
```
PDF 文件
   │  [PyMuPDF]  每页渲染为 200 DPI 的 PNG
   ▼
PNG 图片列表
   │  [DashScope API]  发送到 Qwen-VL-Max 多模态大模型
   ▼
结构化 JSON（中文字段，与 data/*.json 格式一致）
   │  [api_adapter]  转换 + 索引
   ▼
Module B（SQLite + ChromaDB）+ Module C（会话发票列表）
```

### 发票 JSON 格式
解析器输出的 JSON 与 `data/*.json` 完全一致（中文键名）：
```json
{
  "发票类型": "增值税专用发票",
  "发票号码": "12345678",
  "开票日期": "2024-09-23",
  "购买方名称": "某某公司",
  "销售方名称": "供应商有限公司",
  "项目明细": [{"项目名称": "办公用品", "金额": "1000.00", ...}],
  "价税合计小写": "1130.00",
  ...
}
```

---

## 文件结构

```
mini-demo/
├── app.py                    # Streamlit 入口 + 侧边栏配置状态
├── config.py                 # 统一 API 密钥配置（Gemini + DashScope）
├── module_c_agent.py         # Module C: Gemini Agent + 5 技能
├── requirements.txt          # 依赖列表
├── parser/                   # Module A: PDF 发票解析
│   ├── __init__.py
│   └── invoice_parser.py     # Qwen-VL 多模态解析核心逻辑
├── data/                     # 30 张发票 JSON（中文字段）
├── services/
│   └── api_adapter.py        # 适配层：连接 UI ↔ Module A/B/C
├── ui/
│   ├── upload_page.py        # Tab 1: 上传 PDF 并解析
│   ├── search_page.py        # Tab 2: 自然语言搜索
│   ├── tasks_page.py         # Tab 3: 智能分析任务
│   └── dashboard_page.py     # Tab 4: 数据看板
└── .module_b_data/           # Module B 运行时数据（自动创建，可删除重建）
    ├── invoices.db            # SQLite 结构化存储
    └── chroma/                # ChromaDB 向量存储
```

---

## 注意事项

- `.module_b_data/` 目录是运行时自动创建的，删除后下次启动会自动重建并重新索引
- PDF 解析需要调用 DashScope API，每张发票约 5-15 秒
- ChromaDB telemetry 警告（`Failed to send telemetry event`）是无害的，不影响功能
- Module C 的 Gemini API 如果遇到 429 限流，会自动重试并降级到本地计算
