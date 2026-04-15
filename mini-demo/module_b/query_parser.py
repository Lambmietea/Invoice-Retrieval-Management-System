"""LLM-assisted query parsing — natural language → structured filters + semantic phrase.

Architecture (Dense Search RAG):
  1. **LLM parser (primary)** — understands user intent, extracts structured
     filters (date/amount/vendor → SQL) AND a single high-density semantic
     phrase (→ vector store Dense Search).  The phrase is NOT a keyword list
     but a compact noun-phrase that concentrates synonyms and hyponyms into
     one embedding-friendly sentence.
  2. **Rule-based parser (fallback)** — when LLM is unavailable, uses regex to
     extract structured filters, then strips matched spans from the query so
     that only the residual text goes to the vector store.

The key insight: the vector store receives ONE complete phrase, never scattered
keywords.  The retriever then applies absolute distance threshold + dynamic
score-gap cut-off to ensure high precision.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

# ---------------------------------------------------------------------------
# OpenAI-compatible client for Module B (optional — degrades gracefully)
# ---------------------------------------------------------------------------

_llm_client = None
_llm_model: str = ""


def _init_llm() -> None:
    """Lazily initialize the OpenAI-compatible client using centralized config.

    Imports ``config`` at call time so the module can be loaded even when
    ``config.py`` is not on the path (e.g. in unit-test environments).
    """
    global _llm_client, _llm_model
    if _llm_client is not None:
        return
    try:
        from config import MODULE_B_API_KEY, MODULE_B_API_BASE, MODULE_B_MODEL
        _llm_model = MODULE_B_MODEL
        if MODULE_B_API_KEY:
            from openai import OpenAI
            _llm_client = OpenAI(
                api_key=MODULE_B_API_KEY,
                base_url=MODULE_B_API_BASE,
                max_retries=1,
                timeout=20.0,
            )
    except Exception:
        pass


# Attempt init on import — if config.py is importable, client is ready
try:
    _init_llm()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Parsed query result
# ---------------------------------------------------------------------------


class ParsedQuery:
    """Structured output from the query parser."""

    def __init__(
        self,
        filters: dict[str, Any] | None = None,
        semantic_query: str | None = None,
        original: str = "",
    ) -> None:
        self.filters = filters or {}
        self.semantic_query = semantic_query or ""
        self.original = original

    @property
    def has_filters(self) -> bool:
        return bool(self.filters)

    @property
    def has_semantic(self) -> bool:
        return bool(self.semantic_query.strip())

    def __repr__(self) -> str:
        return f"ParsedQuery(filters={self.filters}, semantic={self.semantic_query!r})"


# ---------------------------------------------------------------------------
# LLM-based parser (primary — does ALL the intelligent work)
# ---------------------------------------------------------------------------

_LLM_PROMPT_TEMPLATE = """你是一个专业的发票查询意图翻译官。

你的任务：把用户的自然语言问题解析成 JSON，用于后续的发票检索。
输出 JSON 必须包含以下字段：
1) filters: 结构化过滤条件（用于 SQL 精确筛选）
2) semantic_query: 语义检索短语（用于向量库 Dense Search）
3) query_type: "filter" | "semantic" | "filter+semantic"

## semantic_query 的生成规则（最关键）：

当用户的提问无法完全映射到 filters，需要进行语义检索时，请将用户的
提问翻译成一句【高信息密度的标准财务描述句】。

### 核心规则：语义收敛（概念桶）

为了消除同义词在向量库中的搜索差异，你必须先判断用户意图属于哪个
「概念桶」，然后**强制输出该概念桶对应的标准短语**，绝不允许自由发挥。
同一个概念桶内，无论用户怎么换说法，你的输出必须完全一致（逐字相同）。

已知概念桶及其标准短语：

| 概念桶 | 用户可能的说法 | 标准输出（必须逐字一致） |
|--------|---------------|----------------------|
| 食物/餐饮 | 餐饮、食物、吃的、食品、零食、饮料、水果、下午茶、请客、嘴馋、超市买的吃的 | 关于餐饮、食品、饮料、零食等用于就餐和食用的消费支出 |
| 电子/数码 | 电子产品、电脑、外设、鼠标键盘、数码、充电器、U盘、数据线、手机 | 关于购买电脑、外设、手机、数码配件等电子设备的办公资产支出 |
| 差旅/交通 | 差旅、交通、打车、机票、高铁、住宿、酒店、出差、出行 | 关于机票、火车票、打车费、酒店住宿等差旅和交通出行支出 |
| 办公用品 | 办公用品、文具、打印、纸张、墨盒、办公耗材 | 关于文具、打印耗材、纸张等办公类用品支出 |
| 日用品 | 日用品、生活用品、卫生用品、洗涤、清洁 | 关于纸巾、洗涤用品、清洁用品等日常生活类消费支出 |
| 快递/物流 | 快递、物流、运费、邮费、顺丰、寄件 | 关于快递运输、物流运费等寄送类服务支出 |

### 匹配规则：
1. 先判断用户意图是否属于上述概念桶之一 → 如果是，直接输出该桶的标准短语，一字不改。
2. 如果用户的意图不属于任何已知概念桶 → 参照上述格式，自行浓缩成一句结构化标准财务描述句，格式为："关于 XX、XX、XX 等 YY 类 ZZ 支出"。
3. 如果用户提到的是非常具体的单一物品（如"纸巾""比萨""数据线"），也要归入最近的概念桶。

### 其他规则：
- 绝对不要输出散装关键词列表或数组！
- 去掉所有意图词（查询、搜索、帮我找……）和结构词（发票、票据……）。
- 去掉所有已放入 filters 的信息（日期、金额、供应商等不要重复出现）。
- 纯筛选条件（如日期、金额查询）不需要 semantic_query，设为空字符串。

## filters 字段:
- invoice_id: string
- vendor: string（供应商/卖方关键词，如"京东""山姆"）
- currency: string (如 USD, CNY, HKD)
- category: string（发票类型，如"电子发票""专用发票"）
- status: string
- month: string, 格式 YYYY-MM
- date_from: string, 格式 YYYY-MM-DD
- date_to: string, 格式 YYYY-MM-DD
- min_amount: number（金额下限）
- max_amount: number（金额上限）

## 约束:
- 只输出 JSON，不要输出解释文字，不要 markdown。
- 中文数字转阿拉伯数字（五百=500, 一千=1000）。
- 如果用户只有筛选条件，semantic_query 为空字符串。
- 如果用户只有语义需求，filters 为空对象。

用户问题:
{user_query}
"""

# Mapping from LLM output filter keys to internal filter keys used by storage
_LLM_FILTER_KEY_MAP = {
    "min_amount": "amount_gt",
    "max_amount": "amount_lt",
}


def _normalize_llm_filters(raw_filters: dict[str, Any]) -> dict[str, Any]:
    """Normalize LLM output filter keys to internal storage filter keys.

    Handles:
      - min_amount → amount_gt
      - max_amount → amount_lt
      - Removes keys with None / empty values
    """
    normalized: dict[str, Any] = {}
    for k, v in raw_filters.items():
        if v is None or v == "":
            continue
        mapped_key = _LLM_FILTER_KEY_MAP.get(k, k)
        normalized[mapped_key] = v
    return normalized


# ---------------------------------------------------------------------------
# LLM result cache + rate-limit cooldown
# ---------------------------------------------------------------------------
# Cache LLM results to avoid redundant API calls (especially with 429 limits).
# Also track recent failures to implement a cooldown period before retrying.

_llm_cache: dict[str, ParsedQuery] = {}
_llm_last_fail_time: float = 0.0
_LLM_COOLDOWN_SECONDS: float = 30.0  # wait 30s after a failure before retrying


def parse_query_llm(user_query: str) -> ParsedQuery | None:
    """Attempt LLM-based parsing via OpenAI-compatible API.

    Features:
      - Caches successful results (same query → instant cache hit, no API call)
      - Implements a cooldown after failures (avoids hammering a rate-limited API)

    Returns None on any failure.
    """
    global _llm_last_fail_time

    # Check cache first
    cache_key = user_query.strip()
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    # Cooldown: skip LLM if we recently got a failure (429, timeout, etc.)
    if _llm_last_fail_time > 0:
        elapsed = time.time() - _llm_last_fail_time
        if elapsed < _LLM_COOLDOWN_SECONDS:
            return None

    # Ensure client is initialized
    _init_llm()
    if not _llm_client:
        return None

    prompt = _LLM_PROMPT_TEMPLATE.format(user_query=user_query)
    try:
        response = _llm_client.chat.completions.create(
            model=_llm_model,
            messages=[
                {"role": "system", "content": "你是一个发票检索系统的查询解析助手。只输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        text = (response.choices[0].message.content or "").strip()
        # strip markdown code fences if any
        text = re.sub(r"```json\s*|```\s*", "", text, flags=re.I).strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            return None
        raw_filters = data.get("filters", {})
        normalized_filters = _normalize_llm_filters(raw_filters)
        # LLM outputs semantic_query (高密度短语), not keyword list
        semantic = data.get("semantic_query", "") or data.get("semantic_keywords", "")
        result = ParsedQuery(
            filters=normalized_filters,
            semantic_query=semantic,
            original=user_query,
        )
        # Cache on success
        _llm_cache[cache_key] = result
        _llm_last_fail_time = 0.0  # reset cooldown on success
        return result
    except Exception as e:
        print(f"[query_parser] LLM parsing failed: {e}")
        _llm_last_fail_time = time.time()
        return None


# ---------------------------------------------------------------------------
# Rule-based parser (fallback only — when LLM is unavailable)
# ---------------------------------------------------------------------------
# This parser uses regex to extract structured filters (dates, amounts,
# vendors, categories).  Each regex records its matched span so that
# those substrings can be stripped from the original query.  Whatever
# text remains after stripping is passed as the semantic query.
#
# This is intentionally simple — no hardcoded stop-word lists.  The LLM
# parser handles all the intelligent keyword extraction.  This fallback
# just ensures basic functionality when the LLM API is down.
# ---------------------------------------------------------------------------

_MONTH_MAP_CN = {
    "一月": "01", "二月": "02", "三月": "03", "四月": "04",
    "五月": "05", "六月": "06", "七月": "07", "八月": "08",
    "九月": "09", "十月": "10", "十一月": "11", "十二月": "12",
    "1月": "01", "2月": "02", "3月": "03", "4月": "04",
    "5月": "05", "6月": "06", "7月": "07", "8月": "08",
    "9月": "09", "10月": "10", "11月": "11", "12月": "12",
}

_MONTH_MAP_EN = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

# Chinese numeral → integer conversion

_CN_DIGIT_MAP = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}

_CN_UNIT_MAP = {
    "十": 10, "百": 100, "千": 1000, "万": 10000,
}


def _cn_number_to_int(cn: str) -> int | None:
    """Convert a Chinese numeral string to an integer.

    Handles common patterns like: 五百, 一千, 三万, 一万五千, 两百,
    五十, 十五, 三千五百, etc.

    Returns None if the string cannot be parsed.
    """
    if not cn:
        return None
    total = 0
    current = 0
    for ch in cn:
        if ch in _CN_DIGIT_MAP:
            current = _CN_DIGIT_MAP[ch]
        elif ch in _CN_UNIT_MAP:
            unit = _CN_UNIT_MAP[ch]
            if unit == 10000:  # 万 — multiplies everything accumulated so far
                total = (total + (current if current else 1)) * unit
                current = 0
            else:
                total += (current if current else 1) * unit
                current = 0
        else:
            return None  # unrecognised character
    total += current  # trailing digit without a unit (e.g. 三千五 → the 五)
    return total if total > 0 else None


def parse_query_rules(user_query: str, known_vendors: list[str] | None = None) -> ParsedQuery:
    """Best-effort regex extraction of structured filters (fallback when LLM is down).

    Strategy:
      1. Use regex to extract structured filters (dates, amounts, vendors, categories).
      2. Record matched spans.
      3. Strip matched spans from original query → residual text = semantic query.

    This avoids hardcoded stop-word lists.  The residual text may still contain
    noise, but the vector store distance threshold provides a safety net.

    Args:
        user_query: Raw user input.
        known_vendors: Optional list of known vendor names from the database.
    """
    q = user_query.strip()
    filters: dict[str, Any] = {}
    # Collect (start, end) spans of text consumed by filter extraction
    matched_spans: list[tuple[int, int]] = []

    # --- month detection ---
    m = re.search(r"(\d{4})[年\-/](\d{1,2})月?", q)
    if m:
        filters["month"] = f"{m.group(1)}-{int(m.group(2)):02d}"
        matched_spans.append(m.span())
    else:
        for cn, num in _MONTH_MAP_CN.items():
            idx = q.find(cn)
            if idx >= 0:
                filters["month"] = f"2024-{num}"
                matched_spans.append((idx, idx + len(cn)))
                break
        else:
            ql = q.lower()
            for en, num in _MONTH_MAP_EN.items():
                idx = ql.find(en)
                if idx >= 0:
                    filters["month"] = f"2024-{num}"
                    matched_spans.append((idx, idx + len(en)))
                    break

    # --- amount threshold (Arabic digits) ---
    m_gt = re.search(r"(?:超过|大于|高于|above|over|more\s+than|>)\s*(\d+(?:\.\d+)?)", q, re.I)
    if m_gt:
        filters["amount_gt"] = float(m_gt.group(1))
        matched_spans.append(m_gt.span())
    else:
        m_gt_cn = re.search(
            r"(?:超过|大于|高于)\s*([一二三四五六七八九十百千万零两]+)\s*(?:块|元|万元)?",
            q,
        )
        if m_gt_cn:
            val = _cn_number_to_int(m_gt_cn.group(1))
            if val is not None:
                filters["amount_gt"] = float(val)
                matched_spans.append(m_gt_cn.span())

    m_lt = re.search(r"(?:低于|小于|少于|below|under|less\s+than|<)\s*(\d+(?:\.\d+)?)", q, re.I)
    if m_lt:
        filters["amount_lt"] = float(m_lt.group(1))
        matched_spans.append(m_lt.span())
    else:
        m_lt_cn = re.search(
            r"(?:低于|小于|少于)\s*([一二三四五六七八九十百千万零两]+)\s*(?:块|元|万元)?",
            q,
        )
        if m_lt_cn:
            val = _cn_number_to_int(m_lt_cn.group(1))
            if val is not None:
                filters["amount_lt"] = float(val)
                matched_spans.append(m_lt_cn.span())

    # --- category keywords ---
    category_keywords = {
        "电子发票": "电子发票",
        "普通发票": "普通发票",
        "专用发票": "专用发票",
        "增值税专用": "增值税专用发票",
        "增值税普通": "增值税普通发票",
    }
    for kw, cat in category_keywords.items():
        idx = q.find(kw)
        if idx >= 0:
            filters["category"] = cat
            matched_spans.append((idx, idx + len(kw)))
            break

    # --- vendor detection ---
    vendor_kw = _extract_vendor_keyword(q, known_vendors)
    if vendor_kw:
        filters["vendor"] = vendor_kw
        idx = q.find(vendor_kw)
        if idx >= 0:
            matched_spans.append((idx, idx + len(vendor_kw)))

    # --- Build semantic query: strip matched spans from original ---
    semantic = _strip_spans(q, matched_spans)

    return ParsedQuery(filters=filters, semantic_query=semantic, original=q)


def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Remove matched spans from text, returning the residual.

    If stripping leaves nothing meaningful, returns the original text.
    """
    if not spans:
        return text
    # Sort and merge overlapping spans
    sorted_spans = sorted(spans, key=lambda s: s[0])
    merged: list[tuple[int, int]] = [sorted_spans[0]]
    for start, end in sorted_spans[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    # Build residual
    parts: list[str] = []
    prev_end = 0
    for start, end in merged:
        parts.append(text[prev_end:start])
        prev_end = end
    parts.append(text[prev_end:])
    residual = "".join(parts).strip()
    # Collapse whitespace
    residual = re.sub(r"\s+", " ", residual).strip()
    return residual if residual else text


# --- Vendor extraction helpers -------------------------------------------

# Well-known company / brand keywords that strongly signal a vendor query.
_WELL_KNOWN_BRANDS: list[str] = [
    "京东", "淘宝", "天猫", "拼多多", "苏宁", "国美", "美团", "饿了么",
    "滴滴", "顺丰", "圆通", "中通", "韵达", "申通", "百世", "宜家",
    "山姆", "沃尔玛", "华为", "小米", "联想", "戴尔", "苹果", "腾讯",
    "阿里", "百度", "字节", "抖音", "达美乐", "星巴克", "麦当劳",
    "肯德基", "瑞幸",
]

# Noise words that should NOT be treated as vendor names even if they appear
# in the query (structural words, company suffixes, city names).
_VENDOR_STOP_WORDS: set[str] = {
    "发票", "相关", "的", "有关", "关于", "查询", "搜索", "查找",
    "找", "所有", "全部", "哪些", "什么", "多少", "列出",
    "显示", "看看", "帮我", "请", "我要", "给我",
    "电子", "科技", "技术", "贸易", "商贸", "信息", "有限", "公司",
    "有限公司", "中国", "商务", "服务", "集团", "股份", "工商",
    "文化", "传媒", "网络", "数码", "实业", "投资", "咨询",
    "上海", "北京", "深圳", "广州", "杭州", "南京", "成都", "重庆",
    "昆山", "郑州", "长沙", "武汉", "西安", "天津", "苏州", "宁波",
    "厦门", "青岛", "大连", "廊坊", "江西", "浙江", "广东", "江苏",
}


def _extract_vendor_keyword(
    query: str, known_vendors: list[str] | None = None
) -> str | None:
    """Try to extract a vendor keyword from the user query.

    Two-pass approach:
      1. Check if any known vendor name (from DB) shares a significant
         substring with the query  →  return that substring.
      2. Check if any well-known brand keyword appears in the query.

    Returns the matched vendor keyword, or None.
    """
    if known_vendors:
        best_match: str | None = None
        best_len = 0
        for vendor_name in known_vendors:
            if not vendor_name:
                continue
            common = _longest_common_substring(query, vendor_name)
            if common and len(common) >= 2 and common not in _VENDOR_STOP_WORDS:
                if len(common) > best_len:
                    best_match = common
                    best_len = len(common)
        if best_match:
            return best_match

    for brand in _WELL_KNOWN_BRANDS:
        if brand in query:
            return brand

    return None


def _longest_common_substring(s1: str, s2: str) -> str | None:
    """Return the longest common substring between s1 and s2, or None."""
    if not s1 or not s2:
        return None
    m, n = len(s1), len(s2)
    longest = 0
    end_idx = 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > longest:
                    longest = curr[j]
                    end_idx = i
        prev = curr
    if longest == 0:
        return None
    return s1[end_idx - longest : end_idx]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def parse_query(user_query: str, known_vendors: list[str] | None = None) -> ParsedQuery:
    """
    Parse a natural-language query into structured filters + semantic keywords.

    Pipeline:
      1. Try LLM — it handles ALL intelligent work (intent understanding,
         filter extraction, keyword extraction).
      2. Fall back to rule-based regex — extracts basic filters and uses
         the residual text as semantic query.

    Args:
        user_query: Raw user input.
        known_vendors: Optional list of known vendor names.  Passed through to
            the rule-based parser for dynamic vendor matching.
    """
    # Try LLM first (primary path)
    result = parse_query_llm(user_query)
    if result is not None and (result.has_filters or result.has_semantic):
        return result
    # Fallback to rules (degraded mode)
    return parse_query_rules(user_query, known_vendors=known_vendors)
