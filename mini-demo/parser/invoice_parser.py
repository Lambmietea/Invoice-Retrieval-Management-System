"""
Module A — PDF invoice parser using Alibaba Qwen-VL multimodal LLM.

Pipeline:  PDF → PyMuPDF renders pages to PNG → Qwen-VL-Max extracts structured JSON.

All configuration (API key, model name) is read from ``config.py``.
"""

import os
import re
import json
import time
import hashlib
import tempfile
from datetime import datetime
from typing import Any, Dict, List

import fitz  # PyMuPDF
from dashscope import MultiModalConversation

import config


def calculate_file_md5(file_path: str) -> str:
    """Calculate file MD5 for metadata."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def build_prompt() -> str:
    """Build the prompt for the multimodal model."""
    return """
你是一个发票信息抽取助手。

请从输入的中文发票图片中提取信息，并严格返回 JSON。
只返回合法 JSON，不要输出解释，不要输出 markdown，不要输出代码块。

返回字段必须严格包含以下 JSON 结构：
{
  "发票类型": "",
  "发票号码": "",
  "开票日期": "",
  "购买方名称": "",
  "购买方税号": "",
  "销售方名称": "",
  "销售方税号": "",
  "项目明细": [
    {
      "项目名称": "",
      "规格型号": "",
      "单位": "",
      "数量": "",
      "单价": "",
      "金额": "",
      "税率": "",
      "税额": ""
    }
  ],
  "价税合计大写": "",
  "价税合计小写": "",
  "备注": "",
  "开票人": "",
  "二维码内容": ""
}

规则：
1. 缺失字段填空字符串 ""。
2. 项目明细必须为数组，即使只有一项。
3. 不要遗漏负数金额或负数税额项目。
4. 只输出 JSON。
""".strip()


def pdf_to_image_paths(pdf_path: str, dpi: int = 200) -> List[str]:
    """Render each PDF page to a PNG image, return absolute paths."""
    doc = fitz.open(pdf_path)
    temp_dir = tempfile.mkdtemp(prefix="invoice_pages_")
    image_paths: List[str] = []

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image_path = os.path.abspath(
            os.path.join(temp_dir, f"page_{page_index + 1}.png")
        )
        pix.save(image_path)
        image_paths.append(image_path)

    doc.close()
    return image_paths


def build_multimodal_messages(image_paths: List[str], prompt: str) -> List[Dict[str, Any]]:
    """Build multimodal input messages for DashScope API."""
    content: List[Dict[str, str]] = []

    for image_path in image_paths:
        abs_path = os.path.abspath(image_path)
        content.append({"image": f"file://{abs_path}"})

    content.append({"text": prompt})

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def extract_text_from_dashscope_response(response: Any) -> str:
    """Extract text content from DashScope API response."""
    try:
        choices = response.output.choices
        if choices:
            message = choices[0].message
            content = message.content

            if isinstance(content, str):
                return content.strip()

            if isinstance(content, list):
                text_parts: List[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if text:
                            text_parts.append(text)
                    else:
                        text = getattr(item, "text", None)
                        if text:
                            text_parts.append(text)
                return "\n".join(text_parts).strip()
    except Exception:
        pass

    try:
        if isinstance(response, dict):
            content = response["output"]["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("text"):
                        text_parts.append(item["text"])
                return "\n".join(text_parts).strip()
    except Exception:
        pass

    raise ValueError("Failed to extract text content from DashScope response.")


def clean_json_text(raw_text: str) -> str:
    """Clean model output, extract JSON body."""
    if not isinstance(raw_text, str):
        raise ValueError("Model output is not a string.")

    text = raw_text.strip()

    # Remove markdown code blocks
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Extract outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return text.strip()


def parse_json_safely(raw_text: str) -> Dict[str, Any]:
    """Parse model output into JSON dict."""
    cleaned = clean_json_text(raw_text)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output is not valid JSON:\n{cleaned}") from e

    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON is not an object.")

    return parsed


def call_multimodal_model_once(pdf_path: str, prompt: str) -> Dict[str, Any]:
    """Single call to the multimodal model."""
    api_key = config.DASHSCOPE_API_KEY
    model = config.DASHSCOPE_MODEL

    if not api_key:
        raise EnvironmentError(
            "DASHSCOPE_API_KEY is not set. Please set it in environment variables or config.py."
        )

    image_paths = pdf_to_image_paths(pdf_path)
    messages = build_multimodal_messages(image_paths, prompt)

    response = MultiModalConversation.call(
        api_key=api_key,
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )

    status_code = getattr(response, "status_code", None)
    if status_code is not None and status_code != 200:
        message = getattr(response, "message", "Unknown DashScope error")
        raise RuntimeError(
            f"DashScope API error: status_code={status_code}, message={message}"
        )

    raw_text = extract_text_from_dashscope_response(response)
    return parse_json_safely(raw_text)


def call_multimodal_model(pdf_path: str, prompt: str, max_retries: int = 2) -> Dict[str, Any]:
    """Call multimodal model with retry logic."""
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return call_multimodal_model_once(pdf_path, prompt)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1.5)
            else:
                break

    raise RuntimeError(f"Failed to parse invoice after retries: {last_error}") from last_error


def normalize_item(item: Dict[str, Any]) -> Dict[str, str]:
    """Normalize a single line item, ensuring all fields exist."""
    return {
        "项目名称": str(item.get("项目名称", "") or ""),
        "规格型号": str(item.get("规格型号", "") or ""),
        "单位": str(item.get("单位", "") or ""),
        "数量": str(item.get("数量", "") or ""),
        "单价": str(item.get("单价", "") or ""),
        "金额": str(item.get("金额", "") or ""),
        "税率": str(item.get("税率", "") or ""),
        "税额": str(item.get("税额", "") or ""),
    }


def normalize_result(parsed_result: Dict[str, Any], pdf_path: str) -> Dict[str, Any]:
    """Normalize model output, fill missing fields, append metadata."""
    if not isinstance(parsed_result, dict):
        parsed_result = {}

    items = parsed_result.get("项目明细", [])
    if not isinstance(items, list):
        items = []

    normalized_items: List[Dict[str, str]] = []
    for item in items:
        if isinstance(item, dict):
            normalized_items.append(normalize_item(item))

    file_md5 = calculate_file_md5(pdf_path)
    process_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    normalized = {
        "发票类型": str(parsed_result.get("发票类型", "") or ""),
        "发票号码": str(parsed_result.get("发票号码", "") or ""),
        "开票日期": str(parsed_result.get("开票日期", "") or ""),
        "购买方名称": str(parsed_result.get("购买方名称", "") or ""),
        "购买方税号": str(parsed_result.get("购买方税号", "") or ""),
        "销售方名称": str(parsed_result.get("销售方名称", "") or ""),
        "销售方税号": str(parsed_result.get("销售方税号", "") or ""),
        "项目明细": normalized_items,
        "价税合计大写": str(parsed_result.get("价税合计大写", "") or ""),
        "价税合计小写": str(parsed_result.get("价税合计小写", "") or ""),
        "备注": str(parsed_result.get("备注", "") or ""),
        "开票人": str(parsed_result.get("开票人", "") or ""),
        "二维码内容": str(parsed_result.get("二维码内容", "") or ""),
        "文件路径": pdf_path,
        "处理时间": process_time,
        "文件MD5": file_md5,
    }

    return normalized


def parse_invoice_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Main public API — parse a PDF invoice into structured JSON.

    Args:
        pdf_path: Path to the PDF invoice file.

    Returns:
        dict: Structured Chinese-keyed invoice JSON (same format as data/*.json).
    """
    if not isinstance(pdf_path, str) or not pdf_path.strip():
        raise ValueError("pdf_path must be a non-empty string.")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not os.path.isfile(pdf_path):
        raise ValueError(f"Path is not a file: {pdf_path}")

    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError("Input file must be a PDF file.")

    prompt = build_prompt()
    parsed_result = call_multimodal_model(pdf_path, prompt)
    final_result = normalize_result(parsed_result, pdf_path)

    return final_result
