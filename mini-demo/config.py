"""
Unified API configuration for the mini-demo application.

All API keys and model names are read from environment variables,
with hardcoded defaults for local demo use.

Environment Variables:
    GEMINI_API_KEY    — Google Gemini API key (Module C: LLM agent)
    GEMINI_MODEL      — Gemini model name (default: gemini-2.5-flash)
    DASHSCOPE_API_KEY — Alibaba DashScope API key (Module A: PDF parsing via Qwen-VL)
    DASHSCOPE_MODEL   — DashScope vision model name (default: qwen-vl-max-2025-04-02)
    MODULE_B_API_KEY  — Module B query parser LLM API key (OpenAI-compatible)
    MODULE_B_API_BASE — Module B query parser LLM base URL
    MODULE_B_MODEL    — Module B query parser LLM model name
"""

from __future__ import annotations

import os



GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


MODULE_B_API_KEY: str = os.environ.get("MODULE_B_API_KEY", "")
MODULE_B_API_BASE: str = os.environ.get(
    "MODULE_B_API_BASE", "https://open.bigmodel.cn/api/paas/v4"
)
MODULE_B_MODEL: str = os.environ.get("MODULE_B_MODEL", "glm-4.5-air")


DASHSCOPE_API_KEY: str = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_MODEL: str = os.environ.get(
    "DASHSCOPE_MODEL", "qwen-vl-max-2025-04-02"
)
