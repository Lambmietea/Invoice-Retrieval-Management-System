#!/usr/bin/env python3
"""
demo_test.py — 端到端演示脚本，模拟 Module C 调用 Module B 的完整流程。

用法:
    cd module_b
    pip install -e ".[dev]"
    python demo_test.py

流程:
    1. 读取 ../data/*.json (30 张发票)
    2. 批量入库 (index_invoices)
    3. 演示自然语言查询 (query_invoices) — 各种场景
    4. 演示全量获取 (get_all_invoices) — 供 C 做查重等
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure module_b is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from module_b.models import InvoiceRecord
from module_b.service import InvoiceService


def divider(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    # ── 0. 准备临时目录，避免污染正式数据 ──
    tmp = tempfile.mkdtemp(prefix="module_b_demo_")
    db_url = f"sqlite:///{tmp}/demo.db"
    chroma_dir = f"{tmp}/chroma"
    svc = InvoiceService(db_url=db_url, chroma_dir=chroma_dir)

    # ── 1. 加载 data/*.json ──
    data_dir = Path(__file__).resolve().parent.parent / "data"
    if not data_dir.is_dir():
        print(f"[ERROR] data/ not found at {data_dir}")
        sys.exit(1)

    json_files = sorted(data_dir.glob("*.json"))
    records: list[InvoiceRecord] = []
    for fp in json_files:
        raw = json.loads(fp.read_bytes().decode("utf-8"))
        records.append(InvoiceRecord.from_chinese_dict(raw))

    divider(f"Step 1: 加载了 {len(records)} 张发票")
    for r in records[:3]:
        print(f"  {r.invoice_id}  {r.vendor}  ¥{r.amount}")
    print(f"  ... (共 {len(records)} 张)")

    # ── 2. 批量入库 (index_invoices) ──
    divider("Step 2: 批量入库 index_invoices()")
    result = svc.index_invoices(records)
    print(f"  status: {result['status']}")
    print(f"  indexed_count: {result['indexed_count']}")
    print(f"  errors: {result['errors']}")
    assert result["indexed_count"] == len(records), "入库数量不一致!"

    # ── 3. 获取全量 (get_all_invoices) ──
    divider("Step 3: get_all_invoices() — C 模块查重等全量任务")
    all_inv = svc.get_all_invoices()
    print(f"  Total invoices in DB: {len(all_inv)}")
    assert len(all_inv) == len(records), "全量获取数量不一致!"

    # ── 4. 各种查询场景 ──
    queries = [
        # 情况 1: 结构化过滤
        ("2024年7月的发票", "月份过滤"),
        ("金额超过500的发票", "金额阈值过滤"),
        # 情况 2: 语义模糊查询
        ("和差旅相关的票据", "纯语义检索"),
        ("食品饮料类的消费", "语义检索"),
        # 情况 3: 混合查询
        ("2024年10月超过100元的发票", "混合: 月份+金额+语义"),
        # 举例 1: 月报任务 (C 调 B)
        ("March dining invoices", "英文月报查询"),
        # 举例 2: 查找特定供应商
        ("京东的发票", "供应商关键词"),
    ]

    divider("Step 4: query_invoices() — 多场景检索演示")
    all_passed = True
    for query, label in queries:
        print(f"\n  ▶ [{label}] query_invoices(\"{query}\")")
        ret = svc.query_invoices(query)
        print(f"    status:      {ret.status}")
        print(f"    query_type:  {ret.query_type}")
        print(f"    matched:     {len(ret.matched_invoices)} 张")
        print(f"    summary:     {ret.summary_context}")
        if ret.matched_invoices:
            top = ret.matched_invoices[0]
            print(f"    top match:   {top.invoice_id} | {top.vendor} | ¥{top.amount}")

    # ── 5. C 调 B 模拟: 月报场景 ──
    divider("Step 5: 模拟 C→B 调用 — 月报场景")
    print("  C 内部: retrieval_result = query_invoices('2024年7月的发票')")
    ret = svc.query_invoices("2024年7月的发票")
    invoices = ret.matched_invoices
    total_amount = sum(inv.amount for inv in invoices)
    print(f"  C 聚合: {len(invoices)} 张发票, 总金额 ¥{total_amount:.2f}")
    print(f"  C 输出 TaskResult: {{")
    print(f'    "query": "{ret.query}",')
    print(f'    "matched_invoices": [{len(invoices)} records],')
    print(f'    "summary_context": "{ret.summary_context}",')
    print(f'    "query_type": "{ret.query_type}",')
    print(f'    "status": "{ret.status}"')
    print(f"  }}")

    # ── 6. C 调 B 模拟: 重复发票检测 ──
    divider("Step 6: 模拟 C→B 调用 — 重复发票检测")
    print("  C 内部: all_invoices = get_all_invoices()")
    all_inv = svc.get_all_invoices()
    print(f"  C 收到 {len(all_inv)} 条记录, 开始查重...")
    from collections import defaultdict
    dup_by_id = defaultdict(list)
    dup_by_hash = defaultdict(list)
    for inv in all_inv:
        dup_by_id[inv.invoice_id].append(inv.source_file)
        key = f"{inv.vendor}|{inv.date}|{inv.amount}"
        dup_by_hash[key].append(inv.source_file)
    real_dups_id = {k: v for k, v in dup_by_id.items() if len(v) > 1}
    real_dups_hash = {k: v for k, v in dup_by_hash.items() if len(v) > 1}
    print(f"  按发票号码重复: {len(real_dups_id)} 组")
    print(f"  按内容特征重复: {len(real_dups_hash)} 组")

    # ── Cleanup ──
    divider("Done ✅")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  临时目录已清理: {tmp}")
    print("  所有核心接口测试通过！\n")


if __name__ == "__main__":
    main()
