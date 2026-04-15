from google import genai
import json
import os
import glob
import time
from collections import defaultdict
import re
import csv
import io

import config

# ==========================================
# Gemini API 配置区
# ==========================================
GEMINI_API_KEY = config.GEMINI_API_KEY
GEMINI_MODEL = config.GEMINI_MODEL

try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"[WARNING] 初始化 Gemini Client 失败: {e}")
    gemini_client = None

_GEMINI_MAX_RETRIES = 5
_GEMINI_BASE_WAIT_S = 2.0


def _is_quota_error(exc_or_text: str) -> bool:
    s = str(exc_or_text)
    low = s.lower()
    return (
        "429" in s
        or "resource_exhausted" in low
        or "quota" in low
        or "rate limit" in low
    )


def call_gemini_api(prompt: str) -> str:
    """调用 Gemini；对 429/配额类错误自动退避重试，失败时返回 JSON 错误串（供上层降级）。"""
    if not gemini_client:
        return json.dumps({"error": "API Client 未成功初始化，请检查 API KEY"}, ensure_ascii=False)
    last_err: Exception | None = None
    for attempt in range(_GEMINI_MAX_RETRIES):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            text = (response.text or "").strip()
            return text
        except Exception as e:
            last_err = e
            err_s = str(e)
            if _is_quota_error(err_s) and attempt < _GEMINI_MAX_RETRIES - 1:
                wait = _GEMINI_BASE_WAIT_S * (2**attempt)
                m = re.search(r"retry in ([\d.]+)\s*s", err_s, re.I)
                if m:
                    wait = max(float(m.group(1)), 1.0) + 0.75
                wait = min(wait, 90.0)
                print(
                    f"[WARN] Gemini 配额/限流 ({attempt + 1}/{_GEMINI_MAX_RETRIES})，"
                    f"{wait:.1f}s 后重试…"
                )
                time.sleep(wait)
                continue
            print(f"[ERROR] Gemini API 调用失败: {e}")
            return json.dumps({"error": err_s}, ensure_ascii=False)
    return json.dumps({"error": str(last_err)}, ensure_ascii=False)


def _gemini_text_failed(text: str) -> bool:
    """判断模型输出是否为错误占位或配额报错原文（避免整段 JSON 显示在 UI）。"""
    if not (text or "").strip():
        return True
    t = text.strip()
    if "RESOURCE_EXHAUSTED" in t or ("429" in t and "quota" in t.lower()):
        return True
    if t.startswith("{") and '"error"' in t:
        try:
            obj = json.loads(t)
            if isinstance(obj, dict) and obj.get("error"):
                return True
        except json.JSONDecodeError:
            pass
    return False


def _fallback_narration(task_label: str) -> str:
    return (
        f"⚠️ **{task_label}**：大模型接口暂时不可用（常见原因为免费额度用尽或触发限流）。\n\n"
        "下方 **数据摘要 / 结构化结果** 均为本地计算生成，数据仍然有效。稍后可更换 API Key 或开通计费后重试文字解读。"
    )


def agent_router_keyword_fallback(user_query: str) -> str:
    """Gemini 不可用时，用语义关键词粗分路由（与 Router prompt 技能列表一致）。"""
    q = user_query or ""
    if any(k in q for k in ("报销", "台账", "Excel", "excel", "CSV", "csv", "导出", "贴excel")):
        return "reimbursement_form"
    if any(k in q for k in ("画像", "谁赚", "赚走", "去哪花", "开销去向", "主要花")):
        return "vendor_profiling"
    if any(k in q for k in ("重复", "查重", "一样", "连号")):
        return "duplicate_detection"
    if any(k in q for k in ("异常", "有问题", "离谱", "不全", "缺失")):
        return "anomaly_detection"
    return "aggregation"

# ==========================================
# 大脑层 (Agent Router)
# ==========================================
def agent_router(user_query: str) -> str:
    """Agent的大脑：决定用户的意图属于哪个Skill"""
    prompt = f"""
    你是一个发票数据分析智能体的主路由(Router)。你的任务是理解用户的指令，并将其分配给最合适的专业技能(Skill)。
    可用技能列表：
    1. "aggregation" : 进行求和、计数、统计金额、按类别/销售方汇总等。 (如："一共花了多少钱"、"打印相关的开销")
    2. "duplicate_detection" : 查找重复报销、连号发票、重复的发票等。 (如："有没有重复交的发票？"、"查一下单号一样的")
    3. "anomaly_detection" : 检测异常数据、金额过大、必填项缺失、日期不对等。 (如："发票找一下有没有问题"、"哪张发票金额大得离谱？")
    4. "vendor_profiling" : 消费行为画像，分析钱主要花在哪里了，谁是最大的收款方，消费趋势等。(如："钱都被谁赚走了"、"帮我画个消费画像"、"主要的开销去向")
    5. "reimbursement_form" : 生成报销台账，整理成表格或文件导出格式。(如："帮我生成报销单"、"导出这批数据的台账"、"整理成Excel表")

    用户指令: "{user_query}"
    
    请只输出一个字符串，必须是上述5个技能名之一。不要输出任何其他多余的字符。
    """
    response = call_gemini_api(prompt)
    if not _gemini_text_failed(response):
        skill_name = response.strip().lower()
        valid_skills = [
            "aggregation",
            "duplicate_detection",
            "anomaly_detection",
            "vendor_profiling",
            "reimbursement_form",
        ]
        for valid_skill in valid_skills:
            if valid_skill in skill_name:
                return valid_skill
    return agent_router_keyword_fallback(user_query)

# ==========================================
# 技能 1: 聚合统计 (Aggregation Skill)
# ==========================================
def extract_aggregation_intent(user_query: str) -> dict:
    prompt = f"""
    你是一个数据分析助手。提取聚合统计的关键参数。可用的字段有: ["销售方名称", "购买方名称", "开票日期", "项目名称", "发票类型"] 等。
    用户指令: "{user_query}"
    请输出JSON：
    {{
        "group_by": "项目名称", 
        "calculate_field": "amount", 
        "operation": "sum",
        "filter_keywords": ["打印", "复印"] 
    }}
    如果不分组，group_by 返回 null。若有语义条件请联想拓展 filter_keywords，否则为空数组 []。
    """
    response_text = call_gemini_api(prompt)
    cleaned_text = re.sub(r'```json\s*|```\s*', '', response_text, flags=re.IGNORECASE).strip()
    try:
        data = json.loads(cleaned_text)
        if not isinstance(data, dict) or data.get("error"):
            raise ValueError("invalid intent payload")
        return data
    except (json.JSONDecodeError, ValueError, TypeError):
        return _aggregation_intent_from_keywords(user_query)


def _aggregation_intent_from_keywords(user_query: str) -> dict:
    """Gemini 不可用时，从中文问题里猜分组字段。"""
    q = user_query or ""
    group_by = None
    if any(k in q for k in ("销售方", "商家", "供应商", "卖方", "收款方")):
        group_by = "销售方名称"
    elif "购买方" in q or "买方" in q:
        group_by = "购买方名称"
    elif "项目" in q:
        group_by = "项目名称"
    elif any(k in q for k in ("开票日期", "按日", "按月份", "按月")):
        group_by = "开票日期"
    elif "发票类型" in q or ("类型" in q and "发票" in q):
        group_by = "发票类型"
    return {
        "group_by": group_by,
        "calculate_field": "amount",
        "operation": "sum",
        "filter_keywords": [],
    }

def aggregate_skill(user_query: str, invoices_data: list) -> dict:
    print("[Agent] 🧠 正在执行技能：数据聚合 (Aggregation)")
    intent = extract_aggregation_intent(user_query)
    
    group_by_col = intent.get("group_by")
    operation = intent.get("operation", "sum")
    filter_keywords = intent.get("filter_keywords", [])
    if isinstance(filter_keywords, str): filter_keywords = [filter_keywords]
    
    is_item_level = group_by_col in ["项目名称", "规格型号", "单价", "数量"]
    result_data = defaultdict(float) if operation == "sum" else defaultdict(int)
    
    for inv in invoices_data:
        if is_item_level:
            for item in inv.get("项目明细", []):
                item_str = str(item)
                inv_str = str(inv)
                if filter_keywords and not any(kw in item_str or kw in inv_str for kw in filter_keywords): continue
                amount_val = float(re.sub(r'[^\d.-]', '', str(item.get("金额", "0"))) or 0)
                group_key = item.get(group_by_col, "未归类") if group_by_col else "整体总计"
                result_data[group_key] += amount_val if operation == "sum" else 1
        else:
            inv_str = str(inv)
            if filter_keywords and not any(kw in inv_str for kw in filter_keywords): continue
            amount_val = float(re.sub(r'[^\d.-]', '', str(inv.get("价税合计小写", "0"))) or 0)
            group_key = inv.get(group_by_col, "未归类/整体") if group_by_col else "整体总计"
            result_data[group_key] += amount_val if operation == "sum" else 1
            
    calc_result = {str(k): round(v, 2) for k, v in result_data.items()} if operation == "sum" else dict(result_data)
    
    prompt_report = f'用户问题: "{user_query}"\n统计结果 JSON: {json.dumps(calc_result, ensure_ascii=False)}\n请生成自带Markdown表格的人类可读汇报。'
    message = call_gemini_api(prompt_report)
    status = "success"
    if _gemini_text_failed(message):
        message = _fallback_narration("聚合统计")
        status = "warning"

    return {"task_type": "aggregation", "status": status, "message": message, "result": calc_result}

# ==========================================
# 技能 2: 发票查重 (Duplicate Detection Skill)
# ==========================================
def check_duplicates_skill(user_query: str, invoices_data: list) -> dict:
    print("[Agent] 🧠 正在执行技能：发票查重 (Duplicate Detection)")
    
    # Python 纯逻辑查重
    duplicates_by_id = defaultdict(list)
    duplicates_by_hash = defaultdict(list)
    
    for idx, inv in enumerate(invoices_data):
        inv_id = inv.get("发票号码", "")
        vendor = inv.get("销售方名称", "")
        date = inv.get("开票日期", "")
        amount = inv.get("价税合计小写", "")
        
        # 策略1: 相同的发票号码 (100% 重复)
        if inv_id:
            duplicates_by_id[inv_id].append(inv.get("文件路径", f"未知文件_{idx}"))
            
        # 策略2: 相同的供应商 + 日期 + 金额 (极高概率重复报销)
        if vendor and date and amount:
            key_hash = f"{vendor}|{date}|{amount}"
            duplicates_by_hash[key_hash].append(inv.get("文件路径", f"未知文件_{idx}"))
            
    # 筛选出数量 > 1 的组
    real_duplicates = {
        "完全相同的发票号码": {k: v for k, v in duplicates_by_id.items() if len(v) > 1},
        "内容高度相似的发票(同日期同商家同金额)": {k: v for k, v in duplicates_by_hash.items() if len(v) > 1}
    }
    
    # 统计一下是否找到了重复
    found_duplicates = bool(real_duplicates["完全相同的发票号码"] or real_duplicates["内容高度相似的发票(同日期同商家同金额)"])
    
    prompt_report = f"""
    用户问题: "{user_query}"
    查重系统结果 JSON: {json.dumps(real_duplicates, ensure_ascii=False)}
    请根据发现的重复数据，告诉用户查重结果。如果没有重复，请反馈没发现重复项；如果发现了，请列出具有重复嫌疑的文件路径。
    """
    message = call_gemini_api(prompt_report)
    status = "success"
    if _gemini_text_failed(message):
        message = _fallback_narration("发票查重")
        status = "warning"

    return {
        "task_type": "duplicate_detection",
        "status": status,
        "message": message,
        "result": real_duplicates,
    }

# ==========================================
# 技能 3: 异常检测 (Anomaly Detection Skill)
# ==========================================
def detect_anomalies_skill(user_query: str, invoices_data: list) -> dict:
    print("[Agent] 🧠 正在执行技能：异常检测 (Anomaly Detection)")
    
    anomalies = []
    for idx, inv in enumerate(invoices_data):
        file_path = inv.get("文件路径", f"未知发票_{idx}")
        inv_anomalies = []
        
        # 规则 1: 缺少关键项
        if not inv.get("发票号码"): inv_anomalies.append("缺少'发票号码'")
        if not inv.get("销售方名称"): inv_anomalies.append("缺少'销售方名称'")
        
        # 规则 2: 金额异常判断
        amount_str = str(inv.get("价税合计小写", "0"))
        try:
            amount_val = float(re.sub(r'[^\d.-]', '', amount_str))
            if amount_val > 10000:  
                inv_anomalies.append(f"单张发票金额过大 (¥{amount_val})")
            if amount_val <= 0:
                inv_anomalies.append(f"发票金额为零或负数 (¥{amount_val})")
        except:
             inv_anomalies.append("发票金额格式无法识别")
             
        if inv_anomalies:
            anomalies.append({
                "文件": file_path,
                "开票方": inv.get("销售方名称", "未知"),
                "异常明细": inv_anomalies
            })
            
    prompt_report = f"""
    用户问题: "{user_query}"
    系统发现的异常发票数据 JSON: {json.dumps(anomalies, ensure_ascii=False)}
    请根据上述数据生成一份发票合规异常检测报告。如果没有异常，请安抚用户；如果有异常，请指出哪些文件存在什么异常。
    """
    message = call_gemini_api(prompt_report)
    status = "success"
    if _gemini_text_failed(message):
        message = _fallback_narration("异常检测")
        status = "warning"

    return {
        "task_type": "anomaly_detection",
        "status": status,
        "message": message,
        "result": anomalies,
    }

# ==========================================
# 技能 4: 消费行为画像 (Vendor Profiling Skill)
# ==========================================
def vendor_profiling_skill(user_query: str, invoices_data: list) -> dict:
    print("[Agent] 🧠 正在执行技能：消费行为画像 (Vendor Profiling)")
    
    vendor_stats = defaultdict(lambda: {"amount": 0.0, "count": 0})
    total_amount = 0.0
    
    for inv in invoices_data:
        vendor = inv.get("销售方名称", "未知销售方")
        amount_str = str(inv.get("价税合计小写", "0"))
        try:
            amount_val = float(re.sub(r'[^\d.-]', '', amount_str))
        except:
            amount_val = 0.0
        
        vendor_stats[vendor]["amount"] += amount_val
        vendor_stats[vendor]["count"] += 1
        total_amount += amount_val
        
    # 按金额倒序列出商户
    sorted_vendors = sorted(vendor_stats.items(), key=lambda x: x[1]["amount"], reverse=True)
    # 取 Top 5
    top_vendors = {k: {"总金额": round(v["amount"], 2), "发票张数": v["count"]} for k, v in sorted_vendors[:5]}
    
    summary_data = {
        "总消费金额": round(total_amount, 2),
        "总交易方数量": len(vendor_stats),
        "Top5开销去向": top_vendors
    }
    
    prompt_report = f"""
    用户问题: "{user_query}"
    系统计算的消费画像数据 JSON: {json.dumps(summary_data, ensure_ascii=False)}
    请根据上述数据，生成一段生动有趣的“消费画像报告”（类似于支付宝/微信年度账单的口吻）。
    指出用户最大的开销去向，评价一下发票的集中度。一定要使用 Markdown 格式加粗关键数据，附带一张表格。
    """
    message = call_gemini_api(prompt_report)
    status = "success"
    if _gemini_text_failed(message):
        message = _fallback_narration("消费画像")
        status = "warning"

    return {
        "task_type": "vendor_profiling",
        "status": status,
        "message": message,
        "result": summary_data,
    }

# ==========================================
# 技能 5: 一键生成报销台账 (Reimbursement Form Skill)
# ==========================================
def reimbursement_form_skill(user_query: str, invoices_data: list) -> dict:
    print("[Agent] 🧠 正在执行技能：生成报销台账 (Reimbursement Form)")
    
    records = []
    for inv in invoices_data:
        # 尝试提取主要项目名称（取第一个项目的名称作为代表）
        items = inv.get("项目明细", [])
        main_item = items[0].get("项目名称", "无明细") if items else "无明细"
        
        amount_str = str(inv.get("价税合计小写", "0"))
        try:
            amount_val = float(re.sub(r'[^\d.-]', '', amount_str))
        except:
            amount_val = 0.0
            
        records.append({
            "开票日期": inv.get("开票日期", ""),
            "发票号码": inv.get("发票号码", ""),
            "发票类型": inv.get("发票类型", ""),
            "销售方名称": inv.get("销售方名称", ""),
            "主要消费项目": main_item.strip('*'), # 去掉明细前面可能带的星号
            "总金额(元)": round(amount_val, 2),
            "备注": inv.get("备注", "")
        })
        
    # 时间正序预整理
    records.sort(key=lambda x: x["开票日期"])
    
    # 内存中生成 CSV 字符串，准备送给 UI 下载 (用 io.StringIO)
    output = io.StringIO()
    if records:
        writer = csv.DictWriter(output, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    csv_content = output.getvalue()
    
    total_sum = sum(float(r["总金额(元)"]) for r in records)
    
    prompt_report = f"""
    用户问题: "{user_query}"
    系统已生成包含 {len(records)} 条记录的报销台账。合并总金额为: {round(total_sum, 2)} 元。
    以下是生成的部分数据预览 JSON: {json.dumps(records[:5], ensure_ascii=False)}
    请用专业、贴心的口吻告诉用户报销台账生成完毕：包含合计几张票，多少钱。
    并且用 Markdown 画一个前5条数据的预览表格。在最后提示用户："完整的 CSV 台账数据已在后台准备就绪，可通过 UI 点击下载导出为 Excel"。
    """
    message = call_gemini_api(prompt_report)
    status = "success"
    if _gemini_text_failed(message):
        message = _fallback_narration("报销台账")
        status = "warning"

    return {
        "task_type": "reimbursement_form",
        "status": status,
        "message": message,
        "result": {
            "total_records": len(records),
            "total_amount": round(total_sum, 2),
            "csv_data": csv_content,
        },
    }


# ==========================================
# Module C 主入口 (暴露给 Module D 的唯一接口)
# ==========================================
def run_task(command: str, invoices_data: list) -> dict:
    """Module C 的统一对外入口"""
    print(f"\n[🚀 Agent 启动] 接收到原始指令: '{command}'")
    
    # 1. 意图路由
    skill_name = agent_router(command)
    print(f"[Agent] 🧭 路由决策分配至目标技能: {skill_name}")
    
    # 2. 分配执行
    if skill_name == "duplicate_detection":
        return check_duplicates_skill(command, invoices_data)
    elif skill_name == "anomaly_detection":
        return detect_anomalies_skill(command, invoices_data)
    elif skill_name == "vendor_profiling":
        return vendor_profiling_skill(command, invoices_data)
    elif skill_name == "reimbursement_form":
        return reimbursement_form_skill(command, invoices_data)
    else:
        return aggregate_skill(command, invoices_data)

# ==========================================
# 调试模拟区域
# ==========================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    invoices = []
    # 这里我们加载所有的30张发票用于测试
    for fpath in json_files[:30]:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                invoices.append(json.load(f))
        except:
            pass
            
    print(f"✅ 成功模拟加载了 {len(invoices)} 张发票数据。\n")

    # 模拟用户的提问！Agent自动分发！
    test_queries = [
        "在这个周期内，我们的钱主要是被哪几个公司赚走了呀？给我画个消费画像看看。",
        "把这批发票数据帮我做成一份可以直接贴excel的报销单，需要有商家、日期、内容（如果有多个内容请用+连接）、发票号、金额。"
    ]
    
    for query in test_queries:
        print("="*60)
        # 调用的是对外的纯一接口 run_task！
        final_result = run_task(query, invoices)
        print("\n[🎯 最终结果（已向前端抛出汇报）]:")
        print(final_result.get("message"))
        print("="*60 + "\n")

if __name__ == "__main__":
    main()
