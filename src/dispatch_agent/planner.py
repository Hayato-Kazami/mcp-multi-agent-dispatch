"""指令解析：把一句话变成结构化的意图 + 参数。

**为什么用规则而不是 LLM**：
这个仓库要验证的是编排与工具边界，加一个模型进来会让每个失败都变成
"模型没理解"这种不可定位的原因。规则解析在演示集上是确定性的，
评测数字才可复现。生产里这一层确实是 LoRA 微调的意图分类（见 README「与真实系统的差异」），
两者的接口一致：输入一句话，输出 intent + slots + 缺失项。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

TASK_RE = re.compile(r"T\d{4}")
ROBOT_RE = re.compile(r"R\d{3}")
NODE_RE = re.compile(r"\b([A-D]\d)\b")
LEVEL_RE = re.compile(r"(一级|二级|三级|[123]\s*级)")
WAREHOUSE_RE = re.compile(r"\bW(\d)\b")

INTENT_KEYWORDS = {
    "fault": ["故障", "报修", "坏了", "异常上报", "停机"],
    "path": ["路径", "怎么走", "多久到", "路线", "绕路"],
    "collab": ["协同", "统筹", "一起", "批量", "多个任务", "排队任务"],
    "assign": ["派", "分配", "指派", "给", "安排"],
    "query": ["查", "有哪些", "多少", "状态", "列表", "排队"],
}

LEVEL_MAP = {"一级": 1, "1级": 1, "二级": 2, "2级": 2, "三级": 3, "3级": 3}


@dataclass
class Plan:
    intent: str
    slots: dict = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    reason: str = ""
    write: bool = False


def _detect_intent(text: str) -> tuple[str, str]:
    for intent in ["fault", "path", "collab", "assign", "query"]:
        for keyword in INTENT_KEYWORDS[intent]:
            if keyword in text:
                return intent, f"命中关键词「{keyword}」"
    return "unknown", "没有命中任何意图关键词"


def parse(request: str, default_warehouse: str) -> Plan:
    text = request.strip()
    warehouse_match = WAREHOUSE_RE.search(text)
    warehouse_id = f"W{warehouse_match.group(1)}" if warehouse_match else default_warehouse
    slots: dict = {"warehouse_id": warehouse_id}

    task_match = TASK_RE.search(text)
    if task_match:
        slots["task_id"] = task_match.group(0)
    robot_match = ROBOT_RE.search(text)
    if robot_match:
        slots["robot_id"] = robot_match.group(0)
    nodes = NODE_RE.findall(text)
    if len(nodes) >= 2:
        slots["from_node"], slots["to_node"] = nodes[0], nodes[1]

    intent, reason = _detect_intent(text)
    missing: list[str] = []

    if intent == "path":
        for key in ("from_node", "to_node"):
            if key not in slots:
                missing.append(key)
    elif intent == "assign":
        if "task_id" not in slots:
            missing.append("task_id")
    elif intent == "fault":
        if "robot_id" not in slots:
            missing.append("robot_id")
        level_match = LEVEL_RE.search(text)
        if level_match:
            slots["level"] = LEVEL_MAP.get(level_match.group(1).replace(" ", ""), 2)
        else:
            slots["level"] = 2  # 未说明等级时按降效处理，属于保守默认
    elif intent == "unknown":
        missing.append("intent")

    return Plan(
        intent=intent,
        slots=slots,
        missing=missing,
        reason=reason,
        write=intent in {"assign", "fault", "collab"},
    )
