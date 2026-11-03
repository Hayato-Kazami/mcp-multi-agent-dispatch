"""LangGraph 编排：理解 → 提议 → 审批 → 执行 → 收尾。

三个必须讲清的设计点：

1. **写操作与只读操作走同一条图，但只有写操作进审批**。
   只读动作（查队列、算路径）直接执行，写动作必须经过 review 节点。
   把"要不要审批"做成节点级的判断，而不是让每个 agent 自己记得申请。

2. **人在环用 interrupt，不用"先返回再重新调一次"**。
   interrupt 会把状态存进 checkpointer，进程重启后还能从断点继续；
   这是"断点恢复"和"假装断点恢复"的区别。

3. **防死循环靠两样**：步数上限 + 动作指纹去重。
   只设步数上限会在"两个动作来回跳"时白跑满 8 步；
   只设指纹去重会在"每轮参数都不同"时停不下来。
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .agents import AssignAgent, CollabAgent, PathAgent
from .agents.base import ToolGateway
from .planner import parse

MAX_STEPS = 8
# 停滞断路器的第二个判据：连续这么多次工具调用失败就停手，转人工。
# 只有指纹去重挡不住"每轮参数都不同但一直在错"的情况（书里叫同一错误重复 k 轮）。
STALL_LIMIT = 2


class State(TypedDict, total=False):
    request: str
    user_id: str
    warehouse_id: str
    auto_approve: bool
    intent: str
    slots: dict
    missing: list
    trace: list
    pending: list
    approved: bool
    results: list
    steps: int
    seen: list
    done: bool
    answer: str
    proposals: list
    executed: list
    consecutive_failures: int
    stalled: bool


def _fingerprint(action: dict) -> str:
    return action["tool"] + ":" + json.dumps(action.get("args", {}), sort_keys=True, ensure_ascii=False)


def build_graph(gateway: ToolGateway, user_id: str):
    assign_agent = AssignAgent(gateway, user_id)
    path_agent = PathAgent(gateway, user_id)
    collab_agent = CollabAgent(gateway, user_id)

    async def understand(state: State) -> dict:
        plan = parse(state["request"], state.get("warehouse_id", "W1"))
        return {
            "intent": plan.intent,
            "slots": plan.slots,
            "missing": plan.missing,
            "trace": state.get("trace", []) + [{"step": "understand", "reason": plan.reason}],
            "steps": state.get("steps", 0) + 1,
        }

    def route_after_understand(state: State) -> str:
        if state.get("missing"):
            return "clarify"
        return "propose"

    async def propose(state: State) -> dict:
        plan = parse(state["request"], state.get("warehouse_id", "W1"))
        trace = state.get("trace", [])
        if state["intent"] == "assign":
            actions = await assign_agent.propose(plan, trace)
            agent_name = "AssignAgent"
        elif state["intent"] == "path":
            actions = await path_agent.propose(plan, trace)
            agent_name = "PathAgent"
        else:
            actions = await collab_agent.propose(plan, trace)
            agent_name = "CollabAgent"

        seen = state.get("seen", [])
        fresh: list[dict] = []
        for action in actions:
            described = action.describe() | {"fallbacks": action.fallbacks}
            if _fingerprint(described) in seen:
                continue
            fresh.append(described)

        # 记下"提了什么"，供评测统计提案采纳率与按代理归因。
        # 注意这是 propose 的产出，不是执行结果 —— 两者分开才能看出代理有没有在空转。
        proposals = list(state.get("proposals", []))
        for described in fresh:
            proposals.append(
                {
                    "agent": agent_name,
                    "tool": described["tool"],
                    "write": bool(described.get("write")),
                    "fp": _fingerprint(described),
                }
            )

        return {
            "pending": fresh,
            "trace": trace,
            "proposals": proposals,
            "steps": state.get("steps", 0) + 1,
        }

    def route_after_propose(state: State) -> str:
        pending = state.get("pending") or []
        if not pending:
            return "finalize"
        if any(action.get("write") for action in pending):
            return "review"
        return "execute"

    async def review(state: State) -> dict:
        pending = state.get("pending") or []
        if state.get("auto_approve", True):
            return {"approved": True, "steps": state.get("steps", 0) + 1}

        decision = interrupt(
            {
                "question": "以下写操作需要确认后执行",
                "actions": [
                    {"summary": action.get("summary"), "tool": action.get("tool")} for action in pending
                ],
            }
        )
        approved = bool(decision.get("approved", False)) if isinstance(decision, dict) else bool(decision)
        return {"approved": approved, "steps": state.get("steps", 0) + 1}

    def route_after_review(state: State) -> str:
        return "execute" if state.get("approved") else "finalize"

    async def execute(state: State) -> dict:
        results = list(state.get("results", []))
        seen = list(state.get("seen", []))
        executed = list(state.get("executed", []))
        steps = state.get("steps", 0)
        consecutive = state.get("consecutive_failures", 0)
        stalled = False
        pending = state.get("pending") or []

        for index, action in enumerate(pending):
            if steps >= MAX_STEPS:
                results.append({"action": action, "result": {"ok": False, "code": "STEP_LIMIT"}})
                break
            fingerprint = _fingerprint(action)
            if fingerprint in seen:
                results.append({"action": action, "result": {"ok": False, "code": "DUPLICATE_ACTION"}})
                continue
            seen.append(fingerprint)

            result = await gateway.call(action["tool"], **action["args"])
            calls_here = 1
            steps += 1
            if not result.get("ok"):
                for fallback in action.get("fallbacks") or []:
                    alternative = await gateway.call(action["tool"], **fallback)
                    calls_here += 1
                    steps += 1
                    if alternative.get("ok"):
                        result = alternative | {"retried": True}
                        break
            results.append({"action": action, "result": result})
            executed.append(
                {
                    "fp": fingerprint,
                    "tool": action["tool"],
                    "ok": bool(result.get("ok")),
                    "code": result.get("code"),
                    "retried": bool(result.get("retried")),
                    # 一个提案可能花掉多次工具调用（主方案 + 备选），
                    # 单独记下来才能算"调用放大倍数"，而不是拿全局计数硬除。
                    "calls": calls_here,
                }
            )

            # 停滞断路器：连续失败到阈值就停手。
            # 继续换参数重试只会把步数烧完，人在这时候介入比 Agent 再猜一次便宜。
            if result.get("ok"):
                consecutive = 0
            else:
                consecutive += 1
                if consecutive >= STALL_LIMIT:
                    stalled = True
                    for rest in pending[index + 1 :]:
                        results.append(
                            {"action": rest, "result": {"ok": False, "code": "STALLED_SKIPPED"}}
                        )
                    break

        return {
            "results": results,
            "pending": [],
            "seen": seen,
            "executed": executed,
            "steps": steps,
            "consecutive_failures": consecutive,
            "stalled": stalled,
            "done": True,
        }

    async def clarify(state: State) -> dict:
        missing = state.get("missing") or []
        hints = {
            "task_id": "任务号（形如 T2002）",
            "robot_id": "机器人编号（形如 R101）",
            "from_node": "起点节点（形如 A1）",
            "to_node": "终点节点（形如 B2）",
            "intent": "你想做什么：查任务 / 派单 / 查路径 / 上报故障",
        }
        missing_text = "、".join(hints.get(key, key) for key in missing)
        return {
            "answer": f"信息不足，请补充：{missing_text}",
            "done": True,
            "steps": state.get("steps", 0) + 1,
        }

    async def finalize(state: State) -> dict:
        if state.get("answer"):
            return {}
        if not state.get("approved", True):
            return {"answer": "写操作未获批准，已取消执行。", "done": True}
        results = state.get("results") or []
        if not results:
            return {"answer": "没有可执行的方案。", "done": True}

        lines: list[str] = []
        for row in results:
            action, result = row["action"], row["result"]
            if result.get("ok"):
                data = result.get("data") or {}
                code = result.get("code")
                if action["tool"] == "assign_task":
                    lines.append(
                        f"任务 {data.get('task_id')} 已分配给 {data.get('robot_id')}"
                        f"（{data.get('from')} → {data.get('to')}）"
                        + ("［幂等重放］" if code == "IDEMPOTENT_REPLAY" else "")
                    )
                elif action["tool"] == "plan_path":
                    lines.append(
                        f"路径 {' → '.join(data.get('path', []))}，"
                        f"约 {data.get('distance_m')} 米 / {data.get('eta_seconds')} 秒"
                    )
                elif action["tool"] == "report_fault":
                    lines.append(
                        f"{data.get('robot_id')} 已置为故障态"
                        + (f"，任务 {data.get('released_task')} 已释放待改派" if data.get("released_task") else "")
                        + ("［幂等重放］" if code == "IDEMPOTENT_REPLAY" else "")
                    )
                elif action["tool"] == "query_tasks":
                    lines.append(f"{data.get('warehouse_id')} 共 {data.get('count')} 条任务")
                else:
                    lines.append(f"{action['tool']} 执行成功")
            else:
                lines.append(f"{action['tool']} 被拒绝：{result.get('code')} {result.get('message', '')}".strip())
        if state.get("stalled"):
            lines.append(
                f"连续 {STALL_LIMIT} 次工具调用失败，已停止重试并转人工处理"
                "（避免把步数烧在同一个错误上）"
            )
        return {"answer": "；".join(lines), "done": True}

    builder = StateGraph(State)
    builder.add_node("understand", understand)
    builder.add_node("propose", propose)
    builder.add_node("review", review)
    builder.add_node("execute", execute)
    builder.add_node("clarify", clarify)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "understand")
    builder.add_conditional_edges(
        "understand", route_after_understand, {"clarify": "clarify", "propose": "propose"}
    )
    builder.add_conditional_edges(
        "propose", route_after_propose, {"review": "review", "execute": "execute", "finalize": "finalize"}
    )
    builder.add_conditional_edges("review", route_after_review, {"execute": "execute", "finalize": "finalize"})
    builder.add_edge("execute", "finalize")
    builder.add_edge("clarify", END)
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=MemorySaver())


def initial_state(request: str, user_id: str, warehouse_id: str = "W1", auto_approve: bool = True) -> dict:
    return {
        "request": request,
        "user_id": user_id,
        "warehouse_id": warehouse_id,
        "auto_approve": auto_approve,
        "trace": [],
        "pending": [],
        "results": [],
        "steps": 0,
        "seen": [],
        "done": False,
        "answer": "",
        "proposals": [],
        "executed": [],
        "consecutive_failures": 0,
        "stalled": False,
    }


def summarize(state: dict) -> dict[str, Any]:
    """只保留对外可见的字段，避免把内部 trace 直接抛给调用方。"""
    proposals = state.get("proposals", [])
    executed = state.get("executed", [])
    return {
        "answer": state.get("answer", ""),
        "intent": state.get("intent"),
        "steps": state.get("steps", 0),
        "n_proposals": len(proposals),
        "n_executed": len(executed),
        "stalled": bool(state.get("stalled")),
        "agents": sorted({row["agent"] for row in proposals}),
        "results": [
            {
                "tool": row["action"]["tool"],
                "ok": row["result"].get("ok"),
                "code": row["result"].get("code"),
            }
            for row in state.get("results", [])
        ],
    }
