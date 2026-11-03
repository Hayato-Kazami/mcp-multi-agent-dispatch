"""端到端评测：12 条调度指令，走真 MCP 协议 + 真 LangGraph。

为什么不做成"调一次工具算通过"：
那只能证明工具没写错，证明不了编排对不对。这里的断言落在
**最终回答与每个工具的返回码**上，覆盖四类容易翻车的地方：
越权、状态冲突、信息不足、重复提交。

每条用例前重置内存库，避免用例之间互相污染（这类测试最常见的假通过来源）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dispatch_agent.agents.base import McpGateway
from dispatch_agent.graph import build_graph, initial_state
from dispatch_agent.store import Store
import dispatch_agent.tools as tools

CASES = [
    {
        "id": "c01",
        "request": "查一下 W1 排队的任务",
        "user": "U001",
        "warehouse": "W1",
        "expect": {"intent": "query", "codes": [("query_tasks", "OK")]},
        "why": "只读查询，不应触发审批",
    },
    {
        "id": "c02",
        "request": "把任务 T2002 派给机器人",
        "user": "U001",
        "warehouse": "W1",
        "expect": {"intent": "assign", "codes": [("assign_task", "OK")], "contains": ["已分配给"]},
        "why": "正常派单，Agent 自行选机器人",
    },
    {
        "id": "c03",
        "request": "把任务 T2003 派给机器人 R103",
        "user": "U001",
        "warehouse": "W1",
        "expect": {"codes": [("assign_task", "BATTERY_TOO_LOW")]},
        "why": "点名低电量机器人（18%），必须拒绝而不是悄悄换人",
    },
    {
        "id": "c04",
        "request": "把任务 T2003 派给机器人 R106",
        "user": "U001",
        "warehouse": "W1",
        "expect": {"codes": [("assign_task", "ROBOT_FAULT")]},
        "why": "故障机器人不可派单",
    },
    {
        "id": "c05",
        "request": "把任务 T2001 派给机器人 R105",
        "user": "U001",
        "warehouse": "W1",
        "expect": {"codes": [("assign_task", "TASK_NOT_ASSIGNABLE")]},
        "why": "T2001 已在执行中，不可重复分配",
    },
    {
        "id": "c06",
        "request": "把任务 T2003 派给机器人 R102",
        "user": "U001",
        "warehouse": "W1",
        "expect": {"codes": [("assign_task", "ROBOT_BUSY")]},
        "why": "忙碌机器人不可派单",
    },
    {
        "id": "c07",
        "request": "把任务 T2003 派给机器人 R105",
        "user": "U004",
        "warehouse": "W1",
        "expect": {"codes": [("assign_task", "UNAUTHORIZED")]},
        "why": "viewer 角色无写权限",
    },
    {
        "id": "c08",
        "request": "查一下 W2 的任务",
        "user": "U001",
        "warehouse": "W1",
        "expect": {"codes": [("query_tasks", "OUT_OF_SCOPE")]},
        "why": "仓库级数据隔离：U001 只有 W1",
    },
    {
        "id": "c09",
        "request": "从 A1 到 D1 怎么走，要多久",
        "user": "U001",
        "warehouse": "W1",
        "expect": {"intent": "path", "codes": [("plan_path", "OK")], "contains": ["米"]},
        "why": "只读路径规划",
    },
    {
        "id": "c10",
        "request": "R102 一级故障，马上处理",
        "user": "U002",
        "warehouse": "W1",
        "expect": {"codes": [("report_fault", "OK")], "contains": ["故障态"]},
        "why": "故障上报要释放在途任务",
    },
    {
        "id": "c11",
        "request": "帮我派个单",
        "user": "U001",
        "warehouse": "W1",
        "expect": {"clarify": True},
        "why": "缺少任务号，必须澄清而不是猜",
    },
    {
        "id": "c12",
        "request": "把 W1 排队的任务一起安排掉",
        "user": "U001",
        "warehouse": "W1",
        "expect": {"intent": "collab", "min_assigns": 2},
        "why": "批量协同：一次派多单，且不能把同一台机器人派两次",
    },
]


def _codes(result: dict) -> list[tuple[str, str]]:
    return [
        (row["action"]["tool"], row["result"].get("code") or "")
        for row in result.get("results", [])
    ]


def _check(case: dict, result: dict) -> tuple[bool, list[str]]:
    expect = case["expect"]
    notes: list[str] = []
    ok = True

    if "intent" in expect:
        actual = result.get("intent")
        if actual != expect["intent"]:
            ok = False
            notes.append(f"intent={actual} 期望 {expect['intent']}")

    for tool_name, code in expect.get("codes", []):
        codes = _codes(result)
        if (tool_name, code) not in codes:
            ok = False
            notes.append(f"{tool_name} 期望 {code}，实际 {codes}")

    for keyword in expect.get("contains", []):
        if keyword not in (result.get("answer") or ""):
            ok = False
            notes.append(f"回答缺少「{keyword}」")

    if expect.get("clarify") and "请补充" not in (result.get("answer") or ""):
        ok = False
        notes.append("应当澄清但没有")

    if "min_assigns" in expect:
        assigns = [row for row in result.get("results", []) if row["action"]["tool"] == "assign_task"]
        success = [row for row in assigns if row["result"].get("ok")]
        if len(success) < expect["min_assigns"]:
            ok = False
            notes.append(f"成功派单 {len(success)} 条，期望至少 {expect['min_assigns']}")
        robots = [row["action"]["args"].get("robot_id") for row in success]
        if len(robots) != len(set(robots)):
            ok = False
            notes.append(f"同一台机器人被派了两次: {robots}")

    return ok, notes


async def _run_case(gateway: McpGateway, case: dict) -> dict:
    tools.STORE = Store()  # 每条用例重置内存库
    graph = build_graph(gateway, case["user"])
    config = {"configurable": {"thread_id": case["id"]}}
    state = initial_state(case["request"], case["user"], case["warehouse"], auto_approve=True)
    result = await graph.ainvoke(state, config)
    ok, notes = _check(case, result)
    return {
        "id": case["id"],
        "request": case["request"],
        "why": case["why"],
        "passed": ok,
        "notes": notes,
        "answer": result.get("answer"),
        "steps": result.get("steps"),
        "codes": _codes(result),
        "trace_steps": len(result.get("trace", [])),
    }


async def _run_idempotent(gateway: McpGateway) -> dict:
    """同一条派单指令连跑两次，第二次应命中幂等重放。"""
    tools.STORE = Store()
    graph = build_graph(gateway, "U001")
    request = "把任务 T2002 派给机器人 R101"
    first = await graph.ainvoke(
        initial_state(request, "U001", "W1", auto_approve=True),
        {"configurable": {"thread_id": "c13-a"}},
    )
    second = await graph.ainvoke(
        initial_state(request, "U001", "W1", auto_approve=True),
        {"configurable": {"thread_id": "c13-b"}},
    )
    codes = _codes(second)
    replay = any(code == "IDEMPOTENT_REPLAY" for _, code in codes)
    return {
        "id": "c13",
        "request": request + "（重复提交）",
        "why": "同一幂等键重复提交不得重复执行",
        "passed": replay,
        "notes": [] if replay else [f"未命中幂等重放: {codes}"],
        "answer": second.get("answer"),
        "steps": second.get("steps"),
        "codes": codes,
        "first_codes": _codes(first),
    }


async def main_async(args: argparse.Namespace) -> int:
    gateway = McpGateway(tools.mcp)
    await gateway.connect()

    rows = []
    for case in CASES:
        rows.append(await _run_case(gateway, case))
    rows.append(await _run_idempotent(gateway))
    await gateway.close()

    passed = sum(1 for row in rows if row["passed"])
    steps = [row["steps"] for row in rows if row["steps"]]
    unauthorized = [row for row in rows if row["id"] in {"c07", "c08"}]
    clarify = [row for row in rows if row["id"] == "c11"]

    report = {
        "transport": "mcp",
        "n_cases": len(rows),
        "passed": passed,
        "pass_rate": round(passed / len(rows), 4),
        "avg_steps": round(sum(steps) / len(steps), 2),
        "unauthorized_blocked": f"{sum(1 for r in unauthorized if r['passed'])}/{len(unauthorized)}",
        "clarify_correct": f"{sum(1 for r in clarify if r['passed'])}/{len(clarify)}",
        "tool_calls": gateway.calls,
        "cases": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n用例数 {report['n_cases']}  通过 {passed}  通过率 {report['pass_rate']:.3f}  "
          f"平均步数 {report['avg_steps']}  工具调用 {report['tool_calls']} 次")
    print(f"越权拦截 {report['unauthorized_blocked']}   澄清正确 {report['clarify_correct']}\n")
    print(f"{'id':<5}{'P/F':<6}{'steps':<7}request")
    print("-" * 70)
    for row in rows:
        flag = "PASS" if row["passed"] else "FAIL"
        print(f"{row['id']:<5}{flag:<6}{str(row['steps']):<7}{row['request']}")
        for note in row["notes"]:
            print(f"        ↳ {note}")
    print(f"\n报告已写入 {out}")
    return 0 if passed == len(rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/eval_report.json")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
