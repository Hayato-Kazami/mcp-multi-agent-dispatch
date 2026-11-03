"""工具层测试：用 DirectGateway 直调，不启动 MCP，执行时间在毫秒级。

这里测的是"规则本身"，协议层由 scripts/eval_graph.py 覆盖。
两者分开的目的是：改业务规则时跑单测，改协议/编排时跑端到端。
"""

from __future__ import annotations

import asyncio

import pytest

from dispatch_agent.agents.base import DirectGateway
from dispatch_agent.planner import parse
from dispatch_agent.store import Store


@pytest.fixture()
def gateway():
    return DirectGateway(Store())


def call(gateway: DirectGateway, name: str, **kwargs) -> dict:
    return asyncio.run(gateway.call(name, **kwargs))


def test_tool_count_and_names():
    from dispatch_agent.tools import DIRECT_CALLS

    assert set(DIRECT_CALLS) == {
        "query_tasks",
        "list_robots",
        "get_battery",
        "plan_path",
        "assign_task",
        "report_fault",
    }


def test_viewer_cannot_assign(gateway):
    result = call(gateway, "assign_task", user_id="U004", task_id="T2003",
                  robot_id="R105", idempotency_key="k-viewer-1")
    assert result["ok"] is False
    assert result["code"] == "UNAUTHORIZED"


def test_warehouse_scope_isolation(gateway):
    result = call(gateway, "query_tasks", user_id="U001", warehouse_id="W2")
    assert result["code"] == "OUT_OF_SCOPE"


def test_low_battery_rejected(gateway):
    result = call(gateway, "assign_task", user_id="U001", task_id="T2003",
                  robot_id="R103", idempotency_key="k-battery-1")
    assert result["code"] == "BATTERY_TOO_LOW"


def test_low_soh_rejected(gateway):
    # R106 SOH=79 且处于故障态，先验证故障码优先
    result = call(gateway, "assign_task", user_id="U001", task_id="T2003",
                  robot_id="R106", idempotency_key="k-soh-1")
    assert result["code"] == "ROBOT_FAULT"


def test_assign_then_idempotent_replay(gateway):
    first = call(gateway, "assign_task", user_id="U001", task_id="T2002",
                 robot_id="R101", idempotency_key="k-idem-1")
    assert first["ok"] is True and first["code"] == "OK"
    second = call(gateway, "assign_task", user_id="U001", task_id="T2002",
                  robot_id="R101", idempotency_key="k-idem-1")
    assert second["code"] == "IDEMPOTENT_REPLAY"
    assert second["data"]["task_id"] == "T2002"


def test_path_is_connected(gateway):
    result = call(gateway, "plan_path", user_id="U001", warehouse_id="W1",
                  from_node="A1", to_node="D1")
    assert result["ok"] is True
    path = result["data"]["path"]
    assert path[0] == "A1" and path[-1] == "D1"
    assert result["data"]["distance_m"] > 0


def test_fault_releases_running_task(gateway):
    result = call(gateway, "report_fault", user_id="U002", robot_id="R102",
                  level=1, idempotency_key="k-fault-1")
    assert result["ok"] is True
    assert result["data"]["released_task"] == "T2001"
    assert result["data"]["need_manual"] is True


def test_planner_parses_slots():
    plan = parse("把任务 T2003 派给机器人 R105", default_warehouse="W1")
    assert plan.intent == "assign"
    assert plan.slots["task_id"] == "T2003"
    assert plan.slots["robot_id"] == "R105"
    assert plan.write is True


def test_planner_clarifies_when_missing():
    plan = parse("帮我派个单", default_warehouse="W1")
    assert plan.intent == "assign"
    assert "task_id" in plan.missing


def test_planner_path_intent_needs_two_nodes():
    plan = parse("从 A1 到 D1 怎么走", default_warehouse="W1")
    assert plan.intent == "path"
    assert plan.slots["from_node"] == "A1"
    assert plan.slots["to_node"] == "D1"
    assert plan.missing == []
