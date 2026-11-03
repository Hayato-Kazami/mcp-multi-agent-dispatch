"""CollabAgent：多任务协同与异常处置。

它做的是"批量决策"，难点不在调工具，而在**自己别把系统搞冲突**：
一次派三单，如果不检查同一台机器人被派两次、同一个目标节点超过容量，
就会制造出一批注定被工具拒绝的调用 —— 看起来"系统在运转"，实际在空转。
"""

from __future__ import annotations

from ..agents.action import Action
from ..agents.assign import AssignAgent
from ..agents.base import ToolGateway
from ..planner import Plan

MAX_BATCH = 3
TARGET_CAPACITY = 1  # 批内保守值：同一目标节点一次只派 1 台，避免自造冲突


class CollabAgent:
    name = "collab"

    def __init__(self, gateway: ToolGateway, user_id: str):
        self.gateway = gateway
        self.user_id = user_id
        self.assign = AssignAgent(gateway, user_id)

    async def propose(self, plan: Plan, trace: list[dict]) -> list[Action]:
        if plan.intent == "fault":
            return await self._fault_actions(plan, trace)
        if plan.intent == "query":
            return await self._query_actions(plan, trace)
        return await self._batch_assign(plan, trace)

    async def _fault_actions(self, plan: Plan, trace: list[dict]) -> list[Action]:
        robot_id = plan.slots["robot_id"]
        level = plan.slots.get("level", 2)
        action = Action(
            tool="report_fault",
            args={
                "user_id": self.user_id,
                "robot_id": robot_id,
                "level": level,
                "idempotency_key": f"{self.user_id}-fault-{robot_id}-{level}",
            },
            rationale=f"用户上报 {robot_id} 故障，等级 {level}",
            write=True,
            summary=f"上报 {robot_id} 故障（{level} 级），在途任务将释放为待改派",
        )
        return [action]

    async def _query_actions(self, plan: Plan, trace: list[dict]) -> list[Action]:
        warehouse_id = plan.slots["warehouse_id"]
        action = Action(
            tool="query_tasks",
            args={"user_id": self.user_id, "warehouse_id": warehouse_id},
            rationale="用户查询任务队列",
            write=False,
            summary=f"查询 {warehouse_id} 的任务队列",
        )
        return [action]

    async def _batch_assign(self, plan: Plan, trace: list[dict]) -> list[Action]:
        warehouse_id = plan.slots["warehouse_id"]
        tasks = await self.gateway.call(
            "query_tasks", user_id=self.user_id, warehouse_id=warehouse_id, status="queued"
        )
        trace.append({"tool": "query_tasks", "code": tasks.get("code")})
        if not tasks.get("ok"):
            return []

        queue = tasks["data"]["tasks"]
        occupied_targets: dict[str, int] = {}
        used_robots: set[str] = set()
        actions: list[Action] = []

        for task in queue:
            if len(actions) >= MAX_BATCH:
                break
            target = task["to"]
            if occupied_targets.get(target, 0) >= TARGET_CAPACITY:
                continue

            sub_plan = Plan(
                intent="assign",
                slots={"warehouse_id": warehouse_id, "task_id": task["task_id"]},
            )
            candidates = await self.assign.propose(sub_plan, trace)
            if not candidates:
                continue
            action = candidates[0]

            if action.args["robot_id"] in used_robots:
                replacement = next(
                    (fb for fb in action.fallbacks if fb["robot_id"] not in used_robots), None
                )
                if replacement is None:
                    continue
                action.args = replacement
                action.fallbacks = []

            used_robots.add(action.args["robot_id"])
            occupied_targets[target] = occupied_targets.get(target, 0) + 1
            actions.append(action)
        return actions
