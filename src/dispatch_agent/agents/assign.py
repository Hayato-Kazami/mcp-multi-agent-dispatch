"""AssignAgent：单任务分配。

决策顺序刻意做成"硬约束先于软偏好"：
1. 硬约束（工具层会再校验一遍）：空闲、电量、SOH、目标节点容量；
2. 软偏好：电量高优先（避免派完就要充）、距离近优先（ETA 短）。
把硬约束放在 Agent 里先筛一次，是为了少产生注定被拒的调用 ——
但**不能省略工具层的校验**，因为 Agent 可以被绕过。
"""

from __future__ import annotations

from ..agents.action import Action
from ..agents.base import ToolGateway
from ..planner import Plan


class AssignAgent:
    name = "assign"

    def __init__(self, gateway: ToolGateway, user_id: str):
        self.gateway = gateway
        self.user_id = user_id

    async def propose(self, plan: Plan, trace: list[dict]) -> list[Action]:
        warehouse_id = plan.slots["warehouse_id"]
        task_id = plan.slots.get("task_id")

        # 不按 status 过滤：能否分配由工具判（TASK_NOT_ASSIGNABLE），
        # Agent 预判会让"为什么不能派"这个信息消失在编排层里。
        tasks = await self.gateway.call(
            "query_tasks", user_id=self.user_id, warehouse_id=warehouse_id
        )
        trace.append({"tool": "query_tasks", "code": tasks.get("code")})

        if not tasks.get("ok"):
            return []
        pool_tasks = {row["task_id"]: row for row in tasks["data"]["tasks"]}
        if task_id not in pool_tasks:
            return []

        robots = await self.gateway.call(
            "list_robots", user_id=self.user_id, warehouse_id=warehouse_id
        )
        trace.append({"tool": "list_robots", "code": robots.get("code")})
        if not robots.get("ok"):
            return []

        pool = robots["data"]["robots"]
        assignable = [row for row in pool if row["assignable"]]
        preferred = plan.slots.get("robot_id")
        # 用户点名了机器人就以它为先：即使它忙/电量低，也让它走一遍工具校验，
        # 这样拒绝原因是系统给的（ROBOT_BUSY / BATTERY_TOO_LOW），而不是 Agent 猜的。
        if preferred:
            named = next((row for row in pool if row["robot_id"] == preferred), None)
            if named is not None:
                assignable = [named] + [row for row in assignable if row["robot_id"] != preferred]
        if not assignable:
            return []

        candidates = assignable
        task = pool_tasks[task_id]
        ranked = await self._rank_by_distance(warehouse_id, task["from"], candidates, trace)
        if preferred:
            # 点名必须排第一：排序只在"用户没指定"时才有意义
            ranked = [row for row in ranked if row["robot_id"] == preferred] + [
                row for row in ranked if row["robot_id"] != preferred
            ]

        actions: list[Action] = []
        for order, robot in enumerate(ranked[:3]):
            args = {
                "user_id": self.user_id,
                "task_id": task_id,
                "robot_id": robot["robot_id"],
                "idempotency_key": f"{self.user_id}-{task_id}-{robot['robot_id']}",
            }
            if order == 0:
                # 用户点名时不给备选：失败要如实反馈，不能悄悄换成别的机器人
                fallbacks = (
                    []
                    if preferred
                    else [
                        {
                            "user_id": self.user_id,
                            "task_id": task_id,
                            "robot_id": other["robot_id"],
                            "idempotency_key": f"{self.user_id}-{task_id}-{other['robot_id']}",
                        }
                        for other in ranked[1:3]
                    ]
                )
                actions.append(
                    Action(
                        tool="assign_task",
                        args=args,
                        rationale=(
                            f"{robot['robot_id']} 空闲、电量 {robot['battery']}%、"
                            f"距起运点 {robot.get('distance', '?')} 米，是该任务的最优候选"
                        ),
                        write=True,
                        summary=f"把任务 {task_id} 分配给 {robot['robot_id']}",
                        fallbacks=fallbacks,
                    )
                )
        return actions

    async def _rank_by_distance(
        self, warehouse_id: str, from_node: str, candidates: list[dict], trace: list[dict]
    ) -> list[dict]:
        """只对前 3 个候选算距离：路径规划是 O(V^2)，没必要对全池跑。"""
        scored: list[dict] = []
        for robot in candidates[:3]:
            path = await self.gateway.call(
                "plan_path",
                user_id=self.user_id,
                warehouse_id=warehouse_id,
                from_node=robot["node"],
                to_node=from_node,
            )
            trace.append({"tool": "plan_path", "code": path.get("code")})
            distance = path.get("data", {}).get("distance_m", 10**6) if path.get("ok") else 10**6
            scored.append({**robot, "distance": distance})
        for robot in candidates[3:]:
            scored.append({**robot, "distance": 10**6})
        # 先电量档（>=50 视为充足），再距离
        return sorted(scored, key=lambda row: (row["battery"] < 50, row["distance"]))
