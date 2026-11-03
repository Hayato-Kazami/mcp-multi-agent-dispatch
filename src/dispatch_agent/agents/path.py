"""PathAgent：只读的路径与 ETA。

单独拆出来的理由：它没有副作用，是唯一可以在"审批前"就安全的代理。
真实系统里这类只读代理会被高频调用（调度员随口问"多久到"），
把它和写操作代理分开，才能给不同的超时与限流策略。
"""

from __future__ import annotations

from ..agents.action import Action
from ..agents.base import ToolGateway
from ..planner import Plan


class PathAgent:
    name = "path"

    def __init__(self, gateway: ToolGateway, user_id: str):
        self.gateway = gateway
        self.user_id = user_id

    async def propose(self, plan: Plan, trace: list[dict]) -> list[Action]:
        warehouse_id = plan.slots["warehouse_id"]
        from_node = plan.slots["from_node"]
        to_node = plan.slots["to_node"]
        action = Action(
            tool="plan_path",
            args={
                "user_id": self.user_id,
                "warehouse_id": warehouse_id,
                "from_node": from_node,
                "to_node": to_node,
            },
            rationale=f"用户询问 {from_node} 到 {to_node} 的路线",
            write=False,
            summary=f"规划 {from_node} → {to_node} 的路径",
        )
        return [action]
