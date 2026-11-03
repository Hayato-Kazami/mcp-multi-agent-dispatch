"""内存态业务库：mock 数据 + 幂等表 + 审计日志。

刻意做成"可被工具层直接调用"的同步类，而不是异步 ORM：
这个仓库要证明的是编排与工具边界，不是数据库访问；
换成真库时只要保持这些方法签名不变。
"""

from __future__ import annotations

import heapq
import json
from dataclasses import replace
from pathlib import Path

from .models import ErrorCode, Robot, RobotStatus, Task, TaskStatus, ToolResult, User

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "mock_business.json"

# 规程口径（与 README 的"业务规则"一节一致）
MIN_BATTERY = 20          # 低于 20% 不派单
MIN_SOH = 80              # SOH 低于 80% 退出调度
LANE_CAPACITY = 2         # 同一巷道同时作业上限
CHARGE_THRESHOLD = 25     # 低于 25% 应排队充电


class Store:
    def __init__(self, data_file: Path | None = None):
        raw = json.loads((data_file or DATA_FILE).read_text(encoding="utf-8"))
        self.nodes: dict[str, dict] = raw["nodes"]
        self.edges: dict[str, list[list[str]]] = raw["edges"]
        self.users: dict[str, User] = {
            row["id"]: User(
                id=row["id"],
                name=row["name"],
                role=row["role"],
                warehouses=list(row["warehouses"]),
            )
            for row in raw["users"]
        }
        self.robots: dict[str, Robot] = {
            row["id"]: Robot(
                id=row["id"],
                warehouse_id=row["warehouse_id"],
                battery=row["battery"],
                soh=row["soh"],
                status=RobotStatus(row["status"]),
                node=row["node"],
                current_task=row.get("current_task"),
            )
            for row in raw["robots"]
        }
        self.tasks: dict[str, Task] = {
            row["id"]: Task(
                id=row["id"],
                warehouse_id=row["warehouse_id"],
                priority=row["priority"],
                kind=row["kind"],
                from_node=row["from_node"],
                to_node=row["to_node"],
                status=TaskStatus(row["status"]),
                robot_id=row.get("robot_id"),
                created_at=row.get("created_at", ""),
            )
            for row in raw["tasks"]
        }
        self.idempotency: dict[str, dict] = {}
        self.audit: list[dict] = []

    # ---- 查询 ----
    def get_user(self, user_id: str) -> User | None:
        return self.users.get(user_id)

    def get_robot(self, robot_id: str) -> Robot | None:
        return self.robots.get(robot_id)

    def get_task(self, task_id: str) -> Task | None:
        return self.tasks.get(task_id)

    def list_tasks(self, warehouse_id: str, status: str | None = None) -> list[Task]:
        rows = [t for t in self.tasks.values() if t.warehouse_id == warehouse_id]
        if status:
            rows = [t for t in rows if t.status == status]
        # 调度优先级：数字小的先做，同优先级按创建时间
        return sorted(rows, key=lambda t: (t.priority, t.created_at))

    def units_in_lane(self, warehouse_id: str, node: str) -> int:
        """同一巷道（节点）当前占用数：在执行任务的机器人 + 目标为该节点的机器人。"""
        count = 0
        for robot in self.robots.values():
            if robot.warehouse_id != warehouse_id:
                continue
            if robot.node == node and robot.status == RobotStatus.BUSY:
                count += 1
                continue
            if robot.current_task:
                task = self.tasks.get(robot.current_task)
                if task and task.to_node == node:
                    count += 1
        return count

    # ---- 写入 ----
    def assign_task(self, task: Task, robot: Robot) -> None:
        robot.status = RobotStatus.BUSY
        robot.current_task = task.id
        task.robot_id = robot.id
        task.status = TaskStatus.RUNNING

    def mark_fault(self, robot: Robot, level: int) -> None:
        robot.status = RobotStatus.FAULT
        robot.current_task = None

    def remember(self, key: str, payload: dict) -> None:
        self.idempotency[key] = payload

    def recall(self, key: str) -> dict | None:
        return self.idempotency.get(key)

    def log(self, **kwargs) -> None:
        self.audit.append(kwargs)

    # ---- 路径 ----
    def shortest_path(self, warehouse_id: str, start: str, goal: str) -> tuple[list[str], int]:
        """Dijkstra：边权取欧氏距离，返回节点序列与总长度（取整）。"""
        nodes = self.nodes.get(warehouse_id, {})
        if start not in nodes or goal not in nodes:
            raise KeyError(f"节点不存在: {start} -> {goal}")
        adjacency: dict[str, list[tuple[str, float]]] = {name: [] for name in nodes}
        for left, right in self.edges.get(warehouse_id, []):
            dist = self._dist(nodes[left], nodes[right])
            adjacency[left].append((right, dist))
            adjacency[right].append((left, dist))

        queue: list[tuple[float, str]] = [(0.0, start)]
        best: dict[str, float] = {start: 0.0}
        prev: dict[str, str] = {}
        while queue:
            cost, node = heapq.heappop(queue)
            if node == goal:
                break
            if cost > best.get(node, float("inf")):
                continue
            for neighbor, weight in adjacency[node]:
                new_cost = cost + weight
                if new_cost < best.get(neighbor, float("inf")):
                    best[neighbor] = new_cost
                    prev[neighbor] = node
                    heapq.heappush(queue, (new_cost, neighbor))

        if goal not in best:
            raise KeyError(f"不可达: {start} -> {goal}")
        path = [goal]
        while path[-1] != start:
            path.append(prev[path[-1]])
        path.reverse()
        return path, int(best[goal])

    @staticmethod
    def _dist(left: dict, right: dict) -> float:
        return ((left["x"] - right["x"]) ** 2 + (left["y"] - right["y"]) ** 2) ** 0.5


def validate_idempotency(store: Store, key: str) -> ToolResult | None:
    """幂等键已存在时直接返回首次结果，不重复执行。"""
    if not key or len(key) < 6:
        return ToolResult(False, ErrorCode.BAD_ARGUMENT.value, message="幂等键缺失或过短")
    cached = store.recall(key)
    if cached is not None:
        return ToolResult(True, "IDEMPOTENT_REPLAY", data=cached, message="该幂等键已执行，返回首次结果")
    return None


def clone_robot(robot: Robot) -> Robot:
    return replace(robot)
