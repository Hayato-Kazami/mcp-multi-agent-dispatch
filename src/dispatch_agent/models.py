"""领域模型：只放数据结构与枚举，不放业务规则。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    VIEWER = "viewer"
    DISPATCHER = "dispatcher"
    SUPERVISOR = "supervisor"
    ADMIN = "admin"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    BLOCKED = "blocked"
    DONE = "done"


class RobotStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    FAULT = "fault"
    CHARGING = "charging"


class ErrorCode(str, Enum):
    UNAUTHORIZED = "UNAUTHORIZED"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    NOT_FOUND = "NOT_FOUND"
    ROBOT_BUSY = "ROBOT_BUSY"
    BATTERY_TOO_LOW = "BATTERY_TOO_LOW"
    ROBOT_FAULT = "ROBOT_FAULT"
    LANE_CAPACITY_EXCEEDED = "LANE_CAPACITY_EXCEEDED"
    TASK_NOT_ASSIGNABLE = "TASK_NOT_ASSIGNABLE"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    BAD_ARGUMENT = "BAD_ARGUMENT"


@dataclass
class User:
    id: str
    name: str
    role: Role
    warehouses: list[str] = field(default_factory=list)


@dataclass
class Robot:
    id: str
    warehouse_id: str
    battery: int
    soh: int
    status: RobotStatus
    node: str
    current_task: str | None = None


@dataclass
class Task:
    id: str
    warehouse_id: str
    priority: int
    kind: str
    from_node: str
    to_node: str
    status: TaskStatus
    robot_id: str | None = None
    created_at: str = ""


@dataclass
class ToolResult:
    """所有工具统一的返回结构：成功也给 code，便于上游按码分支。"""

    ok: bool
    code: str
    data: dict | None = None
    message: str = ""

    def to_dict(self) -> dict:
        payload: dict = {"ok": self.ok, "code": self.code}
        if self.data is not None:
            payload["data"] = self.data
        if self.message:
            payload["message"] = self.message
        return payload
