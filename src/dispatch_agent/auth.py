"""权限：工具级 + 仓库级双层校验。

真实系统里用户身份来自 JWT，工具层从请求上下文取；
这里为了可离线演示，把 user_id 作为显式参数传入 —— 边界是一样的：
**授权在工具层做，不在 Agent 层做**。Agent 可以建议任何操作，
但工具自己决定执不执行，这样模型被投毒也改不了数据。
"""

from __future__ import annotations

from .models import ErrorCode, Role, ToolResult
from .store import Store

READ_TOOLS = {"query_tasks", "list_robots", "plan_path", "get_battery"}
WRITE_TOOLS = {"assign_task", "report_fault"}

ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.VIEWER: READ_TOOLS,
    Role.DISPATCHER: READ_TOOLS | WRITE_TOOLS,
    Role.SUPERVISOR: READ_TOOLS | WRITE_TOOLS,
    Role.ADMIN: READ_TOOLS | WRITE_TOOLS,
}


def authorize(
    store: Store,
    user_id: str,
    tool_name: str,
    warehouse_id: str | None = None,
) -> ToolResult | None:
    """返回 None 表示通过；否则返回拒绝结果。"""
    user = store.get_user(user_id)
    if user is None:
        return ToolResult(False, ErrorCode.UNAUTHORIZED.value, message=f"未知用户 {user_id}")

    allowed = ROLE_PERMISSIONS.get(Role(user.role), set())
    if tool_name not in allowed:
        return ToolResult(
            False,
            ErrorCode.UNAUTHORIZED.value,
            message=f"角色 {user.role} 无权调用 {tool_name}",
        )

    if warehouse_id and warehouse_id not in user.warehouses:
        return ToolResult(
            False,
            ErrorCode.OUT_OF_SCOPE.value,
            message=f"用户 {user.id} 无仓库 {warehouse_id} 的权限",
        )
    return None
