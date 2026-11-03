"""MCP 工具层：5 个工具 + 一条明确的执行边界。

三条设计原则（面试可以直接讲）：
1. **工具是唯一能改状态的入口**。Agent 只能"提议"，不能直接写库；
2. **授权在工具里做**，Agent 层不做权限判断——模型被诱导也越不过去；
3. **校验顺序固定**：鉴权 → 幂等 → 资源存在 → 业务规则 → 执行。
   顺序错了会出现"未授权用户通过重复提交探测业务状态"这类问题。
"""

from __future__ import annotations

from fastmcp import FastMCP

from .auth import authorize
from .models import ErrorCode, RobotStatus, TaskStatus, ToolResult
from .store import CHARGE_THRESHOLD, LANE_CAPACITY, MIN_BATTERY, MIN_SOH, Store, validate_idempotency

mcp = FastMCP("dispatch-tools")

# 进程内单例：stdio 传输下每个 server 进程一份，够用且便于演示
STORE = Store()

ROBOT_SPEED_MPS = 1.2   # 机器人平均行驶速度
TURN_PENALTY_S = 4      # 每次转向的额外耗时


# ---------------------------------------------------------------- 实现层
def _query_tasks(store: Store, user_id: str, warehouse_id: str, status: str | None = None) -> ToolResult:
    denied = authorize(store, user_id, "query_tasks", warehouse_id)
    if denied:
        return denied
    rows = store.list_tasks(warehouse_id, status)
    return ToolResult(
        True,
        "OK",
        data={
            "warehouse_id": warehouse_id,
            "count": len(rows),
            "tasks": [
                {
                    "task_id": t.id,
                    "priority": t.priority,
                    "kind": t.kind,
                    "from": t.from_node,
                    "to": t.to_node,
                    "status": t.status.value,
                    "robot_id": t.robot_id,
                }
                for t in rows
            ],
        },
    )


def _list_robots(store: Store, user_id: str, warehouse_id: str, status: str | None = None) -> ToolResult:
    denied = authorize(store, user_id, "list_robots", warehouse_id)
    if denied:
        return denied
    rows = [
        robot
        for robot in store.robots.values()
        if robot.warehouse_id == warehouse_id and (status is None or robot.status == status)
    ]
    rows.sort(key=lambda r: (-r.battery, r.id))
    return ToolResult(
        True,
        "OK",
        data={
            "warehouse_id": warehouse_id,
            "count": len(rows),
            "robots": [
                {
                    "robot_id": r.id,
                    "battery": r.battery,
                    "soh": r.soh,
                    "status": r.status.value,
                    "node": r.node,
                    "current_task": r.current_task,
                    "assignable": r.status == RobotStatus.IDLE
                    and r.battery >= MIN_BATTERY
                    and r.soh >= MIN_SOH,
                }
                for r in rows
            ],
        },
    )


def _get_battery(store: Store, user_id: str, robot_id: str) -> ToolResult:
    robot = store.get_robot(robot_id)
    # 机器人自带仓库信息，先取到再鉴权，避免"仓库参数可伪造"
    warehouse_id = robot.warehouse_id if robot else None
    denied = authorize(store, user_id, "get_battery", warehouse_id)
    if denied:
        return denied
    if robot is None:
        return ToolResult(False, ErrorCode.NOT_FOUND.value, message=f"机器人不存在 {robot_id}")
    return ToolResult(
        True,
        "OK",
        data={
            "robot_id": robot.id,
            "warehouse_id": robot.warehouse_id,
            "battery": robot.battery,
            "soh": robot.soh,
            "status": robot.status.value,
            "node": robot.node,
            "need_charge": robot.battery < CHARGE_THRESHOLD,
            "assignable": robot.status == RobotStatus.IDLE
            and robot.battery >= MIN_BATTERY
            and robot.soh >= MIN_SOH,
        },
    )


def _plan_path(store: Store, user_id: str, warehouse_id: str, from_node: str, to_node: str) -> ToolResult:
    denied = authorize(store, user_id, "plan_path", warehouse_id)
    if denied:
        return denied
    try:
        path, distance = store.shortest_path(warehouse_id, from_node, to_node)
    except KeyError as exc:
        return ToolResult(False, ErrorCode.NOT_FOUND.value, message=str(exc))
    turns = sum(
        1
        for i in range(1, len(path) - 1)
        if _direction(store, warehouse_id, path[i - 1], path[i])
        != _direction(store, warehouse_id, path[i], path[i + 1])
    )
    eta = round(distance / ROBOT_SPEED_MPS + turns * TURN_PENALTY_S, 1)
    return ToolResult(
        True,
        "OK",
        data={
            "path": path,
            "distance_m": distance,
            "turns": turns,
            "eta_seconds": eta,
        },
    )


def _direction(store: Store, warehouse_id: str, left: str, right: str) -> tuple[int, int]:
    nodes = store.nodes[warehouse_id]
    return (nodes[right]["x"] - nodes[left]["x"], nodes[right]["y"] - nodes[left]["y"])


def _assign_task(
    store: Store,
    user_id: str,
    task_id: str,
    robot_id: str,
    idempotency_key: str,
) -> ToolResult:
    task = store.get_task(task_id)
    denied = authorize(store, user_id, "assign_task", task.warehouse_id if task else None)
    if denied:
        return denied

    replay = validate_idempotency(store, idempotency_key)
    if replay:
        return replay

    if task is None:
        return ToolResult(False, ErrorCode.NOT_FOUND.value, message=f"任务不存在 {task_id}")
    if task.status not in {TaskStatus.QUEUED, TaskStatus.BLOCKED}:
        return ToolResult(
            False,
            ErrorCode.TASK_NOT_ASSIGNABLE.value,
            message=f"任务 {task.id} 当前状态 {task.status.value}，不可分配",
        )

    robot = store.get_robot(robot_id)
    if robot is None:
        return ToolResult(False, ErrorCode.NOT_FOUND.value, message=f"机器人不存在 {robot_id}")
    if robot.warehouse_id != task.warehouse_id:
        return ToolResult(
            False,
            ErrorCode.OUT_OF_SCOPE.value,
            message=f"机器人 {robot.id} 与任务不在同一仓库",
        )
    if robot.status == RobotStatus.FAULT:
        return ToolResult(False, ErrorCode.ROBOT_FAULT.value, message=f"机器人 {robot.id} 处于故障态")
    if robot.status == RobotStatus.BUSY:
        return ToolResult(
            False,
            ErrorCode.ROBOT_BUSY.value,
            message=f"机器人 {robot.id} 正在执行 {robot.current_task}",
        )
    if robot.battery < MIN_BATTERY:
        return ToolResult(
            False,
            ErrorCode.BATTERY_TOO_LOW.value,
            message=f"机器人 {robot.id} 电量 {robot.battery}%，低于 {MIN_BATTERY}% 不派单，建议先充电",
        )
    if robot.soh < MIN_SOH:
        return ToolResult(
            False,
            ErrorCode.ROBOT_FAULT.value,
            message=f"机器人 {robot.id} 电池健康度 {robot.soh}%，低于 {MIN_SOH}% 已退出调度",
        )
    occupied = store.units_in_lane(task.warehouse_id, task.to_node)
    if occupied >= LANE_CAPACITY:
        return ToolResult(
            False,
            ErrorCode.LANE_CAPACITY_EXCEEDED.value,
            message=f"目标节点 {task.to_node} 已有 {occupied} 台，达到巷道容量上限 {LANE_CAPACITY}",
        )

    store.assign_task(task, robot)
    payload = {
        "task_id": task.id,
        "robot_id": robot.id,
        "status": task.status.value,
        "from": task.from_node,
        "to": task.to_node,
    }
    store.remember(idempotency_key, payload)
    store.log(
        action="assign_task",
        user_id=user_id,
        task_id=task.id,
        robot_id=robot.id,
        idempotency_key=idempotency_key,
        result="ok",
    )
    return ToolResult(True, "OK", data=payload)


def _report_fault(
    store: Store,
    user_id: str,
    robot_id: str,
    level: int,
    idempotency_key: str,
) -> ToolResult:
    robot = store.get_robot(robot_id)
    denied = authorize(store, user_id, "report_fault", robot.warehouse_id if robot else None)
    if denied:
        return denied

    replay = validate_idempotency(store, idempotency_key)
    if replay:
        return replay

    if robot is None:
        return ToolResult(False, ErrorCode.NOT_FOUND.value, message=f"机器人不存在 {robot_id}")
    if level not in {1, 2, 3}:
        return ToolResult(False, ErrorCode.BAD_ARGUMENT.value, message="故障等级只能是 1/2/3")

    released_task = robot.current_task
    store.mark_fault(robot, level)
    if released_task:
        task = store.get_task(released_task)
        if task:
            task.status = TaskStatus.BLOCKED
            task.robot_id = None

    payload = {
        "robot_id": robot.id,
        "level": level,
        "released_task": released_task,
        "need_manual": level == 1,
        "need_reassign": bool(released_task),
    }
    store.remember(idempotency_key, payload)
    store.log(
        action="report_fault",
        user_id=user_id,
        robot_id=robot.id,
        level=level,
        idempotency_key=idempotency_key,
        result="ok",
    )
    return ToolResult(True, "OK", data=payload)


# ---------------------------------------------------------------- MCP 外观
@mcp.tool
def query_tasks(user_id: str, warehouse_id: str, status: str | None = None) -> dict:
    """按仓库查询任务列表。status 可选 queued/running/blocked/done，不传则返回全部。

    返回按优先级升序排列（数字越小越先执行）。只读工具。
    """
    return _query_tasks(STORE, user_id, warehouse_id, status).to_dict()


@mcp.tool
def list_robots(user_id: str, warehouse_id: str, status: str | None = None) -> dict:
    """列出指定仓库的机器人及其电量、健康度、状态与所在节点。

    status 可选 idle/busy/fault/charging。返回按电量降序，assignable 字段
    直接给出"当前是否可以派单"（空闲 + 电量 >= 20% + SOH >= 80%）。只读工具。
    """
    return _list_robots(STORE, user_id, warehouse_id, status).to_dict()


@mcp.tool
def get_battery(user_id: str, robot_id: str) -> dict:
    """查询指定机器人的电量、电池健康度、当前状态与所在节点，并给出是否可派单。

    只读工具。返回 assignable=false 时不要尝试分配任务，先看 reason 字段。
    """
    return _get_battery(STORE, user_id, robot_id).to_dict()


@mcp.tool
def plan_path(user_id: str, warehouse_id: str, from_node: str, to_node: str) -> dict:
    """计算机器人从 from_node 到 to_node 的最短路径、距离、转向次数与预计耗时。

    只读工具。节点名形如 A1/B2/C1，节点不存在会返回 NOT_FOUND。
    """
    return _plan_path(STORE, user_id, warehouse_id, from_node, to_node).to_dict()


@mcp.tool
def assign_task(user_id: str, task_id: str, robot_id: str, idempotency_key: str) -> dict:
    """把排队中的任务分配给一台空闲机器人。**写操作**，需要 dispatcher 及以上角色。

    idempotency_key 由调用方生成（建议「用户ID-任务ID-时间戳」），
    同一个 key 重复提交不会重复执行，而是返回首次结果。
    可能返回的拒绝码：ROBOT_BUSY / BATTERY_TOO_LOW / ROBOT_FAULT /
    LANE_CAPACITY_EXCEEDED / TASK_NOT_ASSIGNABLE。
    """
    return _assign_task(STORE, user_id, task_id, robot_id, idempotency_key).to_dict()


@mcp.tool
def report_fault(user_id: str, robot_id: str, level: int, idempotency_key: str) -> dict:
    """上报机器人故障。**写操作**，level 取 1（阻断通行）/2（降效）/3（可自恢复）。

    上报后机器人转为故障态，在途任务释放为 blocked 并返回 released_task，
    由上游决定是否改派。同 idempotency_key 重复提交返回首次结果。
    """
    return _report_fault(STORE, user_id, robot_id, level, idempotency_key).to_dict()


# 供单元测试与无 MCP 环境直调（不经过协议层）
DIRECT_CALLS = {
    "query_tasks": _query_tasks,
    "list_robots": _list_robots,
    "get_battery": _get_battery,
    "plan_path": _plan_path,
    "assign_task": _assign_task,
    "report_fault": _report_fault,
}


if __name__ == "__main__":
    mcp.run()
