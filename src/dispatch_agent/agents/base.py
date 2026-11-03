"""工具网关：Agent 与工具之间只认这一层。

两种实现刻意都保留：
- McpGateway   ：真走 MCP 协议（in-memory transport），验证 schema、序列化、错误结构；
- DirectGateway：直调实现函数，用于单元测试与无 MCP 环境的降级。

面试里被问"你怎么证明真用了 MCP 而不是普通函数调用"，
答案就是评测默认跑 McpGateway，DirectGateway 只在测试里出现。
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from ..models import ErrorCode, ToolResult
from ..store import Store
from ..tools import DIRECT_CALLS


class ToolGateway(Protocol):
    async def call(self, name: str, **kwargs: Any) -> dict: ...


def _unwrap(result: Any) -> dict:
    """把 MCP 的 CallToolResult 归一化成工具返回的 dict。

    优先取 structured content（.data），退化时解析第一个文本块。
    """
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"ok": False, "code": ErrorCode.BAD_ARGUMENT.value, "message": text}
    return {"ok": False, "code": ErrorCode.BAD_ARGUMENT.value, "message": "工具返回无法解析"}


class DirectGateway:
    name = "direct"

    def __init__(self, store: Store):
        self.store = store

    async def call(self, name: str, **kwargs: Any) -> dict:
        func = DIRECT_CALLS.get(name)
        if func is None:
            return ToolResult(False, "UNKNOWN_TOOL", message=f"未知工具 {name}").to_dict()
        return func(self.store, **kwargs).to_dict()


class McpGateway:
    name = "mcp"

    def __init__(self, server):
        self.server = server
        self._client = None
        self._context = None
        self.calls = 0

    async def connect(self) -> "McpGateway":
        from fastmcp import Client

        self._context = Client(self.server)
        self._client = await self._context.__aenter__()
        return self

    async def close(self) -> None:
        if self._context is not None:
            await self._context.__aexit__(None, None, None)
            self._context = None
            self._client = None

    async def call(self, name: str, **kwargs: Any) -> dict:
        if self._client is None:
            await self.connect()
        self.calls += 1
        result = await self._client.call_tool(name, kwargs)
        return _unwrap(result)

    async def list_tools(self) -> list[dict]:
        if self._client is None:
            await self.connect()
        tools = await self._client.list_tools()
        return [
            {"name": tool.name, "description": (tool.description or "").strip().splitlines()[0]}
            for tool in tools
        ]
